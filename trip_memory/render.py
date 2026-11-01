"""Turn a list of videos and photos into one finished memory video with FFmpeg.

Every item is first converted into a short "segment" with the same size, frame rate and
sound format. The segments are then joined with transitions, an optional music track is
mixed in, and the result is saved as one MP4.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import shutil
import sys
import tempfile
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont, ImageOps

from . import titles
from .looks import look_filter
from .media import MediaItem
from .transitions import MIXED, STYLES, ZOOM_INTO, ZOOM_OUT_OF, seconds_of, xfade_args
from .tools import FFmpegError, has_encoder, has_filter, probe, run_ffmpeg

# Friendly name -> FFmpeg xfade transition. None means a hard cut.
TRANSITIONS: dict[str, str | None] = {name: style.xfade for name, style in STYLES.items()}

FPS = 30
AUDIO = ["-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-ac", "2"]

ProgressFn = Callable[[float, str], None]

# Bump when the way clips are prepared changes, so old remembered clips are not reused.
SEGMENT_VERSION = 9
OVERLAP_SHARE = 0.5  # a transition takes at most half of either clip around it
TITLE_SECONDS = 3.0
FONTS = [
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    "/System/Library/Fonts/Helvetica.ttc",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
]
CACHE_DAYS = 30
_hardware = {"ok": sys.platform == "darwin"}  # switched off if the Mac's encoder ever fails


def _codec(width: int, height: int, final: bool) -> list[str]:
    """Use the Mac's built-in video chip when available: several times faster than the CPU."""
    if _hardware["ok"] and has_encoder("h264_videotoolbox"):
        bitrate = width * height * (6 if final else 10)  # about 12 Mbit/s final at 1080p
        return ["-c:v", "h264_videotoolbox", "-b:v", str(bitrate), "-allow_sw", "1", "-g", "30"]
    # (a key frame every second or so: the editor can jump and scrub through the video smoothly)
    if final:
        return ["-c:v", "libx264", "-preset", "faster", "-crf", "20", "-g", "30"]
    return ["-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-g", "30"]


def _run_encode(args_for: Callable[[list[str]], list[str]], width: int, height: int, final: bool, **kwargs) -> None:
    """Encode with the best codec; if the Mac's encoder fails, fall back to the CPU for good."""
    try:
        run_ffmpeg(args_for(_codec(width, height, final)), **kwargs)
    except FFmpegError:
        if not _hardware["ok"] or not has_encoder("h264_videotoolbox"):
            raise
        _hardware["ok"] = False
        run_ffmpeg(args_for(_codec(width, height, final)), **kwargs)


@dataclass
class RenderSettings:
    width: int = 1920
    height: int = 1080
    photo_seconds: float = 4.0
    max_clip_seconds: float = 10.0  # 0 keeps whole videos
    transition: str = "Fade"
    transition_seconds: float = 0.8
    keep_clip_sound: bool = True
    clip_volume: float = 1.0
    music_path: str | None = None
    music_volume: float = 0.5
    end_black: float = 0.0  # seconds of black at the very end (the last shot fades into it, the music fades out)
    end_title: dict | None = None  # words shown on that black ending (see titles.py); with "card": its look
    # The opening card over the start of the first shot (it keeps every cut where it was):
    # {"text", "sub", "card" (a key of titles.CARDS), "seconds"}.
    opening: dict | None = None
    # The ending card over the last seconds of the video (the room Auto-edit kept for it), fading in:
    # {"text", "sub", "card" (a key of titles.CARDS, or "" for words on black), "seconds"}.
    closing: dict | None = None
    look: str = "natural"  # the colour mood of the whole video (see looks.py)
    # Titles on the video's own timeline (the editor under the video): {"text", "sub", "style", "where",
    # "length", "fade_in", "fade_out", and "start" (seconds into the video) or "shot" (its number) + "offset"}.
    titles: list = dataclasses.field(default_factory=list)
    # The style package's effects on the whole video (see packages.py): {"flash": [seconds] (a quick flash of light on
    # the music's big hits), "hue": True (the colours swing into each cut), "glitch": True (a short glitch on some
    # cuts)}.
    fx: dict = dataclasses.field(default_factory=dict)


def _fit_with_blur(src: str, w: int, h: int, out: str) -> str:
    """Fit the picture inside w x h; fill empty sides with a blurred copy (for vertical shots)."""
    return (
        f"[{src}]split=2[bg_{out}][fg_{out}];"
        f"[bg_{out}]scale={w // 4}:{h // 4}:force_original_aspect_ratio=increase,"
        f"crop={w // 4}:{h // 4},boxblur=10:2,scale={w}:{h},setsar=1[bgs_{out}];"
        f"[fg_{out}]scale={w}:{h}:force_original_aspect_ratio=decrease,setsar=1[fgs_{out}];"
        f"[bgs_{out}][fgs_{out}]overlay=(W-w)/2:(H-h)/2[{out}]"
    )


def _fill_around(src: str, w: int, h: int, focus: float | None, out: str) -> str:
    """Fill the whole w x h frame with the picture, cutting off its sides, keeping the people (focus, 0..1
    across) in view. For a phone video: a wide shot shows its slice with the people in it."""
    f = 0.5 if focus is None else min(1.0, max(0.0, focus))
    return (f"[{src}]scale={w}:{h}:force_original_aspect_ratio=increase,setsar=1,"
            f"crop={w}:{h}:x='max(0,min(iw-{w},iw*{f:.3f}-{w}/2))':y='(ih-{h})/2'[{out}]")


def tall(s: RenderSettings) -> bool:
    """A phone video (taller than wide)."""
    return s.height > s.width


def _hdr_to_sdr() -> str:
    """Tone-map iPhone HDR footage so colours do not look washed out."""
    if not has_filter("zscale"):
        return ""
    return (
        "zscale=t=linear:npl=100,format=gbrpf32le,zscale=p=bt709,"
        "tonemap=hable:desat=0,zscale=t=bt709:m=bt709:r=tv,"
    )


def clip_range(item: MediaItem, s: RenderSettings) -> tuple[float, float]:
    """The part of a video that is used: your trim if you set one, else the middle part."""
    if item.trim_start is not None and item.trim_end is not None and item.trim_end > item.trim_start:
        return item.trim_start, min(item.trim_end, item.duration)
    if s.max_clip_seconds and item.duration > s.max_clip_seconds:
        start = (item.duration - s.max_clip_seconds) / 2
        return start, start + s.max_clip_seconds
    return 0.0, item.duration


RAMP_SOURCE = 1.4  # a speed ramp plays the middle 40% of the shot twice as fast, so it uses 40% more video
RAMP = (0.3, 0.4, 0.3)  # shares of the shot on screen: normal, fast, normal


def source_factor(effects: dict | None) -> float:
    """How much of the video a shot uses for each second on screen (before its speed)."""
    return RAMP_SOURCE if (effects or {}).get("ramp") else 1.0


def held(effects: dict | None) -> float:
    """Seconds a shot ends frozen on its last picture (a freeze-frame with a name on it), on top of the video."""
    return float((effects or {}).get("freeze") or 0.0)


SHAKE = 0.05  # a shake moves the picture by up to 5% of its size, dying away in about half a second


def _atempo(factor: float) -> str:
    """Speed sound up or down without changing its pitch (one step only goes from half to double speed)."""
    steps = []
    while factor > 2.0:
        steps.append(2.0)
        factor /= 2.0
    while factor < 0.5:
        steps.append(0.5)
        factor /= 0.5
    steps.append(factor)
    return ",".join(f"atempo={f:.4f}" for f in steps if abs(f - 1) > 1e-4)


def _video_segment(item: MediaItem, s: RenderSettings, out: Path) -> float:
    start, end = clip_range(item, s)
    speed = min(2.0, max(0.5, item.speed or 1.0))  # the range the sound can follow too
    effects = item.effects or {}
    moving = (end - start) / speed / source_factor(effects)  # the video playing
    length = moving + held(effects)  # on screen, with a freeze-frame at the end
    smooth = f",framerate=fps={FPS}" if speed < 0.95 else ""  # slow motion: blend frames, so it doesn't stutter
    look = (_hdr_to_sdr() if item.hdr else "") + rotate_filter(item.rotation) + crop_filter(item.crop)
    graph = ""
    src = "[0:v]reverse," if item.reverse else "[0:v]"  # played backwards (only the part used is read in)
    if effects.get("ramp"):
        # Normal speed, then twice as fast through the middle, then normal again.
        a, b = RAMP[0] * length * speed, (RAMP[0] * length + RAMP[1] * length * 2) * speed  # in the video's own time
        graph += (f"{src}setpts=PTS-STARTPTS,split=3[r1][r2][r3];"
                  f"[r1]trim=0:{a:.4f},setpts=(PTS-STARTPTS)/{speed:.4f}{smooth}[q1];"
                  f"[r2]trim={a:.4f}:{b:.4f},setpts=(PTS-STARTPTS)/{2 * speed:.4f}[q2];"
                  f"[r3]trim={b:.4f},setpts=(PTS-STARTPTS)/{speed:.4f}{smooth}[q3];"
                  f"[q1][q2][q3]concat=n=3:v=1:a=0,{look}format=yuv420p[src];")
    elif abs(speed - 1) > 1e-3:
        graph += f"{src}setpts=(PTS-STARTPTS)/{speed:.4f}{smooth},{look}format=yuv420p[src];"
    else:
        graph += f"{src}{look}format=yuv420p[src];"
    graph += (_fill_around("src", s.width, s.height, item.focus, "fit") if tall(s)
              else _fit_with_blur("src", s.width, s.height, "fit"))
    graph += f";[fit]fps={FPS}{_zoom_filter(item, moving)}{_hit_filter(effects, s.width, s.height)}{_freeze_filter(effects, moving)},format=yuv420p[v]"
    inputs = ["-ss", f"{start:.3f}", "-t", f"{end - start:.3f}", "-i", str(item.path)]
    title_png = None
    if item.overlay and (item.overlay.get("text") or item.overlay.get("sub")):
        title_png = _title_input(item.overlay, s.width, s.height, length, out)
    if item.has_audio and item.sound and s.keep_clip_sound and not item.reverse:  # (backwards: no clip sound)
        level = f"aformat=sample_rates=48000:channel_layouts=stereo,volume={item.volume * s.clip_volume:.2f}"
        if effects.get("ramp"):
            parts = [_atempo(speed), _atempo(2 * speed), _atempo(speed)]
            graph += (f";[0:a]asetpts=PTS-STARTPTS,asplit=3[s1][s2][s3];"
                      f"[s1]atrim=0:{a:.4f},asetpts=PTS-STARTPTS{',' + parts[0] if parts[0] else ''}[t1];"
                      f"[s2]atrim={a:.4f}:{b:.4f},asetpts=PTS-STARTPTS{',' + parts[1] if parts[1] else ''}[t2];"
                      f"[s3]atrim={b:.4f},asetpts=PTS-STARTPTS{',' + parts[2] if parts[2] else ''}[t3];"
                      f"[t1][t2][t3]concat=n=3:v=0:a=1,{level}[a]")
        else:
            tempo = _atempo(speed)  # the sound keeps its pitch
            graph += f";[0:a]{tempo + ',' if tempo else ''}{level}[a]"
    else:
        inputs += ["-f", "lavfi", "-t", f"{length:.3f}", "-i", "anullsrc=r=48000:cl=stereo"]
        graph += ";[1:a]anull[a]"
    if title_png is not None:  # the title, fading in and out over the shot
        n = inputs.count("-i")  # the next input's number
        inputs += title_png[0]
        graph = graph.replace(",format=yuv420p[v]", ",format=yuv420p[v0]", 1)
        graph += (f";[{n}:v]format=rgba,setpts=PTS-STARTPTS,{title_png[1]}[title];"
                  f"[v0][title]overlay=0:0:shortest=1:format=auto,format=yuv420p[v]")
    if held(effects) and effects.get("label"):  # the name on the freeze-frame, appearing as it freezes
        name_png = out.with_suffix(".name.png")
        titles.draw({"text": effects["label"], "style": "handwritten", "where": "lower-left"}, s.width, s.height) \
            .save(name_png)
        n = inputs.count("-i")
        inputs += ["-loop", "1", "-t", f"{length + 0.5:.3f}", "-i", str(name_png)]
        graph = graph.replace("format=yuv420p[v]", "format=yuv420p[vf]")
        graph += (f";[{n}:v]format=rgba,setpts=PTS-STARTPTS,fade=t=in:st={moving:.3f}:d=0.25:alpha=1[name];"
                  f"[vf][name]overlay=0:0:shortest=1:format=auto,format=yuv420p[v]")
    _encode_segment(inputs, graph, length, out, s)
    return length


def side_tiles(count: int, w: int, h: int) -> list[tuple[int, int, int, int]]:
    """Upright videos next to each other (x, y, width, height): side by side across a landscape video,
    one above the other in a phone video."""
    g = max(2, round(max(w, h) * COLLAGE_GAP_SHARE))
    if h > w:
        th = (h - g * (count - 1)) // count
        return [(0, k * (th + g), w, th) for k in range(count)]
    tw = (w - g * (count - 1)) // count
    return [(k * (tw + g), 0, tw, h) for k in range(count)]


SIDE_STAGGER = 0.35  # each video next to the first slides in this much later


def _side_segment(item: MediaItem, s: RenderSettings, out: Path) -> float:
    """Two or three upright videos playing next to each other on one cut (the first one's sound)."""
    fx = item.effects or {}
    start, end = clip_range(item, s)
    length = max(0.5, end - start + held(fx))  # (under the ending card a short clip holds its last picture)
    others = list(fx.get("side") or [])
    starts = list(fx.get("starts") or [0.0] * len(others))
    turns = list(fx.get("turns") or [0] * len(others))
    w, h = s.width, s.height
    tiles = side_tiles(1 + len(others), w, h)
    inputs = ["-f", "lavfi", "-t", f"{length:.3f}", "-i", f"color=c=0x0c0c0c:s={w}x{h}:r={FPS}"]
    graph, prev = [], "0:v"
    clips = [(str(item.path), start, item.rotation, (_hdr_to_sdr() if item.hdr else ""))] + \
            [(p, float(a), int(t or 0), "") for p, a, t in zip(others, starts, turns)]
    framed = list(fx.get("frames") or [])
    framed += [None] * (len(clips) - len(framed))
    for k, ((path, at, turn, look), (x, y, tw, th), frame) in enumerate(zip(clips, tiles, framed)):
        n = inputs.count("-i")
        inputs += ["-ss", f"{at:.3f}", "-t", f"{length:.3f}", "-i", path]
        delay = SIDE_STAGGER * k
        f = frame or {}
        z, fx_, fy_ = max(1.0, frame_zoom(f)), float(f.get("x", 0.5)), float(f.get("y", 0.5))
        # (as you framed it: zoomed in, the visible part where you moved it; see frame_crop)
        graph.append(f"[{n}:v]{look}{rotate_filter(turn)}setpts=PTS-STARTPTS+{delay:.3f}/TB,"
                     f"scale={2 * round(tw * z / 2)}:{2 * round(th * z / 2)}:force_original_aspect_ratio=increase,setsar=1,"
                     f"crop={tw}:{th}:(iw-{tw})*{fx_:.4f}:(ih-{th})*{fy_:.4f}[c{k}]")
        left, top, bw, bh = frame_box(tw, th, f)
        if (bw, bh) != (tw, th):  # zoomed out: smaller, on a blurred darker copy of itself (see fit_frame)
            bw, bh = 2 * round(bw / 2), 2 * round(bh / 2)
            graph.append(f"[c{k}]split[cb{k}][cf{k}]")
            graph.append(f"[cb{k}]boxblur={max(8, tw // 30)}:2,eq=brightness=-0.22[bb{k}]")
            graph.append(f"[cf{k}]scale={bw}:{bh}[ff{k}]")
            graph.append(f"[bb{k}][ff{k}]overlay={left}:{top}[c{k}b]")
            src = f"c{k}b"
        else:
            src = f"c{k}"
        graph.append(f"[{src}]fps={FPS},tpad=stop_mode=clone:stop_duration={length:.3f},format=yuva420p[s{k}]")
        graph.append(f"[{prev}][s{k}]overlay={x}:{y}:eof_action=pass:format=auto[o{k}]")
        prev = f"o{k}"
    graph.append(f"[{prev}]trim=duration={length:.3f},setpts=PTS-STARTPTS,format=yuv420p[v]")
    if item.has_audio and item.sound and s.keep_clip_sound:
        level = f"aformat=sample_rates=48000:channel_layouts=stereo,volume={item.volume * s.clip_volume:.2f}"
        graph.append(f"[1:a]asetpts=PTS-STARTPTS,{level},apad,atrim=0:{length:.3f}[a]")
    else:
        n = inputs.count("-i")
        inputs += ["-f", "lavfi", "-t", f"{length:.3f}", "-i", "anullsrc=r=48000:cl=stereo"]
        graph.append(f"[{n}:a]anull[a]")
    _encode_segment(inputs, ";".join(graph), length, out, s)
    return length


def _hit_filter(effects: dict, w: int, h: int) -> str:
    """A camera shake with a flash of light on the music's biggest hits (empty when the shot has none)."""
    hits = effects.get("shake") or []
    if not hits:
        return ""
    # How hard the shake is at time t: a kick at each hit that dies away quickly.
    kick = "+".join(f"gte(t,{t:.3f})*exp(-(t-{t:.3f})*7)" for t in hits)
    m = SHAKE / 2
    return (f",crop=w=iw*{1 - SHAKE:.3f}:h=ih*{1 - SHAKE:.3f}"
            f":x='iw*{m:.4f}*(1+sin(t*83)*({kick}))':y='ih*{m:.4f}*(1+cos(t*71)*({kick}))',"
            f"scale={w}:{h},"
            f"eq=brightness='0.35*({kick.replace('*7)', '*14)')})':eval=frame")


def _freeze_filter(effects: dict, moving: float) -> str:
    """Hold the last picture (with a quick flash as it freezes); the name goes on top in _video_segment."""
    if not held(effects):
        return ""
    if effects.get("still"):  # (held quietly behind the ending card)
        return f",tpad=stop_mode=clone:stop_duration={held(effects):.3f}"
    return (f",tpad=stop_mode=clone:stop_duration={held(effects):.3f}"
            f",eq=brightness='0.4*gte(t,{moving:.3f})*exp(-(t-{moving:.3f})*9)':eval=frame")


ZOOM_FRAMES = 9  # a zoom through a cut takes 0.3 s on each side
REVEAL_FRAMES = 24  # a zoom-out reveal settles over 0.8 s


def _zoom_filter(item: MediaItem, length: float) -> str:
    """Camera-style zooms on the shot, frame by frame and smoothly (not rounded to whole pixels):
    rushing in at its end or landing at its start (zoom transitions), short punch-ins on the music's big
    hits, and a slow push-in for a still, cinematic shot. Empty when the shot has none."""
    effects = item.effects or {}
    n = max(1, round(length * FPS))
    parts = []
    if effects.get("push"):
        parts.append(f"0.08*in/{n}")
    if item.fx_out == "zoom":
        k = min(ZOOM_FRAMES, n // 3)
        parts.append(f"if(gte(in,{n - k}),0.5*pow((in-{n - k})/{k},2),0)")
    if item.fx_in == "zoom":
        k = min(ZOOM_FRAMES, n // 3)
        parts.append(f"if(lt(in,{k}),0.5*pow(1-in/{k},2),0)")
    elif item.fx_in == "reveal":
        k = min(REVEAL_FRAMES, n // 2)
        parts.append(f"if(lt(in,{k}),0.45*pow(1-in/{k},3),0)")
    for t in effects.get("punch") or []:
        f = round(t * FPS)
        if 0 < f < n:
            parts.append(f"if(gte(in,{f}),0.12*exp(-(in-{f})/6),0)")
    if not parts:
        return ""
    z = f"(1+{'+'.join(parts)})"
    corner = {"l": f"W/2-W/(2*{z})", "r": f"W/2+W/(2*{z})", "t": f"H/2-H/(2*{z})", "b": f"H/2+H/(2*{z})"}
    expr = lambda v: v.replace(",", "\\,")  # noqa: E731  (commas inside a filter option)
    return (f",perspective=x0='{expr(corner['l'])}':y0='{expr(corner['t'])}':x1='{expr(corner['r'])}':"
            f"y1='{expr(corner['t'])}':x2='{expr(corner['l'])}':y2='{expr(corner['b'])}':x3='{expr(corner['r'])}':"
            f"y3='{expr(corner['b'])}':interpolation=linear:eval=frame")


def crop_filter(crop: dict | None) -> str:
    """FFmpeg filter for the part of a (turned) video you zoomed in on: crop {"zoom", "x", "y"} as for photos."""
    zoom = float((crop or {}).get("zoom") or 1.0)
    if zoom <= 1.001:
        return ""
    x = min(max(float(crop.get("x", 0.5)), 0.5 / zoom), 1 - 0.5 / zoom) - 0.5 / zoom
    y = min(max(float(crop.get("y", 0.5)), 0.5 / zoom), 1 - 0.5 / zoom) - 0.5 / zoom
    return f"crop=w=trunc(iw/{zoom:.4f}/2)*2:h=trunc(ih/{zoom:.4f}/2)*2:x=iw*{x:.4f}:y=ih*{y:.4f},"


def rotate_filter(rotation: int) -> str:
    """FFmpeg filter that turns a video clockwise by 90, 180 or 270 degrees."""
    return {90: "transpose=1,", 180: "hflip,vflip,", 270: "transpose=2,"}.get(rotation % 360, "")


def _font(size: int) -> ImageFont.FreeTypeFont:
    for path in FONTS:
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default(size)


def _title_still(item: MediaItem, s: RenderSettings, out: Path) -> None:
    """A chapter card: the title over a dark, blurred picture from that day."""
    w, h = s.width, s.height
    try:
        with Image.open(item.path) as img:
            bg = ImageOps.fit(ImageOps.exif_transpose(img).convert("RGB"), (w, h))
        bg = bg.filter(ImageFilter.GaussianBlur(w / 60))
        bg = Image.blend(bg, Image.new("RGB", (w, h), "black"), 0.55)
    except OSError:
        bg = Image.new("RGB", (w, h), (20, 20, 24))
    draw = ImageDraw.Draw(bg)
    heading, _, sub = (item.title or "").partition("\n")
    big, small = _font(h // 8), _font(h // 22)
    draw.text((w / 2, h / 2 - h / 40), heading, font=big, fill="white", anchor="md")
    if sub:
        draw.text((w / 2, h / 2 + h / 40), sub, font=small, fill=(230, 230, 230), anchor="ma")
    bg.save(out, quality=92)


def _photo_segment(item: MediaItem, s: RenderSettings, out: Path, workdir: Path) -> float:
    still = workdir / f"{out.stem}.still.jpg"
    seconds = TITLE_SECONDS if item.kind == "title" else (item.seconds or s.photo_seconds)
    if item.kind == "title":
        _title_still(item, s, still)
    else:
        # Pillow turns the photo the right way up and reads iPhone HEIC files, which FFmpeg may not.
        with Image.open(item.path) as img:
            img = ImageOps.exif_transpose(img).convert("RGB").rotate(-item.rotation, expand=True)
            img = crop_photo(img, item.crop)
            img.thumbnail((3840, 3840))
            img.save(still, quality=92)

    with Image.open(still) as img:
        canvas = _photo_canvas(img.convert("RGB"), s.width, s.height, item.focus if tall(s) else None, tall(s))
    move = "in" if item.kind == "title" else _pick_move(item.path, canvas_is_tall=_is_tall(still))
    frames = max(1, round(seconds * FPS))
    w, h = s.width, s.height
    layer = None
    if item.kind != "title" and item.overlay and (item.overlay.get("text") or item.overlay.get("sub")):
        layer = np.asarray(titles.draw(item.overlay, w, h))

    def feed(pipe) -> None:
        for n in range(frames):
            frame = _photo_frame(canvas, w, h, move, n / max(1, frames - 1), move_strength(seconds))
            if layer is not None:
                frame = _with_title(frame, item.overlay, layer, n / FPS, frames / FPS)
            pipe.write(frame.tobytes())

    inputs = [
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", str(FPS), "-i", "pipe:0",
        "-f", "lavfi", "-t", f"{seconds:.3f}", "-i", "anullsrc=r=48000:cl=stereo",
    ]
    _encode_segment(inputs, "[0:v]format=yuv420p[v];[1:a]anull[a]", seconds, out, s, feed=feed)
    return seconds


POP = 0.18  # seconds the photos of a collage take to pop into place (all together)
COLLAGE_GAP_SHARE = 0.008  # the thin dark line between tiles, as a share of the video's width


# The layouts you can choose for a collage, by how many photos it has ("" = the usual one).
LAYOUTS = {2: {"": "Side by side", "stack": "One above the other"},
           3: {"": "Big left, two small", "big-right": "Big right, two small", "row": "Three in a row"},
           4: {"": "2 × 2 grid", "row": "Four in a row", "big-left": "Big left, three small"}}


def collage_tiles(count: int, w: int, h: int, layout: str = "") -> list[tuple[int, int, int, int]]:
    """Where each photo of a collage goes (x, y, width, height): 2 side by side, 3 as one big and two small,
    4 as a 2x2 grid (or the layout you chose, see LAYOUTS). An upright (phone) video stacks them instead."""
    g = max(2, round(w * COLLAGE_GAP_SHARE))
    upright = h > w
    half_w, half_h = (w - g) // 2, (h - g) // 2
    if layout == "row" and count >= 3 and not upright:
        tw = (w - g * (count - 1)) // count
        return [(k * (tw + g), 0, tw, h) for k in range(count)]
    if layout == "stack" and count == 2:
        return [(0, 0, w, half_h), (0, h - half_h, w, half_h)]
    if layout == "big-right" and count == 3 and not upright:
        return [(w - half_w, 0, half_w, h), (0, 0, half_w, half_h), (0, h - half_h, half_w, half_h)]
    if layout == "big-left" and count == 4 and not upright:
        third = (h - 2 * g) // 3
        return [(0, 0, half_w, h)] + [(w - half_w, k * (third + g), half_w, third) for k in range(3)]
    if count == 2:
        return [(0, 0, w, half_h), (0, h - half_h, w, half_h)] if upright else \
            [(0, 0, half_w, h), (w - half_w, 0, half_w, h)]
    if count == 3:
        if upright:
            return [(0, 0, w, half_h), (0, h - half_h, half_w, half_h), (w - half_w, h - half_h, half_w, half_h)]
        return [(0, 0, half_w, h), (w - half_w, 0, half_w, half_h), (w - half_w, h - half_h, half_w, half_h)]
    return [(0, 0, half_w, half_h), (w - half_w, 0, half_w, half_h),
            (0, h - half_h, half_w, half_h), (w - half_w, h - half_h, half_w, half_h)]


def frame_crop(iw: int, ih: int, tw: int, th: int, frame: dict | None) -> tuple[float, float, float, float]:
    """The part of a picture (iw x ih) that shows in a place (tw x th), as you framed it: zoomed in by "z" (1 = just
    filling it) with the visible part at "x", "y" (0 = left/top, 0.5 = the middle, 1 = right/bottom), like CSS
    object-fit: cover with object-position x% y% and scale(z) around that point."""
    f = frame or {}
    z = max(1.0, float(f.get("z") or 1.0))
    x, y = float(f.get("x", 0.5)), float(f.get("y", 0.5))
    scale = max(tw / iw, th / ih) * z
    w, h = tw / scale, th / scale
    left, top = x * (iw - w), y * (ih - h)
    return left, top, left + w, top + h


SMALLEST = 0.4  # a picture can shrink to 40% of its place (a soft, blurred copy of it fills around it)


def frame_zoom(frame: dict | None) -> float:
    return min(4.0, max(SMALLEST, float((frame or {}).get("z") or 1.0)))


def frame_box(tw: int, th: int, frame: dict | None) -> tuple[int, int, int, int]:
    """Where a picture smaller than its place (zoomed out, z < 1) sits in it: (left, top, width, height), at "x",
    "y" of the room around it (like CSS scale(z) with transform-origin x% y%). The whole place when z >= 1."""
    f = frame or {}
    z = frame_zoom(f)
    if z >= 1.0:
        return 0, 0, tw, th
    w, h = max(2, round(tw * z)), max(2, round(th * z))
    return round(float(f.get("x", 0.5)) * (tw - w)), round(float(f.get("y", 0.5)) * (th - h)), w, h


def fit_frame(img: Image.Image, tw: int, th: int, frame: dict | None) -> np.ndarray:
    full = img.resize((tw, th), Image.LANCZOS, box=frame_crop(img.width, img.height, tw, th, frame))
    left, top, w, h = frame_box(tw, th, frame)
    if (w, h) == (tw, th):
        return np.asarray(full)
    # Zoomed out: the picture smaller in its place, on a blurred, darker copy of itself.
    from PIL import ImageEnhance, ImageFilter
    back = ImageEnhance.Brightness(full.filter(ImageFilter.GaussianBlur(max(8, tw // 30)))).enhance(0.55)
    back.paste(full.resize((w, h), Image.LANCZOS), (left, top))
    return np.asarray(back)


def _with_title(frame: np.ndarray, title: dict, layer: np.ndarray, t: float, length: float) -> np.ndarray:
    """A shot's title over one of its frames: fading in and out, or (with an animation) coming on its own way."""
    if title.get("anim"):
        start = titles.timing(length)[0]
        return titles.blend(frame, titles.anim_layer(title, layer, t - start), titles.anim_strength(title, t, length))
    return titles.blend(frame, layer, titles.strength(t, length))


def _title_input(title: dict, w: int, h: int, length: float, out: Path) -> tuple[list[str], str]:
    """The title of a video shot as an FFmpeg input and the filter that shows it: one picture that fades in and out,
    or (animated) the few pictures of its animation, each for its moment, then fading out."""
    layer = titles.draw(title, w, h)
    if not title.get("anim"):
        png = out.with_suffix(".title.png")
        layer.save(png)
        return ["-loop", "1", "-t", f"{length + 0.5:.3f}", "-i", str(png)], titles.ffmpeg_fades(length)
    full = np.asarray(layer)
    start, end = titles.timing(length)
    steps, t, last, k = [], 0.0, None, 0
    lines = ["ffconcat version 1.0"]
    blank = out.with_suffix(".t-blank.png")
    Image.new("RGBA", (w, h), (0, 0, 0, 0)).save(blank)
    lines += [f"file '{blank.name}'", f"duration {start:.4f}"]
    span = titles.anim_length(title) + 0.4
    while t < span:  # (a picture each time it changes, at 30 a second)
        shown = titles.anim_share(title, t)
        if shown != last:
            png = out.with_suffix(f".t{k}.png")
            Image.fromarray(titles.anim_layer(title, full, t)).save(png)
            lines += [f"file '{png.name}'", f"duration {1 / FPS:.4f}"]
            last, k = shown, k + 1
        else:  # (the same picture a frame longer)
            dur = float(lines[-1].split()[1]) + 1 / FPS
            lines[-1] = f"duration {dur:.4f}"
        t += 1 / FPS
    png = out.with_suffix(".title.png")
    layer.save(png)
    lines += [f"file '{png.name}'", f"duration {max(0.1, length + 0.5 - start - span):.4f}", f"file '{png.name}'"]
    listing = out.with_suffix(".title.txt")
    listing.write_text("\n".join(lines) + "\n")
    fade_out = min(titles.FADE_OUT, (end - start) / 3)
    return (["-f", "concat", "-safe", "0", "-i", str(listing)],
            f"fps={FPS},fade=t=out:st={end - fade_out:.3f}:d={fade_out:.3f}:alpha=1")


COLLAGE_DRIFT = 0.07  # how far each photo of a collage gently zooms (7%) while it's on screen
COLLAGE_MOVES = ["in", "out", "left", "right", "in-up", "out-down"]


def _ease(p: float) -> float:
    p = min(1.0, max(0.0, p))
    return p * p * (3 - 2 * p) * 0.35 + p * 0.65


def drift(frame: dict | None, move: str, p: float, strength: float = 1.0) -> dict:
    """How a collage photo is framed at moment p (0 = start, 1 = end) of its gentle move: your framing (x, y, z),
    zoomed a little in or out and drifting a little sideways or up/down around it."""
    f = dict(frame or {})
    z = frame_zoom(f)
    e = _ease(p)
    grow = 1 + COLLAGE_DRIFT * strength * (e if move.startswith("in") or move in ("left", "right") else 1 - e)
    x, y = float(f.get("x", 0.5)), float(f.get("y", 0.5))
    pan = min(0.8, 0.3 * strength)
    if move == "left":
        x += pan * (0.5 - e)
    elif move == "right":
        x -= pan * (0.5 - e)
    elif move == "in-up":
        y += pan * (0.5 - e)
    elif move == "out-down":
        y -= pan * (0.5 - e)
    return {"x": min(1.0, max(0.0, x)), "y": min(1.0, max(0.0, y)), "z": z * grow}


def _warp(pic: np.ndarray, tw: int, th: int, frame: dict) -> np.ndarray:
    """The picture as framed (z >= 1) in a tw x th place, with sub-pixel positions (it glides, never shakes)."""
    ih, iw = pic.shape[:2]
    left, top, right, bottom = frame_crop(iw, ih, tw, th, {**frame, "z": max(1.0, frame["z"])})
    m = np.float32([[(right - left) / tw, 0, left], [0, (bottom - top) / th, top]])
    return cv2.warpAffine(pic, m, (tw, th), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_REFLECT)


def collage_tile(pic: np.ndarray, back: np.ndarray | None, tw: int, th: int, frame: dict | None, move: str,
                 p: float, strength: float = 1.0) -> np.ndarray:
    """One place of a collage at moment p of its move: filled as you framed it, or (zoomed out) the smaller
    picture on its blurred copy (back), the picture itself still moving gently inside its box."""
    f = drift(frame, move, p, strength)
    if frame_zoom(frame) >= 1.0:
        return _warp(pic, tw, th, f)
    left, top, w, h = frame_box(tw, th, frame)
    out = back.copy()
    out[top:top + h, left:left + w] = _warp(pic, w, h, {"x": 0.5, "y": 0.5, "z": f["z"] / frame_zoom(frame)})
    return out


def _collage_segment(item: MediaItem, s: RenderSettings, out: Path) -> float:
    """Several photos on screen together: they all come on at once, then each one moves gently on its own
    (zooming a little in or out, drifting a little), around the way you framed it."""
    fx = item.effects or {}
    paths = [item.path, *map(Path, fx["collage"])][:4]
    turns = [item.rotation, *(fx.get("turns") or [0] * len(fx["collage"]))]
    seconds = item.seconds or s.photo_seconds
    w, h = s.width, s.height
    places = collage_tiles(len(paths), w, h, fx.get("layout") or "")
    framed = list(fx.get("frames") or [])
    framed += [None] * (len(paths) - len(framed))
    pictures, backs, moves = [], [], []
    used: set[str] = set()
    for k, (path, turn, frame, (x, y, tw, th)) in enumerate(zip(paths, turns, framed, places)):
        with Image.open(path) as img:
            img = ImageOps.exif_transpose(img).convert("RGB").rotate(-turn, expand=True)
        # Just big enough for the closest it gets (sharp, and quick to move every frame).
        need = max(tw / img.width, th / img.height) * max(1.0, frame_zoom(frame)) * (1 + COLLAGE_DRIFT * move_strength(seconds)) * 1.15
        if need < 1:
            img = img.resize((max(2, round(img.width * need)), max(2, round(img.height * need))), Image.LANCZOS)
        pictures.append(np.asarray(img))
        backs.append(fit_frame(img, tw, th, frame) if frame_zoom(frame) < 1.0 else None)
        move = COLLAGE_MOVES[int(hashlib.sha1(str(path).encode()).hexdigest(), 16) % len(COLLAGE_MOVES)]
        while move in used and len(used) < len(COLLAGE_MOVES):  # (neighbours move differently)
            move = COLLAGE_MOVES[(COLLAGE_MOVES.index(move) + 1) % len(COLLAGE_MOVES)]
        used.add(move)
        moves.append(move)
    frames = max(1, round(seconds * FPS))
    layer = np.asarray(titles.draw(item.overlay, w, h)) if item.overlay and (
        item.overlay.get("text") or item.overlay.get("sub")) else None

    def feed(pipe) -> None:
        for n in range(frames):
            t = n / FPS
            p = t / max(seconds, 1e-6)
            grow = 1 - (1 - min(1.0, t / POP)) ** 3  # all pop in together, quick at first, settling
            frame = np.full((h, w, 3), 12, np.uint8)
            for pic, back, f, move, (x, y, tw, th) in zip(pictures, backs, framed, moves, places):
                tile = collage_tile(pic, back, tw, th, f, move, p, move_strength(seconds))
                if grow >= 1.0:
                    frame[y:y + th, x:x + tw] = tile
                    continue
                scale = 0.82 + 0.18 * grow
                small = cv2.resize(tile, (max(1, round(tw * scale)), max(1, round(th * scale))))
                oy, ox = y + (th - small.shape[0]) // 2, x + (tw - small.shape[1]) // 2
                region = frame[oy:oy + small.shape[0], ox:ox + small.shape[1]]
                frame[oy:oy + small.shape[0], ox:ox + small.shape[1]] = (region * (1 - grow) + small * grow).astype(np.uint8)
            if layer is not None:
                frame = _with_title(frame, item.overlay, layer, t, seconds)
            pipe.write(np.ascontiguousarray(frame).tobytes())

    inputs = [
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", str(FPS), "-i", "pipe:0",
        "-f", "lavfi", "-t", f"{seconds:.3f}", "-i", "anullsrc=r=48000:cl=stereo",
    ]
    _encode_segment(inputs, "[0:v]format=yuv420p[v];[1:a]anull[a]", seconds, out, s, feed=feed)
    return seconds


# Photos move gently while on screen (the "Ken Burns" look). Each photo gets one of these moves, chosen from
# its file name so it stays the same every time you make the video.
ZOOM = 1.15  # how far in the moves go (15%) for a photo on screen up to 2 seconds ...
STRONGER_PER_SECOND = 0.5  # ... and this much more for each second longer (a 5 s photo moves 2.5 times as far)
STRONGEST = 2.6


def move_strength(seconds: float) -> float:
    """How far a photo moves, from 1 (up to 2 s on screen: a gentle move) to 2.6 (6 s or more: a bold one),
    so a longer photo keeps moving instead of nearly standing still."""
    return min(STRONGEST, max(1.0, 1 + (float(seconds or 0) - 2.0) * STRONGER_PER_SECOND))
MOVES = ["in", "out", "left", "right", "in-up", "out-down"]
CANVAS = 1.45  # the photo is prepared this much bigger than the video, so zooming in stays sharp


def _is_tall(still: Path) -> bool:
    with Image.open(still) as img:
        return img.height > img.width


def _pick_move(path, canvas_is_tall: bool) -> str:
    moves = ["in", "out", "in-up", "out-down"] if canvas_is_tall else MOVES  # tall photos: no sideways pans
    return moves[int(hashlib.sha1(str(path).encode()).hexdigest(), 16) % len(moves)]


def crop_photo(img: Image.Image, crop: dict | None) -> Image.Image:
    """The part of a photo you zoomed in on: crop {"zoom", "x", "y"} (its middle as fractions of the picture)."""
    zoom = float((crop or {}).get("zoom") or 1.0)
    if zoom <= 1.001:
        return img
    w, h = img.width / zoom, img.height / zoom
    x = min(max(float(crop.get("x", 0.5)), 0.5 / zoom), 1 - 0.5 / zoom) * img.width
    y = min(max(float(crop.get("y", 0.5)), 0.5 / zoom), 1 - 0.5 / zoom) * img.height
    return img.crop((round(x - w / 2), round(y - h / 2), round(x + w / 2), round(y + h / 2)))


def _photo_canvas(img: Image.Image, w: int, h: int, focus: float | None = None, fill: bool = False) -> np.ndarray:
    """The photo fitted into the video frame (bigger, for zooming); empty sides filled with a blurred copy.
    fill (a phone video): the photo fills the frame instead, its sides cut off around the people (focus)."""
    cw, ch = round(w * CANVAS), round(h * CANVAS)
    if fill:
        f = 0.5 if focus is None else focus
        wide = img.width * ch / img.height  # its width once scaled to fill the height
        across = min(1.0, max(0.0, (f * wide - cw / 2) / (wide - cw))) if wide > cw else 0.5  # the people in the middle
        return np.asarray(ImageOps.fit(img, (cw, ch), Image.LANCZOS, centering=(across, 0.5)))
    bg = ImageOps.fit(img, (max(1, cw // 8), max(1, ch // 8))).filter(ImageFilter.GaussianBlur(4)).resize((cw, ch))
    fg = ImageOps.contain(img, (cw, ch), Image.LANCZOS)
    bg.paste(fg, ((cw - fg.width) // 2, (ch - fg.height) // 2))
    return np.asarray(bg)


def _photo_frame(canvas: np.ndarray, w: int, h: int, move: str, t: float, strength: float = 1.0) -> np.ndarray:
    """One frame of the move, t from 0 (start) to 1 (end); strength (see move_strength) makes it go further.
    Positions are not rounded to whole pixels, so the picture glides instead of shaking."""
    ch, cw = canvas.shape[:2]
    t = t * t * (3 - 2 * t) * 0.35 + t * 0.65  # a little gentler at the start and end
    far = 1 + (ZOOM - 1) * strength
    zoom_in = 1 + (far - 1) * t
    zoom_out = far - (far - 1) * t
    zoom = {"in": zoom_in, "in-up": zoom_in, "out": zoom_out, "out-down": zoom_out}.get(move, 1 + 0.1 * strength)
    vw, vh = cw / zoom, ch / zoom  # the part of the canvas in view
    room_x, room_y = cw - vw, ch - vh
    x0, y0 = room_x / 2, room_y / 2
    tilt = min(0.45, 0.2 * strength)  # longer photos also travel up or down while zooming
    if move == "left":
        x0 = room_x * (1 - t)  # the view travels left: the picture slides right
    elif move == "right":
        x0 = room_x * t
    elif move == "in-up":
        y0 = room_y * (0.5 - tilt * t)
    elif move == "out-down":
        y0 = room_y * (0.5 + tilt * (t - 1) + tilt)
    m = np.float32([[vw / w, 0, x0], [0, vh / h, y0]])
    return cv2.warpAffine(canvas, m, (w, h), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_REFLECT)


def _encode_segment(inputs: list[str], graph: str, length: float, out: Path, s: RenderSettings, feed=None) -> None:
    # Exactly the planned number of frames (repeat the last one if the source falls a frame short),
    # so cuts land where they were planned, e.g. on the music's beat.
    frames = max(1, round(length * FPS))
    graph += ";[v]tpad=stop_mode=clone:stop_duration=1[vfull];[a]apad[afull]"
    _run_encode(
        lambda codec: [
            *inputs,
            "-filter_complex", graph,
            "-map", "[vfull]", "-map", "[afull]",
            "-frames:v", str(frames), "-t", f"{frames / FPS:.4f}",
            *codec, "-pix_fmt", "yuv420p", "-r", str(FPS),
            *AUDIO,
            str(out),
        ],
        s.width,
        s.height,
        final=False,
        feed=feed,
    )


def _segment_key(item: MediaItem, s: RenderSettings) -> str:
    """Name a prepared clip by everything that changes how it looks, so it can be reused safely."""
    stat = item.path.stat()
    parts = [SEGMENT_VERSION, item.kind, item.path, stat.st_mtime, stat.st_size, s.width, s.height, item.rotation,
             sorted((item.overlay or {}).items()), item.focus if tall(s) else None]
    if item.kind == "photo":
        parts += [item.seconds or s.photo_seconds, sorted((item.effects or {}).items()),
                  sorted((item.crop or {}).items())]
    elif item.kind == "title":
        parts += [item.title]
    else:
        parts += [clip_range(item, s), item.has_audio and item.sound and s.keep_clip_sound, item.volume * s.clip_volume,
                  item.speed, sorted((item.effects or {}).items()), item.fx_in, item.fx_out, item.reverse]
        if item.crop:
            parts += [sorted(item.crop.items())]
    return hashlib.sha1(repr(parts).encode()).hexdigest()[:20]


def _prune(cache_dir: Path) -> None:
    cutoff = time.time() - CACHE_DAYS * 86400
    for f in cache_dir.glob("*.mp4"):
        if f.stat().st_mtime < cutoff:
            f.unlink(missing_ok=True)


def _prepare_segment(item: MediaItem, s: RenderSettings, cache_dir: Path, workdir: Path) -> Path:
    seg = cache_dir / f"{_segment_key(item, s)}.mp4"
    if seg.exists():
        os.utime(seg)  # mark as recently used
        return seg
    tmp = workdir / f"{seg.stem}.mp4"
    if item.kind == "video" and (item.effects or {}).get("side"):
        _side_segment(item, s, tmp)
    elif item.kind == "video":
        _video_segment(item, s, tmp)
    elif (item.effects or {}).get("collage"):
        _collage_segment(item, s, tmp)
    else:
        _photo_segment(item, s, tmp, workdir)
    shutil.move(tmp, seg)
    return seg


def _segment_length(path: Path) -> float:
    """Length of a prepared clip's picture (its sound can run a few milliseconds longer)."""
    video = next(st for st in probe(str(path))["streams"] if st["codec_type"] == "video")
    if video.get("nb_frames", "").isdigit():
        return int(video["nb_frames"]) / FPS
    return float(video.get("duration") or probe(str(path))["format"]["duration"])


def render(
    items: list[MediaItem],
    settings: RenderSettings,
    out_path: str | Path,
    on_progress: ProgressFn | None = None,
    cache_dir: str | Path | None = None,
    timing: str | Path | None = None,
) -> Path:
    """Build the memory video. Progress is reported as (0..1, message).

    timing: a file to write where each shot starts and how long it is in the video ([{"start", "length"}]).

    Prepared clips are kept in cache_dir (when given), so making the video again after a small change
    only prepares what is new.
    """
    if not items:
        raise ValueError("There are no videos or photos to use.")
    report = on_progress or (lambda _f, _m: None)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    workdir = Path(tempfile.mkdtemp(prefix="trip-memory-"))
    cache = Path(cache_dir) if cache_dir else workdir / "segments"
    cache.mkdir(parents=True, exist_ok=True)
    _prune(cache)
    prep_share = 0.7  # rough share of the total time spent preparing clips
    workers = max(2, min(6, (os.cpu_count() or 4) // 2))

    try:
        # Zoom transitions are drawn on the shots on either side of the cut.
        items = [
            dataclasses.replace(
                item,
                fx_in=ZOOM_INTO.get((items[k - 1].transition or settings.transition) if k else "") if item.kind == "video" else None,
                fx_out=ZOOM_OUT_OF.get(item.transition or settings.transition) if item.kind == "video" and k < len(items) - 1 else None,
            )
            for k, item in enumerate(items)
        ]
        # The same file used twice is prepared once.
        unique = {_segment_key(item, settings): item for item in items}
        ready: dict[str, Path] = {}
        new = sum(1 for key in unique if not (cache / f"{key}.mp4").exists())
        report(0.0, f"Preparing {new} new or changed clips" + (f" (reusing {len(unique) - new} made before)"
                                                                 if new < len(unique) else ""))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            jobs = {pool.submit(_prepare_segment, item, settings, cache, workdir): key for key, item in unique.items()}
            for done, job in enumerate(as_completed(jobs), 1):
                ready[jobs[job]] = job.result()
                report(prep_share * done / len(unique), f"Prepared {done} of {len(unique)} clips"
                       + (f" ({len(unique) - new} reused)" if new < len(unique) else ""))
        segments = [ready[_segment_key(item, settings)] for item in items]
        lengths = [_segment_length(seg) for seg in segments]
        styles = [_transition_between(items[k - 1], k, settings) for k in range(1, len(items))]
        times = shot_times(lengths, styles)
        if timing:
            Path(timing).write_text(json.dumps(times))
        settings = dataclasses.replace(settings, titles=place_titles(settings.titles, times))
        if settings.end_black > 0:
            # The last shot fades to black; the end title fades in on the black. (An ending card with a
            # blurred look: the last picture blurs into the card instead.)
            seconds = settings.end_black + END_FADE
            card = (settings.end_title or {}).get("card")
            blurred = titles.CARDS.get(card, {}).get("bg") == "blur"
            last = segments[-1] if blurred else None
            black = cache / f"black-{_black_key(settings, seconds, last)}.mp4"
            if not black.exists():
                _black_segment(settings, seconds, workdir / "black.mp4", last)
                shutil.move(workdir / "black.mp4", black)
            segments.append(black)
            lengths.append(_segment_length(black))
            styles.append(("fade" if blurred else "fadeblack", END_FADE))

        report(prep_share, "Joining everything together")
        _join(segments, lengths, styles, settings, out_path,
              lambda f, message="Joining everything together": report(prep_share + (1 - prep_share) * f, message))
        report(1.0, "Done")
        return out_path
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


END_FADE = 1.2  # seconds for the last shot to fade to black


def _black_key(s: RenderSettings, seconds: float, last: Path | None = None) -> str:
    return hashlib.sha1(repr([SEGMENT_VERSION, s.width, s.height, round(seconds, 3),
                              sorted((k, repr(v)) for k, v in (s.end_title or {}).items()),
                              last.name if last else None]).encode()).hexdigest()[:20]


def _black_segment(s: RenderSettings, seconds: float, out: Path, last: Path | None = None) -> None:
    """The black ending, with the end title fading in once the picture has gone dark. With last (the last
    clip): its last picture, blurred in the ending card's look, instead of black."""
    title = s.end_title or {}
    card = title.get("card")
    if last is not None:
        still = out.with_suffix(".last.png")
        run_ffmpeg(["-sseof", "-0.25", "-i", str(last), "-frames:v", "1", "-update", "1", str(still)])
        inputs = ["-loop", "1", "-framerate", str(FPS), "-t", f"{seconds:.3f}", "-i", str(still)]
        background = f"[0:v]scale={s.width}:{s.height},setsar=1,{titles.card_filter(card, s.height)},fps={FPS},format=yuv420p"
    else:
        inputs = ["-f", "lavfi", "-t", f"{seconds:.3f}", "-i", f"color=black:s={s.width}x{s.height}:r={FPS}"]
        background = "[0:v]format=yuv420p"
    inputs += ["-f", "lavfi", "-t", f"{seconds:.3f}", "-i", "anullsrc=r=48000:cl=stereo"]
    graph = f"{background}[v];[1:a]anull[a]"
    if title.get("text") or title.get("sub") or titles.CARDS.get(card, {}).get("bars"):
        png = out.with_suffix(".title.png")
        if card in titles.CARDS:
            layer = titles.card_layer(card, title, s.width, s.height)
            if titles.CARDS[card]["bg"] == "black":  # (black already: just the words)
                layer = titles.draw(titles.card_words(card, title), s.width, s.height)
            layer.save(png)
        else:
            titles.draw(title, s.width, s.height).save(png)
        inputs += ["-loop", "1", "-t", f"{seconds:.3f}", "-i", str(png)]
        appear = min(END_FADE * 0.8, seconds / 3)  # as the last shot is nearly gone
        graph = (f"{background}[v0];[2:v]format=rgba,setpts=PTS-STARTPTS,"
                 f"fade=t=in:st={appear:.3f}:d=0.8:alpha=1[title];"
                 f"[v0][title]overlay=0:0:shortest=1:format=auto,format=yuv420p[v];[1:a]anull[a]")
    _encode_segment(inputs, graph, seconds, out, s)


def _transition_between(before: MediaItem, k: int, s: RenderSettings) -> tuple[str | None, float]:
    """The FFmpeg transition into clip k and how long it lasts: the one set after the previous clip, else the
    default."""
    return style_between(before.transition, k, s.transition, s.transition_seconds, before.transition_seconds)


def style_between(name: str | None, k: int, default: str | None, seconds: float,
                  own: float | None = None) -> tuple[str | None, float]:
    """The transition into clip k after a clip whose own transition is name (None: the video's default), lasting
    own seconds if you set it."""
    name = name or default
    style = TRANSITIONS.get(name, "fade")
    seconds = seconds_of(name if name in TRANSITIONS else "Fade", seconds)
    if own and style:
        seconds = float(own)
    return (MIXED[(k - 1) % len(MIXED)] if style == "mixed" else style), seconds


def overlaps(lengths: list[float], styles: list[tuple[str | None, float]]) -> list[float]:
    """How many seconds each transition overlaps the two clips around it (as shot_times counts it)."""
    return [(max(1, round(min(sec, lengths[k] * OVERLAP_SHARE, lengths[k + 1] * OVERLAP_SHARE) * FPS)) if style else 1) / FPS
            for k, (style, sec) in enumerate(styles)]


def _join(
    segments: list[Path],
    lengths: list[float],
    styles: list[tuple[str | None, float]],
    s: RenderSettings,
    out: Path,
    progress: Callable[[float], None],
) -> None:
    """Join the prepared clips. If the result comes out too short, try again more carefully: on the computer's
    main processor, then with every clip re-made in one plain format. A report is saved next to the video."""
    total = _planned_length(lengths, styles, s)
    made = _join_once(segments, lengths, styles, s, out, progress)
    if made >= total - 1.0:
        return
    tries = [f"First try: {made:.2f} s"]
    _write_report(out, segments, lengths, styles, total, tries)  # straight away, in case the retries take long
    saved = _hardware["ok"]
    _hardware["ok"] = False
    try:
        second = "Second try (it came out too short): on the main processor, slower"
        made = _join_once(segments, lengths, styles, s, out, lambda f: progress(f, second))
        tries.append(f"Main processor: {made:.2f} s")
        if made < total - 1.0:
            plain_dir = Path(tempfile.mkdtemp(prefix="trip-memory-plain-"))
            try:
                third = "Third try: re-making every clip in a plain format"
                plain = []
                for k, seg in enumerate(segments):
                    progress(k / len(segments) / 2, third)
                    plain.append(_plain_copy(seg, plain_dir / f"{k}.mp4"))
                plain_lengths = [_segment_length(p) for p in plain]
                made = _join_once(plain, plain_lengths, styles, s, out, lambda f: progress(0.5 + f / 2, third))
                tries.append(f"Plain clips: {made:.2f} s")
            finally:
                shutil.rmtree(plain_dir, ignore_errors=True)
    finally:
        _hardware["ok"] = saved
    report = _write_report(out, segments, lengths, styles, total, tries)
    if made < total - 1.0:  # never hand over a video that silently stops early
        raise FFmpegError(f"The finished video is only {made:.1f} s long instead of {total:.1f} s. "
                          f"Please send me the report saved at {report}.")


def place_titles(found: list[dict], times: list[dict]) -> list[dict]:
    """Timeline titles with their start in the video: a title linked to a shot starts that far into it."""
    out = []
    for t in found or []:
        if t.get("shot") is not None:
            if not 0 <= t["shot"] < len(times):
                continue
            start = times[t["shot"]]["start"] + float(t.get("offset") or 0)
        else:
            start = float(t.get("start") or 0)
        if (t.get("text") or t.get("sub")) and float(t.get("length") or 0) > 0.2:
            out.append(t | {"start": max(0.0, start)})
    return out


def title_filter(t: dict, n: int, label: str) -> str:
    """The title picture (input n) fading in and out at its own time in the joined video."""
    a, length = t["start"], float(t["length"])
    fade_in = min(float(t.get("fade_in", titles.FADE_IN) or 0), length / 2)
    fade_out = min(float(t.get("fade_out", titles.FADE_OUT) or 0), length / 2)
    fades = (f",fade=t=in:st={a:.3f}:d={fade_in:.3f}:alpha=1" if fade_in > 0 else "") + \
            (f",fade=t=out:st={a + length - fade_out:.3f}:d={fade_out:.3f}:alpha=1" if fade_out > 0 else "")
    return f"[{n}:v]format=rgba,setpts=PTS-STARTPTS{fades}[{label}]"


def shot_times(lengths: list[float], styles: list[tuple[str | None, float]]) -> list[dict]:
    """Where each shot starts in the joined video (a transition overlaps it with the one before) and how long
    it is seen, the same way _planned_length counts."""
    out, at = [], 0
    for k, length in enumerate(lengths):
        if k:
            style, seconds = styles[k - 1]
            at -= max(1, round(min(seconds, lengths[k - 1] * OVERLAP_SHARE, length * OVERLAP_SHARE) * FPS)) if style else 1
        out.append({"start": round(at / FPS, 3), "length": round(length, 3)})
        at += round(length * FPS)
    return out


def _planned_length(lengths: list[float], styles: list[tuple[str | None, float]], s: RenderSettings) -> float:
    total = round(lengths[0] * FPS)
    for k in range(1, len(lengths)):
        style, seconds = styles[k - 1]
        t = max(1, round(min(seconds, lengths[k - 1] * OVERLAP_SHARE, lengths[k] * OVERLAP_SHARE) * FPS)) if style else 1
        total += round(lengths[k] * FPS) - t
    return total / FPS


def _plain_copy(seg: Path, out: Path) -> Path:
    """The clip re-made in the plainest way: steady 30 pictures a second, a clean clock, standard sound."""
    run_ffmpeg([
        "-i", str(seg), "-vf", f"settb=AVTB,setpts=PTS-STARTPTS,fps={FPS},format=yuv420p,setsar=1",
        "-af", "asetpts=PTS-STARTPTS,aresample=48000", "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
        "-r", str(FPS), *AUDIO, str(out),
    ])
    return out


def _write_report(out: Path, segments, lengths, styles, total: float, tries: list[str]) -> Path:
    """What went into a video that came out too short, so the problem can be found."""
    lines = [f"Planned {total:.2f} s", *tries, ""]
    for k, (seg, length) in enumerate(zip(segments, lengths)):
        try:
            info = probe(str(seg))
            streams = [
                {key: st.get(key) for key in ("codec_type", "codec_name", "width", "height", "sample_aspect_ratio",
                                              "r_frame_rate", "avg_frame_rate", "time_base", "start_time", "duration",
                                              "nb_frames", "sample_rate")}
                for st in info["streams"]
            ]
        except Exception as exc:
            streams = [f"could not read: {exc}"]
        style = styles[k - 1] if k else None
        lines.append(f"{k}: {length:.3f} s, transition in {style}, {streams}")
    path = out.with_suffix(".report.txt")
    path.write_text("\n".join(lines))
    return path


HUE_SWING = 0.3  # seconds before and after a cut that the colours swing round
GLITCH = 0.13  # seconds a glitch lasts
FLASH_DECAY = 12  # how quickly a flash fades (higher: shorter)


def package_fx(fx: dict, cuts: list[float], total: float) -> str:
    """The style package's effects as one FFmpeg filter over the joined video ("" for none)."""
    chain = []
    flashes = [t for t in fx.get("flash") or [] if 0 < t < total]
    if flashes:
        kick = "+".join(f"gte(t,{t:.3f})*exp(-(t-{t:.3f})*{FLASH_DECAY})" for t in flashes[:60])
        chain.append(f"eq=brightness='min(0.5,0.45*({kick}))':eval=frame")
    if fx.get("hue") and cuts:
        swing = "+".join(f"max(0,1-abs(t-{c:.3f})/{HUE_SWING})" for c in cuts[:120])
        chain.append(f"hue=h='120*({swing})'")
    if fx.get("glitch") and cuts:
        picked = cuts[1::3][:40]  # (every third cut: often enough to feel it, not so often it tires)
        if picked:
            on = "+".join(f"between(t,{c - GLITCH / 2:.3f},{c + GLITCH / 2:.3f})" for c in picked)
            if has_filter("rgbashift"):
                chain.append(f"rgbashift=rh=-14:bh=14:gv=4:enable='{on}'")
            chain.append(f"noise=alls=28:allf=t:enable='{on}'")
    return ",".join(chain)


def _join_once(
    segments: list[Path],
    lengths: list[float],
    styles: list[tuple[str | None, float]],
    s: RenderSettings,
    out: Path,
    progress: Callable[[float], None],
) -> float:
    n = len(segments)
    # Everything is counted in whole frames, so every transition starts and ends exactly on a frame.
    frames = [max(1, round(length * FPS)) for length in lengths]
    t_frames = [max(1, round(min(sec, lengths[k] * OVERLAP_SHARE, lengths[k + 1] * OVERLAP_SHARE) * FPS))
                if style else 1 for k, (style, sec) in enumerate(styles[:n - 1])]  # (cut k: after clip k)
    cuts: list[float] = []  # (the middle of each cut, in the joined video)
    at = frames[0]
    for k in range(1, n):
        cuts.append((at - t_frames[k - 1] / 2) / FPS)
        at += frames[k] - t_frames[k - 1]
    # A long video is joined in parts first (a few clips each), then the parts: joining a hundred clips at once
    # needs more memory than a laptop has (FFmpeg stopped without a word).
    tmp = None
    if n > JOIN_PART:
        tmp = Path(tempfile.mkdtemp(prefix="trip-memory-join-"))
        groups = [list(range(a, min(n, a + JOIN_PART))) for a in range(0, n, JOIN_PART)]
        pieces, piece_frames = [], []
        for g, idx in enumerate(groups):
            piece = tmp / f"part{g}.mkv"
            _join_plain([segments[k] for k in idx], [frames[k] for k in idx], [styles[k] for k in idx[:-1]],
                        [t_frames[k] for k in idx[:-1]], s, piece,
                        lambda f, g=g: progress(0.5 * (g + f) / len(groups)))
            pieces.append(piece)
            piece_frames.append(sum(frames[k] for k in idx) - sum(t_frames[k] for k in idx[:-1]))
        joins = [(styles[idx[-1]], t_frames[idx[-1]]) for idx in groups[:-1]]
        segments, frames, n = pieces, piece_frames, len(pieces)
        outer = progress
        progress = lambda f: outer(0.5 + 0.5 * f)  # noqa: E731
    else:
        joins = [(styles[k], t_frames[k]) for k in range(n - 1)]
    inputs: list[str] = []
    for seg in segments:
        inputs += ["-i", str(seg)]
    parts: list[str] = []
    v_prev, a_prev, total_frames = _xfade_chain(n, frames, joins, parts)
    total = total_frames / FPS
    fx = package_fx(s.fx or {}, cuts, total)
    if fx:
        parts.append(f"[{v_prev}]{fx}[pfx]")
        v_prev = "pfx"

    # The opening card over the start of the first shot: the shot blurred (or black) with the trip's title,
    # then it clears to the shot.
    pngs = []
    card = s.opening or {}
    if card.get("card") in titles.CARDS:
        spec = titles.CARDS[card["card"]]
        length = max(1.0, min(float(card.get("seconds") or 3.0), lengths[0] - 0.3 if n > 1 else total - 0.5, total * 0.6))
        fade = min(titles.CARD_FADE, length / 2)
        v_in = v_prev
        if spec["bg"] == "blur":
            parts.append(f"[{v_prev}]split=2[ocm][ocb]")
            parts.append(f"[ocb]trim=end={length:.3f},setpts=PTS-STARTPTS,{titles.card_filter(card['card'], s.height)},"
                         f"format=rgba,fade=t=out:st={length - fade:.3f}:d={fade:.3f}:alpha=1[ocbg]")
            parts.append("[ocm][ocbg]overlay=0:0:eof_action=pass:format=auto,format=yuv420p[ocv]")
            v_in = "ocv"
        png = out.with_name(f"{out.stem}.opening.png")
        titles.card_layer(card["card"], card, s.width, s.height).save(png)
        pngs.append(png)
        n_in = inputs.count("-i")
        inputs += ["-loop", "1", "-t", f"{length:.3f}", "-i", str(png)]
        parts.append(f"[{n_in}:v]format=rgba,setpts=PTS-STARTPTS,fade=t=in:st=0.15:d=0.5:alpha=1,"
                     f"fade=t=out:st={length - fade:.3f}:d={fade:.3f}:alpha=1[ocw]"
                     if spec["bg"] == "blur" else
                     f"[{n_in}:v]format=rgba,setpts=PTS-STARTPTS,fade=t=out:st={length - fade:.3f}:d={fade:.3f}:alpha=1[ocw]")
        parts.append(f"[{v_in}][ocw]overlay=0:0:eof_action=pass:format=auto,format=yuv420p[ocd]")
        v_prev = "ocd"

    # The ending card over the last seconds: the shot blurred (or black) with your words, fading in.
    card = s.closing or {}
    if card and (card.get("text") or card.get("sub") or card.get("card")):
        spec = titles.CARDS.get(card.get("card") or "") or {"bg": "black"}
        length = max(1.0, min(float(card.get("seconds") or 3.0), lengths[-1] - 0.3 if n > 1 else total - 0.5,
                              total * 0.6))
        start = max(0.0, total - length)
        fade = min(titles.CARD_FADE, length / 2)
        v_in = v_prev
        if card.get("card") in titles.CARDS and spec["bg"] == "blur":
            parts.append(f"[{v_prev}]split=2[ecm][ecb]")
            parts.append(f"[ecb]trim=start={start:.3f},setpts=PTS-STARTPTS,"
                         f"{titles.card_filter(card['card'], s.height)},format=rgba,"
                         f"fade=t=in:st=0:d={fade:.3f}:alpha=1,setpts=PTS+{start:.3f}/TB[ecbg]")
            parts.append("[ecm][ecbg]overlay=0:0:eof_action=pass:format=auto,format=yuv420p[ecv]")
            v_in = "ecv"
        png = out.with_name(f"{out.stem}.closing.png")
        if card.get("card") in titles.CARDS:
            titles.card_layer(card["card"], card, s.width, s.height).save(png)
        else:  # words on black
            layer = Image.new("RGBA", (s.width, s.height), (0, 0, 0, 255))
            if card.get("text") or card.get("sub"):
                layer = Image.alpha_composite(layer, titles.draw(
                    {"text": card.get("text") or "", "sub": card.get("sub") or "", "style": "elegant",
                     "where": "center"}, s.width, s.height))
            layer.save(png)
        pngs.append(png)
        n_in = inputs.count("-i")
        inputs += ["-loop", "1", "-t", f"{length:.3f}", "-i", str(png)]
        parts.append(f"[{n_in}:v]format=rgba,fade=t=in:st=0:d={fade:.3f}:alpha=1,"
                     f"setpts=PTS-STARTPTS+{start:.3f}/TB[ecw]")
        parts.append(f"[{v_in}][ecw]overlay=0:0:eof_action=pass:format=auto,format=yuv420p[ecd]")
        v_prev = "ecd"

    # Titles on the video's timeline (the editor's), each fading in and out at its own time.
    for k, t in enumerate(s.titles or []):
        if t["start"] >= total:
            continue
        png = out.with_name(f"{out.stem}.title{k}.png")
        titles.draw(t, s.width, s.height).save(png)
        pngs.append(png)
        n_in = inputs.count("-i")
        inputs += ["-loop", "1", "-t", f"{total:.3f}", "-i", str(png)]
        parts.append(title_filter(t, n_in, f"tt{k}"))
        end = min(total, t["start"] + float(t["length"]))
        parts.append(f"[{v_prev}][tt{k}]overlay=0:0:enable='between(t,{t['start']:.3f},{end:.3f})':"
                     f"format=auto,format=yuv420p[tv{k}]")
        v_prev = f"tv{k}"
    # Gentle fade in from black at the start and out at the end.
    edge = min(1.0, total / 4)
    parts.append(f"[{v_prev}]format=yuv420p{look_filter(s.look)},format=yuv420p,"
                 f"fade=t=in:d={edge:.2f},fade=t=out:st={total - edge:.3f}:d={edge:.2f}[vout]")
    sound = f"[{a_prev}]afade=t=in:d={edge:.2f},afade=t=out:st={total - edge:.3f}:d={edge:.2f}"

    music_in = inputs.count("-i")
    if s.music_path:
        inputs += ["-stream_loop", "-1", "-i", s.music_path]
        fade = min(max(3.0, s.end_black + END_FADE), total / 4)  # the music fades out across the black ending
        parts.append(f"{sound}[clips]")
        parts.append(
            f"[{music_in}:a]aformat=sample_rates=48000:channel_layouts=stereo,atrim=0:{total:.3f},"
            f"volume={s.music_volume},afade=t=out:st={total - fade:.3f}:d={fade:.2f}[music]"
        )
        parts.append("[clips][music]amix=inputs=2:duration=first:normalize=0[aout]")
    else:
        parts.append(f"{sound}[aout]")

    _run_encode(
        lambda codec: [
            *inputs,
            "-filter_complex", ";".join(parts),
            "-map", "[vout]", "-map", "[aout]",
            *codec, "-pix_fmt", "yuv420p",
            *AUDIO,
            "-movflags", "+faststart",
            str(out),
        ],
        s.width,
        s.height,
        final=True,
        total_seconds=total,
        on_progress=progress,
    )
    for png in pngs:
        png.unlink(missing_ok=True)
    if tmp is not None:
        shutil.rmtree(tmp, ignore_errors=True)
    return _segment_length(out)


JOIN_PART = 12  # clips joined together at a time in a long video (see _join_once)


def _xfade_chain(n: int, frames: list[int], joins: list[tuple[tuple[str | None, float], int]],
                 parts: list[str]) -> tuple[str, str, int]:
    """The filters that put inputs 0..n-1 one after the other, each cut overlapping the clips by its transition
    (a hard cut is a one-frame fade, so all cuts are timed the same way). Returns (video, audio, frames)."""
    # Every clip starts at time 0 with the same clock and frame rate. (Clips from the Mac's video chip can
    # start a little after 0, which made joining stop after the first clip.)
    for k in range(n):
        parts.append(f"[{k}:v]settb=AVTB,setpts=PTS-STARTPTS,fps={FPS},format=yuv420p[sv{k}]")
        parts.append(f"[{k}:a]asetpts=PTS-STARTPTS,aresample=48000[sa{k}]")
    v_prev, a_prev, total_frames = "sv0", "sa0", frames[0]
    for k in range(1, n):
        (style, _), t_frames = joins[k - 1]
        offset_frames = total_frames - t_frames
        t, offset = t_frames / FPS, offset_frames / FPS
        parts.append(f"[{v_prev}][sv{k}]xfade={xfade_args(style or 'fade')}:duration={t:.6f}:offset={offset:.6f}[v{k}]")
        parts.append(f"[{a_prev}][sa{k}]acrossfade=d={t:.6f}[a{k}]")
        v_prev, a_prev = f"v{k}", f"a{k}"
        total_frames = offset_frames + frames[k]
    return v_prev, a_prev, total_frames


def _join_plain(segments: list[Path], frames: list[int], styles: list[tuple[str | None, float]],
                t_frames: list[int], s: RenderSettings, out: Path, progress: Callable[[float], None]) -> None:
    """A few clips joined with their transitions, nothing else (a part of a long video, see _join_once). Kept in
    high quality with plain sound, so joining the parts loses nothing and stays in time."""
    inputs: list[str] = []
    for seg in segments:
        inputs += ["-i", str(seg)]
    parts: list[str] = []
    v, a, total_frames = _xfade_chain(len(segments), frames, list(zip(styles, t_frames)), parts)
    parts.append(f"[{v}]format=yuv420p[pv]")
    parts.append(f"[{a}]anull[pa]")
    _run_encode(
        lambda codec: [*inputs, "-filter_complex", ";".join(parts), "-map", "[pv]", "-map", "[pa]",
                       *codec, "-pix_fmt", "yuv420p", "-c:a", "pcm_s16le", "-ar", "48000", "-ac", "2", str(out)],
        s.width, s.height, final=False, total_seconds=total_frames / FPS, on_progress=progress)
