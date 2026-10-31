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
from .transitions import MIXED, STYLES, ZOOM_INTO, ZOOM_OUT_OF, seconds_of
from .tools import FFmpegError, has_encoder, has_filter, probe, run_ffmpeg

# Friendly name -> FFmpeg xfade transition. None means a hard cut.
TRANSITIONS: dict[str, str | None] = {name: style.xfade for name, style in STYLES.items()}

FPS = 30
AUDIO = ["-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-ac", "2"]

ProgressFn = Callable[[float, str], None]

# Bump when the way clips are prepared changes, so old remembered clips are not reused.
SEGMENT_VERSION = 7
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
    end_title: dict | None = None  # words shown on that black ending (see titles.py)
    look: str = "natural"  # the colour mood of the whole video (see looks.py)
    # Titles on the video's own timeline (the editor under the video): {"text", "sub", "style", "where",
    # "length", "fade_in", "fade_out", and "start" (seconds into the video) or "shot" (its number) + "offset"}.
    titles: list = dataclasses.field(default_factory=list)


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
    look = (_hdr_to_sdr() if item.hdr else "") + rotate_filter(item.rotation)
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
        title_png = out.with_suffix(".title.png")
        titles.draw(item.overlay, s.width, s.height).save(title_png)
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
        inputs += ["-loop", "1", "-t", f"{length + 0.5:.3f}", "-i", str(title_png)]
        graph = graph.replace(",format=yuv420p[v]", ",format=yuv420p[v0]", 1)
        graph += (f";[{n}:v]format=rgba,setpts=PTS-STARTPTS,{titles.ffmpeg_fades(length)}[title];"
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
            frame = _photo_frame(canvas, w, h, move, n / max(1, frames - 1))
            if layer is not None:
                frame = titles.blend(frame, layer, titles.strength(n / FPS, frames / FPS))
            pipe.write(frame.tobytes())

    inputs = [
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", str(FPS), "-i", "pipe:0",
        "-f", "lavfi", "-t", f"{seconds:.3f}", "-i", "anullsrc=r=48000:cl=stereo",
    ]
    _encode_segment(inputs, "[0:v]format=yuv420p[v];[1:a]anull[a]", seconds, out, s, feed=feed)
    return seconds


POP = 0.18  # seconds a collage tile takes to pop into place
COLLAGE_GAP_SHARE = 0.008  # the thin dark line between tiles, as a share of the video's width


def collage_tiles(count: int, w: int, h: int) -> list[tuple[int, int, int, int]]:
    """Where each photo of a collage goes (x, y, width, height): 2 side by side, 3 as one big and two small,
    4 as a 2x2 grid. An upright (phone) video stacks them instead."""
    g = max(2, round(w * COLLAGE_GAP_SHARE))
    upright = h > w
    half_w, half_h = (w - g) // 2, (h - g) // 2
    if count == 2:
        return [(0, 0, w, half_h), (0, h - half_h, w, half_h)] if upright else \
            [(0, 0, half_w, h), (w - half_w, 0, half_w, h)]
    if count == 3:
        if upright:
            return [(0, 0, w, half_h), (0, h - half_h, half_w, half_h), (w - half_w, h - half_h, half_w, half_h)]
        return [(0, 0, half_w, h), (w - half_w, 0, half_w, half_h), (w - half_w, h - half_h, half_w, half_h)]
    return [(0, 0, half_w, half_h), (w - half_w, 0, half_w, half_h),
            (0, h - half_h, half_w, half_h), (w - half_w, h - half_h, half_w, half_h)]


def _collage_segment(item: MediaItem, s: RenderSettings, out: Path) -> float:
    """Several photos on screen together, popping in one after another (on the beats where they'd have been cut)."""
    fx = item.effects or {}
    paths = [item.path, *map(Path, fx["collage"])][:4]
    turns = [item.rotation, *(fx.get("turns") or [0] * len(fx["collage"]))]
    appear = list(fx.get("appear") or [0.0] * len(paths))
    seconds = item.seconds or s.photo_seconds
    w, h = s.width, s.height
    tiles = []
    for path, turn, (x, y, tw, th) in zip(paths, turns, collage_tiles(len(paths), w, h)):
        with Image.open(path) as img:
            img = ImageOps.exif_transpose(img).convert("RGB").rotate(-turn, expand=True)
            img.thumbnail((2400, 2400))
            tiles.append((x, y, np.asarray(ImageOps.fit(img, (tw, th), Image.LANCZOS))))
    frames = max(1, round(seconds * FPS))
    layer = np.asarray(titles.draw(item.overlay, w, h)) if item.overlay and (
        item.overlay.get("text") or item.overlay.get("sub")) else None

    def feed(pipe) -> None:
        for n in range(frames):
            t = n / FPS
            frame = np.full((h, w, 3), 12, np.uint8)
            for (x, y, tile), start in zip(tiles, appear):
                if t < start:
                    continue
                grow = min(1.0, (t - start) / POP)
                grow = 1 - (1 - grow) ** 3  # quick at first, settling into place
                if grow >= 1.0:
                    frame[y:y + tile.shape[0], x:x + tile.shape[1]] = tile
                    continue
                scale = 0.82 + 0.18 * grow
                th, tw = tile.shape[:2]
                small = cv2.resize(tile, (max(1, round(tw * scale)), max(1, round(th * scale))))
                oy, ox = y + (th - small.shape[0]) // 2, x + (tw - small.shape[1]) // 2
                region = frame[oy:oy + small.shape[0], ox:ox + small.shape[1]]
                frame[oy:oy + small.shape[0], ox:ox + small.shape[1]] = (region * (1 - grow) + small * grow).astype(np.uint8)
            if layer is not None:
                frame = titles.blend(frame, layer, titles.strength(t, seconds))
            pipe.write(frame.tobytes())

    inputs = [
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", str(FPS), "-i", "pipe:0",
        "-f", "lavfi", "-t", f"{seconds:.3f}", "-i", "anullsrc=r=48000:cl=stereo",
    ]
    _encode_segment(inputs, "[0:v]format=yuv420p[v];[1:a]anull[a]", seconds, out, s, feed=feed)
    return seconds


# Photos move gently while on screen (the "Ken Burns" look). Each photo gets one of these moves, chosen from
# its file name so it stays the same every time you make the video.
ZOOM = 1.15  # how far in the moves go (15%)
MOVES = ["in", "out", "left", "right", "in-up", "out-down"]
CANVAS = 1.25  # the photo is prepared this much bigger than the video, so zooming in stays sharp


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


def _photo_frame(canvas: np.ndarray, w: int, h: int, move: str, t: float) -> np.ndarray:
    """One frame of the move, t from 0 (start) to 1 (end). Positions are not rounded to whole pixels,
    so the picture glides instead of shaking."""
    ch, cw = canvas.shape[:2]
    t = t * t * (3 - 2 * t) * 0.35 + t * 0.65  # a little gentler at the start and end
    zoom_in = 1 + (ZOOM - 1) * t
    zoom = {"in": zoom_in, "in-up": zoom_in, "out": ZOOM - (ZOOM - 1) * t, "out-down": ZOOM - (ZOOM - 1) * t}.get(move, 1.1)
    vw, vh = cw / zoom, ch / zoom  # the part of the canvas in view
    room_x, room_y = cw - vw, ch - vh
    x0, y0 = room_x / 2, room_y / 2
    if move == "left":
        x0 = room_x * (1 - t)  # the view travels left: the picture slides right
    elif move == "right":
        x0 = room_x * t
    elif move == "in-up":
        y0 = room_y * 0.3
    elif move == "out-down":
        y0 = room_y * 0.7
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
    if item.kind == "video":
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
            # The last shot fades to black; the end title fades in on the black.
            seconds = settings.end_black + END_FADE
            black = cache / f"black-{_black_key(settings, seconds)}.mp4"
            if not black.exists():
                _black_segment(settings, seconds, workdir / "black.mp4")
                shutil.move(workdir / "black.mp4", black)
            segments.append(black)
            lengths.append(_segment_length(black))
            styles.append(("fadeblack", END_FADE))

        report(prep_share, "Joining everything together")
        _join(segments, lengths, styles, settings, out_path,
              lambda f, message="Joining everything together": report(prep_share + (1 - prep_share) * f, message))
        report(1.0, "Done")
        return out_path
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


END_FADE = 1.2  # seconds for the last shot to fade to black


def _black_key(s: RenderSettings, seconds: float) -> str:
    return hashlib.sha1(repr([SEGMENT_VERSION, s.width, s.height, round(seconds, 3),
                              sorted((s.end_title or {}).items())]).encode()).hexdigest()[:20]


def _black_segment(s: RenderSettings, seconds: float, out: Path) -> None:
    """The black ending, with the end title fading in once the picture has gone dark."""
    inputs = ["-f", "lavfi", "-t", f"{seconds:.3f}", "-i", f"color=black:s={s.width}x{s.height}:r={FPS}",
              "-f", "lavfi", "-t", f"{seconds:.3f}", "-i", "anullsrc=r=48000:cl=stereo"]
    graph = "[0:v]format=yuv420p[v];[1:a]anull[a]"
    title = s.end_title or {}
    if title.get("text") or title.get("sub"):
        png = out.with_suffix(".title.png")
        titles.draw(title, s.width, s.height).save(png)
        inputs += ["-loop", "1", "-t", f"{seconds:.3f}", "-i", str(png)]
        appear = min(END_FADE * 0.8, seconds / 3)  # as the last shot is nearly gone
        graph = (f"[0:v]format=yuv420p[v0];[2:v]format=rgba,setpts=PTS-STARTPTS,"
                 f"fade=t=in:st={appear:.3f}:d=0.8:alpha=1[title];"
                 f"[v0][title]overlay=0:0:shortest=1:format=auto,format=yuv420p[v];[1:a]anull[a]")
    _encode_segment(inputs, graph, seconds, out, s)


def _transition_between(before: MediaItem, k: int, s: RenderSettings) -> tuple[str | None, float]:
    """The FFmpeg transition into clip k and how long it lasts: the one set after the previous clip, else the
    default."""
    name = before.transition or s.transition
    style = TRANSITIONS.get(name, "fade")
    seconds = seconds_of(name if name in TRANSITIONS else "Fade", s.transition_seconds)
    return (MIXED[(k - 1) % len(MIXED)] if style == "mixed" else style), seconds


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
            at -= max(1, round(min(seconds, lengths[k - 1] * 0.4, length * 0.4) * FPS)) if style else 1
        out.append({"start": round(at / FPS, 3), "length": round(length, 3)})
        at += round(length * FPS)
    return out


def _planned_length(lengths: list[float], styles: list[tuple[str | None, float]], s: RenderSettings) -> float:
    total = round(lengths[0] * FPS)
    for k in range(1, len(lengths)):
        style, seconds = styles[k - 1]
        t = max(1, round(min(seconds, lengths[k - 1] * 0.4, lengths[k] * 0.4) * FPS)) if style else 1
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


def _join_once(
    segments: list[Path],
    lengths: list[float],
    styles: list[tuple[str | None, float]],
    s: RenderSettings,
    out: Path,
    progress: Callable[[float], None],
) -> float:
    n = len(segments)
    inputs: list[str] = []
    for seg in segments:
        inputs += ["-i", str(seg)]

    parts: list[str] = []
    # Every clip starts at time 0 with the same clock and frame rate. (Clips from the Mac's video chip can
    # start a little after 0, which made joining stop after the first clip.)
    for k in range(n):
        parts.append(f"[{k}:v]settb=AVTB,setpts=PTS-STARTPTS,fps={FPS},format=yuv420p[sv{k}]")
        parts.append(f"[{k}:a]asetpts=PTS-STARTPTS,aresample=48000[sa{k}]")
    # Every cut overlaps the clips by the transition's length; a hard cut is a one-frame fade. (Joining
    # all cuts the same way keeps their timing predictable, e.g. for cutting on the music's beat.)
    # Everything is counted in whole frames, so every transition starts and ends exactly on a frame.
    frames = [max(1, round(length * FPS)) for length in lengths]
    v_prev, a_prev, total_frames = "sv0", "sa0", frames[0]
    for k in range(1, n):
        style, seconds = styles[k - 1]
        # A transition is never longer than ~40% of either clip.
        t_frames = max(1, round(min(seconds, lengths[k - 1] * 0.4, lengths[k] * 0.4) * FPS)) if style else 1
        offset_frames = total_frames - t_frames
        t, offset = t_frames / FPS, offset_frames / FPS
        parts.append(f"[{v_prev}][sv{k}]xfade=transition={style or 'fade'}:duration={t:.6f}:offset={offset:.6f}[v{k}]")
        parts.append(f"[{a_prev}][sa{k}]acrossfade=d={t:.6f}[a{k}]")
        v_prev, a_prev = f"v{k}", f"a{k}"
        total_frames = offset_frames + frames[k]
    total = total_frames / FPS

    # Titles on the video's timeline (the editor's), each fading in and out at its own time.
    pngs = []
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
    return _segment_length(out)
