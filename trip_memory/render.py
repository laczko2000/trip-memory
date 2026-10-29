"""Turn a list of videos and photos into one finished memory video with FFmpeg.

Every item is first converted into a short "segment" with the same size, frame rate and
sound format. The segments are then joined with transitions, an optional music track is
mixed in, and the result is saved as one MP4.
"""

from __future__ import annotations

import hashlib
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

from .media import MediaItem
from .transitions import MIXED, STYLES, seconds_of
from .tools import FFmpegError, has_encoder, has_filter, probe, run_ffmpeg

# Friendly name -> FFmpeg xfade transition. None means a hard cut.
TRANSITIONS: dict[str, str | None] = {name: style.xfade for name, style in STYLES.items()}

FPS = 30
AUDIO = ["-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-ac", "2"]

ProgressFn = Callable[[float, str], None]

# Bump when the way clips are prepared changes, so old remembered clips are not reused.
SEGMENT_VERSION = 5
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
        return ["-c:v", "h264_videotoolbox", "-b:v", str(bitrate), "-allow_sw", "1"]
    if final:
        return ["-c:v", "libx264", "-preset", "faster", "-crf", "20"]
    return ["-c:v", "libx264", "-preset", "veryfast", "-crf", "18"]


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


def _fit_with_blur(src: str, w: int, h: int, out: str) -> str:
    """Fit the picture inside w x h; fill empty sides with a blurred copy (for vertical shots)."""
    return (
        f"[{src}]split=2[bg_{out}][fg_{out}];"
        f"[bg_{out}]scale={w // 4}:{h // 4}:force_original_aspect_ratio=increase,"
        f"crop={w // 4}:{h // 4},boxblur=10:2,scale={w}:{h},setsar=1[bgs_{out}];"
        f"[fg_{out}]scale={w}:{h}:force_original_aspect_ratio=decrease,setsar=1[fgs_{out}];"
        f"[bgs_{out}][fgs_{out}]overlay=(W-w)/2:(H-h)/2[{out}]"
    )


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


def _video_segment(item: MediaItem, s: RenderSettings, out: Path) -> float:
    start, end = clip_range(item, s)
    speed = min(2.0, max(0.5, item.speed or 1.0))  # the range the sound can follow too
    length = (end - start) / speed  # on screen
    pre = (_hdr_to_sdr() if item.hdr else "") + rotate_filter(item.rotation)
    if abs(speed - 1) > 1e-3:
        pre = f"setpts=(PTS-STARTPTS)/{speed:.4f}," + pre  # slow motion or sped up
    graph = f"[0:v]{pre}format=yuv420p[src];" + _fit_with_blur("src", s.width, s.height, "fit")
    graph += f";[fit]fps={FPS},format=yuv420p[v]"
    inputs = ["-ss", f"{start:.3f}", "-t", f"{end - start:.3f}", "-i", str(item.path)]
    if item.has_audio and item.sound and s.keep_clip_sound:
        tempo = f"atempo={speed:.4f}," if abs(speed - 1) > 1e-3 else ""  # the sound keeps its pitch
        graph += f";[0:a]{tempo}aformat=sample_rates=48000:channel_layouts=stereo,volume={item.volume * s.clip_volume:.2f}[a]"
    else:
        inputs += ["-f", "lavfi", "-t", f"{length:.3f}", "-i", "anullsrc=r=48000:cl=stereo"]
        graph += ";[1:a]anull[a]"
    _encode_segment(inputs, graph, length, out, s)
    return length


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
            img.thumbnail((3840, 3840))
            img.save(still, quality=92)

    with Image.open(still) as img:
        canvas = _photo_canvas(img.convert("RGB"), s.width, s.height)
    move = "in" if item.kind == "title" else _pick_move(item.path, canvas_is_tall=_is_tall(still))
    frames = max(1, round(seconds * FPS))
    w, h = s.width, s.height

    def feed(pipe) -> None:
        for n in range(frames):
            pipe.write(_photo_frame(canvas, w, h, move, n / max(1, frames - 1)).tobytes())

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


def _photo_canvas(img: Image.Image, w: int, h: int) -> np.ndarray:
    """The photo fitted into the video frame (bigger, for zooming); empty sides filled with a blurred copy."""
    cw, ch = round(w * CANVAS), round(h * CANVAS)
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
    parts = [SEGMENT_VERSION, item.kind, item.path, stat.st_mtime, stat.st_size, s.width, s.height, item.rotation]
    if item.kind == "photo":
        parts += [item.seconds or s.photo_seconds]
    elif item.kind == "title":
        parts += [item.title]
    else:
        parts += [clip_range(item, s), item.has_audio and item.sound and s.keep_clip_sound, item.volume * s.clip_volume,
                  item.speed]
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
) -> Path:
    """Build the memory video. Progress is reported as (0..1, message).

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
        # The same file used twice is prepared once.
        unique = {_segment_key(item, settings): item for item in items}
        ready: dict[str, Path] = {}
        report(0.0, f"Preparing {len(unique)} clips")
        with ThreadPoolExecutor(max_workers=workers) as pool:
            jobs = {pool.submit(_prepare_segment, item, settings, cache, workdir): key for key, item in unique.items()}
            for done, job in enumerate(as_completed(jobs), 1):
                ready[jobs[job]] = job.result()
                report(prep_share * done / len(unique), f"Prepared {done} of {len(unique)} clips")
        segments = [ready[_segment_key(item, settings)] for item in items]
        lengths = [_segment_length(seg) for seg in segments]

        report(prep_share, "Joining everything together")
        styles = [_transition_between(items[k - 1], k, settings) for k in range(1, len(items))]
        _join(segments, lengths, styles, settings, out_path, lambda f: report(prep_share + (1 - prep_share) * f, "Joining everything together"))
        report(1.0, "Done")
        return out_path
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


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
    saved = _hardware["ok"]
    _hardware["ok"] = False
    try:
        made = _join_once(segments, lengths, styles, s, out, progress)
        tries.append(f"Main processor: {made:.2f} s")
        if made < total - 1.0:
            plain_dir = Path(tempfile.mkdtemp(prefix="trip-memory-plain-"))
            try:
                plain = [_plain_copy(seg, plain_dir / f"{k}.mp4") for k, seg in enumerate(segments)]
                plain_lengths = [_segment_length(p) for p in plain]
                made = _join_once(plain, plain_lengths, styles, s, out, progress)
                tries.append(f"Plain clips: {made:.2f} s")
            finally:
                shutil.rmtree(plain_dir, ignore_errors=True)
    finally:
        _hardware["ok"] = saved
    report = _write_report(out, segments, lengths, styles, total, tries)
    if made < total - 1.0:  # never hand over a video that silently stops early
        raise FFmpegError(f"The finished video is only {made:.1f} s long instead of {total:.1f} s. "
                          f"Please send me the report saved at {report}.")


def _planned_length(lengths: list[float], styles: list[tuple[str | None, float]], s: RenderSettings) -> float:
    total = lengths[0]
    for k in range(1, len(lengths)):
        style, seconds = styles[k - 1]
        t = min(seconds, lengths[k - 1] * 0.4, lengths[k] * 0.4) if style else 1 / FPS
        total += lengths[k] - t
    return total


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
    v_prev, a_prev, offset, total = "sv0", "sa0", 0.0, lengths[0]
    for k in range(1, n):
        style, seconds = styles[k - 1]
        # A transition is never longer than ~40% of either clip.
        t = min(seconds, lengths[k - 1] * 0.4, lengths[k] * 0.4) if style else 1 / FPS
        offset = total - t
        parts.append(f"[{v_prev}][sv{k}]xfade=transition={style or 'fade'}:duration={t:.3f}:offset={offset:.3f}[v{k}]")
        parts.append(f"[{a_prev}][sa{k}]acrossfade=d={t:.3f}[a{k}]")
        v_prev, a_prev = f"v{k}", f"a{k}"
        total = offset + lengths[k]

    # Gentle fade in from black at the start and out at the end.
    edge = min(1.0, total / 4)
    parts.append(f"[{v_prev}]fade=t=in:d={edge:.2f},fade=t=out:st={total - edge:.3f}:d={edge:.2f}[vout]")
    sound = f"[{a_prev}]afade=t=in:d={edge:.2f},afade=t=out:st={total - edge:.3f}:d={edge:.2f}"

    if s.music_path:
        inputs += ["-stream_loop", "-1", "-i", s.music_path]
        fade = min(3.0, total / 4)
        parts.append(f"{sound}[clips]")
        parts.append(
            f"[{n}:a]aformat=sample_rates=48000:channel_layouts=stereo,atrim=0:{total:.3f},"
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
    return _segment_length(out)
