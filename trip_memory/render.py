"""Turn a list of videos and photos into one finished memory video with FFmpeg.

Every item is first converted into a short "segment" with the same size, frame rate and
sound format. The segments are then joined with transitions, an optional music track is
mixed in, and the result is saved as one MP4.
"""

from __future__ import annotations

import shutil
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageOps

from .media import MediaItem
from .tools import has_filter, probe, run_ffmpeg

# Friendly name -> FFmpeg xfade transition. None means a hard cut.
TRANSITIONS: dict[str, str | None] = {
    "Fade": "fade",
    "Fade through black": "fadeblack",
    "Dissolve": "dissolve",
    "Slide": "slideleft",
    "Wipe": "wipeleft",
    "Circle": "circleopen",
    "Zoom": "zoomin",
    "Mixed": "mixed",
    "None (hard cut)": None,
}
MIXED = ["fade", "slideleft", "circleopen", "dissolve", "wipeleft", "fadeblack", "zoomin", "slideup"]

FPS = 30
AUDIO = ["-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-ac", "2"]

ProgressFn = Callable[[float, str], None]


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


def _video_segment(item: MediaItem, s: RenderSettings, out: Path) -> float:
    length = item.duration
    start = 0.0
    if s.max_clip_seconds and length > s.max_clip_seconds:
        start = (length - s.max_clip_seconds) / 2  # keep the middle part
        length = s.max_clip_seconds
    pre = _hdr_to_sdr() if item.hdr else ""
    graph = f"[0:v]{pre}format=yuv420p[src];" + _fit_with_blur("src", s.width, s.height, "fit")
    graph += f";[fit]fps={FPS},format=yuv420p[v]"
    inputs = ["-ss", f"{start:.3f}", "-t", f"{length:.3f}", "-i", str(item.path)]
    if item.has_audio and s.keep_clip_sound:
        graph += f";[0:a]aformat=sample_rates=48000:channel_layouts=stereo,volume={s.clip_volume}[a]"
    else:
        inputs += ["-f", "lavfi", "-t", f"{length:.3f}", "-i", "anullsrc=r=48000:cl=stereo"]
        graph += ";[1:a]anull[a]"
    _encode_segment(inputs, graph, length, out)
    return length


def _photo_segment(item: MediaItem, s: RenderSettings, out: Path, workdir: Path) -> float:
    # Pillow turns the photo the right way up and reads iPhone HEIC files, which FFmpeg may not.
    still = workdir / f"{out.stem}.jpg"
    with Image.open(item.path) as img:
        img = ImageOps.exif_transpose(img).convert("RGB")
        img.thumbnail((3840, 3840))
        img.save(still, quality=92)

    frames = round(s.photo_seconds * FPS)
    w2, h2 = s.width * 2, s.height * 2  # work at double size so the slow zoom stays smooth
    graph = (
        _fit_with_blur("0:v", w2, h2, "fit")
        + f";[fit]zoompan=z='1+0.08*on/{frames}':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)'"
        f":d={frames}:s={s.width}x{s.height}:fps={FPS},format=yuv420p[v];[1:a]anull[a]"
    )
    inputs = ["-i", str(still), "-f", "lavfi", "-t", f"{s.photo_seconds:.3f}", "-i", "anullsrc=r=48000:cl=stereo"]
    _encode_segment(inputs, graph, s.photo_seconds, out)
    return s.photo_seconds


def _encode_segment(inputs: list[str], graph: str, length: float, out: Path) -> None:
    run_ffmpeg(
        [
            *inputs,
            "-filter_complex", graph,
            "-map", "[v]", "-map", "[a]",
            "-t", f"{length:.3f}",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-r", str(FPS),
            *AUDIO,
            str(out),
        ]
    )


def _segment_length(path: Path) -> float:
    return float(probe(str(path))["format"]["duration"])


def render(
    items: list[MediaItem],
    settings: RenderSettings,
    out_path: str | Path,
    on_progress: ProgressFn | None = None,
) -> Path:
    """Build the memory video. Progress is reported as (0..1, message)."""
    if not items:
        raise ValueError("There are no videos or photos to use.")
    report = on_progress or (lambda _f, _m: None)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    workdir = Path(tempfile.mkdtemp(prefix="trip-memory-"))
    prep_share = 0.7  # rough share of the total time spent preparing clips

    try:
        segments: list[Path] = []
        lengths: list[float] = []
        for i, item in enumerate(items):
            report(prep_share * i / len(items), f"Preparing {i + 1} of {len(items)}: {item.name}")
            seg = workdir / f"seg_{i:04d}.mp4"
            if item.kind == "video":
                _video_segment(item, settings, seg)
            else:
                _photo_segment(item, settings, seg, workdir)
            segments.append(seg)
            lengths.append(_segment_length(seg))

        report(prep_share, "Joining everything together")
        _join(segments, lengths, settings, out_path, lambda f: report(prep_share + (1 - prep_share) * f, "Joining everything together"))
        report(1.0, "Done")
        return out_path
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


def _join(segments: list[Path], lengths: list[float], s: RenderSettings, out: Path, progress: Callable[[float], None]) -> None:
    style = TRANSITIONS.get(s.transition, "fade")
    n = len(segments)
    # A transition can never be longer than ~40% of the shortest clip.
    t = min(s.transition_seconds, min(lengths) * 0.4) if style and n > 1 else 0.0

    inputs: list[str] = []
    for seg in segments:
        inputs += ["-i", str(seg)]

    parts: list[str] = []
    if t > 0:
        v_prev, a_prev, offset = "0:v", "0:a", 0.0
        for k in range(1, n):
            offset += lengths[k - 1] - t
            kind = MIXED[(k - 1) % len(MIXED)] if style == "mixed" else style
            parts.append(f"[{v_prev}][{k}:v]xfade=transition={kind}:duration={t:.3f}:offset={offset:.3f}[v{k}]")
            parts.append(f"[{a_prev}][{k}:a]acrossfade=d={t:.3f}[a{k}]")
            v_prev, a_prev = f"v{k}", f"a{k}"
        total = sum(lengths) - t * (n - 1)
    else:
        streams = "".join(f"[{k}:v][{k}:a]" for k in range(n))
        parts.append(f"{streams}concat=n={n}:v=1:a=1[v{n}][a{n}]")
        v_prev, a_prev, total = f"v{n}", f"a{n}", sum(lengths)

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

    run_ffmpeg(
        [
            *inputs,
            "-filter_complex", ";".join(parts),
            "-map", "[vout]", "-map", "[aout]",
            "-c:v", "libx264", "-preset", "medium", "-crf", "20", "-pix_fmt", "yuv420p",
            *AUDIO,
            "-movflags", "+faststart",
            str(out),
        ],
        total_seconds=total,
        on_progress=progress,
    )
