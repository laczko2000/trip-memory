"""Find the FFmpeg programs and run them."""

from __future__ import annotations

import functools
import shutil
import subprocess
from collections.abc import Callable


class FFmpegError(RuntimeError):
    pass


@functools.cache
def ffmpeg_paths() -> tuple[str, str]:
    """Return (ffmpeg, ffprobe). Uses an installed copy, else downloads a free static build once."""
    ffmpeg, ffprobe = shutil.which("ffmpeg"), shutil.which("ffprobe")
    if ffmpeg and ffprobe:
        return ffmpeg, ffprobe
    from static_ffmpeg import run

    return run.get_or_fetch_platform_executables_else_raise()


@functools.cache
def has_filter(name: str) -> bool:
    out = subprocess.run([ffmpeg_paths()[0], "-hide_banner", "-filters"], capture_output=True, text=True).stdout
    return any(line.split()[1:2] == [name] for line in out.splitlines() if line.strip())


def probe(path: str) -> dict:
    import json

    result = subprocess.run(
        [ffmpeg_paths()[1], "-v", "error", "-print_format", "json", "-show_streams", "-show_format", path],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise FFmpegError(f"Could not read {path}: {result.stderr.strip()}")
    return json.loads(result.stdout)


def run_ffmpeg(args: list[str], total_seconds: float = 0, on_progress: Callable[[float], None] | None = None) -> None:
    """Run ffmpeg, reporting progress (0..1) when the expected output length is known."""
    cmd = [ffmpeg_paths()[0], "-hide_banner", "-loglevel", "error", "-y", "-progress", "pipe:1", "-nostats", *args]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    for line in proc.stdout:
        key, _, value = line.strip().partition("=")
        if key == "out_time_us" and on_progress and total_seconds > 0 and value.isdigit():
            on_progress(min(1.0, int(value) / 1e6 / total_seconds))
    err = proc.stderr.read()
    if proc.wait() != 0:
        raise FFmpegError(err.strip()[-2000:] or "ffmpeg failed")
