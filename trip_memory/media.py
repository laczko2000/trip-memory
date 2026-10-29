"""Find the videos and photos in a trip folder and work out when each was taken."""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image

from .tools import FFmpegError, probe

try:
    from pillow_heif import register_heif_opener

    register_heif_opener()
except ImportError:  # HEIC photos are skipped without it
    pass

VIDEO_EXTS = {".mp4", ".mov", ".m4v", ".avi", ".mkv", ".mts", ".3gp", ".webm"}
PHOTO_EXTS = {".jpg", ".jpeg", ".png", ".heic", ".heif", ".webp", ".tif", ".tiff"}

EXIF_IFD = 0x8769
DATETIME_ORIGINAL = 36867
OFFSET_TIME_ORIGINAL = 36881
DATETIME = 306


@dataclass
class MediaItem:
    path: Path
    kind: str  # "video" or "photo"
    taken_at: datetime  # always timezone-aware
    duration: float = 0.0  # seconds, videos only
    has_audio: bool = False
    hdr: bool = False

    @property
    def name(self) -> str:
        return self.path.name


def _local(dt: datetime) -> datetime:
    """Treat a date without a timezone as the computer's local time."""
    return dt if dt.tzinfo else dt.astimezone()


def _file_time(path: Path) -> datetime:
    return datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)


def _parse_iso(text: str) -> datetime | None:
    text = text.strip().replace("Z", "+00:00")
    # Apple writes offsets as +0200; fromisoformat wants +02:00
    if len(text) > 5 and text[-5] in "+-" and text[-4:].isdigit():
        text = f"{text[:-2]}:{text[-2:]}"
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        return None


def photo_taken_at(path: Path) -> datetime:
    try:
        with Image.open(path) as img:
            exif = img.getexif()
            sub = exif.get_ifd(EXIF_IFD)
            raw = sub.get(DATETIME_ORIGINAL) or exif.get(DATETIME)
            if raw:
                dt = datetime.strptime(str(raw).strip(), "%Y:%m:%d %H:%M:%S")
                offset = sub.get(OFFSET_TIME_ORIGINAL)
                if offset:
                    dt = _parse_iso(dt.isoformat() + str(offset).strip()) or dt
                return _local(dt)
    except (OSError, ValueError):
        pass
    return _file_time(path)


def read_video(path: Path) -> MediaItem:
    info = probe(str(path))
    fmt_tags = {k.lower(): v for k, v in info.get("format", {}).get("tags", {}).items()}
    streams = info.get("streams", [])
    video = next((s for s in streams if s.get("codec_type") == "video"), None)
    if video is None:
        raise FFmpegError(f"{path.name} has no picture")

    # iPhones store local time with its offset here; the plain creation_time is UTC.
    taken = None
    for key in ("com.apple.quicktime.creationdate", "creation_time"):
        if key in fmt_tags:
            taken = _parse_iso(fmt_tags[key])
            if taken:
                break
    duration = float(info.get("format", {}).get("duration") or video.get("duration") or 0)
    return MediaItem(
        path=path,
        kind="video",
        taken_at=_local(taken) if taken else _file_time(path),
        duration=duration,
        has_audio=any(s.get("codec_type") == "audio" for s in streams),
        hdr=video.get("color_transfer") in ("arib-std-b67", "smpte2084"),
    )


def scan_folder(folder: str | Path) -> tuple[list[MediaItem], list[str]]:
    """Return the trip's videos and photos sorted by when they were taken, plus any problems found."""
    items: list[MediaItem] = []
    problems: list[str] = []
    for root, dirs, files in os.walk(folder):
        dirs[:] = [d for d in dirs if not d.startswith(".")]
        for filename in files:
            if filename.startswith("."):
                continue
            path = Path(root) / filename
            ext = path.suffix.lower()
            try:
                if ext in VIDEO_EXTS:
                    item = read_video(path)
                    if item.duration > 0.2:
                        items.append(item)
                elif ext in PHOTO_EXTS:
                    items.append(MediaItem(path=path, kind="photo", taken_at=photo_taken_at(path)))
            except (FFmpegError, OSError) as exc:
                problems.append(f"Skipped {filename}: {exc}")
    items.sort(key=lambda i: (i.taken_at, i.name))
    return items, problems
