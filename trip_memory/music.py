"""Find free music online (Creative Commons songs from Jamendo, ccMixter and others, via Openverse).

Openverse is a free search engine for openly licensed media run by the WordPress Foundation; it needs no
account. Only songs whose licence allows using them in your own video are shown. Most ask you to credit
the artist, so the app saves a credits file next to every video that uses one.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import httpx

API = "https://api.openverse.org/v1/audio/"
HEADERS = {"User-Agent": "TripMemory/1.0 (personal holiday video app)"}
MOODS = ["Happy", "Summer", "Chill", "Acoustic", "Upbeat", "Cinematic", "Piano", "Tropical", "Road trip", "Epic"]


@dataclass
class Song:
    id: str
    title: str
    artist: str
    seconds: float
    url: str  # the audio file
    page: str  # where the song lives online
    licence: str  # e.g. "CC BY 4.0"
    licence_url: str
    source: str

    @property
    def credit(self) -> str:
        return f'"{self.title}" by {self.artist} ({self.page}), licensed {self.licence} ({self.licence_url})'


def search(words: str, page: int = 1) -> list[Song]:
    """Songs matching the words, long enough for a video, whose licence allows using them in it."""
    params = {
        "q": words or "happy",
        "category": "music",
        "license_type": "modification",  # setting a song to pictures counts as adapting it
        "page_size": 20,
        "page": page,
        "mature": "false",
    }
    response = httpx.get(API, params=params, headers=HEADERS, timeout=20, follow_redirects=True)
    response.raise_for_status()
    return parse(response.json())


def parse(data: dict) -> list[Song]:
    songs = []
    for r in data.get("results", []):
        seconds = (r.get("duration") or 0) / 1000
        if not r.get("url") or seconds < 45:
            continue  # sound effects and jingles are too short
        licence = f"CC {str(r.get('license', '')).upper()} {r.get('license_version') or ''}".strip()
        if str(r.get("license", "")).lower() in ("cc0", "pdm"):
            licence = "Public domain" if r["license"].lower() == "pdm" else "CC0 (no credit needed)"
        songs.append(Song(
            id=str(r.get("id")),
            title=r.get("title") or "Untitled",
            artist=r.get("creator") or "Unknown artist",
            seconds=seconds,
            url=r["url"],
            page=r.get("foreign_landing_url") or r.get("url"),
            licence=licence,
            licence_url=r.get("license_url") or "",
            source=r.get("source") or r.get("provider") or "",
        ))
    return songs


def download(song: Song, folder: Path) -> Path:
    """Save the song in the app's music folder (once) and remember its credit."""
    folder.mkdir(parents=True, exist_ok=True)
    suffix = Path(song.url.split("?")[0]).suffix.lower()
    if suffix not in (".mp3", ".ogg", ".wav", ".flac", ".m4a", ".aac", ".opus"):
        suffix = ".mp3"
    safe = "".join(c for c in f"{song.artist} - {song.title}" if c.isalnum() or c in " -_").strip()[:80]
    path = folder / f"{safe or 'song'} [{hashlib.sha1(song.id.encode()).hexdigest()[:6]}]{suffix}"
    if not path.exists():
        part = path.with_suffix(path.suffix + ".part")
        with httpx.stream("GET", song.url, headers=HEADERS, timeout=60, follow_redirects=True) as response:
            response.raise_for_status()
            with open(part, "wb") as f:
                for chunk in response.iter_bytes():
                    f.write(chunk)
        part.rename(path)
    credits = _load_credits(folder)
    credits[path.name] = asdict(song) | {"credit": song.credit}
    (folder / "credits.json").write_text(json.dumps(credits, indent=1))
    return path


def credit_for(music_path: str | Path) -> str | None:
    """The credit line for a downloaded song, if the music is one."""
    path = Path(music_path)
    entry = _load_credits(path.parent).get(path.name)
    return entry["credit"] if entry else None


def _load_credits(folder: Path) -> dict:
    try:
        return json.loads((folder / "credits.json").read_text())
    except (OSError, ValueError):
        return {}
