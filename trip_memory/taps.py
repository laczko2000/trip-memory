"""Your own cut points for a song: you tap along while it plays, the app places each tap exactly on the hit
you meant, and Auto-edit cuts there."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from .beats import RATE, read_song

EARLY, LATE = 0.2, 0.1  # the hit you meant can be this far before or after your tap (taps tend to come late)
FINE_HOP = 64  # look every 6 ms
FINE_WINDOW = 512
SEEN_EARLY = 0.012  # a hit starts to show in the sound a little before it really starts


def _file(song: str, folder: Path) -> Path:
    stat = Path(song).stat()
    key = hashlib.sha1(f"{song}|{stat.st_mtime}|{stat.st_size}".encode()).hexdigest()[:16]
    return folder / f"{key}.json"


def load(song: str, folder: Path) -> dict:
    """{"taps": [seconds...], "final": bool} for this song (empty if you haven't tapped it yet)."""
    try:
        data = json.loads(_file(song, folder).read_text())
        return {"taps": sorted(float(t) for t in data.get("taps", [])), "final": bool(data.get("final"))}
    except (OSError, ValueError):
        return {"taps": [], "final": False}


def save(song: str, folder: Path, taps: list[float], final: bool) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    clean = sorted({round(float(t), 3) for t in taps if t > 0})
    _file(song, folder).write_text(json.dumps({"song": song, "taps": clean, "final": final}))


def final_taps(song: str, folder: Path) -> list[float] | None:
    data = load(song, folder)
    return data["taps"] if data["final"] and data["taps"] else None


def snap(samples: np.ndarray, taps: list[float]) -> list[float]:
    """Move each tap onto the hit it was meant for: the sharpest rise in sound (a drum, a note starting)
    close to the tap. A tap with no clear hit near it stays where it is."""
    window = np.hanning(FINE_WINDOW).astype(np.float32)
    weights = None
    result = []
    for t in taps:
        a = int((t - EARLY) * RATE) - FINE_WINDOW // 2
        b = int((t + LATE) * RATE) + FINE_WINDOW // 2
        part = samples[max(0, a) : max(0, b)]
        if len(part) < FINE_WINDOW + 2 * FINE_HOP:
            result.append(round(t, 3))
            continue
        frames = np.lib.stride_tricks.sliding_window_view(part, FINE_WINDOW)[::FINE_HOP] * window
        spec = np.sqrt(np.abs(np.fft.rfft(frames, axis=1)))  # gently compressed, so silence doesn't count as a hit
        if weights is None:
            hz = np.fft.rfftfreq(FINE_WINDOW, 1 / RATE)
            weights = np.where(hz < 150, 2.0, 1.0)  # drums and bass count a bit more
        rise = (np.maximum(0, np.diff(spec, axis=0)) * weights).sum(axis=1)
        # The time each rise happens at: the middle of the later of the two looks.
        times = (max(0, a) + FINE_WINDOW // 2 + FINE_HOP * np.arange(1, len(frames))) / RATE + SEEN_EARLY
        near = 1 - 0.4 * np.abs(times - t) / max(EARLY, LATE)  # prefer hits closer to the tap
        best = int(np.argmax(rise * near))
        clear = rise[best] > 1.5 * np.median(rise) and rise[best] > 0
        result.append(round(float(times[best]) if clear else t, 3))
    return result


def snap_song(path: str, taps: list[float]) -> list[float]:
    if not taps:
        return []
    samples = read_song(path, seconds=max(taps) + 2)
    snapped: list[float] = []
    for t in sorted(snap(samples, sorted(taps))):
        if not snapped or t - snapped[-1] > 0.05:  # two taps for the same hit become one pin
            snapped.append(t)
    return snapped


def shot_lengths(taps: list[float], duration: float) -> list[float]:
    cuts = [0.0, *sorted(t for t in taps if 0 < t < duration), duration]
    return [b - a for a, b in zip(cuts, cuts[1:])]
