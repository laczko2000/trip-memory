"""Your own cut points for a song: you tap along while it plays, the app places each tap exactly on the hit
you meant, and Auto-edit cuts there."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from .beats import RATE, read_song

EARLY, LATE = 0.2, 0.1  # without enough taps to learn from: the hit can be this far before or after a tap
USUAL_EARLY, USUAL_LATE = 0.4, 0.15  # how far off anyone's taps can be (slow reactions, Bluetooth headphones)
NEAR = 0.1  # once your usual delay is known, each tap's hit is looked for this close to where it should be
ON_GRID = 0.035  # a hit this close to a beat (or half beat) of the song's beat grid is preferred
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


def _rises(samples: np.ndarray, a: float, b: float) -> tuple[np.ndarray, np.ndarray]:
    """How sharply the sound rises (a drum, a note starting) every 6 ms between a and b seconds."""
    lo = int(a * RATE) - FINE_WINDOW // 2
    part = samples[max(0, lo) : max(0, int(b * RATE) + FINE_WINDOW // 2)]
    if len(part) < FINE_WINDOW + 2 * FINE_HOP:
        return np.zeros(0), np.zeros(0)
    frames = np.lib.stride_tricks.sliding_window_view(part, FINE_WINDOW)[::FINE_HOP] * _HANN
    spec = np.sqrt(np.abs(np.fft.rfft(frames, axis=1)))  # gently compressed, so silence doesn't count as a hit
    rise = (np.maximum(0, np.diff(spec, axis=0)) * _WEIGHTS).sum(axis=1)
    # The time each rise happens at: the middle of the later of the two looks.
    times = (max(0, lo) + FINE_WINDOW // 2 + FINE_HOP * np.arange(1, len(frames))) / RATE + SEEN_EARLY
    return times, rise


_HANN = np.hanning(FINE_WINDOW).astype(np.float32)
_HZ = np.fft.rfftfreq(FINE_WINDOW, 1 / RATE)
_WEIGHTS = np.where(_HZ < 150, 2.0, np.where(_HZ < 2500, 1.0, 0.4))  # kick and bass count more, hi-hats less


def _best_hit(samples: np.ndarray, centre: float, early: float, late: float, grid: np.ndarray | None) -> float | None:
    times, rise = _rises(samples, centre - early, centre + late)
    if not len(rise):
        return None
    score = rise * (1 - 0.4 * np.abs(times - centre) / max(early, late))  # prefer hits closer to where it should be
    if grid is not None and len(grid):
        nearest = np.minimum(np.abs(times - grid[np.clip(np.searchsorted(grid, times) - 1, 0, len(grid) - 1)]),
                             np.abs(times - grid[np.clip(np.searchsorted(grid, times), 0, len(grid) - 1)]))
        score = score * np.where(nearest < ON_GRID, 1.6, 1.0)  # the music's own beat is the likeliest cut
    best = int(np.argmax(score))
    clear = rise[best] > 1.5 * np.median(rise) and rise[best] > 0
    return float(times[best]) if clear else None


def usual_delay(samples: np.ndarray, taps: list[float]) -> float:
    """How late (positive) or early you usually tap compared with the hits: everyone has a steady delay
    (reaction time, and the sound reaching your ears late, e.g. through Bluetooth headphones). It is the shift
    that puts the most of your taps, together, onto strong hits."""
    if len(taps) < 4:
        return 0.0
    times, rise = _rises(samples, max(0.0, min(taps) - USUAL_EARLY - 0.05), max(taps) + USUAL_LATE + 0.05)
    if not len(rise):
        return 0.0
    rise = rise / (np.percentile(rise, 99) or 1)
    reach = int(round(0.02 * RATE / FINE_HOP))  # taps wobble a little around your usual delay
    strongest = np.array([rise[max(0, k - reach) : k + reach + 1].max() for k in range(len(rise))])
    step = FINE_HOP / RATE
    delays = np.arange(-USUAL_LATE, USUAL_EARLY + 1e-9, 0.003)
    scores = []
    for d in delays:
        k = np.clip(np.round((np.asarray(taps) - d - times[0]) / step).astype(int), 0, len(rise) - 1)
        scores.append(strongest[k].sum() * (1 - 0.3 * abs(d) / USUAL_EARLY))  # a smaller delay is likelier
    return float(delays[int(np.argmax(scores))])


def snap(samples: np.ndarray, taps: list[float], grid: np.ndarray | None = None) -> list[float]:
    """Move each tap onto the hit it was meant for. First learn your usual delay from all your taps, then look
    for each hit close to where that delay says it is, preferring the song's beats. A tap with no clear hit
    near it just has your usual delay taken off."""
    delay = usual_delay(samples, taps)
    early, late = (NEAR, NEAR) if delay else (EARLY, LATE)
    result = []
    for t in taps:
        hit = _best_hit(samples, t - delay, early, late, grid)
        result.append(round(hit if hit is not None else max(0.0, t - delay), 3))
    return result


def snap_song(path: str, taps: list[float]) -> tuple[list[float], float]:
    """Your taps snapped onto the song's hits, and your usual delay (seconds; positive = you tap late)."""
    if not taps:
        return [], 0.0
    from .beats import beat_of

    samples = read_song(path, seconds=max(taps) + 2)
    beat = beat_of(samples, len(samples) / RATE)
    grid = np.arange(beat.first, len(samples) / RATE, beat.period / 2) if beat.clear else None  # beats and half beats
    snapped: list[float] = []
    for t in sorted(snap(samples, sorted(taps), grid)):
        if not snapped or t - snapped[-1] > 0.05:  # two taps for the same hit become one pin
            snapped.append(t)
    return snapped, usual_delay(samples, sorted(taps))


def shot_lengths(taps: list[float], duration: float) -> list[float]:
    cuts = [0.0, *sorted(t for t in taps if 0 < t < duration), duration]
    return [b - a for a, b in zip(cuts, cuts[1:])]
