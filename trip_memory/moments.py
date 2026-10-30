"""Moments: a particular thing inside a theme, like "big jumps" in your kitesurfing. The app looks at a frame every
second inside each video (the theme only knows what a shot is about as a whole) and finds the second that fits your
words best. You tick the right ones; the app then also looks for shots like the ones you ticked.

Free and on this Mac: the same CLIP picture model as the themes.
"""

from __future__ import annotations

import numpy as np

EVERY = 1.0  # seconds between the frames looked at
MOST = 40  # frames per video at most (long clips: spread out)
AROUND = 3.0  # seconds of the clip kept around a moment
TICKED_WEIGHT = 1.5  # how much your ticked examples count against your words


def frame_times(duration: float) -> list[float]:
    """When to look inside a video: every second (spread out over long clips), not the very first and last frames."""
    if not duration or duration < 1.0:
        return [max(0.0, (duration or 0) / 2)]
    count = min(MOST, max(1, int(duration / EVERY)))
    step = duration / count
    return [round(step * (k + 0.5), 2) for k in range(count)]


def find(words: np.ndarray | None, shots: dict[str, tuple[list[float], np.ndarray]],
         ticked: list[np.ndarray] | None = None) -> list[tuple[str, float, float]]:
    """Each shot's best second for the moment, best first: (path, second, score).

    words: what the moment looks like (a CLIP text vector). shots: path -> (frame times, one CLIP vector per frame).
    ticked: the frames you ticked as right: the frames most like them count too."""
    parts = []
    if words is not None:
        parts.append(np.asarray(words, np.float32))
    if ticked:
        mean = np.mean(np.asarray(ticked, np.float32), axis=0)
        parts.append(TICKED_WEIGHT * mean / (np.linalg.norm(mean) or 1))
    if not parts:
        return []
    query = np.sum(parts, axis=0)
    query = query / (np.linalg.norm(query) or 1)
    out = []
    for path, (times, vectors) in shots.items():
        if not len(times):
            continue
        sims = np.asarray(vectors, np.float32) @ query
        # A moment is a few seconds: smooth over neighbours so one odd frame doesn't win.
        if len(sims) >= 3:
            sims = np.convolve(np.pad(sims, 1, mode="edge"), [0.25, 0.5, 0.25], mode="valid")
        best = int(np.argmax(sims))
        out.append((path, float(times[best]), float(sims[best])))
    return sorted(out, key=lambda m: -m[2])


def likely(found: list[tuple[str, float, float]]) -> set[str]:
    """The matches worth a tick at first: clearly better than the typical shot of the theme."""
    if not found:
        return set()
    scores = np.array([s for _, _, s in found])
    bar = max(np.median(scores) + 0.5 * scores.std(), np.percentile(scores, 60))
    keep = {p for p, _, s in found if s >= bar}
    return keep or {found[0][0]}


def window(second: float, duration: float, start: float = 0.0, end: float | None = None,
           length: float = AROUND) -> tuple[float, float]:
    """The part of the clip kept around a moment (inside the part you kept, if you cut the clip)."""
    end = duration if end is None else end
    length = min(length, end - start)
    begin = min(max(start, second - length / 2), end - length)
    return round(begin, 2), round(begin + length, 2)
