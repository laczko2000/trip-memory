"""Moments: a particular thing inside a theme, like "big jumps" in your kitesurfing. The app looks at a frame every
second inside each video (the theme only knows what a shot is about as a whole) and finds the second that fits your
words best. You tick the right ones; the app then also looks for shots like the ones you ticked.

Free and on this Mac: the same CLIP picture model as the themes.
"""

from __future__ import annotations

import numpy as np

EVERY = 1.0  # seconds between the frames looked at
MOST = 30  # frames per video at most (long clips: spread out)
AROUND = 3.0  # seconds of the clip kept around a moment


def frame_times(duration: float) -> list[float]:
    """When to look inside a video: every second (spread out over long clips), not the very first and last frames."""
    if not duration or duration < 1.0:
        return [max(0.0, (duration or 0) / 2)]
    count = min(MOST, max(1, int(duration / EVERY)))
    step = duration / count
    return [round(step * (k + 0.5), 2) for k in range(count)]


def pick_frames(found: list[float], duration: float) -> list[int]:
    """Of the frames a video has ready (its key frames, fast to read), the ones nearest to every second; empty if
    they are too far apart to find a moment (then the app reads frames at those seconds itself)."""
    if len(found) < 2 or (duration or 0) / len(found) > 2.5:
        return []
    out = []
    for t in frame_times(duration):
        k = min(range(len(found)), key=lambda k: abs(found[k] - t))
        if k not in out:
            out.append(k)
    return sorted(out)


def find(words: np.ndarray | None, shots: dict[str, tuple[list[float], np.ndarray]],
         ticked: list[np.ndarray] | None = None) -> list[tuple[str, float, float]]:
    """Each shot's best second for the moment, best first: (path, second, score).

    words: what the moment looks like (a CLIP text vector). shots: path -> (frame times, one CLIP vector per frame).
    ticked: the frames you ticked as right. A frame that looks like any one of them counts (jumps filmed from
    different sides don't blur into one average); the more you tick, the less the words count."""
    paths = [p for p, (times, _) in shots.items() if len(times)]
    if not paths or (words is None and not ticked):
        return []
    frames = np.concatenate([np.asarray(shots[p][1], np.float32) for p in paths])
    owner = np.concatenate([[k] * len(shots[p][0]) for k, p in enumerate(paths)])
    score = np.zeros(len(frames))
    weight_ticks = min(0.85, 0.45 + 0.1 * len(ticked)) if ticked else 0.0
    if words is not None:
        score += (1 - weight_ticks) * _z(frames @ np.asarray(words, np.float32))
    if ticked:
        near = (frames @ np.asarray(ticked, np.float32).T).max(axis=1)  # like the closest of your ticks
        score += weight_ticks * _z(near)
    out = []
    for k, path in enumerate(paths):
        mine = score[owner == k]
        # A moment is a few seconds: smooth over neighbours so one odd frame doesn't win.
        if len(mine) >= 3:
            mine = np.convolve(np.pad(mine, 1, mode="edge"), [0.25, 0.5, 0.25], mode="valid")
        best = int(np.argmax(mine))
        out.append((path, float(shots[path][0][best]), round(float(mine[best]), 3)))
    return sorted(out, key=lambda m: -m[2])


def _z(values: np.ndarray) -> np.ndarray:
    """How far above the usual each value is (so word and picture likeness can be added up)."""
    spread = values.std()
    return (values - values.mean()) / spread if spread > 1e-6 else values * 0


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
