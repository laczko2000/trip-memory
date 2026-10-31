"""Learning your highlights: every ⭐ you put on a video teaches the app what a great moment looks like to you.

For each highlight the app keeps what that second looks like (the CLIP picture vector of the frame) and how it moves
(movement, camera and loudness, compared with the rest of that video). Kept in the app's folder, so what you taught
on one trip helps on the next. On a new trip it looks at a frame every second of the videos you have not marked yet
and suggests the seconds most like your highlights (in orange). Keep one and it becomes a highlight (and teaches it
more); drop one and it learns that too.

Free and on this Mac: the same picture model as the themes and moments.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

FILE = "learned-highlights.json"
START_FROM = 5  # highlights you marked (in all trips) before the app suggests any
MOST_KEPT = 1500  # examples remembered at most (the oldest go first)
NEAR = 2.0  # seconds: a suggestion this close to a highlight (or one you dropped) is the same moment
STEPS = 5  # the movement curves have five values a second
MOTION_WEIGHT = 0.15  # how much the movement counts next to the look


def load(folder: Path) -> list[dict]:
    """[{"key", "trip", "keep", "vec", "motion", "theme"}]: every highlight you kept (keep true) or dropped."""
    try:
        return json.loads((Path(folder) / FILE).read_text()).get("examples") or []
    except (OSError, ValueError, AttributeError):
        return []


def save(folder: Path, examples: list[dict]) -> None:
    try:
        Path(folder).mkdir(parents=True, exist_ok=True)
        (Path(folder) / FILE).write_text(json.dumps({"examples": examples[-MOST_KEPT:]}))
    except OSError:
        pass


def example(key: str, trip: str, keep: bool, vec: np.ndarray, motion: list[float], theme: str | None) -> dict:
    return {"key": key, "trip": trip, "keep": bool(keep), "vec": [round(float(x), 4) for x in vec],
            "motion": [round(float(x), 3) for x in motion], "theme": theme}


def motion_at(curves: dict | None, t: float) -> list[float]:
    """How much is going on at that second compared with the rest of the video: movement, camera, loudness
    (0 = as usual, 1 = one spread above it)."""
    out = []
    for name in ("action", "camera", "loud"):
        values = np.asarray((curves or {}).get(name) or [], np.float32)
        if len(values) < 3:
            out.append(0.0)
            continue
        k = int(round(t * STEPS))
        here = values[max(0, k - STEPS // 2): k + STEPS // 2 + 1]
        spread = float(values.std())
        out.append(float((here.mean() - values.mean()) / spread) if len(here) and spread > 1e-6 else 0.0)
    return [max(-3.0, min(3.0, x)) for x in out]


def counts(examples: list[dict]) -> tuple[int, int]:
    """(highlights kept, suggestions dropped) the app has learned from."""
    kept = sum(1 for e in examples if e["keep"])
    return kept, len(examples) - kept


def suggest(examples: list[dict], frames: dict[str, tuple[list[float], np.ndarray]],
            motions: dict[str, dict | None], avoid: dict[str, list[float]] | None = None,
            most: int = 30) -> list[dict]:
    """The seconds of these videos most like your highlights, best first: [{"path", "t", "score"}].

    frames: path -> (times, one CLIP vector per frame). motions: path -> the video's movement curves.
    avoid: path -> seconds already marked or dropped. One suggestion per video at most, and only the ones clearly
    better than a usual second of your videos."""
    good = np.array([e["vec"] for e in examples if e["keep"]], np.float32)
    bad = np.array([e["vec"] for e in examples if not e["keep"]], np.float32)
    if len(good) < START_FROM:
        return []
    good_motion = np.array([e["motion"] for e in examples if e["keep"]], np.float32).mean(axis=0)
    paths = [p for p, (times, _) in frames.items() if len(times)]
    if not paths:
        return []
    best: list[tuple[str, float, float]] = []
    every: list[np.ndarray] = []
    for path in paths:
        times, vectors = frames[path]
        vectors = np.asarray(vectors, np.float32)
        like = (vectors @ good.T).max(axis=1)  # like the closest of your highlights (jumps filmed many ways)
        if len(bad):
            unlike = (vectors @ bad.T).max(axis=1)
            like = like - 0.5 * np.maximum(0.0, unlike - like)  # more like one you dropped: less likely
        moving = np.array([np.dot(motion_at(motions.get(path), t), good_motion) for t in times], np.float32)
        score = like + MOTION_WEIGHT * 0.05 * moving
        if len(score) >= 3:  # a moment is a few seconds: one odd frame doesn't win
            score = np.convolve(np.pad(score, 1, mode="edge"), [0.25, 0.5, 0.25], mode="valid")
        every.append(score)
        taken = (avoid or {}).get(path) or []
        order = np.argsort(-score)
        pick = next((k for k in order if all(abs(times[k] - t) > NEAR for t in taken)), None)
        if pick is not None:
            best.append((path, float(times[pick]), float(score[pick])))
    if not best:
        return []
    allscores = np.concatenate(every)
    bar = max(float(np.percentile(allscores, 75)), float(allscores.mean() + 0.5 * allscores.std()))
    out = [{"path": p, "t": round(t, 2), "score": round(s, 4)} for p, t, s in best if s >= bar]
    return sorted(out, key=lambda o: -o["score"])[:most]
