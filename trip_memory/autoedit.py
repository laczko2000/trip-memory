"""Auto-edit: make a first cut of the trip at the length you want, with cuts on the music's beat.

1. For every video, find its best moment: where things move and sound lively, without camera shake.
2. Pick what goes in: one shot per set of look-alike shots, every story represented, then the best of
   the rest until the wanted length is reached. Everything stays in date order.
3. Decide how long each shot stays on screen (shorter for action, longer for calm moments) and nudge
   every cut onto a beat of the music.
4. Hard cuts between lively shots, soft transitions elsewhere.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass, field

import numpy as np

from .beats import Beat
from .motion import FPS as STEPS_PER_SECOND
from .tools import ffmpeg_paths

HARD_CUT = "None (hard cut)"
MIN_SHOT = 2.2  # seconds; a transition needs room on both sides
PHOTO_SECONDS = 3.0
VIDEO_SECONDS = {"Fast action": 3.0, "Lively": 4.0, "Calm": 5.0, "Still": 4.0}
MAX_VIDEO_SECONDS = 10.0
MAX_PHOTO_SECONDS = 5.0


@dataclass
class Shot:
    path: str
    kind: str
    start: float = 0.0  # for videos: where the used part starts
    seconds: float = 0.0  # how long it is on screen (including its transition into the next shot)
    transition: str | None = None  # into the next shot; None = the default from Make video


@dataclass
class Plan:
    shots: list[Shot] = field(default_factory=list)
    left_out: list[str] = field(default_factory=list)
    length: float = 0.0
    beat: Beat | None = None


def loudness_steps(path: str) -> list[float]:
    """How loud a video is, five times a second (in dB; quiet is around -60)."""
    rate = 8000
    raw = subprocess.run(
        [ffmpeg_paths()[0], "-loglevel", "error", "-i", path, "-vn", "-ac", "1", "-ar", str(rate), "-f", "s16le", "-"],
        capture_output=True,
    ).stdout
    samples = np.frombuffer(raw, np.int16).astype(np.float32) / 32768
    step = rate // STEPS_PER_SECOND
    n = len(samples) // step
    if not n:
        return []
    rms = np.sqrt((samples[: n * step].reshape(n, step) ** 2).mean(axis=1) + 1e-9)
    return [round(float(v), 1) for v in 20 * np.log10(rms)]


def interest(action: list[float], shake: list[float], loud: list[float], n: int) -> np.ndarray:
    """A score five times a second for how good that moment looks: movement and sound up, shake down."""

    def norm(values: list[float], length: int) -> np.ndarray:
        v = np.zeros(length) if not values else np.resize(np.asarray(values, dtype=float), length)
        spread = v.max() - v.min()
        return (v - v.min()) / spread if spread > 0 else np.zeros(length)

    score = 0.5 * norm(np.minimum(action, 30).tolist() if action else [], n)
    score += 0.35 * norm(loud, n)
    score -= 0.3 * norm(shake, n)
    # The very start and end of a phone video are usually the camera being raised or lowered.
    edge = min(n // 4, STEPS_PER_SECOND)
    score[:edge] -= 0.3
    score[n - edge :] -= 0.3
    return score


def best_start(score: np.ndarray, duration: float, seconds: float) -> float:
    """Where a window of the given length catches the best moment."""
    window = int(round(seconds * STEPS_PER_SECOND))
    if seconds >= duration or window >= len(score) or window <= 0:
        return max(0.0, (duration - seconds) / 2)
    sums = np.convolve(score, np.ones(window), mode="valid")
    start = int(np.argmax(sums)) / STEPS_PER_SECOND
    return float(min(start, duration - seconds))


def shot_value(row: dict) -> float:
    """How much a file deserves a place in the short version."""
    tags = row.get("tags_final") or []
    value = 1.0 + (row.get("peak") or 0.0)
    if row["kind"] == "video":
        value += 0.4
    if row.get("people_tag") in ("Solo", "Couple", "Group"):
        value += 0.4
    if row.get("people"):
        value += 0.3  # someone you named
    if "Fast action" in tags or "Lively" in tags:
        value += 0.3
    if "Shaky" in tags:
        value -= 0.4
    value += 0.1 * np.log1p(row.get("sharpness") or 0) / 8
    return value


def wanted_seconds(row: dict) -> float:
    if row["kind"] != "video":
        return PHOTO_SECONDS
    tags = row.get("tags_final") or []
    pace = next((VIDEO_SECONDS[t] for t in ("Fast action", "Lively", "Calm", "Still") if t in tags), 4.0)
    return min(pace, row["duration"] or pace)


def choose(rows: list[dict], target: float | None, transition_seconds: float) -> tuple[list[dict], list[dict]]:
    """Which files go in (date order) and which are left out."""
    # One shot per set of look-alike shots: the one marked best.
    sets: dict = {}
    for r in rows:
        sets.setdefault(r.get("similar_set"), []).append(r)
    candidates = []
    extra = []
    for key, members in sets.items():
        if key is None or len(members) == 1:
            candidates += members
        else:
            best = [m for m in members if m.get("best")] or [max(members, key=shot_value)]
            candidates.append(best[0])
            extra += [m for m in members if m is not best[0]]
    if target is None:
        keep = candidates
    else:
        on_screen = lambda r: wanted_seconds(r) - transition_seconds  # noqa: E731
        keep, used = [], 0.0
        by_story: dict = {}
        for r in candidates:
            by_story.setdefault(r.get("story"), []).append(r)
        firsts = sorted((max(m, key=shot_value) for m in by_story.values()), key=shot_value, reverse=True)
        rest = sorted((r for r in candidates if r not in firsts), key=shot_value, reverse=True)
        for r in firsts + rest:
            if used >= target:
                break
            keep.append(r)
            used += on_screen(r)
    kept = {id(r) for r in keep}
    keep.sort(key=lambda r: (r["taken_at"], r["path"]))
    return keep, [r for r in rows if id(r) not in kept]


def plan(
    rows: list[dict],
    curves: dict[str, np.ndarray],
    target: float | None,
    transition_seconds: float,
    beat: Beat | None,
    title_seconds: float | None = None,
) -> Plan:
    """rows: the chosen files (from Trip.items()); curves: interest scores of the videos (by path).

    title_seconds: length of the day title cards shown before each day's first shot (None = no cards);
    they are counted so the cuts after them still land on the beat."""
    for r in rows:
        if r["path"] in curves and len(curves[r["path"]]):
            r["peak"] = float(np.clip(np.percentile(curves[r["path"]], 90), 0, 1))
    keep, out = choose(rows, target, transition_seconds)
    if not keep:
        return Plan(left_out=[r["path"] for r in out], beat=beat)

    wanted = [wanted_seconds(r) for r in keep]
    if target is not None:
        # Too short? Let shots stay a little longer (up to a limit) to fill the wanted length.
        total = sum(wanted) - transition_seconds * (len(keep) - 1)
        if total < target:
            room = [
                (min(MAX_VIDEO_SECONDS, r["duration"] or 0) if r["kind"] == "video" else MAX_PHOTO_SECONDS) - w
                for r, w in zip(keep, wanted)
            ]
            spare = sum(max(0, x) for x in room)
            if spare > 0:
                share = min(1.0, (target - total) / spare)
                wanted = [w + max(0, x) * share for w, x in zip(wanted, room)]

    transitions = []
    for a, b in zip(keep, keep[1:]):
        lively = {"Fast action", "Lively"}
        fast = lively & set(a.get("tags_final") or []) and lively & set(b.get("tags_final") or [])
        new_day = a["taken_at"][:10] != b["taken_at"][:10]
        transitions.append(None if new_day else (HARD_CUT if fast else None))

    min_shot = max(MIN_SHOT, transition_seconds / 0.4 + 0.05)  # the video maker shortens longer transitions
    shots: list[Shot] = []
    begin = 0.0  # where the current shot starts in the finished video
    for i, (r, want) in enumerate(zip(keep, wanted)):
        if title_seconds and (i == 0 or keep[i - 1]["taken_at"][:10] != r["taken_at"][:10]):
            begin += title_seconds - min(transition_seconds, 0.4 * title_seconds, 0.4 * want)
        longest = (r["duration"] or want) if r["kind"] == "video" else MAX_PHOTO_SECONDS
        shortest = min(min_shot, longest)
        seconds = max(min(want, longest), shortest)
        if i == len(keep) - 1:
            t_out = 0.0
        elif transitions[i] == HARD_CUT:
            t_out = 1 / 30
        else:  # as the video maker does: never more than 40% of either shot
            nxt = keep[i + 1]
            next_len = min(wanted[i + 1], nxt["duration"] or wanted[i + 1]) if nxt["kind"] == "video" else wanted[i + 1]
            t_out = min(transition_seconds, 0.4 * seconds, 0.4 * next_len)
        if beat and beat.clear and i < len(keep) - 1:
            # Put the middle of the change-over on the beat nearest to where it would be anyway.
            mid = begin + seconds - t_out / 2
            low = begin + shortest - t_out / 2
            high = begin + longest - t_out / 2
            beats = beat.times(high + beat.period)
            beats = beats[(beats >= low - 1e-6) & (beats <= high + 1e-6)]
            if len(beats):
                mid = float(beats[np.argmin(np.abs(beats - mid))])
                seconds = mid + t_out / 2 - begin
        seconds = round(seconds * 30) / 30  # whole video frames
        start = 0.0
        if r["kind"] == "video":
            curve = curves.get(r["path"])
            start = best_start(curve, r["duration"], seconds) if curve is not None else max(0.0, (r["duration"] - seconds) / 2)
        shots.append(Shot(path=r["path"], kind=r["kind"], start=round(start, 2), seconds=round(seconds, 3),
                          transition=transitions[i] if i < len(transitions) else None))
        begin += seconds - t_out
    return Plan(shots=shots, left_out=[r["path"] for r in out], length=begin, beat=beat)
