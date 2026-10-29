"""Auto-edit: make a first cut of the trip where the music leads and the shots follow.

1. The song is cut into shot-sized pieces on its beats: quick ones in lively parts, long ones in calm
   parts, and a cut at every change in the music, all within your limits (longest clip, photo, shot).
2. For every video, find its best moment: where things move and sound lively, without camera shake.
3. Fill the pieces in the order the trip happened, one shot per set of look-alike shots, matching
   action clips to lively music and calm or cinematic clips and photos to calm music.
4. Hard cuts on the beat; soft transitions only inside calm parts.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass, field

import numpy as np

from .beats import Beat
from .motion import FPS as STEPS_PER_SECOND
from .tools import ffmpeg_paths

HARD_CUT = "None (hard cut)"


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


def choose(rows: list[dict]) -> list[dict]:
    """The shots to choose from: everything you ticked, in the order it was taken.

    (Choose files already ticks only the best of each set of look-alike shots for you; anything else you
    ticked yourself, you want in.)"""
    return sorted(rows, key=lambda r: (r["taken_at"], r["path"]))


# ---------- music-driven editing ----------


@dataclass
class Limits:
    """Your limits for the shots (seconds)."""

    max_video: float = 4.0
    max_photo: float = 2.5
    min_shot: float = 1.0


@dataclass
class Slot:
    start: float  # where the shot starts in the music (the cut before it)
    end: float  # the cut after it
    energy: float  # how lively the music is here, 0..1
    soft_before: bool = False  # a soft transition (fade) into this slot instead of a hard cut

    @property
    def seconds(self) -> float:
        return self.end - self.start


# How many beats a shot lasts, by how lively the music is: quick cuts when it's busy, long ones when calm.
BEATS_FOR = {"Lively": 2, "Medium": 4, "Calm": 8}


def music_slots(song, limits: Limits, end: float, longest: float | None = None) -> list[Slot]:
    """Cut the music into shot-sized pieces: every cut on a beat, every change in the music a cut."""
    longest = longest or max(limits.max_video, limits.max_photo)
    beats = song.beat_times()
    slots: list[Slot] = []
    for section in song.sections:
        if section.start >= end:
            break
        n = BEATS_FOR.get(section.level, 4)
        while n > 1 and n * song.period > longest * 1.03:  # a hair over the limit is fine (tempo isn't exact)
            n //= 2
        while n * song.period < limits.min_shot - 1e-6:
            n *= 2
        stop = min(section.end, end)
        # Cuts every n beats, counted from the bar lines, so they line up with the music's bars.
        cuts = [section.start]
        for k, b in enumerate(beats):
            if section.start + 1e-3 < b < stop - 1e-3 and (k - song.downbeat) % n == 0 and b - cuts[-1] >= limits.min_shot - 1e-6:
                cuts.append(float(b))
        cuts.append(stop)
        # Too short a piece at the end joins the one before it, when the joined piece is allowed.
        if len(cuts) > 2 and cuts[-1] - cuts[-2] < limits.min_shot and cuts[-1] - cuts[-3] <= longest:
            cuts.pop(-2)
        slots += [Slot(a, b, section.energy) for a, b in zip(cuts, cuts[1:]) if b - a > 1e-3]
    # Any piece still longer than allowed is split evenly on beats.
    result: list[Slot] = []
    for s in slots:
        pieces = int(np.ceil(s.seconds / (longest * 1.03)))
        if pieces <= 1:
            result.append(s)
            continue
        inner = beats[(beats > s.start + 1e-3) & (beats < s.end - 1e-3)]
        cuts = [s.start]
        for k in range(1, pieces):
            want = s.start + k * s.seconds / pieces
            cuts.append(float(inner[np.argmin(np.abs(inner - want))]) if len(inner) else want)
        cuts.append(s.end)
        cuts = sorted(set(cuts))
        result += [Slot(a, b, s.energy) for a, b in zip(cuts, cuts[1:])]
    return result


def merge_slots(slots: list[Slot], count: int, longest: float) -> list[Slot]:
    """With fewer shots than pieces of music, join neighbouring pieces (in the same part of the song, and
    never longer than allowed) so the shots you have stretch further."""
    slots = list(slots)
    while len(slots) > count:
        pairs = [
            (slots[k].seconds + slots[k + 1].seconds, k)
            for k in range(len(slots) - 1)
            if slots[k].energy == slots[k + 1].energy and slots[k].seconds + slots[k + 1].seconds <= longest * 1.03
        ]
        if not pairs:
            break
        _, k = min(pairs)
        slots[k : k + 2] = [Slot(slots[k].start, slots[k + 1].end, slots[k].energy)]
    return slots


def pace_of(row: dict) -> tuple[float, bool]:
    """How lively a shot is (0 still .. 1 fast action) and whether it is a smooth, cinematic camera move."""
    if row["kind"] != "video":
        return 0.15, False
    action, camera, shake = row.get("motion_action"), row.get("motion_camera"), row.get("motion_shake")
    if action is None:
        tags = row.get("tags_final") or []
        level = {"Fast action": 0.95, "Lively": 0.7, "Calm": 0.4, "Still": 0.2}
        return next((level[t] for t in level if t in tags), 0.5), "Cinematic" in tags
    pace = float(np.clip(np.log1p(action) / np.log1p(15), 0, 1))
    cinematic = camera is not None and camera >= 1.0 and (shake or 0) < 0.6 and action < 3
    return (0.3 if cinematic else pace), cinematic


def fill_slots(slots: list[Slot], candidates: list[dict], limits: Limits) -> list[int]:
    """Choose a shot for every slot, keeping the trip in time order, matching lively shots to lively music.

    Returns the candidate index for each slot that could be filled (stops early if it runs out)."""
    n_slots, n = len(slots), len(candidates)
    if not n_slots or not n:
        return []
    paces = [pace_of(c)[0] for c in candidates]
    values = [shot_value(c) for c in candidates]
    neg = -1e9
    score = np.full((n_slots, n), neg)
    back = np.full((n_slots, n), -1, dtype=int)
    for j, slot in enumerate(slots):
        best_prev, best_idx = (0.0, -1) if j == 0 else (neg, -1)
        for i, c in enumerate(candidates):
            if j > 0 and i > 0 and score[j - 1, i - 1] > best_prev:
                best_prev, best_idx = score[j - 1, i - 1], i - 1
            if best_prev <= neg / 2:
                continue
            if c["kind"] == "video":
                if (c["duration"] or 0) < slot.seconds - 0.05 or slot.seconds > limits.max_video * 1.03 + 0.05:
                    continue
            elif slot.seconds > limits.max_photo * 1.03 + 0.05:
                continue
            fit = 1.0 - abs(slot.energy - paces[i])
            score[j, i] = best_prev + 3.0 + 0.5 * values[i] + 2.0 * fit
            back[j, i] = best_idx
    filled = [j for j in range(n_slots) if score[j].max() > neg / 2]  # always the first few slots
    if not filled:
        return []
    last = max(filled)
    picks: list[int] = []
    i = int(np.argmax(score[last]))
    for jj in range(last, -1, -1):
        picks.append(i)
        i = int(back[jj, i])
    return picks[::-1]


def plan_to_music(
    rows: list[dict],
    curves: dict[str, np.ndarray],
    song,
    limits: Limits,
    target: float | None,
    transition_seconds: float,
    soft_transition: str,
) -> Plan:
    """The music-driven first cut: the song decides where the cuts are, the shots are matched to it."""
    for r in rows:
        if r["path"] in curves and len(curves[r["path"]]):
            r["peak"] = float(np.clip(np.percentile(curves[r["path"]], 90), 0, 1))
    candidates = choose(rows)
    end = song.duration if target is None or target >= song.duration else target
    if target is not None and target < song.duration:
        bars = song.bar_starts()
        later = bars[bars >= max(limits.min_shot, target - 2 * song.period)]
        end = float(later[0]) if len(later) else target  # end on a bar line
    best: tuple[list[Slot], list[int]] = ([], [])
    for longest in (None, limits.max_photo, max(limits.min_shot, limits.max_photo / 2)):
        slots = music_slots(song, limits, end, longest)
        slots = merge_slots(slots, len(candidates), longest or max(limits.max_video, limits.max_photo))
        picks = fill_slots(slots, candidates, limits)
        if len(picks) == len(slots):
            best = (slots, picks)
            break
        if sum(s.seconds for s in slots[: len(picks)]) > sum(s.seconds for s in best[0][: len(best[1])]):
            best = (slots, picks)
    slots, picks = best
    slots = slots[: len(picks)]
    if not slots:
        return Plan(left_out=[r["path"] for r in rows], beat=song.beat)

    # Soft transitions only inside calm music, where both shots are long enough for it to fit fully.
    t = transition_seconds
    for k in range(1, len(slots)):
        a, b = slots[k - 1], slots[k]
        same_section = abs(a.energy - b.energy) < 1e-9
        slots[k].soft_before = same_section and a.energy < 0.38 and t <= 0.4 * a.seconds and t <= 0.4 * b.seconds
    overlap = [t if s.soft_before else 1 / 30 for s in slots[1:]] + [0.0]  # into the next shot

    shots: list[Shot] = []
    begin = 0  # in video frames, where this shot starts
    for k, (slot, i) in enumerate(zip(slots, picks)):
        c = candidates[i]
        # The change-over to the next shot is centred on the cut; counted in whole frames so nothing drifts.
        finish = round((slot.end + overlap[k] / 2) * 30)
        seconds = (finish - begin) / 30
        begin = finish - round(overlap[k] * 30)
        start = 0.0
        if c["kind"] == "video":
            curve = curves.get(c["path"])
            longest_start = max(0.0, (c["duration"] or seconds) - seconds)
            start = best_start(curve, c["duration"], seconds) if curve is not None else longest_start / 2
            start = min(max(0.0, start), longest_start)
        transition = None if k == len(slots) - 1 else (soft_transition if slots[k + 1].soft_before else HARD_CUT)
        shots.append(Shot(path=c["path"], kind=c["kind"], start=round(start, 2), seconds=seconds, transition=transition))
    used = {s.path for s in shots}
    length = begin / 30
    return Plan(shots=shots, left_out=[r["path"] for r in rows if r["path"] not in used], length=length, beat=song.beat)
