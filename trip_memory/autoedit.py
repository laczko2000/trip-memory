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

from . import transitions
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
    why: str | None = None  # why that transition was suggested
    speed: float = 1.0  # 0.5 = slow motion, 1.5 = sped up
    speed_why: str | None = None
    effects: dict = field(default_factory=dict)  # {"punch": [seconds], "ramp": bool, "push": bool}
    effects_why: str | None = None


@dataclass
class Plan:
    shots: list[Shot] = field(default_factory=list)
    left_out: list[str] = field(default_factory=list)
    length: float = 0.0
    beat: Beat | None = None
    pins: int = 0  # how many of your pins (Music tab) the cuts follow; 0 = cut on the music by itself
    kept_pins: list[float] | None = None  # too few shots for all your pins: the ones this cut used
    # Short of shots, a long clip gives two or three: "clip.mov#2" is its second part. path -> (the file, where in
    # it (seconds into the part of the clip Auto-edit saw) the part starts).
    parts: dict[str, tuple[str, float]] = field(default_factory=dict)
    outside: list[str] = field(default_factory=list)  # selected, but never offered (a repeat, or in no chapter)


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


BAD = -2.0  # a moment that should never be on screen


def interest(action: list[float], shake: list[float], loud: list[float], n: int, sharp: list[float] | None = None,
             camera: list[float] | None = None, bright: list[float] | None = None) -> np.ndarray:
    """A score five times a second for how good that moment looks.

    Movement in the scene and sound count up. Shake, blur, a camera swinging about and sudden changes in
    brightness count down, and the moments when the phone is being raised or lowered at the start and end
    are left out altogether."""

    def norm(values: list[float], length: int) -> np.ndarray:
        v = np.zeros(length) if not values else np.resize(np.asarray(values, dtype=float), length)
        spread = v.max() - v.min()
        return (v - v.min()) / spread if spread > 0 else np.zeros(length)

    def fit(values: list[float] | None, length: int) -> np.ndarray | None:
        if not values:
            return None
        v = np.asarray(values, dtype=float)
        return np.concatenate([v, np.full(max(0, length - len(v)), v[-1])])[:length]

    score = 0.5 * norm(np.minimum(action, 30).tolist() if action else [], n)
    score += 0.35 * norm(loud, n)
    score -= 0.3 * norm(shake, n)
    bad = np.zeros(n, bool)
    clear = fit(sharp, n)
    if clear is not None:
        # Compared with the moments around it (3 s either way), since some scenes simply have less detail.
        reach = 3 * STEPS_PER_SECOND
        around = np.array([np.percentile(clear[max(0, i - reach) : i + reach + 1], 80) for i in range(n)])
        crisp = clear / np.maximum(around, 1e-6)  # 1 = as sharp as it gets around here
        score -= 0.6 * np.clip(1 - crisp, 0, 1)
        bad |= crisp < 0.35  # blurred (moving too fast, or out of focus)
    swing = fit(camera, n)
    if swing is not None:
        score -= 0.4 * np.clip((swing - 4) / 8, 0, 1)  # a calm pan is fine; flailing about is not
        bad |= swing > 14
    light = fit(bright, n)
    if light is not None:
        bad |= np.abs(np.diff(light, prepend=light[0])) > 25  # the exposure jumping
    # The start and end of a phone video are usually the camera being raised or lowered: leave out the
    # first and last moments, and more of them while the camera is still swinging or the picture is blurred.
    edge = min(n // 4, STEPS_PER_SECOND)
    score[:edge] -= 0.3
    score[n - edge :] -= 0.3
    # (A steady pan that runs through the whole clip is meant: only movement well beyond the clip's usual counts.)
    usual = float(np.median(swing[n // 4 : n - n // 4])) if swing is not None and n >= 4 else 0.0
    for order in (range(n), range(n - 1, -1, -1)):
        for k, i in enumerate(order):
            if k >= min(n // 3, 3 * STEPS_PER_SECOND):
                break
            settling = (swing is not None and swing[i] > max(3.0, 2 * usual)) or bad[i]
            if not settling and k >= STEPS_PER_SECOND // 2:
                break
            bad[i] = True
    score[bad] = BAD
    return score


def quality(score: np.ndarray, seconds: float = 2.0) -> float:
    """How good a clip's best stretch of the given length is (about -2 bad .. 1 great)."""
    window = max(1, int(round(seconds * STEPS_PER_SECOND)))
    if len(score) == 0:
        return 0.0
    if window >= len(score):
        return float(score.mean())
    return float(np.convolve(score, np.ones(window) / window, mode="valid").max())


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
    if row.get("ai_score"):
        value += 0.25 * (row["ai_score"] - 3)  # the AI director's view: a moment worth remembering, or a dull one
    if row.get("plan_role") == "highlight":
        value += 1.0
    elif row.get("plan_pos") is None and row.get("planned_story"):
        value -= 1.0  # the story's plan left this shot out: used only if the song needs more shots
    value += 0.9 * (row.get("wish") or 0.0)  # the people and things you asked for when typing your video
    if row.get("quality") is not None:
        value += 0.8 * min(0.0, row["quality"])  # a clip with no steady, sharp stretch is a poor choice
    return value


# How Auto-edit arranges the shots.
STORYLINES = {
    "teaser": "Teaser, then the stories",  # a few of the most exciting shots first, then story by story
    "stories": "Story by story",  # each story opens with a wide, calm view; the video ends on a calm shot
    "date": "Plain date order",
}
TEASER_SHOTS = 4


def excitement(row: dict) -> float:
    pace, _ = pace_of(row)
    return pace + 0.5 * (row.get("peak") or 0.0) + 0.2 * shot_value(row) + (0.5 if row.get("plan_role") == "highlight" else 0)


def _establishing(row: dict) -> float:
    """How well a shot sets the scene: a wide, steady, calm view (or a smooth camera move over one)."""
    tags = row.get("tags_final") or []
    pace, cinematic = pace_of(row)
    score = (1 - pace) + (0.6 if cinematic else 0) + (0.3 if row.get("people_tag") in (None, "No people") else 0)
    return score + 0.3 * sum(t in tags for t in ("Landscape", "Cityscape", "Beach", "Mountains", "Sunset"))


def arrange(candidates: list[dict], style: str) -> list[dict]:
    """Put the shots in story order with some shape: an opener per story, no two look-alikes back to back,
    a calm last shot, and (for the teaser) a few of the most exciting shots up front.

    Marks shots that should certainly be used (c["must"] = True)."""
    rows = [dict(c, must=False) for c in candidates]
    if style == "date" or len(rows) < 6:
        return rows
    stories: list[list[dict]] = []
    for r in rows:
        if stories and stories[-1][0].get("story_rank", 0) == r.get("story_rank", 0) and \
                stories[-1][0].get("story") == r.get("story"):
            stories[-1].append(r)
        else:
            stories.append([r])
    teaser: list[dict] = []
    if style == "teaser" and len(rows) >= 3 * TEASER_SHOTS:
        # The most exciting shot of each of the busiest stories (no more than one per story).
        best = sorted((max(st, key=excitement) for st in stories if len(st) >= 3), key=excitement, reverse=True)
        teaser = sorted(best[:TEASER_SHOTS], key=lambda r: r["taken_at"])
        chosen = {id(r) for r in teaser}
        stories = [[r for r in st if id(r) not in chosen] for st in stories]
        for r in teaser:
            r["must"] = True
    shaped: list[dict] = []
    for st in stories:
        if any(r.get("plan_role") for r in st):  # the story director's plan sets the order
            for r in st:
                r["must"] = r["must"] or (r.get("plan_role") == "highlight")
            opening = next((r for r in st if r.get("plan_role") == "opening"), None)
            if opening is not None:
                opening["must"] = True
            shaped += st
            continue
        if len(st) >= 4:
            opener = max(st[: max(2, len(st) // 3)], key=_establishing)  # from the early part of the story
            st = [opener, *[r for r in st if r is not opener]]
        shaped += _spread_look_alikes(st)
    if stories and any(r.get("plan_role") for r in stories[-1]):
        for r in stories[-1]:
            r["must"] = r["must"] or r.get("plan_role") == "ending"  # the plan's own ending closes the video
    elif len(shaped) >= 4:
        # End on a calm, steady shot from the last story.
        last_story = shaped[-max(2, len(stories[-1])):]
        closer = max(last_story, key=_establishing)
        shaped = [r for r in shaped if r is not closer] + [closer]
        closer["must"] = True
        # Keep the last story free of back-to-back look-alikes, working back from the closer (which stays last).
        tail = len(last_story)
        shaped[-tail:] = _spread_look_alikes(shaped[-tail:][::-1])[::-1]
    return teaser + shaped


def _spread_look_alikes(rows: list[dict]) -> list[dict]:
    """Move a shot that looks like the one before it a little later, so look-alikes don't come back to back."""
    rows = list(rows)
    for k in range(1, len(rows)):
        group = rows[k].get("look_group")
        if group is not None and rows[k - 1].get("look_group") == group:
            later = next((j for j in range(k + 1, len(rows)) if rows[j].get("look_group") != group), None)
            if later is not None:
                rows.insert(k, rows.pop(later))
                continue
            # Nothing different after it: tuck it in earlier, between two shots that don't look like it.
            shot = rows.pop(k)
            spot = next((p for p in range(k - 1, 0, -1) if group not in (rows[p - 1].get("look_group"),
                                                                         rows[p].get("look_group"))), None)
            rows.insert(spot if spot is not None else k, shot)
    return rows


def choose(rows: list[dict]) -> list[dict]:
    """The shots to choose from: everything you ticked, story by story in your story order, then as taken.

    (Choose files already ticks only the best of each set of look-alike shots for you; anything else you
    ticked yourself, you want in.)"""
    planned = {(r.get("story_rank", 0), r.get("story")) for r in rows if r.get("plan_role")}
    for r in rows:
        r["planned_story"] = (r.get("story_rank", 0), r.get("story")) in planned
    # In a planned story, the plan's order first; shots the plan left out after it, as taken.
    return sorted(rows, key=lambda r: (r.get("story_rank", 0), r.get("plan_pos") is None,
                                       r.get("plan_pos") or 0, r["taken_at"], r["path"]))


FINALE_SHARE = 0.12  # of the video, for the finale (the most exciting shots)
FINALE_FROM = 12  # shots needed before a finale is worth it
REST_WEIGHT = 0.6  # "Everything else" gets a little less time than a chapter you named
SPAN_PENALTY = 20.0  # how strongly a chapter's shots keep to their part of the video


def by_chapters(rows: list[dict], chapters: list[dict]) -> list[dict]:
    """The shots in your chapters' order (row["chapter"] says which), each chapter in the order it was taken. Every
    shot gets the part of the video its chapter should fill (r["span"], fractions 0..1): bigger chapters get more
    time. The finale takes the most exciting shots of all and ends on the most exciting one."""
    rows = [dict(r, must=False) for r in rows]
    finale = next((j for j, c in enumerate(chapters) if c["kind"] == "finale"), None)
    groups: dict[int, list[dict]] = {}
    for r in rows:
        if r.get("chapter") is not None:
            groups.setdefault(r["chapter"], []).append(r)
    if finale is not None and len(rows) < FINALE_FROM:
        finale = None  # too few shots for a finale of their own: they stay where they belong
    if finale is not None:
        size = int(np.clip(round(0.1 * len(rows)), 2, 10))
        best = sorted(rows, key=excitement, reverse=True)[:size]
        for r in best:
            if r.get("chapter") is not None and r["chapter"] != finale:
                groups[r["chapter"]].remove(r)
            r["chapter"] = finale
        groups[finale] = sorted(best, key=lambda r: (excitement(r), r["taken_at"]))  # building up to the biggest
        if groups[finale]:
            groups[finale][-1]["must"] = True
    order = [j for j in range(len(chapters)) if groups.get(j)]
    weights = {j: (np.sqrt(len(groups[j])) * (REST_WEIGHT if chapters[j]["kind"] == "rest" else 1.0))
               for j in order if j != finale}
    room = 1.0 - (FINALE_SHARE if finale in order and weights else 0.0)
    total = sum(weights.values()) or 1.0
    out: list[dict] = []
    at = 0.0
    for j in order:
        share = FINALE_SHARE if j == finale and weights else room * weights.get(j, total) / total
        members = groups[j] if j == finale else _spread_look_alikes(
            sorted(groups[j], key=lambda r: (r["taken_at"], r["path"])))
        for r in members:
            r["span"] = (at, at + share)
        out += members
        at += share
    return out


# ---------- music-driven editing ----------


@dataclass
class Limits:
    """Your limits for the shots (seconds)."""

    max_video: float = 4.0
    max_photo: float = 2.5
    min_shot: float = 1.0
    cut_on: str = "strong"  # which beats the cuts go on: a key of CUT_ON
    storyline: str = "teaser"  # how the shots are arranged: a key of STORYLINES
    speed_effects: bool = True  # slow motion in calm moments, sped-up shots in lively ones
    camera_effects: bool = True  # zoom transitions, punch-ins on big hits, slow push-ins, speed ramps


@dataclass
class Slot:
    start: float  # where the shot starts in the music (the cut before it)
    end: float  # the cut after it
    energy: float  # how lively the music is here, 0..1
    soft_before: bool = False  # a soft transition (fade) into this slot instead of a hard cut
    tapped: bool = False  # both its cuts are your pins (Music tab): never joined, never split, however long

    @property
    def seconds(self) -> float:
        return self.end - self.start


# How many beats a shot lasts, by how lively the music is: quick cuts when it's busy, long ones when calm.
BEATS_FOR = {"Lively": 2, "Medium": 4, "Calm": 8}

# Where the cuts go. Every one of them still lands on a beat, and every change in the music is a cut.
CUT_ON = {
    "taps": "My taps (Music tab)",  # the cuts you tapped along to the song yourself
    "strong": "Strong beats",  # the hardest hits, keeping roughly the pace the music asks for
    "bass": "Bass hits",  # the kick drum and bass notes
    "changes": "Big moments only",  # few cuts, held as long as allowed, on the biggest hits
    "beats": "Every few beats",  # a steady count of beats, line up with the bars
}
CUT_STRENGTH = {"strong": 1.5, "bass": 1.5, "changes": 3.0}  # how far a cut may drift from the pace for a hit


def music_slots(song, limits: Limits, end: float, longest: float | None = None) -> list[Slot]:
    """Cut the music into shot-sized pieces: every cut on a beat, every change in the music a cut."""
    longest = longest or max(limits.max_video, limits.max_photo)
    beats = song.beat_times()
    taps = getattr(song, "taps", None)  # your pins (Music tab) always lead; "Cut on" is for a song without them
    # Your taps decide the cuts up to your last tap; any part of the song after it is cut automatically.
    tapped = min(end, max(taps)) if taps else 0.0
    slots: list[Slot] = _tap_slots(song, taps, tapped) if taps else []
    for section in song.sections:
        if section.start >= end:
            break
        begin = max(section.start, tapped)
        if section.end <= begin + 1e-3:
            continue
        n = BEATS_FOR.get(section.level, 4)
        while n > 1 and n * song.period > longest * 1.03:  # a hair over the limit is fine (tempo isn't exact)
            n //= 2
        while n * song.period < limits.min_shot - 1e-6:
            n *= 2
        stop = min(section.end, end)
        cuts = _cuts_on_hits(song, limits, begin, stop, n, longest)
        if cuts is None:
            # Cuts every n beats, counted from the bar lines, so they line up with the music's bars.
            cuts = [begin]
            for k, b in enumerate(beats):
                if begin + 1e-3 < b < stop - 1e-3 and (k - song.downbeat) % n == 0 and b - cuts[-1] >= limits.min_shot - 1e-6:
                    cuts.append(float(b))
            cuts.append(stop)
        # Too short a piece at the end joins the one before it, when the joined piece is allowed.
        if len(cuts) > 2 and cuts[-1] - cuts[-2] < limits.min_shot and cuts[-1] - cuts[-3] <= longest:
            cuts.pop(-2)
        slots += [Slot(a, b, section.energy) for a, b in zip(cuts, cuts[1:]) if b - a > 1e-3]
    # Any piece still longer than allowed is split evenly on beats; never the time between two of your pins.
    result: list[Slot] = []
    for s in slots:
        pieces = 1 if s.tapped else int(np.ceil(s.seconds / (longest * 1.03)))
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
        result += [Slot(a, b, s.energy, tapped=s.tapped) for a, b in zip(cuts, cuts[1:])]
    return result


def _tap_slots(song, taps: list[float], end: float) -> list[Slot]:
    """Your own cuts: exactly where you put the pins, as close together as a tenth of a second."""
    cuts = [0.0]
    for t in sorted(taps):
        if 0.1 <= t - cuts[-1] and t < end - 0.1:
            cuts.append(float(t))
    cuts.append(end)
    return [Slot(a, b, song.energy_at((a + b) / 2), tapped=True) for a, b in zip(cuts, cuts[1:])]


def _cuts_on_hits(song, limits: Limits, start: float, stop: float, n: int, longest: float) -> list[float] | None:
    """Cuts on the music's hits: the best set of beats to cut on, trading how hard each one hits against
    keeping close to the pace (n beats a shot) and within your limits. None = cut on a steady count instead."""
    mode = "strong" if limits.cut_on == "taps" else limits.cut_on  # no taps for this song: its strong beats
    mode = mode if mode in CUT_STRENGTH else None
    strength = getattr(song, "bass" if mode == "bass" else "accents", None)
    beats = song.beat_times()
    if mode is None or not strength or len(strength) != len(beats):
        return None
    strength = np.asarray(strength, float)
    pace = longest if mode == "changes" else n * song.period
    if mode == "changes":
        strength = np.where(strength >= 0.6, strength, 0.0)  # only the big hits count
    inside = [(float(t), float(strength[k])) for k, t in enumerate(beats) if start + 1e-3 < t < stop - 1e-3]
    points = [(start, 0.0), *inside, (stop, 0.0)]
    shortest, most = limits.min_shot - 1e-6, longest * 1.03
    weight = CUT_STRENGTH[mode]
    best = np.full(len(points), -np.inf)
    back = np.zeros(len(points), int)
    best[0] = 0.0
    for j in range(1, len(points)):
        tj, hit = points[j]
        for i in range(j - 1, -1, -1):
            d = tj - points[i][0]
            if d > most:
                break
            if d < shortest or best[i] == -np.inf:
                continue
            score = best[i] + weight * hit - ((d - pace) / pace) ** 2
            if score > best[j]:
                best[j], back[j] = score, i
    if best[-1] == -np.inf:
        return None
    cuts, j = [], len(points) - 1
    while j > 0:
        cuts.append(points[j][0])
        j = back[j]
    return [start, *cuts[::-1]]


def _measure(r: dict, curve) -> None:
    """How good a video looks overall, at its best, and its longest good stretch."""
    if curve is None or not len(curve):
        return
    r["peak"] = float(np.clip(np.percentile(curve, 90), 0, 1))
    r["quality"] = quality(curve)
    good = np.concatenate([[False], curve > BAD, [False]]).astype(int)
    edges = np.flatnonzero(np.diff(good))
    runs = edges[1::2] - edges[::2]
    r["usable"] = float(runs.max()) / STEPS_PER_SECOND if len(runs) else 0.0  # longest good stretch


PART_FROM = 12.0  # seconds: a clip this long can give two shots (twice this: three)
PART_GAP = 2.0  # seconds left out between two parts of a clip, so they never show the same moment


def split_long(rows: list[dict], curves: dict, short: int) -> tuple[list[dict], dict[str, tuple[str, float]]]:
    """Short of `short` shots: the longest clips each give two or three shots, from different parts of the clip
    that never overlap (the clip itself keeps only its first part). curves gets each part's own curve."""
    parts: dict[str, tuple[str, float]] = {}
    split: dict[str, list[dict]] = {}
    for r in sorted((r for r in rows if r["kind"] == "video" and (r["duration"] or 0) >= PART_FROM),
                    key=lambda r: -(r["duration"] or 0)):
        if short <= 0:
            break
        length = r["duration"]
        n = min(3 if length >= 2 * PART_FROM else 2, short + 1)
        each = (length - (n - 1) * PART_GAP) / n
        curve = curves.get(r["path"])
        pieces = []
        for k in range(n):
            path = r["path"] if k == 0 else f"{r['path']}#{k + 1}"
            begin = k * (each + PART_GAP)
            piece = r | {"path": path, "duration": each, "part_of": r["path"] if k else None}
            if curve is not None and len(curve):
                curves[path] = curve[int(round(begin * STEPS_PER_SECOND)):int(round((begin + each) * STEPS_PER_SECOND))]
                _measure(piece, curves[path])
            parts[path] = (r["path"], begin)
            pieces.append(piece)
        split[r["path"]] = pieces
        short -= n - 1
    result = [p for r in rows for p in split.get(r["path"], [r])]
    return result, parts


def fit_pins(song, pins: list[float], count: int, longest: float) -> list[float]:
    """Too few shots for all your pins: keep `count` of them. The least important go first, so the pattern
    stays (a pin every 2 s becomes one every 4 s); pins on big hits and where the music changes, and your first
    and last pin, stay longest. No shot is made longer than allowed (so a few more may stay)."""
    pins = sorted(pins)
    if len(pins) <= max(count, 2):
        return pins
    beats = song.beat_times()
    accents = np.asarray(getattr(song, "accents", None) or [], float)
    changes = [s.start for s in song.sections[1:]]

    def weight(p: float) -> float:
        w = 0.0
        if any(abs(p - c) <= max(0.35, song.period) for c in changes):
            w += 2.0
        if len(accents) == len(beats) and len(beats):
            k = int(np.argmin(np.abs(beats - p)))
            if abs(beats[k] - p) < 0.1:
                w += float(accents[k])
        return w

    weights = [weight(p) for p in pins]
    usual = float(np.median(np.diff(pins))) or 1.0
    while len(pins) > count:
        best, drop = None, None
        for i in range(1, len(pins) - 1):  # never the first or the last
            joined = pins[i + 1] - pins[i - 1]
            if joined > longest * 1.03:
                continue
            cost = joined / usual + 1.5 * weights[i]
            if best is None or cost < best - 1e-9:
                best, drop = cost, i
        if drop is None:
            break  # every shot is as long as allowed already
        del pins[drop], weights[drop]
    return pins


def merge_slots(slots: list[Slot], count: int, longest: float) -> list[Slot]:
    """With fewer shots than pieces of music, join neighbouring pieces (in the same part of the song, and
    never longer than allowed) so the shots you have stretch further."""
    slots = list(slots)
    while len(slots) > count:
        pairs = [
            (slots[k].seconds + slots[k + 1].seconds, k)
            for k in range(len(slots) - 1)
            if slots[k].energy == slots[k + 1].energy and slots[k].seconds + slots[k + 1].seconds <= longest * 1.03
            and not (slots[k].tapped or slots[k + 1].tapped)  # your own cuts always stay
        ]
        if not pairs:
            break
        _, k = min(pairs)
        slots[k : k + 2] = [Slot(slots[k].start, slots[k + 1].end, slots[k].energy)]
    return slots


def pace_of(row: dict) -> tuple[float, bool]:
    """How lively a shot is (0 still .. 1 fast action) and whether it is a smooth, cinematic camera move."""
    role = row.get("plan_role")
    if role == "montage":
        return 0.9, False  # cut quickly on the beat: goes to the lively parts of the song, in short shots
    if role == "slow moment":
        return 0.15, True  # goes to calm music: slow motion or a slow push-in
    if role in ("opening", "ending"):
        pace, cinematic = pace_of({**row, "plan_role": None})
        return min(pace, 0.35), cinematic
    if row["kind"] != "video":
        return 0.15, False
    action, camera, shake = row.get("motion_action"), row.get("motion_camera"), row.get("motion_shake")
    if action is None:
        tags = row.get("tags_final") or []
        level = {"Fast action": 0.95, "Lively": 0.7, "Calm": 0.4, "Still": 0.2}
        return next((level[t] for t in level if t in tags), 0.5), "Cinematic" in tags
    pace = float(np.clip(np.log1p(action) / np.log1p(15), 0, 1))
    cinematic = camera is not None and camera >= 1.0 and (shake or 0) < 0.6 and action < 3
    if {"Whip pan", "Fast camera", "Turn around"} & set(row.get("tags_final") or []):
        return max(pace, 0.75), False  # a swinging camera feels fast even when the scene is calm
    return (0.3 if cinematic else pace), cinematic


SLOW, FAST_SPEED = 0.5, 1.5


def speed_for(row: dict, slot: Slot, seconds: float) -> tuple[float, str | None]:
    """Slow motion for a calm, cinematic moment in calm music; a speed-up for a slow shot in lively music."""
    if row["kind"] != "video":
        return 1.0, None
    pace, cinematic = pace_of(row)
    if slot.energy < 0.38 and (cinematic or pace < 0.35) and seconds >= 2.0:
        if (row.get("fps") or 30) >= 50:
            return SLOW, "Slow motion: a calm, cinematic moment"
        return 0.75, "A little slower: a calm, cinematic moment"  # gentler, so normal videos stay smooth
    if slot.energy >= 0.62 and pace < 0.45 and (row.get("duration") or 0) >= seconds * FAST_SPEED + 0.1:
        return FAST_SPEED, "Sped up to match the lively music"
    return 1.0, None


COLLAGE_MAX = 4  # photos in one collage (2x2)
COLLAGE_GAP = 12.0  # at least this many seconds between two collages
COLLAGE_SHOT = 1.6  # only quick shots are gathered into a collage


def collages(shots: list[Shot], slots: list[Slot], rows: dict[str, dict], transition_seconds: float) -> list[Shot]:
    """In the lively parts, a quick run of 3 or 4 photos from one story becomes one collage: the photos pop in one
    after another on the same beats where they would have been cut, and stay together on screen."""
    result: list[Shot] = []
    k, last_end, at = 0, -99.0, 0.0  # at: where shot k starts in the video (seconds)

    def fits(n: int, story) -> bool:
        shot = shots[n]
        return (shot.kind == "photo" and not shot.effects and shot.seconds <= COLLAGE_SHOT and n < len(slots)
                and slots[n].energy >= 0.62 and rows.get(shot.path, {}).get("story") == story)

    while k < len(shots):
        story = rows.get(shots[k].path, {}).get("story")
        run = k
        while run < len(shots) and run - k < COLLAGE_MAX and fits(run, story) and (
                run == k or shots[run - 1].transition in (None, HARD_CUT)):
            run += 1
        if run - k >= 3 and at - last_end >= COLLAGE_GAP:
            group = shots[k:run]
            inside = [transitions.seconds_of(g.transition, transition_seconds) for g in group[:-1]]  # cut overlaps
            appear, t = [], 0.0
            for g, overlap in zip(group, [*inside, 0.0]):
                appear.append(round(t, 3))
                t += g.seconds - overlap
            lead = group[0]
            result.append(Shot(path=lead.path, kind="photo", seconds=round(t, 4), transition=group[-1].transition,
                               why=group[-1].why,
                               effects={"collage": [g.path for g in group[1:]], "appear": appear},
                               effects_why=f"Collage of {len(group)} photos, popping in on the beat"))
            last_end = at + t
            at += t
            k = run
            continue
        at += shots[k].seconds - transitions.seconds_of(shots[k].transition, transition_seconds)
        result.append(shots[k])
        k += 1
    return result


FREEZE = 1.0  # seconds a freeze-frame holds, with the person's name on it
FREEZE_GAP = 20.0  # at most one freeze-frame in this many seconds


def freeze_for(row: dict, slot: Slot, seconds: float, introduced: set[str], since: float) -> tuple[dict, str | None]:
    """The first good video of someone you named ends frozen for a moment with their name, like a film
    introducing its cast. Not in the busiest music, and not too often."""
    names = [p for p in row.get("people") or [] if not p.startswith("Person ") and p not in introduced]
    if (row["kind"] != "video" or not names or slot.energy >= 0.75 or seconds < 2.2 or since < FREEZE_GAP
            or row.get("people_tag") not in ("Solo", "Couple", "Group")):
        return {}, None
    names = names[:2]
    introduced.update(names)
    return {"freeze": FREEZE, "label": " & ".join(names)}, f"Freeze-frame introducing {' & '.join(names)}"


def effects_for(row: dict, slot: Slot, seconds: float, speed: float, hits: list[tuple[float, float]], turn: int,
                shot_start: float | None = None) -> tuple[dict, str | None]:
    # turn: how many shots since the last one with an effect (punch-ins and ramps are kept apart)
    """Camera-style moves inside a shot, to suit the music: a punch-in on a big hit or a speed ramp when it's
    lively, a slow push-in on a still shot when it's calm. Used on some shots only, so they stay special."""
    if row["kind"] != "video":
        return {}, None
    pace, cinematic = pace_of(row)
    tags = set(row.get("tags_final") or [])
    if slot.energy >= 0.62:
        inside = [(t - slot.start, h) for t, h in hits if slot.start + 0.3 < t < slot.end - 0.3 and h >= 0.75]
        if inside and turn >= 2:
            t, h = max(inside, key=lambda x: x[1])
            t += slot.start - (slot.start if shot_start is None else shot_start)  # from where the shot starts on screen
            action = bool({"Fast action", "Lively", "Shaky"} & tags) or (row.get("motion_action") or 0) >= 0.6
            if h >= 0.9 and action:
                return {"shake": [round(t, 3)]}, "Camera shake and a flash on the hardest hit"
            return {"punch": [round(t, 3)]}, "Punch-in on a big hit"
        room = row.get("usable") or row.get("duration") or 0
        if speed == 1.0 and seconds >= 2.0 and room >= seconds * 1.4 + 0.1 and turn >= 3:
            return {"ramp": True}, "Speed ramp: fast through the middle of the shot"
    if slot.energy < 0.38 and not cinematic and ({"Static camera", "Still", "Calm"} & tags or pace < 0.3) \
            and seconds >= 2.0:
        return {"push": True}, "Slow push-in: a still shot made cinematic"
    return {}, None


MIN_FILL_SPEED = 0.5  # a video a little shorter than the time between two pins may be slowed down this far


def fill_slots(slots: list[Slot], candidates: list[dict], limits: Limits) -> list[int]:
    """Choose a shot for every slot, keeping the trip in time order, matching lively shots to lively music.

    Returns the candidate index for each slot that could be filled (stops early if it runs out)."""
    n_slots, n = len(slots), len(candidates)
    if not n_slots or not n:
        return []
    paces = [pace_of(c)[0] for c in candidates]
    wants = [c.get("pace_want") for c in candidates]
    values = [shot_value(c) + (4.0 if c.get("must") else 0.0) + 0.3 * min(3.0, c.get("chapter_fit") or 0.0)
              for c in candidates]
    # Chapters: a shot outside the part of the video its chapter should fill costs points, the further the more.
    total = slots[-1].end - slots[0].start or 1.0
    spans = np.array([c.get("span") or (0.0, 1.0) for c in candidates])
    neg = -1e9
    score = np.full((n_slots, n), neg)
    back = np.full((n_slots, n), -1, dtype=int)
    for j, slot in enumerate(slots):
        best_prev, best_idx = (0.0, -1) if j == 0 else (neg, -1)
        where = ((slot.start + slot.end) / 2 - slots[0].start) / total
        off = SPAN_PENALTY * np.maximum(0.0, np.maximum(spans[:, 0] - where, where - spans[:, 1]))
        for i, c in enumerate(candidates):
            if j > 0 and i > 0 and score[j - 1, i - 1] > best_prev:
                best_prev, best_idx = score[j - 1, i - 1], i - 1
            if best_prev <= neg / 2:
                continue
            if slot.tapped:
                # Your pins come first, whatever your length limits: any photo, or any video that fills the time
                # (slowed down to half speed at most if it is a little short).
                if c["kind"] == "video" and (c["duration"] or 0) < slot.seconds * MIN_FILL_SPEED - 0.05:
                    continue
            elif c["kind"] == "video":
                # Fits in the clip's longest good stretch (or, if that is very short, the clip as a whole).
                room = c.get("usable") if (c.get("usable") or 0) >= limits.min_shot else c["duration"]
                if (room or 0) < slot.seconds - 0.05 or slot.seconds > limits.max_video * 1.03 + 0.05:
                    continue
            elif slot.seconds > limits.max_photo * 1.03 + 0.05:
                continue
            fit = 1.0 - abs(slot.energy - paces[i])
            if wants[i] is not None:  # a story part that should feel calm or lively: shots of that pace fit better
                fit = 0.5 * fit + 0.5 * (1.0 - abs(wants[i] - paces[i]))
            score[j, i] = best_prev + 3.0 + 0.5 * values[i] + 2.0 * fit - off[i]
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
    chapters: list[dict] | None = None,
) -> Plan:
    """The music-driven first cut: the song decides where the cuts are, the shots are matched to it.
    chapters: your chapters (rows then carry "chapter"): they set the order instead of the storyline."""
    curves = dict(curves)
    for r in rows:
        _measure(r, curves.get(r["path"]))

    def order(rows: list[dict]) -> list[dict]:
        return by_chapters(rows, chapters) if chapters else arrange(choose(rows), limits.storyline)

    candidates = order(rows)
    end = song.duration if target is None or target >= song.duration else target
    if target is not None and target < song.duration:
        taps = getattr(song, "taps", None) or []
        near = [t for t in taps if abs(t - target) <= max(4.0, 0.15 * target) and t >= limits.min_shot]
        if near:  # a shorter video ends on your pin closest to the length you asked for
            end = float(min(near, key=lambda t: abs(t - target)))
        else:
            bars = song.bar_starts()
            later = bars[bars >= max(limits.min_shot, target - 2 * song.period)]
            end = float(later[0]) if len(later) else target  # end on a bar line
    def attempt() -> tuple[list[Slot], list[int]]:
        best: tuple[list[Slot], list[int]] = ([], [])
        for longest in (None, limits.max_photo, max(limits.min_shot, limits.max_photo / 2)):
            slots = music_slots(song, limits, end, longest)
            slots = merge_slots(slots, len(candidates), longest or max(limits.max_video, limits.max_photo))
            picks = fill_slots(slots, candidates, limits)
            if len(picks) == len(slots):
                return slots, picks
            if sum(s.seconds for s in slots[: len(picks)]) > sum(s.seconds for s in best[0][: len(best[1])]):
                best = (slots, picks)
        return best

    kept = None
    taps = sorted(getattr(song, "taps", None) or [])
    inside = [t for t in taps if 0 < t < end - 1e-3]
    after = [t for t in taps if t >= end - 1e-3]
    cap = max(limits.max_video, limits.max_photo)
    # Fewer shots than the music needs: a long clip gives two or three, from different parts of it.
    need = len(inside) + 1 if inside else int(np.ceil(end / ((limits.min_shot + cap) / 2)))
    parts: dict[str, tuple[str, float]] = {}
    if len(candidates) < need:
        rows, parts = split_long(rows, curves, need - len(candidates))
        candidates = order(rows)
    if inside and len(candidates) - 1 < len(inside):  # fewer shots than your pins: a first guess
        kept = fit_pins(song, inside, max(1, len(candidates) - 1), cap)
        song.taps = kept + after
    slots, picks = attempt()

    def covered(slots: list[Slot], picks: list[int]) -> float:
        return sum(s.seconds for s in slots[: len(picks)])

    tries = [(slots, picks, kept)]
    # Still short of shots (some clips too short for the time between pins, or the part after your last pin
    # needs shots too): use fewer pins again, until the video lasts as long as it should. Rather a few longer
    # shots than a video that stops early.
    for _ in range(8):
        if not inside or len(picks) >= len(slots):
            break
        now = kept or inside
        want = max(1, len(now) - (len(slots) - len(picks)))
        fewer = fit_pins(song, inside, want, cap)
        while len(fewer) > want and cap < 60:  # as far as your longest-shot setting allows: allow a bit longer
            cap *= 1.5
            fewer = fit_pins(song, inside, want, cap)
        if len(fewer) >= len(now):
            break
        kept = fewer
        song.taps = kept + after
        slots, picks = attempt()
        tries.append((slots, picks, kept))
    # The try that fills (nearly) the most of the music, with as many shots as possible.
    most = max(covered(t[0], t[1]) for t in tries)
    slots, picks, kept = max((t for t in tries if covered(t[0], t[1]) >= most - max(3.0, 0.03 * most)),
                             key=lambda t: len(t[1]))
    song.taps = (kept or inside) + after
    slots = slots[: len(picks)]
    if not slots:
        return Plan(left_out=[r["path"] for r in rows], beat=song.beat)

    # A suggested transition at every cut, from the music, the stories and the shots, each sized to fit.
    beats_at = song.beat_times()
    accents = np.asarray(getattr(song, "accents", None) or [], float)

    def hit_at(t: float) -> float:
        if len(accents) != len(beats_at) or not len(accents):
            return 0.0
        k = int(np.argmin(np.abs(beats_at - t)))
        return float(accents[k]) if abs(beats_at[k] - t) < 0.06 else 0.0

    cuts = [
        transitions.Cut(before=candidates[picks[k - 1]], after=candidates[picks[k]],
                        energy_before=slots[k - 1].energy, energy_after=slots[k].energy,
                        room=0.4 * min(slots[k - 1].seconds, slots[k].seconds), hit=hit_at(slots[k].start))
        for k in range(1, len(slots))
    ]
    chosen = transitions.recommend(cuts, transition_seconds)
    if not limits.camera_effects:  # no zoom transitions either: a clean cut instead
        chosen = [(HARD_CUT, "Clean cut on the beat") if name in transitions.ZOOM_INTO else (name, why)
                  for name, why in chosen]
    hits = [(float(t), float(a)) for t, a in zip(beats_at, accents)] if len(accents) == len(beats_at) else []
    overlap = [transitions.seconds_of(name, transition_seconds) for name, _ in chosen] + [0.0]  # into the next shot

    shots: list[Shot] = []
    since_fx = 99
    introduced: set[str] = set()  # people already shown with their name on a freeze-frame
    last_freeze = -99.0
    previous_story = None
    begin = 0  # in video frames, where this shot starts
    for k, (slot, i) in enumerate(zip(slots, picks)):
        c = candidates[i]
        # The change-over to the next shot is centred on the cut; counted in whole frames so nothing drifts.
        finish = round((slot.end + overlap[k] / 2) * 30)
        seconds = (finish - begin) / 30
        begin = finish - round(overlap[k] * 30)
        start = 0.0
        speed, speed_why = speed_for(c, slot, seconds) if limits.speed_effects else (1.0, None)
        effects, effects_why = (effects_for(c, slot, seconds, speed, hits, since_fx, finish / 30 - seconds)
                                if limits.camera_effects else ({}, None))
        new_story = c.get("story") != previous_story  # its first shot carries the story's title already
        previous_story = c.get("story")
        if limits.camera_effects and not effects and not new_story:
            effects, effects_why = freeze_for(c, slot, seconds, introduced, slot.start - last_freeze)
            if effects:
                last_freeze = slot.start
        since_fx = 0 if effects.get("punch") or effects.get("ramp") or effects.get("shake") else since_fx + 1
        if effects.get("ramp") and not limits.speed_effects:
            effects, effects_why = {}, None
        if slot.tapped and c["kind"] == "video" and (c["duration"] or 0) < seconds * speed * (1.4 if effects.get("ramp") else 1.0) - 0.01:
            # Too short for the time between your pins: slow it down so it fills it, rather than add a cut.
            if effects.get("ramp"):
                effects, effects_why = {}, None
            speed = max(MIN_FILL_SPEED, (c["duration"] or 0) / seconds - 0.001)
            speed_why = "Slowed down to fill the time between your pins"
        if c["kind"] == "video":
            curve = curves.get(c["path"])
            used = (seconds - (effects.get("freeze") or 0)) * speed * (1.4 if effects.get("ramp") else 1.0)  # of the video
            longest_start = max(0.0, (c["duration"] or used) - used)
            start = best_start(curve, c["duration"], used) if curve is not None else longest_start / 2
            start = min(max(0.0, start), longest_start)
        transition, why = chosen[k] if k < len(chosen) else (None, None)
        shots.append(Shot(path=c["path"], kind=c["kind"], start=round(start, 2), seconds=seconds, transition=transition,
                          why=why, speed=speed, speed_why=speed_why, effects=effects, effects_why=effects_why))
    if limits.camera_effects:
        shots = collages(shots, slots, {r["path"]: r for r in candidates}, transition_seconds)
    used = {s.path for s in shots} | {p for s in shots for p in s.effects.get("collage", [])}
    used |= {parts[p][0] for p in used if p in parts}
    length = begin / 30
    pins = sum(1 for k in range(1, len(slots)) if slots[k].tapped or slots[k - 1].tapped)
    return Plan(shots=shots, left_out=[r["path"] for r in rows if r["path"] not in used and not r.get("part_of")],
                length=length, beat=song.beat, pins=pins, kept_pins=kept if kept and len(kept) < len(inside) else None,
                parts={p: parts[p] for p in {s.path for s in shots} if p in parts})
