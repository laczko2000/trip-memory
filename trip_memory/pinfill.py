"""Finish my pins: you pin the start of a song by hand, the app learns how you pin and suggests pins for the rest.

What it learns from your pins: how many beats apart they are in calm, medium and lively stretches, where in the bar
they fall, whether you like the beats that hit hard, and whether you pin where the music changes. It then walks
on through the rest of the song choosing the beats that fit that pattern best. You listen, change what you like
and keep them.
"""

from __future__ import annotations

import numpy as np

from .songmap import BEATS_PER_BAR, SongMap

LEAST = 4  # pins needed before the app can learn anything from them


def _mood_at(stretches: list[dict], t: float) -> str:
    for s in stretches:
        if s["start"] <= t < s["end"]:
            return s["mood"]
    return stretches[-1]["mood"] if stretches else "Medium"


def learn(song: SongMap, pins: list[float], stretches: list[dict]) -> dict | None:
    """Your way of pinning, from the pins you placed. None: too few pins to tell."""
    pins = sorted(p for p in pins if 0 <= p <= song.duration)
    if len(pins) < LEAST:
        return None
    beats = song.beat_times()
    period = song.period
    gaps: dict[str, list[float]] = {}
    for a, b in zip(pins, pins[1:]):
        gaps.setdefault(_mood_at(stretches, a), []).append((b - a) / period)
    pattern = {
        "gaps": {mood: float(np.median(g)) for mood, g in gaps.items()},
        "gap": float(np.median([g for v in gaps.values() for g in v])),
        "until": pins[-1],
        "offset": 0.0, "bar": [1.0] * BEATS_PER_BAR, "hits": 0.0, "changes": 0.0,
        "on_beat": False,
    }
    if not song.clear_beat or len(beats) < 8:
        return pattern  # no steady beat to follow: only the spacing counts
    near = np.abs(beats[None, :] - np.asarray(pins)[:, None]).argmin(axis=1)
    off = np.asarray(pins) - beats[near]
    on = np.abs(off) <= max(0.08, 0.25 * period)
    pattern["on_beat"] = bool(on.mean() >= 0.6)
    if pattern["on_beat"]:
        pattern["offset"] = float(np.median(off[on]))  # you may pin a touch early or late: keep that feel
        where = (near[on] - song.downbeat) % BEATS_PER_BAR
        counts = np.bincount(where, minlength=BEATS_PER_BAR) + 0.5
        pattern["bar"] = [float(c) for c in counts / counts.max()]
        accents = np.asarray(song.accents, float) if len(song.accents) == len(beats) else None
        if accents is not None:
            mine = accents[near[on]].mean()
            pattern["hits"] = float(np.clip((mine - accents.mean()) / (accents.std() + 1e-6), 0, 2))
    changes = [s.start for s in song.sections[1:] if pins[0] < s.start < pins[-1]]
    if changes:
        reach = max(0.35, period)
        pattern["changes"] = sum(any(abs(p - c) <= reach for p in pins) for c in changes) / len(changes)
    return pattern


def suggest(song: SongMap, pins: list[float], stretches: list[dict], until: float | None = None) -> list[float]:
    """Pins for the rest of the song (after your last pin, up to `until` or the end), in your way of pinning."""
    pattern = learn(song, pins, stretches)
    if pattern is None:
        return []
    return walk(song, pattern, stretches, until)


def walk(song: SongMap, pattern: dict, stretches: list[dict], until: float | None = None) -> list[float]:
    """Pins from pattern["until"] on (0: the whole song), choosing the beats that fit the pattern best."""
    end = min(until or song.duration, song.duration)
    beats = song.beat_times()
    period = song.period
    accents = np.asarray(song.accents, float) if len(song.accents) == len(beats) else np.zeros(len(beats))
    changes = [s.start for s in song.sections[1:] if s.start > pattern["until"]] if pattern["changes"] >= 0.5 else []
    result: list[float] = []
    t = pattern["until"]
    while True:
        gap = pattern["gaps"].get(_mood_at(stretches, t), pattern["gap"])  # in beats
        if not pattern["on_beat"]:
            nxt = t + gap * period
        else:
            lo, hi = t + max(1.0, 0.6 * gap) * period - 1e-3, t + 1.5 * gap * period
            ks = np.flatnonzero((beats > lo) & (beats <= hi))
            if not len(ks):
                break
            change = next((c for c in changes if lo < c <= hi), None)
            if change is not None:  # you pin where the music changes: so does the app
                k = int(ks[np.abs(beats[ks] - change).argmin()])
            else:
                fit = -np.abs(np.log((beats[ks] - t) / (gap * period)))
                bar = np.asarray(pattern["bar"])[(ks - song.downbeat) % BEATS_PER_BAR]
                score = 2.0 * fit + 0.6 * np.log(bar) + 0.5 * pattern["hits"] * accents[ks]
                k = int(ks[score.argmax()])
            nxt = float(beats[k]) + pattern["offset"]
        if nxt >= end - 0.3:
            break
        result.append(round(nxt, 3))
        t = nxt
    return result


# Your cut style across songs: what was learned from each song you pinned (at least STYLE_FROM pins) is kept in a
# small file, so a new song can be pinned your way before you've placed a single pin on it.
STYLE_FROM = 8
KEEP_SONGS = 30


def remember(file, song_key: str, song: SongMap, pins: list[float], stretches: list[dict]) -> bool:
    """Keep what this song's pins say about your style (False: too few pins to learn from)."""
    import json
    from pathlib import Path

    if len([p for p in pins if 0 <= p <= song.duration]) < STYLE_FROM:
        return False
    pattern = learn(song, pins, stretches)
    if pattern is None:
        return False
    file = Path(file)
    try:
        saved = json.loads(file.read_text())
    except (OSError, ValueError):
        saved = {}
    saved.pop(song_key, None)
    saved[song_key] = {k: v for k, v in pattern.items() if k != "until"} | {"pins": len(pins)}
    saved = dict(list(saved.items())[-KEEP_SONGS:])
    file.parent.mkdir(parents=True, exist_ok=True)
    file.write_text(json.dumps(saved))
    return True


def my_style(file) -> dict | None:
    """Your way of pinning across every song you pinned (None: none yet). Songs with more pins count more."""
    import json
    from pathlib import Path

    try:
        saved = list(json.loads(Path(file).read_text()).values())
    except (OSError, ValueError):
        return None
    if not saved:
        return None
    weight = np.array([max(1, s.get("pins", 1)) for s in saved], float)

    def wmedian(values, w):
        order = np.argsort(values)
        values, w = np.asarray(values, float)[order], np.asarray(w, float)[order]
        return float(values[np.searchsorted(np.cumsum(w), w.sum() / 2)])

    moods = {m for s in saved for m in s.get("gaps", {})}
    gaps = {}
    for m in moods:
        have = [(s["gaps"][m], wt) for s, wt in zip(saved, weight) if m in s.get("gaps", {})]
        gaps[m] = wmedian([g for g, _ in have], [wt for _, wt in have])
    beat = [s for s in saved if s.get("on_beat")]
    bw = np.array([max(1, s.get("pins", 1)) for s in beat], float)
    bar = (np.average([s["bar"] for s in beat], axis=0, weights=bw) if beat else np.ones(BEATS_PER_BAR))
    return {"gaps": gaps, "gap": wmedian([s["gap"] for s in saved], weight), "until": 0.0,
            "offset": wmedian([s["offset"] for s in beat], bw) if beat else 0.0,
            "bar": [float(b) for b in bar / max(bar.max(), 1e-6)],
            "hits": float(np.average([s["hits"] for s in saved], weights=weight)),
            "changes": float(np.average([s["changes"] for s in saved], weights=weight)),
            "on_beat": weight[[bool(s.get("on_beat")) for s in saved]].sum() >= weight.sum() / 2,
            "songs": len(saved)}


def suggest_my_way(song: SongMap, style: dict, stretches: list[dict], pins: list[float] | None = None,
                   until: float | None = None) -> list[float]:
    """Pins for a song in your style learned from other songs: after the pins you already placed (if any), else
    from the start. A song without a steady beat is pinned by spacing only."""
    pattern = dict(style)
    pattern["until"] = max(pins) if pins else 0.0
    if not song.clear_beat or len(song.beat_times()) < 8:
        pattern["on_beat"] = False
    return walk(song, pattern, stretches, until)
