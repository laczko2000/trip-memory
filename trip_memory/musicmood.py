"""Reading the music: the song's calm, medium and lively stretches (lined up with your pins), what fits in each,
and the style of the whole song. You can change any of it; Auto-edit and the AI follow it."""

from __future__ import annotations

from .songmap import Section, SongMap

MOODS = {"Calm": 0.25, "Medium": 0.5, "Lively": 0.8}  # the energy Auto-edit reads for a stretch you set yourself
COLOURS = {"Calm": "#93c5fd", "Medium": "#60a5fa", "Lively": "#f97316"}
FITS = {
    "Calm": "wide views, photos, slow motion",
    "Medium": "people, walking, everyday moments",
    "Lively": "action clips, quick cuts",
}
STYLES = {
    "cinematic": {"label": "Cinematic", "pace": "calm", "pin_gap": 3.0,
                  "what": "long, flowing shots and soft dissolves; cuts only on your strongest pins"},
    "action": {"label": "Action", "pace": "lively", "pin_gap": 0.0,
               "what": "quick cuts on every pin, hard cuts and camera moves"},
    "discovery": {"label": "Discovery", "pace": "medium", "pin_gap": 0.0,
                  "what": "a journey: calm openings that build to lively moments"},
    "calm": {"label": "Calm memories", "pace": "calm", "pin_gap": 2.0,
             "what": "gentle, unhurried shots, photos and slow motion"},
}
SNAP = 2  # a change in the music this many bars from one of your pins moves onto the pin
SHORTEST = 2.0  # seconds: shorter stretches join their neighbour


def bands(song: SongMap, pins: list[float] | None) -> list[dict]:
    """The song's stretches as the app hears them: [{"start", "end", "mood", "energy"}], each change moved onto
    your nearest pin when there is one close by."""
    if not song.sections:
        return [{"start": 0.0, "end": song.duration, "mood": "Medium", "energy": 0.5}]
    reach = SNAP * 4 * song.period
    edges = [0.0]
    for s in song.sections[1:]:
        t = s.start
        near = [p for p in pins or [] if abs(p - t) <= reach]
        if near:
            t = min(near, key=lambda p: abs(p - t))
        if t - edges[-1] >= SHORTEST:
            edges.append(float(t))
    edges.append(song.duration)
    result: list[dict] = []
    for a, b in zip(edges, edges[1:]):
        mid = (a + b) / 2
        section = next((s for s in song.sections if s.start <= mid < s.end), song.sections[-1])
        if result and result[-1]["mood"] == section.level:
            result[-1]["end"] = b  # the same mood again: one stretch
            continue
        result.append({"start": a, "end": b, "mood": section.level, "energy": section.energy})
    return result


def suggest_style(stretches: list[dict]) -> str:
    """The style the music suits, from how much of it is calm or lively."""
    total = sum(b["end"] - b["start"] for b in stretches) or 1.0
    share = {m: sum(b["end"] - b["start"] for b in stretches if b["mood"] == m) / total for m in MOODS}
    if share["Lively"] >= 0.55:
        return "action"
    if share["Calm"] >= 0.55:
        return "cinematic" if share["Lively"] < 0.1 else "calm"
    return "discovery"


def apply(song: SongMap, stretches: list[dict]) -> None:
    """Auto-edit hears the song through these stretches (yours, if you changed any)."""
    song.sections = [Section(start=b["start"], end=b["end"], energy=b.get("energy", MOODS[b["mood"]]), level=b["mood"])
                     for b in stretches]


def thin(pins: list[float], gap: float) -> list[float]:
    """For a calmer style: keep a pin only when it comes at least `gap` seconds after the last one kept."""
    kept: list[float] = []
    for p in sorted(pins):
        if not kept or p - kept[-1] >= gap:
            kept.append(p)
    return kept


def reading(song: SongMap, song_path: str, folder) -> dict:
    """The music as Auto-edit will use it: your stretches if you changed any, else the app's; and the style
    (yours, else the one suggested)."""
    from . import taps

    pins = taps.final_taps(song_path, folder) or []
    heard = bands(song, pins)
    mine = taps.extra(song_path, folder, "moods")
    stretches = mine or heard
    style = taps.extra(song_path, folder, "style")
    return {"stretches": stretches, "mine": bool(mine), "suggested": suggest_style(stretches),
            "style": style if style in STYLES else suggest_style(stretches), "style_mine": style in STYLES}


def words(stretches: list[dict]) -> str:
    """The song's stretches in words, for the AI: "0:00-0:40 calm, 0:40-1:30 lively"."""
    def mmss(t: float) -> str:
        return f"{int(t // 60)}:{int(t % 60):02d}"

    return ", ".join(f"{mmss(b['start'])}-{mmss(b['end'])} {b['mood'].lower()}" for b in stretches)
