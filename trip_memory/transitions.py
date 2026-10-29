"""The transitions between shots, and which one suits each cut.

Every cut gets a suggestion with a short reason ("Next story: Fish & aquarium"), worked out from the music
(lively or calm, where it lifts or drops), the trip (a new story, a new day) and the shots (a fast camera move).
You can change any of them in the Storyboard.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

HARD_CUT = "None (hard cut)"


@dataclass(frozen=True)
class Style:
    xfade: str | None  # FFmpeg's name for it; None = a hard cut
    quick: float | None = None  # a quick effect lasts this long (seconds, whole frames); None = your transition length


# Friendly name -> how FFmpeg draws it.
STYLES: dict[str, Style] = {
    HARD_CUT: Style(None),
    "Fade": Style("fade"),
    "Dissolve": Style("dissolve"),
    "Fade through black": Style("fadeblack"),
    "Dip to white": Style("fadewhite"),
    "Smooth wipe": Style("smoothleft"),
    "Circle": Style("circleopen"),
    "Radial": Style("radial"),
    "Slide": Style("slideleft"),
    "Wipe": Style("wipeleft"),
    "Zoom": Style("zoomin"),
    "Flash": Style("fadewhite", quick=0.3),
    "Whip left": Style("slideleft", quick=8 / 30),
    "Whip right": Style("slideright", quick=8 / 30),
    "Whip up": Style("slideup", quick=8 / 30),
    "Zoom punch": Style("zoomin", quick=10 / 30),
    "Blur": Style("hblur", quick=0.4),
    "Pixelate": Style("pixelize", quick=0.4),
    "Zoom through": Style(None),  # a hard cut: the first shot rushes in, the next one lands zoomed and settles
    "Zoom out reveal": Style(None),  # a hard cut: the next shot starts close up and pulls back
    "Mixed": Style("mixed"),
}
# Transitions drawn by zooming the shots themselves (around a clean cut), and how each side moves.
ZOOM_OUT_OF = {"Zoom through": "zoom"}  # the shot before the cut
ZOOM_INTO = {"Zoom through": "zoom", "Zoom out reveal": "reveal"}  # the shot after it
MIXED = ["fade", "slideleft", "circleopen", "dissolve", "wipeleft", "fadeblack", "zoomin", "slideup"]


def seconds_of(name: str | None, default: float) -> float:
    """How long a transition overlaps the two shots (a hard cut: one frame)."""
    style = STYLES.get(name or "", Style("fade"))
    if style.xfade is None:
        return 1 / 30
    return style.quick if style.quick is not None else default


# Families to rotate through, so the same effect doesn't come twice in a row.
STORY_LIVELY = ["Zoom through", "Flash", "Whip left", "Zoom punch", "Whip up"]
STORY_CALM = ["Zoom out reveal", "Smooth wipe", "Circle", "Dissolve", "Radial"]
WHIPS = ["Whip left", "Whip right"]
CALM_BLENDS = ["Dissolve", "Fade"]
LIVELY, CALM = 0.62, 0.38


@dataclass
class Cut:
    before: dict  # the shot before the cut (a catalogue row)
    after: dict
    energy_before: float  # the music's liveliness, 0..1, on each side
    energy_after: float
    room: float  # the longest transition that fits (seconds)
    hit: float = 0.0  # how hard the music hits right at the cut, 0..1 (0 if unknown)


def recommend(cuts: list[Cut], soft_seconds: float) -> list[tuple[str, str]]:
    """A transition and the reason for it, for every cut."""
    result: list[tuple[str, str]] = []
    turn = {"story_lively": 0, "story_calm": 0, "whip": 0, "blend": 0}
    since_effect = 99

    def take(family: list[str], key: str) -> str:
        name = family[turn[key] % len(family)]
        turn[key] += 1
        return name

    for cut in cuts:
        a, b = cut.before, cut.after
        energy = (cut.energy_before + cut.energy_after) / 2
        name, why = HARD_CUT, "On the beat"
        new_day = _day(a) != _day(b)
        new_story = a.get("story") != b.get("story")
        lift = cut.energy_after - cut.energy_before
        if new_day:
            name, why = "Fade through black", f"A new day: {_day_name(b)}"
        elif new_story:
            family, key = (STORY_LIVELY, "story_lively") if energy >= 0.5 else (STORY_CALM, "story_calm")
            name = take(family, key)
            why = f"Next story: {b['story_name']}" if b.get("story_name") else "Next story"
        elif lift > 0.2:
            name, why = "Zoom through", "The music lifts here: zoom into it"
        elif lift < -0.2:
            name, why = "Blur", "The music calms down"
        elif energy >= LIVELY:
            tags = set(a.get("tags_final") or [])
            moving = bool(tags & {"Whip pan", "Fast camera", "Pan left", "Pan right"}) or (a.get("motion_camera") or 0) >= 1.2
            if moving and since_effect >= 3:
                # Carry the camera's swing on into the next shot: a pan to the right pushes the picture left.
                name = "Whip left" if "Pan right" in tags else "Whip right" if "Pan left" in tags else take(WHIPS, "whip")
                why = "A fast camera move: a whip pan keeps it flowing"
            elif "Zoom in" in tags and since_effect >= 3:
                name, why = "Zoom through", "The shot zooms in: carry the zoom through the cut"
            elif cut.hit >= 0.8 and since_effect >= 4:
                name, why = "Zoom through", "A big hit in the music"
            else:
                name, why = HARD_CUT, "Quick cut on the beat"
        elif energy < CALM:
            if "Zoom in" in (a.get("tags_final") or []):
                name, why = "Zoom", "The shot zooms in: carry on into the next one"
            else:
                name, why = take(CALM_BLENDS, "blend"), "Calm music: a soft blend"
        # Make sure it fits between the two cuts; otherwise a quicker version, or a clean cut.
        if seconds_of(name, soft_seconds) > cut.room and name in (*CALM_BLENDS, "Zoom"):
            name, why = HARD_CUT, "Shots too short for a blend: a clean cut"
        if seconds_of(name, soft_seconds) > cut.room:
            quick = "Flash" if name in ("Fade through black", "Dip to white") else "Blur" if STYLES[name].quick is None else name
            name = quick if seconds_of(quick, soft_seconds) <= cut.room else HARD_CUT
            if name == HARD_CUT:
                why = "Shots too short for an effect: a clean cut"
        since_effect = 0 if name != HARD_CUT else since_effect + 1
        result.append((name, why))
    return result


def _day(row: dict) -> str:
    try:
        return datetime.fromisoformat(row["taken_at"]).astimezone().strftime("%Y-%m-%d")
    except (KeyError, TypeError, ValueError):
        return ""


def _day_name(row: dict) -> str:
    try:
        return datetime.fromisoformat(row["taken_at"]).astimezone().strftime("%a %d %b")
    except (KeyError, TypeError, ValueError):
        return ""
