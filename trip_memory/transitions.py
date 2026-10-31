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
    # (more, 0.91)
    "Fade to grey": Style("fadegrays"),
    "Slide right": Style("slideright"),
    "Slide up": Style("slideup"),
    "Slide down": Style("slidedown"),
    "Squeeze": Style("squeezeh"),
    "Circle close": Style("circleclose"),
    "Diagonal": Style("diagtl"),
    "Doors open": Style("horzopen"),
    "Blinds": Style("hlslice"),
    "Wipe up": Style("wipeup"),
    "Box": Style("rectcrop"),
    "Spin": Style("custom:spin"),
    "Bounce in": Style("custom:bounce"),
    "Drop in": Style("custom:drop"),
}
# How the picker groups them (the order shown).
GROUPS: dict[str, list[str]] = {
    "Soft": ["Fade", "Dissolve", "Fade through black", "Dip to white", "Fade to grey", "Blur"],
    "Move": ["Slide", "Slide right", "Slide up", "Slide down", "Smooth wipe", "Whip left", "Whip right", "Whip up",
             "Squeeze"],
    "Shape": ["Circle", "Circle close", "Radial", "Diagonal", "Doors open", "Blinds", "Wipe", "Wipe up", "Box",
              "Pixelate"],
    "Fun": ["Zoom", "Zoom punch", "Zoom through", "Zoom out reveal", "Flash", "Spin", "Bounce in", "Drop in"],
}


def _plane(fn: str, x: str, y: str) -> str:
    return (f"if(eq(PLANE,0),{fn}0({x},{y}),if(eq(PLANE,1),{fn}1({x},{y}),"
            f"if(eq(PLANE,2),{fn}2({x},{y}),{fn}3({x},{y}))))")


def _back(q: str) -> str:  # eases in with a little overshoot, then settles (0 -> 1)
    return f"(1+2.70158*pow({q}-1,3)+1.70158*pow({q}-1,2))"


def custom_expr(kind: str) -> str:
    """FFmpeg xfade's own-drawn transitions (P runs from 1 to 0; no stored variables: the picture is drawn in
    parallel slices)."""
    q = "(1-P)"
    if kind == "spin":  # the next shot spins in from the middle, growing
        s, th = f"max({q},0.001)", "(P*6.2832)"
        u = f"(((X-W/2)*cos({th})-(Y-H/2)*sin({th}))/{s}+W/2)"
        v = f"(((X-W/2)*sin({th})+(Y-H/2)*cos({th}))/{s}+H/2)"
        return f"if(between({u},0,W-1)*between({v},0,H-1),{_plane('b', u, v)},{_plane('a', 'X', 'Y')})"
    if kind == "bounce":  # the next shot pushes in from the right, overshoots a little and settles
        dx = f"(W*(1-{_back(q)}))"
        return (f"if(gte(X,{dx}),{_plane('b', f'min(max(X-{dx},0),W-1)', 'Y')},"
                f"{_plane('a', f'min(X-{dx}+W,W-1)', 'Y')})")
    if kind == "drop":  # the next shot drops in from the top and lands with a little bounce
        y0 = f"(-H*(1-{_back(q)}))"
        return f"if(lt(Y,{y0}+H),{_plane('b', 'X', f'min(max(Y-{y0},0),H-1)')},{_plane('a', 'X', 'Y')})"
    raise ValueError(kind)


def xfade_args(style: str) -> str:
    """The xfade filter's transition options for a style (a custom one carries its expression)."""
    if style.startswith("custom:"):
        return f"transition=custom:expr='{custom_expr(style.split(':', 1)[1])}'"
    return f"transition={style}"


# How the editor's instant player shows each (CSS, roughly like the finished video).
LOOKS: dict[str, str] = {
    "Fade": "fade", "Dissolve": "fade", "Mixed": "fade", "Fade through black": "black", "Dip to white": "white",
    "Fade to grey": "grey", "Blur": "blur", "Pixelate": "blur", "Slide": "slide-l", "Whip left": "slide-l",
    "Slide right": "slide-r", "Whip right": "slide-r", "Slide up": "slide-u", "Whip up": "slide-u",
    "Slide down": "slide-d", "Smooth wipe": "wipe-l", "Wipe": "wipe-l", "Wipe up": "wipe-u", "Squeeze": "squeeze",
    "Circle": "circle", "Circle close": "circle", "Radial": "circle", "Diagonal": "diag", "Doors open": "doors",
    "Blinds": "fade", "Box": "box", "Zoom": "zoom", "Zoom punch": "zoom", "Flash": "white", "Spin": "spin",
    "Bounce in": "bounce", "Drop in": "drop", "Zoom through": "zoom", "Zoom out reveal": "reveal",
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


FUN = ["Zoom punch", "Spin", "Whip left", "Bounce in", "Flash", "Drop in", "Whip up", "Squeeze", "Zoom through"]
FANCY_CALM = ["Circle", "Smooth wipe", "Dissolve", "Doors open", "Radial", "Box"]


def exciting(cuts: list[Cut], soft_seconds: float) -> list[tuple[str, str]]:
    """Like recommend, with more going on: on lively music every second or third clean cut becomes a fun effect
    (spin, bounce, zoom punch...), on calm music the soft blends take turns with shapes. Each still fits."""
    out = []
    turn = {"fun": 0, "calm": 0}
    since = 99
    for cut, (name, why) in zip(cuts, recommend(cuts, soft_seconds)):
        energy = (cut.energy_before + cut.energy_after) / 2
        if name == HARD_CUT and energy >= 0.5 and since >= (2 if cut.hit >= 0.5 else 3):
            for _ in range(len(FUN)):
                pick = FUN[turn["fun"] % len(FUN)]
                turn["fun"] += 1
                if seconds_of(pick, min(soft_seconds, 0.5)) <= max(cut.room, 1 / 30):
                    name, why = pick, "Lively music: something fun" + (" on the big hit" if cut.hit >= 0.5 else "")
                    break
        elif name in CALM_BLENDS and turn["calm"] % 2:
            pick = FANCY_CALM[(turn["calm"] // 2) % len(FANCY_CALM)]
            if seconds_of(pick, soft_seconds) <= cut.room:
                name, why = pick, "Calm music: a soft shape"
        if name in CALM_BLENDS or name in FANCY_CALM:
            turn["calm"] += 1
        since = 0 if name != HARD_CUT else since + 1
        out.append((name, why))
    return out
