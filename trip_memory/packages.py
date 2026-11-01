"""Style packages: one choice sets the whole video's style. Each package picks the transitions (and how long they
are), the colour mood, the titles' and cards' look, automatic place titles (with their own animation), and, for
the livelier ones, effects on the music: flashes on the big hits, a colour shift into the cuts, a short glitch.

You can change anything a package set afterwards (a transition, the look, a title); choosing a package again
sets it all back to the package's way."""

from __future__ import annotations

from .transitions import HARD_CUT, Cut, STYLES, seconds_of, recommend

PACKAGES: dict[str, dict] = {
    "calm": {
        "label": "Calm & cinematic", "emoji": "🎬",
        "what": "Long soft dissolves and shapes, film colours, cinema bars on the opening, place names that slowly "
                "appear letter by letter.",
        "look": "film", "seconds": 1.0, "hard": 0.25,
        "calm": ["Dissolve", "Fade", "Smooth wipe", "Fade through black", "Circle", "Radial"],
        "lively": ["Dissolve", "Smooth wipe", "Zoom out reveal"],
        "card": "cinematic", "end_card": "cinematic", "style": "cinematic", "story_style": "cinematic",
        "place_titles": True, "anim": "reveal", "music_style": "cinematic",
        "fx": {},
    },
    "neon": {
        "label": "Neon energy", "emoji": "⚡",
        "what": "Bright colours, flashes on the big hits, a colour shift into the cuts, short glitches, spins and "
                "zooms; place names flicker on like neon.",
        "look": "vivid", "seconds": 0.5, "hard": 0.4,
        "calm": ["Dip to white", "Zoom", "Pixelate", "Circle"],
        "lively": ["Zoom punch", "Spin", "Flash", "Bounce in", "Whip left", "Pixelate", "Drop in"],
        "card": "bold", "end_card": "bold", "style": "bold", "story_style": "bold",
        "place_titles": True, "anim": "flicker", "music_style": "action",
        "fx": {"flash": True, "hue": True, "glitch": True},
    },
    "action": {
        "label": "Fast action", "emoji": "🏄",
        "what": "Quick hard cuts on the beat, whips and zoom punches, punchy colours, a flash on the biggest hits; "
                "place names flash in.",
        "look": "vivid", "seconds": 0.3, "hard": 0.6,
        "calm": ["Whip left", "Whip right", "Zoom", "Slide"],
        "lively": ["Zoom through", "Whip left", "Whip up", "Zoom punch", "Flash", "Whip right"],
        "card": "bold", "end_card": "bold", "style": "bold", "story_style": "bold",
        "place_titles": True, "anim": "flicker-rl", "music_style": "action",
        "fx": {"flash": True},
    },
    "journal": {
        "label": "Travel journal", "emoji": "📔",
        "what": "Warm summer colours, slides and soft wipes like turning pages, handwritten titles; each new place "
                "is typed on screen with the day.",
        "look": "summer", "seconds": 0.7, "hard": 0.2,
        "calm": ["Slide", "Dissolve", "Smooth wipe", "Slide up", "Fade"],
        "lively": ["Slide", "Whip left", "Slide right", "Squeeze"],
        "card": "journal", "end_card": "journal", "style": "handwritten", "story_style": "typewriter",
        "place_titles": True, "anim": "type", "music_style": "discovery",
        "fx": {},
    },
}
ANIMS = {"fade": "Fade in", "reveal": "Slow reveal (left to right)", "type": "Typed letter by letter",
         "flicker": "Neon flicker (left to right)", "flicker-rl": "Flicker (right to left)"}


def transitions_for(package: str, cuts: list[Cut], soft_seconds: float | None = None) -> list[tuple[str, str]]:
    """A transition for every cut in the package's way: its soft set where the music is calm, its lively set where
    it's lively (and on the big hits), clean cuts on the beat as often as the package likes. Each one fits."""
    p = PACKAGES[package]
    seconds = soft_seconds or p["seconds"]
    out, turn, since = [], {"calm": 0, "lively": 0}, 99
    base = recommend(cuts, seconds)
    for k, (cut, (name, why)) in enumerate(zip(cuts, base)):
        energy = (cut.energy_before + cut.energy_after) / 2
        lively = energy >= 0.5 or cut.hit >= 0.5
        # Clean cuts on the beat: the share the package likes (more for action, few for calm).
        if name == HARD_CUT and since < 99 and (k * 0.618) % 1 < p["hard"] and not cut.hit >= 0.7:
            since += 1
            out.append((HARD_CUT, f"{p['label']}: a clean cut on the beat"))
            continue
        pool = p["lively"] if lively else p["calm"]
        key = "lively" if lively else "calm"
        pick = None
        for _ in range(len(pool)):
            option = pool[turn[key] % len(pool)]
            turn[key] += 1
            if seconds_of(option, seconds) <= max(cut.room, 1 / 30) or STYLES.get(option, STYLES[HARD_CUT]).xfade is None:
                pick = option
                break
        if pick is None:
            out.append((HARD_CUT, f"{p['label']}: no room for a transition here"))
            since += 1
            continue
        out.append((pick, f"{p['label']}: " + ("lively music" + (", on the big hit" if cut.hit >= 0.5 else "")
                                                if lively else "calm music")))
        since = 0
    return out


def place_changes(board: list[dict], least_gap: int = 3) -> list[int]:
    """Where a new place starts on the board (indexes): the place differs from the last one named, it lasts at
    least two shots, and it's at least least_gap shots after the last place title. Not on the first shot (the
    opening title is there)."""
    out, last, last_at = [], None, -99
    for k, r in enumerate(board):
        place = r.get("place") or r.get("country") or ""
        if not place or place == last:
            continue
        nxt = board[k + 1] if k + 1 < len(board) else None
        lasting = nxt is not None and (nxt.get("place") or nxt.get("country") or "") == place
        if k > 0 and lasting and k - last_at >= least_gap:
            out.append(k)
            last_at = k
        last = place
    return out


def flash_times(beats, accents, sections, length: float, most: int = 40) -> list[float]:
    """When the lively packages flash: the biggest hits of the music's lively parts (not in the first and last
    two seconds), at most `most`, never two within a second."""
    import numpy as np

    beats, accents = np.asarray(beats, float), np.asarray(accents, float)
    if not len(beats) or len(accents) != len(beats):
        return []
    lively = [(s.start, s.end) for s in sections if s.level == "Lively" or s.energy >= 0.65]
    chosen: list[float] = []
    for k in np.argsort(-accents):
        t = float(beats[k])
        if accents[k] < 0.6 or len(chosen) >= most:
            break
        if not (2.0 < t < length - 2.0) or not any(a <= t < b for a, b in lively):
            continue
        if all(abs(t - c) >= 1.0 for c in chosen):
            chosen.append(round(t, 3))
    return sorted(chosen)
