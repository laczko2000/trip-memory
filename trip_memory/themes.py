"""Themes: every shot gets one main activity (what is it: kitesurfing, a boat tour, food...), plus who is in it, where
and when. The app suggests the activity from the picture as a whole (a ship far behind a kitesurfer doesn't make it a
boat shot); you confirm or move shots, and the app learns from the shots you confirmed: the rest are compared with
them, picture with picture, which is far more reliable than words.

Free and on this Mac: it uses the CLIP picture fingerprints the app already made for every shot.
"""

from __future__ import annotations

from datetime import datetime

import numpy as np

OTHER = "Other"

# Activity -> (emoji, what the pictures show, for CLIP). The order is the order shown.
DEFAULTS: dict[str, tuple[str, list[str]]] = {
    "Water sports": ("🏄", ["kitesurfing on the sea with a kite", "windsurfing or surfing on a wave",
                           "a person riding a jet ski or wakeboard"]),
    "Boat": ("⛵", ["people on the deck of a boat", "a boat trip on the sea", "a sailing boat"]),
    "Beach & sea": ("🏖️", ["a sandy beach by the sea", "people relaxing on a beach", "the sea and the coast"]),
    "Swimming & pool": ("🏊", ["people swimming", "a swimming pool", "snorkelling underwater"]),
    "Food & drink": ("🍝", ["a plate of food on a table", "drinks and cocktails", "people eating at a restaurant"]),
    "Sightseeing": ("🏰", ["a famous building or monument", "a castle, church or old ruins", "tourists sightseeing"]),
    "City streets": ("🏙️", ["a busy city street", "narrow streets of an old town", "a town square"]),
    "Nature & views": ("🏞️", ["mountains and a wide landscape", "a forest or green countryside", "a panoramic view"]),
    "Party & night": ("🌙", ["a party with people dancing", "a bar or a city at night", "a concert at night"]),
    "Travel": ("✈️", ["an airport or an airplane", "a view from a car on the road", "a train, bus or ferry"]),
    "Relaxing": ("😌", ["people relaxing at a hotel or holiday home", "a hotel room or balcony", "a sunbed by the pool"]),
    "People & portraits": ("🧑‍🤝‍🧑", ["a close-up portrait of a person", "a group of friends posing for a photo",
                                     "a selfie"]),
}
SURE = 0.55  # below this share of the vote, a suggestion is marked "?" (worth a look)


def default_activities() -> list[dict]:
    return [{"name": n, "emoji": e, "describe": d, "vector": None} for n, (e, d) in DEFAULTS.items()] + \
        [{"name": OTHER, "emoji": "❔", "describe": [], "vector": None}]


def new_activity(name: str) -> dict:
    """An activity you added yourself: the app looks for pictures of it, then learns from the shots you move in."""
    return {"name": name, "emoji": "⭐", "describe": [name.lower()], "vector": None}


def day_part(taken_at: str) -> str:
    hour = datetime.fromisoformat(taken_at).astimezone().hour
    return "morning" if 5 <= hour < 12 else "afternoon" if hour < 17 else "evening" if hour < 21 else "night"


def suggest(vectors: np.ndarray, activities: list[dict], confirmed: list[tuple[int, str]],
            earlier: list[tuple[np.ndarray, str]] = ()) -> list[tuple[str, float]]:
    """For each shot (one CLIP vector per row): the most likely activity and how sure (0..1).

    confirmed: (row, activity) for the shots you confirmed. Those keep their activity, and the others are compared
    with them: a shot that looks like your confirmed boat shots is a boat shot, whatever the words say.
    earlier: (vector, activity) of shots you confirmed in earlier trips, which count as examples too."""
    names = [a["name"] for a in activities]
    n = len(vectors)
    if not n or not names:
        return []
    # 1. The words: how well each picture fits each activity's description (CLIP), "Other" when nothing fits.
    words = np.full((n, len(names)), -np.inf)
    for j, a in enumerate(activities):
        if a.get("vector") is not None:
            words[:, j] = 100 * (vectors @ np.asarray(a["vector"], np.float32))
    known = np.isfinite(words[0])
    if OTHER in names and known.any():
        words[:, names.index(OTHER)] = np.percentile(words[:, known], 40, axis=1) + OTHER_BAR
    p = _softmax(words) if np.isfinite(words).any() else np.full((n, len(names)), 1 / len(names))
    # 2. Your examples: the activity whose confirmed shots this picture looks most like. The closer it is to any of
    # them, the more that counts.
    examples: dict[int, list[np.ndarray]] = {}
    for row, name in confirmed:
        if name in names and 0 <= row < n:
            examples.setdefault(names.index(name), []).append(vectors[row])
    for vector, name in earlier:
        if name in names and len(vector) == vectors.shape[1]:
            examples.setdefault(names.index(name), []).append(np.asarray(vector, np.float32))
    if examples:
        near = np.full((n, len(names)), -np.inf)
        for j, shots in examples.items():
            sims = np.sort(vectors @ np.array(shots).T, axis=1)[:, ::-1][:, :3]
            near[:, j] = sims.mean(axis=1)
        closest = near.max(axis=1, keepdims=True)
        trust = np.clip((closest - LOOKS_LIKE) / (SAME - LOOKS_LIKE), 0, 1)
        p = (1 - trust) * p + trust * _softmax(30 * near)
    out = [(names[int(j)], float(p[i, j])) for i, j in enumerate(p.argmax(axis=1))]
    for row, name in confirmed:
        if 0 <= row < n:
            out[row] = (name, 1.0)
    return out


OTHER_BAR = 1.0  # how far above an average fit a picture must be to get an activity rather than "Other"
LOOKS_LIKE, SAME = 0.72, 0.9  # picture-to-picture likeness: from "a bit like" your example to "the same kind of shot"


def _softmax(logits: np.ndarray) -> np.ndarray:
    logits = np.where(np.isfinite(logits), logits, -1e9)
    logits = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(logits)
    return e / e.sum(axis=1, keepdims=True)


def by_tags(tags: list[str], scene: str | None) -> str:
    """Without picture fingerprints (scenes not analysed yet): a first guess from the shot's tags."""
    known = set(tags or []) | {scene or ""}
    rules = [
        ("Water sports", {"Surfing", "Kayak & paddle board", "Jet ski & speedboat", "Snorkelling & diving"}),
        ("Boat", {"Boat trip", "Sea & boats"}),
        ("Food & drink", {"Eating & drinking", "Restaurant & café", "Food & drinks", "Cooking & barbecue"}),
        ("Swimming & pool", {"Swimming", "Pool"}),
        ("Beach & sea", {"Beach", "Sea", "Relaxing & sunbathing"}),
        ("Sightseeing", {"Landmark", "Sightseeing", "Museum"}),
        ("City streets", {"City street", "Old town", "City"}),
        ("Nature & views", {"Mountains", "Forest & countryside", "Lake & river", "Nature", "Hiking"}),
        ("Party & night", {"Dancing & party", "Night", "Party & nightlife", "Concert & show"}),
        ("Travel", {"Airport & transport", "Road", "Driving", "Travel day", "Car & road trip"}),
        ("Relaxing", {"Hotel & home", "Hotel"}),
        ("People & portraits", {"Posing", "Group photo", "Selfie & portrait"}),
    ]
    for name, hits in rules:
        if known & hits:
            return name
    return OTHER


# ---------- the storyline: blocks of themes in your order ----------

TIME, THEME, DAY, STORY = "time", "theme", "day", "story"
ORDERS = {TIME: "As it happened", THEME: "By theme", STORY: "My story (AI) ✨"}  # (day by day: a switch of TIME)
PICKS = {"all": "all shots", "early": "first half", "late": "second half", "action": "most exciting half",
         "moment": "a moment… 🔍"}
PACES = {"calm": 0.2, "medium": 0.55, "lively": 0.9}  # how lively the shots of a story part should be, 0..1


def block_key(activity: str, person: str | None) -> str:
    return f"{activity}|{person or ''}"


def block_of(row: dict, split: set[str]) -> str:
    """The block a shot belongs to: its activity, and for an activity you split by person, the first person in it."""
    activity = row.get("theme") or OTHER
    if activity in split:
        named = [p for p in row.get("people") or [] if not p.startswith("Person ")]
        return block_key(activity, named[0] if named else "everyone else")
    return block_key(activity, None)


def blocks_for(rows: list[dict], saved: list[dict], split: set[str]) -> list[dict]:
    """The storyline's blocks: your saved order first (with whether each is on), then any new ones in the order
    their first shot was taken. Each: {"key", "activity", "person", "on", "count"}."""
    counts: dict[str, int] = {}
    first: dict[str, str] = {}
    for r in rows:
        k = block_of(r, split)
        counts[k] = counts.get(k, 0) + 1
        first.setdefault(k, r["taken_at"])
    out = []
    seen = set()
    for b in saved:
        if b["key"] in counts and b["key"] not in seen:
            out.append(b | {"count": counts[b["key"]]})
            seen.add(b["key"])
    for k in sorted(counts, key=lambda k: first[k]):
        if k not in seen:
            activity, _, person = k.partition("|")
            out.append({"key": k, "activity": activity, "person": person or None, "on": True, "count": counts[k]})
    return out


def chapters_for(storyline: dict, rows: list[dict]) -> tuple[list[dict], list[int | None]]:
    """Your storyline as chapters for Auto-edit, and each shot's chapter (None: left out).

    As it happened: one chapter in time order. By theme: one chapter per block, in your order. Day by day: for each
    day, its blocks in your order. A finale (the most exciting shots) closes the video if it is on."""
    order = storyline.get("order", TIME)
    split = set(storyline.get("split") or [])
    blocks = blocks_for(rows, storyline.get("blocks") or [], split)
    on = [b["key"] for b in blocks if b["on"]]
    chapters: list[dict] = []
    where: list[int | None] = []
    story = storyline.get("story") or {}
    if order == STORY and not story.get("parts"):
        order = TIME  # no story made yet: as it happened
    if order == STORY:  # your parts in your order, the last one ends the video (no finale after it)
        return _story_chapters(story, rows, split)
    if order == TIME and storyline.get("by_day"):
        order = DAY  # as it happened, each day's themes kept together
    if order == TIME:
        chapters.append({"kind": "theme", "what": "The whole trip"})
        where = [0 if block_of(r, split) in on else None for r in rows]
    elif order == THEME:
        index = {k: n for n, k in enumerate(on)}
        chapters += [{"kind": "theme", "what": k} for k in on]
        where = [index.get(block_of(r, split)) for r in rows]
    else:
        days = sorted({r["taken_at"][:10] for r in rows})
        index = {}
        for d in days:
            for k in on:
                if any(r["taken_at"][:10] == d and block_of(r, split) == k for r in rows):
                    index[(d, k)] = len(chapters)
                    chapters.append({"kind": "theme", "what": f"{d} {k}"})
        where = [index.get((r["taken_at"][:10], block_of(r, split))) for r in rows]
    if storyline.get("finale", True):
        chapters.append({"kind": "finale", "what": "Finale"})
    return chapters, where


def _story_chapters(story: dict, rows: list[dict], split: set[str]) -> tuple[list[dict], list[int | None]]:
    """My story: every theme of every part is a chapter, in the story's order. A theme's shots go to its first
    entry that wants them ("early": the first half by time, "late": the second, "action": the most exciting half);
    a theme can come back later in the story with the shots not used yet."""
    from .autoedit import excitement

    members: dict[str, list[int]] = {}
    for i, r in enumerate(rows):
        members.setdefault(block_of(r, split), []).append(i)
    for k in members:
        members[k].sort(key=lambda i: rows[i]["taken_at"])
    where: list[int | None] = [None] * len(rows)
    chapters: list[dict] = []
    # The shots you unticked in a theme's window stay out of the video; the ones you ticked must be in it.
    dropped = {p for part in story.get("parts") or [] for e in part.get("entries") or [] for p in e.get("drop") or []}
    for n_part, part in enumerate(story.get("parts") or []):
        for entry in part.get("entries") or []:
            keep = set(entry.get("keep") or [])
            free = [i for i in members.get(entry["key"], []) if where[i] is None and rows[i]["path"] not in dropped]
            if not free:
                continue
            pick = entry.get("pick", "all")
            # the half of the whole theme (claimed or not), so "early" and "late" split it fairly
            whole = members[entry["key"]]
            half = max(1, (len(whole) + 1) // 2)
            if pick == "early":
                want = set(whole[:half])
            elif pick == "late":
                want = set(whole[len(whole) - half:])
            elif pick == "moment" and entry.get("shots"):
                # the shots you ticked for the moment (a big jump...), each with its second
                seconds = {s["path"]: s["t"] for s in entry["shots"]}
                want = {i for i in whole if rows[i]["path"] in seconds}
                for i in want:
                    if where[i] is None:
                        rows[i]["moment_at"] = seconds[rows[i]["path"]]
            elif pick in ("action", "moment"):  # a moment not looked for yet: the most exciting half meanwhile
                want = set(sorted(whole, key=lambda i: excitement(rows[i]), reverse=True)[:half])
            else:
                want = set(whole)
            # (a clip with a moment you locked always comes along)
            take = [i for i in free if i in want or rows[i]["path"] in keep or rows[i].get("locks")]
            if not take:
                continue
            for i in take:
                where[i] = len(chapters)
                if rows[i]["path"] in keep:
                    rows[i]["forced"] = True
            chapters.append({"kind": "theme", "what": f"{part.get('name', '')}: {entry['key']}",
                             "pace": PACES.get(part.get("pace"), None), "part": n_part, "count": len(take)})
    # Your time per part: shared by its themes (bigger ones get more), so Auto-edit gives each exactly that time.
    for n_part, part in enumerate(story.get("parts") or []):
        mine = [c for c in chapters if c["part"] == n_part]
        if part.get("seconds") and mine:
            total = sum(c["count"] ** 0.5 for c in mine)
            for c in mine:
                c["weight"] = part["seconds"] * c["count"] ** 0.5 / total
    return chapters, where


FINALE_WORDS = ("finale", "highlights", "best bits", "best moments", "best shots", "most exciting")
REST_WORDS = ("everything else", "the rest", "all the rest", "anything else", "other things")


def story_from_words(idea: str, blocks: list[dict]) -> dict:
    """Without an AI: the themes your words name, in the order you name them (one part each), then "the rest" where
    you say it. blocks: {"key", "activity", "person"}. Same shape as an AI story (entries carry the block's key)."""
    text = " " + " ".join(idea.lower().split()) + " "

    def spots(b: dict) -> list[int]:
        words = {b["activity"].lower()} | {w for w in b["activity"].lower().replace("&", " ").split() if len(w) > 3}
        for line in b.get("describe") or DEFAULTS.get(b["activity"], ("", []))[1]:
            words |= {w.strip(",.") for w in line.lower().split() if len(w) > 4 and w not in COMMON}
        found = [m for w in words for m in [_find_word(text, w)] if m >= 0]
        person = b.get("person")
        if person and person != "everyone else":
            near = _find_word(text, person.lower())
            if near < 0:
                return []  # a block of one person: only when you name them
            found = [near] + found
        return found

    named = []
    for b in blocks:
        at = spots(b)
        if at:
            named.append((min(at), b))
    named.sort(key=lambda t: t[0])
    rest_at = min([text.find(w) for w in REST_WORDS if text.find(w) >= 0] or [-1])
    parts = []
    used = {b["key"] for _, b in named}
    rest = [b for b in blocks if b["key"] not in used]
    placed_rest = False
    for at, b in named:
        if rest_at >= 0 and not placed_rest and rest_at < at and rest:
            parts.append({"name": "Everything else", "pace": "medium",
                          "entries": [{"key": r["key"], "pick": "all"} for r in rest]})
            placed_rest = True
        parts.append({"name": block_name(b), "pace": "medium", "entries": [{"key": b["key"], "pick": "all"}]})
    if rest_at >= 0 and not placed_rest and rest:
        parts.append({"name": "Everything else", "pace": "medium",
                      "entries": [{"key": r["key"], "pick": "all"} for r in rest]})
    if parts:
        parts[0]["pace"] = "calm"
        if len(parts) > 2:
            parts[-1]["pace"] = "lively"
    return {"title": "Your story", "parts": parts, "finale": any(w in text for w in FINALE_WORDS)}


COMMON = {"people", "person", "with", "relaxing", "photo", "close-up", "group", "friends", "busy", "narrow", "famous",
          "trip", "view", "views", "riding", "small"}


def _find_word(text: str, word: str) -> int:
    """Where a word (or the start of one: "kite" finds "kitesurfing") is in the text, -1 if it isn't."""
    import re

    m = re.search(r"\b" + re.escape(word), text)
    return m.start() if m else -1


def block_name(b: dict) -> str:
    return b["activity"] + (f" · {b['person']}" if b.get("person") else "")


def story_keys(story: dict, blocks: list[dict]) -> dict:
    """An AI answer (entries by block number) with each entry's block key, ready to save."""
    out = dict(story)
    out["parts"] = [dict(p, entries=[{"key": blocks[e["block"]]["key"], "pick": e["pick"]}
                                     | ({"moment": e["moment"]} if e.get("moment") else {})
                                     for e in p["entries"] if 0 <= e["block"] < len(blocks)])
                    for p in story.get("parts") or []]
    out["parts"] = [p for p in out["parts"] if p["entries"]]
    return out


def left_out(story: dict, blocks: list[dict]) -> list[dict]:
    """The blocks (that are on) your story doesn't use."""
    used = {e["key"] for p in story.get("parts") or [] for e in p.get("entries") or []}
    return [b for b in blocks if b.get("on", True) and b["key"] not in used]


def add_missing(story: dict, blocks: list[dict], when: dict[str, float]) -> dict:
    """Your story with every theme in it: a theme the AI (or your words) left out goes into the part whose themes
    were filmed nearest in time to it (never into the last part, which ends the video, unless it is the only one),
    marked "added" so you can see the app put it there. when: block key -> its middle time (seconds)."""
    parts = [dict(p, entries=list(p.get("entries") or [])) for p in story.get("parts") or []]
    missing = left_out({"parts": parts}, blocks)
    if not parts or not missing:
        return dict(story, parts=parts)

    def middle(part: dict) -> float | None:
        times = sorted(when[e["key"]] for e in part["entries"] if e["key"] in when)
        return times[len(times) // 2] if times else None

    choices = list(range(len(parts) - 1)) or [0]
    for block in sorted(missing, key=lambda b: when.get(b["key"], 0.0)):
        at = when.get(block["key"])
        mids = {k: middle(parts[k]) for k in choices}
        known = [k for k in choices if mids[k] is not None]
        k = min(known, key=lambda k: abs(mids[k] - at)) if at is not None and known else choices[-1]
        entry = {"key": block["key"], "pick": "all", "added": True}
        # inside the part, by time among the other added ones, after what the AI chose (that order is yours)
        spot = len(parts[k]["entries"])
        while spot > 0 and parts[k]["entries"][spot - 1].get("added") and \
                when.get(parts[k]["entries"][spot - 1]["key"], 0.0) > (at or 0.0):
            spot -= 1
        if k == len(parts) - 1 and parts[k]["entries"]:  # the very last theme stays last
            spot = min(spot, len(parts[k]["entries"]) - 1)
        parts[k]["entries"].insert(spot, entry)
    return dict(story, parts=parts)


# ---------- time per part ----------

FINALE_SHARE = 0.12  # the finale's share of the video (as in Auto-edit)
SHORTEST_PART = 3.0  # seconds


def part_room(total: float, finale: bool) -> float:
    """The seconds the story's parts share: the whole video, less the finale."""
    return total * (1 - FINALE_SHARE) if finale else total


def fit_times(parts: list[dict], room: float) -> list[float]:
    """Each part's seconds, adding up to room: your (or the AI's) times scaled to fit, or, where a part has none, a
    share by how many shots it has."""
    if not parts:
        return []
    raw = [float(p.get("seconds") or 0) for p in parts]
    sizes = [max(1, len(p.get("entries") or [])) for p in parts]
    if not all(raw):
        known = sum(raw) / max(1, sum(1 for r in raw if r))  # the average of the ones set
        raw = [r or (known or room / len(parts)) * sizes[k] ** 0.5 / (sum(sizes) / len(sizes)) ** 0.5
               for k, r in enumerate(raw)]
    scale = room / sum(raw)
    times = [max(SHORTEST_PART, r * scale) for r in raw]
    over = sum(times) - room
    if over > 0.01:  # the shortest parts were lifted: the others give that time back
        flexible = [k for k, t in enumerate(times) if t > SHORTEST_PART]
        spare = sum(times[k] - SHORTEST_PART for k in flexible) or 1
        times = [t - over * (t - SHORTEST_PART) / spare if k in flexible else t for k, t in enumerate(times)]
    return [round(t, 1) for t in times]


def change_time(times: list[float], n: int, seconds: float) -> list[float]:
    """Part n gets this many seconds; the others give or take the difference (in proportion), so the total stays."""
    room = sum(times)
    others = [k for k in range(len(times)) if k != n]
    if not others:
        return times
    seconds = min(max(SHORTEST_PART, seconds), room - SHORTEST_PART * len(others))
    extra = room - seconds - SHORTEST_PART * len(others)  # what the others have above their shortest
    above = {k: max(0.0, times[k] - SHORTEST_PART) for k in others}
    have = sum(above.values())
    out = list(times)
    out[n] = seconds
    for k in others:
        out[k] = SHORTEST_PART + extra * (above[k] / have if have else 1 / len(others))
    return [round(t, 1) for t in out]
