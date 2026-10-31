"""New themes for a new trip: the app looks at the shots you chose and proposes the trip's own themes (and
sub-themes), instead of the ones of your last trip.

1. It puts the shots that look alike together (their CLIP picture fingerprints, made when Analyse runs).
2. It names each group from a long list of things holidays show (temples, street food, neon streets at night,
   kitesurfing...): the name whose pictures the group fits clearly better than the rest of the trip does.
   A group that looks like the shots of a theme you confirmed on an earlier trip keeps that theme's name.
3. Inside a bigger group, the shots that fit a narrower name (torii gates, ramen) become sub-themes.
4. If an AI is set up (Claude through your Pro, or Ollama on this Mac), it may give the groups better names from
   what it knows of the shots (the descriptions it wrote, places); without it the names above stay.

Free and on this Mac, except step 4 when you chose Claude.
"""

from __future__ import annotations

import numpy as np

# Theme -> (emoji, [(sub-theme, what its pictures show)]). The sub-themes' sentences are what CLIP compares.
VOCAB: dict[str, tuple[str, list[tuple[str, str]]]] = {
    "Temples & shrines": ("⛩️", [("Torii gates", "rows of red torii gates"), ("Temple", "a buddhist temple building"),
                                ("Shrine", "a shinto shrine with a wooden gate"), ("Pagoda", "a tall pagoda"),
                                ("Buddha statue", "a big buddha statue")]),
    "Churches & monuments": ("🏛️", [("Church", "a church or cathedral"), ("Monument", "a famous monument or statue"),
                                    ("Ruins", "ancient ruins and columns")]),
    "Castles & palaces": ("🏯", [("Castle", "a castle on a hill"), ("Palace", "a royal palace and its courtyard")]),
    "Gardens & parks": ("🌳", [("Garden", "a traditional garden with a pond"), ("Zen garden", "a zen rock garden"),
                               ("Park", "people walking in a city park")]),
    "Cherry blossoms & autumn": ("🌸", [("Cherry blossoms", "pink cherry blossom trees"),
                                       ("Autumn leaves", "red and orange autumn leaves")]),
    "Food": ("🍜", [("Ramen & noodles", "a bowl of ramen noodles"), ("Sushi", "sushi on a plate"),
                    ("Street food", "street food stalls and snacks"), ("Restaurant meal", "a meal on a restaurant table"),
                    ("Desserts & sweets", "desserts, cakes and ice cream"), ("Pizza & pasta", "pizza and pasta")]),
    "Cafés & drinks": ("☕", [("Café", "coffee and cake in a café"), ("Drinks", "cocktails and drinks at a bar")]),
    "Markets & shopping": ("🛍️", [("Market", "a busy food market with stalls"),
                                  ("Shopping street", "a shopping street with shop signs"),
                                  ("Souvenirs", "souvenirs on a shop shelf")]),
    "City at night": ("🌃", [("Neon streets", "a street full of neon signs at night"),
                             ("Night skyline", "a city skyline lit up at night"), ("Night crossing", "a busy crossing at night")]),
    "City streets": ("🏙️", [("Skyscrapers", "tall skyscrapers in a big city"), ("Busy crossing", "crowds crossing a city street"),
                            ("Old town", "narrow old town streets and alleys"), ("Towers & views", "the view from a tall tower")]),
    "Traditional villages": ("🏘️", [("Wooden houses", "old wooden houses in a traditional street"),
                                    ("Village", "a small village in the countryside")]),
    "Trains & travel": ("🚄", [("Bullet train", "a high speed bullet train"), ("Train station", "a train station platform"),
                               ("Plane", "an airport or the view from a plane window"), ("Bus & car", "the view from a car or bus")]),
    "Mountains & hiking": ("⛰️", [("Mountain view", "snowy mountains and a wide view"), ("Hiking", "people hiking on a trail"),
                                  ("Volcano", "a volcano mountain peak")]),
    "Forests & bamboo": ("🎋", [("Bamboo forest", "a tall bamboo forest path"), ("Forest", "a green forest with tall trees")]),
    "Lakes, rivers & waterfalls": ("🏞️", [("Lake", "a calm lake with mountains"), ("River", "a river through a town or valley"),
                                          ("Waterfall", "a waterfall")]),
    "Beach & sea": ("🏖️", [("Beach", "a sandy beach by the sea"), ("Coast", "a rocky coast and the sea"),
                           ("Sunset by the sea", "a sunset over the sea")]),
    "Boats": ("⛵", [("Boat trip", "people on the deck of a boat"), ("Sailing", "a sailing boat"), ("Ferry", "a ferry or a harbour")]),
    "Water sports": ("🏄", [("Kitesurfing", "kitesurfing on the sea with a kite"), ("Surfing", "surfing on a wave"),
                            ("Jet ski", "a person riding a jet ski"), ("Snorkelling", "snorkelling underwater")]),
    "Swimming & pools": ("🏊", [("Pool", "people swimming in a pool"), ("Swimming", "people swimming in the sea")]),
    "Hot springs & spa": ("♨️", [("Onsen", "a steaming outdoor hot spring bath"), ("Spa", "a spa and wellness room")]),
    "Snow & skiing": ("⛷️", [("Skiing", "people skiing down a slope"), ("Snow", "a snowy landscape")]),
    "Animals": ("🦌", [("Deer", "deer in a park"), ("Monkeys", "monkeys"), ("Cats & dogs", "a cat or a dog"),
                       ("Wildlife", "wild animals in nature"), ("Aquarium", "fish in an aquarium")]),
    "Museums & art": ("🖼️", [("Museum", "inside a museum with exhibits"), ("Art", "paintings and art in a gallery")]),
    "Culture & festivals": ("🏮", [("Lanterns", "paper lanterns"), ("Kimono", "people wearing kimono"),
                                   ("Festival", "a street festival with crowds"), ("Show", "a traditional show or dance")]),
    "Theme parks & fun": ("🎢", [("Theme park", "a theme park with rides"), ("Arcade", "an arcade with games and lights")]),
    "Nightlife & parties": ("🎉", [("Party", "people dancing at a party"), ("Concert", "a concert with a stage")]),
    "Hotel & relaxing": ("🛏️", [("Hotel room", "a hotel room"), ("Balcony view", "the view from a hotel balcony"),
                                ("Relaxing", "people relaxing on sunbeds")]),
    "People & portraits": ("🧑‍🤝‍🧑", [("Selfie", "a selfie of friends"), ("Group photo", "a group of people posing for a photo"),
                                     ("Portrait", "a close-up portrait of a person")]),
    "Kids & family": ("👨‍👩‍👧", [("Kids playing", "children playing"), ("Family", "a family together")]),
    "Sports": ("⚽", [("Ball games", "people playing football or volleyball"), ("Cycling", "people cycling"),
                     ("Climbing", "people climbing a rock")]),
    "Sunsets & views": ("🌅", [("Sunset", "a colourful sunset sky"), ("Panorama", "a wide panoramic view")]),
}

SAME_AS_EARLIER = 0.86  # a group this like the shots of an earlier trip's theme keeps that theme's name
SUB_SHARE = 0.18  # a sub-theme needs this share of its group's shots (and at least 2)


def phrases() -> list[tuple[str, str, str]]:
    """(theme, sub-theme, sentence) for every sentence CLIP compares."""
    return [(name, sub, sentence) for name, (_, subs) in VOCAB.items() for sub, sentence in subs]


def _unit(x: np.ndarray) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-9)


def groups(vectors: np.ndarray, k: int | None = None, seed: int = 0) -> np.ndarray:
    """Which group each shot is in: shots that look alike together (k-means on the fingerprints, by likeness)."""
    n = len(vectors)
    if n == 0:
        return np.zeros(0, int)
    k = k or int(np.clip(round(np.sqrt(n / 2)), 2, 12))
    k = max(1, min(k, n))
    x = _unit(np.asarray(vectors, np.float32))
    rng = np.random.default_rng(seed)
    centres = [x[rng.integers(n)]]
    for _ in range(1, k):  # spread out the starting points (k-means++)
        far = 1 - np.max(x @ np.array(centres).T, axis=1)
        far = np.maximum(far, 0) ** 2
        centres.append(x[rng.choice(n, p=far / far.sum())] if far.sum() > 0 else x[rng.integers(n)])
    c = np.array(centres)
    label = np.zeros(n, int)
    for _ in range(30):
        new = np.argmax(x @ c.T, axis=1)
        if _ > 0 and (new == label).all():
            break
        label = new
        for j in range(k):
            if (label == j).any():
                c[j] = _unit(x[label == j].mean(axis=0))
    return label


def propose(vectors: np.ndarray, sentence_vectors: np.ndarray,
            earlier: dict[str, tuple[str, list[np.ndarray]]] | None = None, k: int | None = None) -> list[dict]:
    """The proposed themes, biggest first: [{"name", "emoji", "describe", "rows", "sample", "subs": [{"name",
    "rows"}], "from": "looks" | "earlier"}]. rows: indexes into vectors. sentence_vectors: one per phrases().
    earlier: theme -> (emoji, the vectors of shots you confirmed for it on earlier trips)."""
    n = len(vectors)
    if n == 0:
        return []
    x = _unit(np.asarray(vectors, np.float32))
    table = phrases()
    s = x @ np.asarray(sentence_vectors, np.float32).T  # how well each shot fits each sentence
    # (compared with how well the whole trip fits it: "people" fits everything, temples only the temple shots)
    lift = s - s.mean(axis=0, keepdims=True)
    names = list(VOCAB)
    by_name = np.stack([lift[:, [i for i, p in enumerate(table) if p[0] == name]].max(axis=1) for name in names], 1)
    label = groups(x, k)
    found: dict[str, dict] = {}
    for j in sorted(set(label.tolist())):
        rows = np.flatnonzero(label == j)
        centre = _unit(x[rows].mean(axis=0))
        name, emoji, source = None, "", "looks"
        for theme, (mark, shots) in (earlier or {}).items():  # like an earlier trip's theme: that name
            if len(shots):
                near = np.sort(np.asarray(shots, np.float32) @ centre)[::-1][:3].mean()
                if near >= SAME_AS_EARLIER and (name is None or near > best):
                    name, emoji, source, best = theme, mark, "earlier", near
        if name is None:
            score = by_name[rows].mean(axis=0)
            pick = int(np.argmax(score))
            name, emoji = names[pick], VOCAB[names[pick]][0]
        if name in found:  # two groups of the same thing: one theme
            found[name]["rows"] = np.concatenate([found[name]["rows"], rows])
        else:
            found[name] = {"name": name, "emoji": emoji, "rows": rows, "from": source}
    out = []
    for theme in found.values():
        rows = theme["rows"]
        centre = _unit(x[rows].mean(axis=0))
        sample = rows[np.argsort(-(x[rows] @ centre))][:8]
        subs = []
        if theme["name"] in VOCAB and len(rows) >= 4:
            mine = [i for i, p in enumerate(table) if p[0] == theme["name"]]
            best = np.array(mine)[np.argmax(lift[np.ix_(rows, mine)], axis=1)]
            for i in mine:
                members = rows[best == i]
                if len(members) >= max(2, SUB_SHARE * len(rows)):
                    subs.append({"name": table[i][1], "rows": members.tolist()})
            if len(subs) < 2:
                subs = []
        describe = [p[2] for p in table if p[0] == theme["name"]][:3] or [theme["name"].lower()]
        out.append({"name": theme["name"], "emoji": theme["emoji"] or "⭐", "describe": describe,
                    "rows": sorted(rows.tolist()), "sample": sample.tolist(), "subs": subs, "from": theme["from"]})
    return sorted(out, key=lambda t: -len(t["rows"]))


NAMES_SCHEMA = {
    "type": "object",
    "properties": {"themes": {"type": "array", "items": {
        "type": "object",
        "properties": {"id": {"type": "integer"}, "name": {"type": "string"}, "emoji": {"type": "string"},
                       "subs": {"type": "array", "items": {"type": "object", "properties": {
                           "id": {"type": "integer"}, "name": {"type": "string"}}, "required": ["id", "name"]}}},
        "required": ["id", "name"]}}},
    "required": ["themes"],
}


def names_prompt(proposals: list[dict], facts: list[dict]) -> str:
    """A question for the AI: better names for the groups, from what is known about their shots.
    facts: per proposal {"captions": [...], "places": [...], "days": "..."}."""
    lines = ["These are groups of holiday photos and videos from one trip, put together because they look alike. "
             "Give each group a short, natural theme name (one to three words, like 'Temples', 'Street food', "
             "'Tokyo at night', 'Kitesurfing'), one fitting emoji, and short names for its sub-groups. Keep each "
             "group's id. Names must be different from each other. Use the language of the names given.", ""]
    for n, (p, f) in enumerate(zip(proposals, facts)):
        lines.append(f"Group {n}: {len(p['rows'])} shots. The picture model thinks: {p['name']}"
                     + (f" (sub-groups: " + ", ".join(f"{k}: {s['name']}" for k, s in enumerate(p["subs"])) + ")"
                        if p["subs"] else ""))
        if f.get("captions"):
            lines.append("  What some shots show: " + " | ".join(f["captions"][:8]))
        if f.get("places"):
            lines.append("  Places: " + ", ".join(f["places"][:5]))
        if f.get("days"):
            lines.append("  When: " + f["days"])
    return "\n".join(lines)


def apply_names(proposals: list[dict], answer: dict) -> list[dict]:
    """The AI's names on the groups (anything missing, empty or doubled keeps the picture model's name)."""
    taken = set()
    by_id = {t.get("id"): t for t in (answer or {}).get("themes") or [] if isinstance(t, dict)}
    out = []
    for n, p in enumerate(proposals):
        got = by_id.get(n) or {}
        name = " ".join(str(got.get("name") or "").split())[:30]
        p = dict(p)
        if name and name.lower() not in taken:
            p["name"] = name
            if got.get("emoji") and len(str(got["emoji"])) <= 4:
                p["emoji"] = str(got["emoji"])
        taken.add(p["name"].lower())
        subs = {s.get("id"): " ".join(str(s.get("name") or "").split())[:30]
                for s in got.get("subs") or [] if isinstance(s, dict)}
        p["subs"] = [s | ({"name": subs[k]} if subs.get(k) else {}) for k, s in enumerate(p["subs"])]
        out.append(p)
    return out
