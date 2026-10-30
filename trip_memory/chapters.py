"""Chapters: your note ("kiting first, then the people, food at the end, close with action") becomes a list of
chapters, and the video tells them in that order. The music still decides how many shots there are and where
the cuts go; the chapters decide which shots come where.

Free and on this Mac: the note is read by a small word reader (or the local AI, when there is one), and the shots
are matched to each chapter by their tags, the AI's captions, the people you named and the CLIP model the app
already uses for scenes (it compares the words with what each picture shows).
"""

from __future__ import annotations

import re

import numpy as np

WORDS, PEOPLE, PERSON, FINALE, REST = "words", "people", "person", "finale", "rest"
KINDS = (WORDS, PEOPLE, PERSON, FINALE, REST)
FINALE_NAME = "Finale: the most action"
REST_NAME = "Everything else"
PEOPLE_NAME = "People"

# A word you might use -> what CLIP should look for, and the app's tags that mean the same.
FOOD = ["Eating & drinking", "Restaurant & café", "Food & drinks", "Cooking & barbecue", "Breakfast", "Pizza", "Pasta",
        "Seafood", "Sushi", "Ice cream", "Cake & dessert", "Burgers", "Street food", "Coffee", "Cocktails", "Beer",
        "Wine", "Dinner out", "Fruit", "Restaurant", "Cafe", "Barbecue", "Cooking", "Picnic", "Wine tasting"]
HINTS: dict[str, tuple[str, list[str]]] = {
    "kit": ("kitesurfing: a kite surfer riding over the sea pulled by a big kite", ["Surfing"]),
    "surf": ("surfing on a wave", ["Surfing"]),
    "food": ("food and drinks on a table", FOOD), "eat": ("people eating food at a table", FOOD),
    "dinner": ("dinner at a restaurant table", FOOD), "lunch": ("lunch at a table", FOOD),
    "breakfast": ("breakfast with coffee", FOOD), "restaurant": ("inside a restaurant", FOOD),
    "drink": ("drinks and cocktails", FOOD), "meal": ("a meal on a table", FOOD), "cocktail": ("cocktails", FOOD),
    "beach": ("a sandy beach by the sea", ["Beach", "Relaxing & sunbathing", "Sunbathing"]),
    "sea": ("the sea", ["Sea", "Sea view", "Beach", "Boat trip"]),
    "sunset": ("a sunset with golden light", ["Sunset & sunrise", "Sunset", "Sunrise", "Sunset"]),
    "sunrise": ("a sunrise", ["Sunset & sunrise", "Sunrise"]),
    "swim": ("people swimming", ["Swimming", "Pool", "Water park"]),
    "pool": ("a swimming pool", ["Pool"]),
    "boat": ("a boat on the water", ["Boat trip", "Sailing", "Jet ski & speedboat", "Harbour", "Ferry"]),
    "hik": ("people hiking in the mountains", ["Hiking", "Mountains"]),
    "mountain": ("mountains", ["Mountains", "Hiking"]),
    "city": ("a city street", ["City street", "Old town", "Skyline", "Square", "Landmark"]),
    "town": ("streets of an old town", ["Old town", "City street", "Square"]),
    "sightsee": ("tourists sightseeing", ["Sightseeing", "Landmark", "Old town", "Museum visit"]),
    "night": ("a scene at night", ["Night", "Night lights", "Bar & nightlife"]),
    "party": ("a party with people dancing", ["Dancing & party", "Party", "Dancing", "Bar & nightlife"]),
    "danc": ("people dancing", ["Dancing & party", "Dancing"]),
    "kid": ("children playing", ["Kids playing", "Kids", "Playground", "Sandcastles"]),
    "child": ("children playing", ["Kids playing", "Kids", "Playground", "Sandcastles"]),
    "animal": ("animals", ["Animals"]), "dog": ("a dog", ["Dogs"]),
    "snow": ("snow", ["Snow", "Skiing"]), "ski": ("skiing on a slope", ["Skiing", "Snow"]),
    "drive": ("a road trip in a car", ["Driving", "Road", "Road trip", "Car"]),
    "road": ("a road trip in a car", ["Road", "Road trip", "Driving", "Car"]),
    "car": ("a car", ["Car", "Road", "Driving"]),
    "flight": ("an airplane flight", ["Airport & transport", "Flight", "Airport", "Travel day"]),
    "plane": ("an airplane", ["Airport & transport", "Flight", "Airport", "Travel day"]),
    "airport": ("an airport", ["Airport & transport", "Airport", "Travel day"]),
    "travel": ("travelling by plane, train or car", ["Airport & transport", "Travel day", "Road"]),
    "hotel": ("a hotel", ["Hotel & home", "Hotel", "Villa & holiday home"]),
    "landscape": ("a wide landscape view", ["Landscape", "Viewpoint", "Mountains", "Forest & countryside"]),
    "view": ("a panoramic view", ["Viewpoint", "Landscape", "Sea view"]),
    "nature": ("nature, forest and countryside", ["Forest & countryside", "Forest", "Countryside", "Lake & river"]),
    "underwater": ("underwater with fish", ["Underwater", "Snorkelling & diving", "Snorkelling", "Scuba diving"]),
    "snorkel": ("snorkelling underwater", ["Snorkelling & diving", "Snorkelling", "Underwater"]),
    "div": ("scuba diving underwater", ["Snorkelling & diving", "Scuba diving", "Underwater"]),
    "market": ("a market with stalls", ["Shopping & market", "Market", "Shopping"]),
    "shop": ("people shopping", ["Shopping & market", "Shopping", "Market"]),
    "sport": ("people doing sport", ["Ball games & sport", "Sports & activities"]),
    "football": ("people playing football", ["Ball games & sport", "Playing football"]),
    "bike": ("people riding bicycles", ["Cycling"]), "cycl": ("people riding bicycles", ["Cycling"]),
    "kayak": ("people kayaking", ["Kayak & paddle board", "Kayaking", "Paddle boarding"]),
    "paddle": ("stand up paddle boarding", ["Kayak & paddle board", "Paddle boarding"]),
    "jet": ("riding a jet ski", ["Jet ski & speedboat", "Jet ski"]),
    "museum": ("inside a museum", ["Museum visit", "Museum", "Art gallery"]),
    "church": ("a church or cathedral", ["Church & cathedral", "Landmark"]),
    "castle": ("a castle", ["Castle", "Landmark"]),
    "pool party": ("a pool party", ["Pool", "Dancing & party"]),
}
PEOPLE_WORDS = {"people", "person", "us", "we", "friends", "friend", "family", "together", "faces", "portraits",
                "selfies", "selfie", "group", "crew", "gang", "everyone", "everybody"}
REST_PHRASES = ("everything else", "the rest", "all the rest", "anything else", "other shots", "other stuff",
                "whatever is left", "the others", "everything in between", "in between")
ACTION_WORDS = {"action", "actions", "exciting", "excitement", "epic", "fast", "best", "highlights", "highlight",
                "craziest", "wildest", "energy", "energetic", "bang", "adrenaline", "jumps", "tricks"}
START_WORDS = ("start", "begin", "beginning", "first", "open", "opening", "intro", "kick off")
END_WORDS = ("end", "ending", "finish", "finally", "last", "close", "closing", "lastly", "wrap", "at the end",
             "in the end")
FILLER = set("""
i i'd i'm id im want wanna would like love to have has put all the a an more of with some something it its it's then
and also please should be is are was were video videos clip clips shot shots picture pictures photo photos footage
part parts bit lot lots little few many much most at in on into for from by after before that this these those
separate separately beginning begin start starting first open opening end ending finish finishing finally last
lastly close closing wrap up next followed afterwards afterward show showing me my our your make let do go come
just really very maybe kind sort stuff things thing like scene scenes moments moment section sections where there
can could will gonna one ones whole entire full main again same time times when while um uh so okay ok yeah
""".split())

SPLIT = re.compile(r"(?:[.;!?\n]+|,\s*(?:and\s+)?then\b|,|\bthen\b|\bafter that\b|\bafterwards\b|\bfollowed by\b|"
                   r"\bnext\b|\band finally\b|\bfinally\b|\blastly\b|\bto finish\b|\band to close\b)", re.I)


def chapter(kind: str, what: str = "", search: str = "") -> dict:
    """One chapter: kind is one of KINDS, what its name, search what the pictures should show (for CLIP)."""
    names = {FINALE: FINALE_NAME, REST: REST_NAME, PEOPLE: PEOPLE_NAME}
    return {"kind": kind, "what": what or names.get(kind, ""), "search": search, "vector": None}


def stem(word: str) -> str:
    """'kiting' and 'kites' -> 'kit', 'beaches' -> 'beach', 'dinner' stays."""
    w = word.lower().strip("'\"")
    for end in ("ing", "ers", "es", "er", "s", "e"):
        if len(w) - len(end) >= 3 and w.endswith(end) and not (end == "er" and w.endswith("nner")):
            w = w[: -len(end)]
            break
    return w


def words_of(text: str) -> list[str]:
    return re.findall(r"[a-zà-ÿ][a-zà-ÿ'&-]*", text.lower())


def hint_for(text: str) -> tuple[str, list[str]]:
    """What CLIP should look for, and tags that mean the same, for a chapter's words."""
    searches, tags = [], []
    for w in words_of(text):
        s = stem(w)
        for key, (search, hint_tags) in HINTS.items():
            if s.startswith(key) or (len(s) >= 4 and key.startswith(s)):
                searches.append(search)
                tags += hint_tags
                break
    return ("; ".join(dict.fromkeys(searches)) or text), list(dict.fromkeys(tags))


def read_note(note: str, people: list[str]) -> list[dict]:
    """Your note as chapters, by the app's own word reader (no AI needed)."""
    known = {p.lower(): p for p in people if p and not p.startswith("Person ")}
    parts = [p.strip() for p in SPLIT.split(note or "") if p and p.strip()]
    found: list[tuple[int, int, dict]] = []  # (where: -1 start, 0 middle, 1 end; order in the note; chapter)
    for n, part in enumerate(parts):
        low = " " + " ".join(words_of(part)) + " "
        where = (1 if any(f" {w} " in low for w in END_WORDS) else
                 -1 if any(f" {w} " in low for w in START_WORDS) else 0)
        words = [w for w in words_of(part)]
        if any(f" {p} " in low for p in REST_PHRASES):
            found.append((where, n, chapter(REST)))
            continue
        who = [known[w] for w in words if w in known]
        content = [w for w in words if w not in FILLER and w not in known and w not in ACTION_WORDS]
        if who:
            for name in dict.fromkeys(who):
                found.append((where, n, chapter(PERSON, name)))
            continue
        if any(w in ACTION_WORDS for w in words) and not [w for w in content if w not in PEOPLE_WORDS]:
            found.append((1 if where >= 0 else where, n, chapter(FINALE)))
            continue
        if content and all(w in PEOPLE_WORDS for w in content):
            found.append((where, n, chapter(PEOPLE)))
            continue
        content = [w for w in content if w not in PEOPLE_WORDS] or content
        if not content:
            continue
        what = " ".join(content[:4])
        search, _tags = hint_for(what)
        found.append((where, n, chapter(WORDS, what, search)))
    # "at the beginning" and "at the end" win over the order you said things in; a finale always closes.
    found.sort(key=lambda f: (f[2]["kind"] == FINALE, f[0], f[1]))
    return tidy([c for _w, _n, c in found])


def tidy(chapters: list[dict]) -> list[dict]:
    """No repeats, one finale (last), and an "Everything else" chapter so no shot is lost without you saying so."""
    out: list[dict] = []
    seen = set()
    for c in chapters:
        key = (c["kind"], c["what"].lower()) if c["kind"] in (WORDS, PERSON) else (c["kind"],)
        if key in seen:
            continue
        seen.add(key)
        out.append(c)
    finale = [c for c in out if c["kind"] == FINALE]
    out = [c for c in out if c["kind"] != FINALE]
    content = [c for c in out if c["kind"] != REST]
    if content and not any(c["kind"] == REST for c in out):
        # Before the last thing you named, so the ending is yours ("food at the end") and the rest comes before.
        at = out.index(content[-1]) if len(content) >= 2 else len(out)
        out.insert(at, chapter(REST))
    return out + finale[:1]


# ---------- matching shots to chapters ----------


def row_words(row: dict) -> set[str]:
    """Every word known about a shot: its tags, scene, subjects, the AI's caption and the place."""
    text = " ".join([*(row.get("tags_final") or []), row.get("scene_final") or "", row.get("ai_caption") or "",
                     row.get("place") or "", *(row.get("subject_names") or [])])
    return {w for w in words_of(text)}


def _keyword_hit(chapter_words: list[str], words: set[str]) -> bool:
    stems = [stem(w) for w in chapter_words if w not in FILLER and len(w) >= 3]
    return any(w.startswith(s) for s in stems for w in words)


def scores(rows: list[dict], chapters: list[dict], vectors: dict[str, np.ndarray]) -> np.ndarray:
    """How well each shot fits each chapter (rows x chapters); 1 or more is a match. Finale and "Everything else"
    take no shots here (the finale takes the most exciting shots when cutting; the rest gets what is left)."""
    out = np.zeros((len(rows), len(chapters)))
    known = np.array([p in vectors for p in (r["path"] for r in rows)])
    image = np.array([vectors[r["path"]] for r in rows if r["path"] in vectors]) if known.any() else None
    for j, c in enumerate(chapters):
        if c["kind"] == PERSON:
            out[:, j] = [3.0 if c["what"] in (r.get("people") or []) else 0.0 for r in rows]
        elif c["kind"] == PEOPLE:
            out[:, j] = [(1.0 + 0.3 * bool(r.get("people"))) if (r.get("people") or r.get("people_tag") in
                                                                 ("Solo", "Couple", "Group")) else 0.0 for r in rows]
        elif c["kind"] == WORDS:
            _search, tags = hint_for(c["what"])
            words = words_of(c["what"])
            for i, r in enumerate(rows):
                own = set(r.get("tags_final") or []) | {r.get("scene_final")} | set(r.get("subject_names") or [])
                if _keyword_hit(words, row_words(r)) or own & set(tags):
                    out[i, j] += 2.0
            if c.get("vector") is not None and image is not None and len(image) >= 3:
                sims = image @ np.asarray(c["vector"], np.float32)
                z = (sims - sims.mean()) / (sims.std() + 1e-6)
                out[known, j] += np.clip(z, 0, 3) * 0.8  # a picture that looks like the words, compared with the rest
    return out


def assign(rows: list[dict], chapters: list[dict], vectors: dict[str, np.ndarray]) -> list[tuple[int | None, float]]:
    """For each shot: the chapter it goes in (None: left out) and how well it fits. A shot that fits several goes to
    the one it fits best (on a tie: the earlier one); a shot that fits none goes to "Everything else" if you have it."""
    fit = scores(rows, chapters, vectors)
    rest = next((j for j, c in enumerate(chapters) if c["kind"] == REST), None)
    out: list[tuple[int | None, float]] = []
    for i in range(len(rows)):
        j = int(np.argmax(fit[i])) if len(chapters) else 0
        if len(chapters) and fit[i, j] >= 1.0:
            out.append((j, float(fit[i, j])))
        else:
            out.append((rest, 0.0))
    return out
