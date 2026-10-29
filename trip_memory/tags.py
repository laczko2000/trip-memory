"""Many tags per photo or video: the kind of place, what people are doing, and the light.

Uses the same free CLIP model as the scene suggestion. Each tag has a few descriptions; every picture
(every few seconds of a video) is compared with them, group by group, and the clear matches become tags.
A clip that starts on the beach and ends in the water gets both.
"""

from __future__ import annotations

from collections import Counter

import numpy as np

from .motion import MOVEMENT_TAGS

# Group -> tag -> descriptions. A tag starting with "_" is a "none of these" option and is never shown.
GROUPS: dict[str, dict[str, list[str]]] = {
    "Place": {
        "Beach": ["a sandy beach", "people on a beach by the sea"],
        "Sea": ["the open sea", "the sea seen from a boat"],
        "Underwater": ["an underwater view with fish", "under the sea surface"],
        "Pool": ["a swimming pool"],
        "Mountains": ["mountains", "a mountain view"],
        "Forest & countryside": ["a forest", "green countryside and fields"],
        "Lake & river": ["a lake", "a river"],
        "City street": ["a busy city street", "a modern city"],
        "Old town": ["narrow streets of an old town", "a historic town square"],
        "Landmark": ["a famous landmark or monument", "a castle, cathedral or temple"],
        "Restaurant & café": ["inside a restaurant", "a cafe terrace"],
        "Hotel & home": ["a hotel room", "inside a house or apartment"],
        "Airport & transport": ["an airport or an airplane", "a train, bus or ferry"],
        "Road": ["a road seen from a car", "inside a car"],
        "Snow": ["a snowy landscape", "a ski slope"],
        "Desert": ["a desert with sand dunes"],
        "Garden & park": ["a garden or park"],
    },
    "Activity": {
        "Walking": ["people walking along a street or path"],
        "Hiking": ["people hiking on a mountain trail"],
        "Swimming": ["people swimming in the water"],
        "Snorkelling & diving": ["a person snorkelling", "a scuba diver underwater"],
        "Surfing": ["a person surfing a wave"],
        "Kayak & paddle board": ["people kayaking", "a person on a stand up paddle board"],
        "Jet ski & speedboat": ["a person riding a jet ski", "a fast speedboat"],
        "Boat trip": ["people on a boat trip"],
        "Cycling": ["people riding bicycles"],
        "Skiing": ["people skiing or snowboarding"],
        "Eating & drinking": ["people eating at a table", "people having drinks"],
        "Cooking & barbecue": ["people cooking or having a barbecue"],
        "Dancing & party": ["people dancing at a party"],
        "Kids playing": ["children playing"],
        "Sightseeing": ["tourists sightseeing and taking pictures"],
        "Relaxing & sunbathing": ["people relaxing and sunbathing"],
        "Driving": ["a person driving a car"],
        "Posing": ["people posing and smiling for a photo"],
        "Ball games & sport": ["people playing football or volleyball", "people doing sport"],
        "Shopping & market": ["people shopping at a market"],
        "Concert & show": ["a concert or a show on a stage"],
        "Animals": ["animals", "people with animals"],
        "_none": ["a landscape without people", "a building", "an object on a table"],
    },
    "Light": {
        "Daylight": ["a picture taken in bright daylight"],
        "Sunset & sunrise": ["a sunset with golden and orange light", "a sunrise"],
        "Night": ["a picture taken at night", "a dark scene with artificial lights"],
    },
}
PEOPLE_TAGS = ["No people", "Solo", "Couple", "Group"]

# How sure (0..1, compared with the other tags of the same group) a match must be.
SURE = {"Place": 0.35, "Activity": 0.30, "Light": 0.5}
MAX_PER_PICTURE = {"Place": 1, "Activity": 2, "Light": 1}

TAG_GROUP = {tag: group for group, tags in GROUPS.items() for tag in tags if not tag.startswith("_")}
TAG_GROUP.update({t: "Movement" for t in MOVEMENT_TAGS})
TAG_GROUP.update({t: "People" for t in PEOPLE_TAGS})
GROUP_ORDER = ["Movement", "Activity", "Place", "Light", "People"]
ALL_TAGS = sorted(TAG_GROUP, key=lambda t: (GROUP_ORDER.index(TAG_GROUP[t]), t))
EDITABLE_TAGS = [t for t in ALL_TAGS if TAG_GROUP[t] != "People"]


class Tagger:
    def __init__(self, scenes):
        """scenes: the SceneTagger (for its CLIP text model)."""
        self.groups = {}
        for group, tags in GROUPS.items():
            names, rows = [], []
            for tag, descriptions in tags.items():
                vectors = scenes.describe([f"a holiday photo of {d}" for d in descriptions])
                mean = vectors.mean(axis=0)
                names.append(tag)
                rows.append(mean / np.linalg.norm(mean))
            self.groups[group] = (names, np.array(rows))

    def tag(self, frame_vectors: np.ndarray) -> list[str]:
        return tag_frames(frame_vectors, self.groups)


def tag_frames(frame_vectors: np.ndarray, groups: dict[str, tuple[list[str], np.ndarray]]) -> list[str]:
    """Tags for a photo (one vector) or a video (one vector per frame), most frequent first."""
    seen: Counter = Counter()
    for vector in frame_vectors:
        for group, (names, matrix) in groups.items():
            logits = 100 * matrix @ vector
            probs = np.exp(logits - logits.max())
            probs /= probs.sum()
            for i in np.argsort(probs)[::-1][: MAX_PER_PICTURE[group]]:
                if probs[i] >= SURE[group] and not names[i].startswith("_"):
                    seen[names[i]] += 1
    return [tag for tag, _ in seen.most_common()]


def people_tag(face_count: int) -> str:
    return PEOPLE_TAGS[min(face_count, 3)]


def look_alike_groups(vectors: dict[str, np.ndarray], threshold: float = 0.86) -> dict[str, int]:
    """Put photos and videos that look alike in the same group, wherever they are in the trip.

    Greedy: each file joins the most similar group if close enough, else starts a new one.
    Groups of one are left out."""
    centres: list[np.ndarray] = []
    members: list[list[str]] = []
    for path, vec in vectors.items():
        if centres:
            sims = np.array(centres) @ vec
            best = int(sims.argmax())
            if sims[best] >= threshold:
                members[best].append(path)
                centre = np.mean([vectors[p] for p in members[best]], axis=0)
                centres[best] = centre / np.linalg.norm(centre)
                continue
        centres.append(vec)
        members.append([path])
    result = {}
    number = 0
    for group in members:
        if len(group) > 1:
            number += 1
            result.update({p: number for p in group})
    return result


def group_name(tag_lists: list[list[str]]) -> str:
    """Name a look-alike group after what its members have in common, e.g. "Snorkelling & diving · Sea"."""
    counts: Counter = Counter(t for tags in tag_lists for t in tags if TAG_GROUP.get(t) in ("Activity", "Place"))
    common = [t for t, n in counts.most_common() if n * 2 >= len(tag_lists)]
    activity = next((t for t in common if TAG_GROUP[t] == "Activity"), None)
    place = next((t for t in common if TAG_GROUP[t] == "Place"), None)
    return " · ".join(t for t in (activity, place) if t) or "Similar scenes"
