"""Themes: the app suggests one activity per shot, learns from the ones you confirm, and builds the storyline."""

import numpy as np

from trip_memory import themes


def _unit(v):
    v = np.asarray(v, np.float32)
    return v / np.linalg.norm(v)


def test_suggestions_learn_from_your_confirmed_shots():
    acts = [{"name": "Water sports", "vector": list(_unit([1, 0, 0]))}, {"name": "Boat", "vector": list(_unit([0, 1, 0]))},
            {"name": themes.OTHER, "vector": None}]
    # Kitesurfing with a ship behind: the words lean towards "Boat" a little...
    kite = [_unit([0.5, 0.55, 0.3]) for _ in range(4)]
    boat = [_unit([0.1, 1.0, 0.1]) for _ in range(3)]
    vectors = np.array(kite + boat)
    first = themes.suggest(vectors, acts, [])
    assert first[0][0] == "Boat" and first[5][0] == "Boat"
    # ...until you confirm one kitesurfing shot: the ones that look like it follow.
    after = themes.suggest(vectors, acts, [(0, "Water sports")])
    assert [a for a, _ in after[:4]] == ["Water sports"] * 4
    assert [a for a, _ in after[4:]] == ["Boat"] * 3
    assert after[0] == ("Water sports", 1.0)


def test_a_shot_that_fits_nothing_is_other():
    acts = [{"name": "Food & drink", "vector": list(_unit([1, 0, 0]))}, {"name": "Boat", "vector": list(_unit([0, 1, 0]))},
            {"name": themes.OTHER, "vector": None}]
    vectors = np.array([_unit([1, 0, 0.05]), _unit([0, 1, 0.05]), _unit([0.02, 0.02, 1]), _unit([0.9, 0, 0.1])])
    assert [a for a, _ in themes.suggest(vectors, acts, [])] == ["Food & drink", "Boat", themes.OTHER, "Food & drink"]


def _row(n, theme, day="2025-07-03", people=()):
    return {"path": f"/t/{n}.jpg", "taken_at": f"{day}T10:{n:02d}:00", "theme": theme, "people": list(people)}


def test_the_storyline_orders_the_shots_three_ways():
    rows = [_row(0, "Boat"), _row(1, "Food & drink"), _row(2, "Water sports", people=["Laszlo"]),
            _row(3, "Food & drink", day="2025-07-04"), _row(4, "Water sports", day="2025-07-04", people=["Gosha"])]
    line = {"order": themes.TIME, "blocks": [], "split": [], "finale": True}
    chapters, where = themes.chapters_for(line, rows)
    assert [c["kind"] for c in chapters] == ["theme", "finale"] and where == [0] * 5
    blocks = [{"key": k, "activity": k.split("|")[0], "person": None, "on": True}
              for k in ("Water sports|", "Food & drink|", "Boat|")]
    chapters, where = themes.chapters_for(line | {"order": themes.THEME, "blocks": blocks, "finale": False}, rows)
    assert where == [2, 1, 0, 1, 0]  # water sports first, then food, then the boat
    chapters, where = themes.chapters_for(line | {"order": themes.DAY, "blocks": blocks, "finale": False}, rows)
    assert [c["what"] for c in chapters] == ["2025-07-03 Water sports|", "2025-07-03 Food & drink|", "2025-07-03 Boat|",
                                             "2025-07-04 Water sports|", "2025-07-04 Food & drink|"]
    assert where == [2, 1, 0, 4, 3]
    # Split by person, and a block switched off.
    split = themes.blocks_for(rows, [], {"Water sports"})
    assert {b["key"] for b in split} >= {"Water sports|Laszlo", "Water sports|Gosha"}
    off = [b | {"on": b["activity"] != "Boat"} for b in blocks]
    _, where = themes.chapters_for(line | {"order": themes.THEME, "blocks": off, "finale": False}, rows)
    assert where[0] is None


def test_a_first_guess_from_tags():
    assert themes.by_tags(["Eating & drinking"], None) == "Food & drink"
    assert themes.by_tags([], "Sea & boats") == "Boat"
    assert themes.by_tags(["Daylight"], None) == themes.OTHER


def test_my_story_puts_the_themes_in_its_parts():
    rows = [_row(n, "Water sports", people=["Laszlo"]) for n in range(6)] + \
        [_row(10 + n, "Boat") for n in range(4)] + [_row(20 + n, "Food & drink") for n in range(2)]
    for n, r in enumerate(rows[:6]):
        r |= {"kind": "video", "motion_action": 0.9 if n in (1, 4, 5) else 0.1}
    for r in rows[6:]:
        r["kind"] = "photo"
    story = {"parts": [{"name": "Calm start", "pace": "calm", "entries": [{"key": "Boat|", "pick": "all"}]},
                       {"name": "Warm up", "pace": "medium",
                        "entries": [{"key": "Water sports|", "pick": "early"}]},
                       {"name": "Big jumps", "pace": "lively",
                        "entries": [{"key": "Water sports|", "pick": "action"}]}]}
    chapters, where = themes.chapters_for({"order": themes.STORY, "story": story, "finale": False}, rows)
    assert [c["pace"] for c in chapters] == [0.2, 0.55, 0.9]
    assert where[6:10] == [0] * 4  # the boat opens
    assert where[10:] == [None, None]  # food isn't in the story
    assert [where[i] for i in range(6)] == [1, 1, 1, None, 2, 2]  # first half, then the exciting rest
    # No story made yet: as it happened.
    chapters, where = themes.chapters_for({"order": themes.STORY, "finale": False}, rows)
    assert len(chapters) == 1 and set(where) == {0}


def test_without_an_ai_the_story_follows_the_themes_you_name():
    blocks = [{"key": "Water sports|Laszlo", "activity": "Water sports", "person": "Laszlo"},
              {"key": "Water sports|Anna", "activity": "Water sports", "person": "Anna"},
              {"key": "Boat|", "activity": "Boat", "person": None},
              {"key": "Food & drink|", "activity": "Food & drink", "person": None},
              {"key": "Beach & sea|", "activity": "Beach & sea", "person": None}]
    story = themes.story_from_words("Start with the boat, then everything else, and finish with Laszlo kitesurfing",
                                    blocks)
    keys = [[e["key"] for e in p["entries"]] for p in story["parts"]]
    assert keys == [["Boat|"], ["Water sports|Anna", "Food & drink|", "Beach & sea|"], ["Water sports|Laszlo"]]
    assert story["parts"][0]["pace"] == "calm" and story["parts"][-1]["pace"] == "lively"
    assert not story["finale"]
    assert themes.left_out(themes.story_from_words("food then the boat", blocks), blocks)[0]["key"] == \
        "Water sports|Laszlo"


def test_an_ai_story_is_cleaned_and_keyed():
    from trip_memory import director

    blocks = [{"key": "Boat|"}, {"key": "Food & drink|"}]
    answer = {"title": " Sea  days ", "finale": True, "parts": [
        {"name": "Open", "pace": "fast", "themes": [{"theme": 1, "pick": "all"}, {"theme": 9, "pick": "all"}]},
        {"name": "", "pace": "calm", "themes": [{"theme": 2, "pick": "late"}, {"theme": 1, "pick": "all"}]},
        {"name": "Empty", "pace": "calm", "themes": []}]}
    story = themes.story_keys(director.clean_story(answer, 2), blocks)
    assert story["title"] == "Sea days" and story["finale"]
    assert story["parts"] == [
        {"name": "Open", "pace": "medium", "entries": [{"key": "Boat|", "pick": "all"}]},
        {"name": "Part 2", "pace": "calm", "entries": [{"key": "Food & drink|", "pick": "late"}]}]
    assert "1. Boat" in director.story_prompt("boat first", [{"name": "Boat", "count": 4}])

