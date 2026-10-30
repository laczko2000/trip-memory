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
