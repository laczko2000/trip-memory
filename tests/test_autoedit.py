"""Auto-edit: picks shots up to a length, finds the best moment, and lands the cuts on the beat."""

import numpy as np

from trip_memory import autoedit
from trip_memory.beats import Beat


def _row(n, kind="video", duration=12.0, story=0, similar=None, best=False, tags=()):
    return {
        "path": f"/trip/{n:02d}.mov", "kind": kind, "duration": duration if kind == "video" else 0,
        "taken_at": f"2025-07-03T10:{n:02d}:00+00:00", "story": story, "similar_set": similar if similar is not None else n,
        "best": best, "tags_final": list(tags), "people_tag": None, "people": [], "sharpness": 100.0,
    }


def test_best_moment_is_found():
    score = np.zeros(60)  # 12 s at five steps a second
    score[35:45] = 1.0  # the action is at 7-9 s
    start = autoedit.best_start(score, 12.0, 3.0)
    assert 6.0 <= start <= 7.0


def test_cuts_land_on_the_beat_and_length_is_respected():
    rows = [_row(n, story=n // 4) for n in range(16)] + [_row(20 + n, kind="photo", story=5) for n in range(4)]
    beat = Beat(period=0.5, first=0.12, duration=180, clear=True)
    plan = autoedit.plan(rows, {}, target=30.0, transition_seconds=0.8, beat=beat)
    assert 27 <= plan.length <= 40
    assert {s.path for s in plan.shots}.isdisjoint(plan.left_out)
    assert len({r["story"] for r in rows if r["path"] in {s.path for s in plan.shots}}) == 5  # every story
    begin = 0.0
    for shot in plan.shots[:-1]:
        t = 1 / 30 if shot.transition == autoedit.HARD_CUT else min(0.8, 0.4 * shot.seconds)
        middle = begin + shot.seconds - t / 2
        assert abs((middle - beat.first) / beat.period - round((middle - beat.first) / beat.period)) * beat.period < 0.05
        begin += shot.seconds - t
        assert shot.seconds >= 2.0


def test_one_shot_per_set_of_look_alikes():
    rows = [_row(0, similar=7, best=False), _row(1, similar=7, best=True), _row(2, similar=7)]
    plan = autoedit.plan(rows, {}, target=None, transition_seconds=0.8, beat=None)
    assert [s.path for s in plan.shots] == ["/trip/01.mov"]


def test_lively_neighbours_get_hard_cuts():
    rows = [_row(0, tags=["Fast action"]), _row(1, tags=["Lively"]), _row(2, tags=["Calm"])]
    plan = autoedit.plan(rows, {}, target=None, transition_seconds=0.8, beat=None)
    assert [s.transition for s in plan.shots] == [autoedit.HARD_CUT, None, None]


def test_day_titles_are_counted_so_cuts_stay_on_the_beat():
    rows = [_row(n) for n in range(4)]
    rows[2]["taken_at"] = rows[3]["taken_at"] = "2025-07-04T10:00:00+00:00"
    beat = Beat(period=0.5, first=0.1, duration=60, clear=True)
    plan = autoedit.plan(rows, {}, target=None, transition_seconds=0.8, beat=beat, title_seconds=3.0)
    begin = 3.0 - 0.8  # the first day's title card
    for n, shot in enumerate(plan.shots[:-1]):
        middle = begin + shot.seconds - 0.4
        assert abs(((middle - 0.1) / 0.5) - round((middle - 0.1) / 0.5)) * 0.5 < 0.05
        begin += shot.seconds - 0.8
        if n == 1:
            begin += 3.0 - 0.8  # the second day's title card
