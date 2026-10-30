"""Moments: finding a particular thing (a big jump...) inside the shots of a theme."""

import numpy as np

from trip_memory import moments


def _unit(v):
    v = np.asarray(v, np.float32)
    return v / np.linalg.norm(v)


def test_the_best_second_inside_each_clip():
    jump = _unit([1, 0, 0])
    water = _unit([0, 1, 0])
    shots = {
        "a.mp4": ([0.5, 1.5, 2.5, 3.5, 4.5], np.array([water, water, _unit([0.9, 0.3, 0]), jump, water])),
        "b.mp4": ([0.5, 1.5, 2.5], np.array([water, water, water])),
        "c.mp4": ([0.5, 1.5, 2.5], np.array([water, _unit([0.6, 0.8, 0]), water])),
    }
    found = moments.find(jump, shots)
    assert [p for p, _, _ in found] == ["a.mp4", "c.mp4", "b.mp4"]
    assert found[0][1] == 3.5
    assert moments.likely(found) == {"a.mp4"}
    # Your ticks count: a shot like the one you ticked comes up.
    other = _unit([0, 0, 1])
    shots["d.mp4"] = ([0.5, 1.5], np.array([other, other]))
    assert moments.find(jump, shots, [other])[0][0] == "d.mp4"


def test_frames_and_the_kept_window():
    assert moments.frame_times(4) == [0.5, 1.5, 2.5, 3.5]
    assert len(moments.frame_times(600)) == moments.MOST
    assert moments.window(1.0, 20) == (0.0, 3.0)
    assert moments.window(10.0, 20) == (8.5, 11.5)
    assert moments.window(19.5, 20) == (17.0, 20.0)
    assert moments.window(5.0, 20, start=6.0, end=12.0) == (6.0, 9.0)  # inside the part you kept


def test_key_frames_are_used_when_close_enough():
    assert moments.pick_frames([0.0, 1.0, 2.0, 3.0], 4) == [0, 1, 2, 3]
    assert moments.pick_frames([0.0, 8.3], 12) == []  # too far apart: the app reads frames itself
    assert len(moments.pick_frames([k * 0.5 for k in range(40)], 20)) == 20  # about one a second


def test_ticks_from_different_sides_each_find_their_look_alikes():
    rng = np.random.default_rng(1)
    side_a, side_b = _unit([1, 0, 0, 0, 0, 0]), _unit([0, 1, 0, 0, 0, 0])
    other = [_unit(rng.normal(size=6) * [0.1, 0.1, 1, 1, 1, 1]) for _ in range(8)]
    shots = {f"a{k}": ([1.0], np.array([_unit(side_a + rng.normal(size=6) * 0.15)])) for k in range(3)}
    shots |= {f"b{k}": ([1.0], np.array([_unit(side_b + rng.normal(size=6) * 0.15)])) for k in range(3)}
    shots |= {f"o{k}": ([1.0], np.array([v])) for k, v in enumerate(other)}
    found = moments.find(None, shots, [shots["a0"][1][0], shots["b0"][1][0]])
    assert {p for p, _, _ in found[:6]} == {"a0", "a1", "a2", "b0", "b1", "b2"}
    assert {"a1", "a2", "b1", "b2"} <= moments.likely(found)
