import numpy as np

from trip_memory import packages, titles
from trip_memory.render import package_fx
from trip_memory.transitions import HARD_CUT, Cut


def test_every_package_gives_every_cut_a_transition_from_its_own_sets():
    cuts = [Cut(before={}, after={}, energy_before=e, energy_after=e, hit=0.0, room=1.0)
            for e in (0.2, 0.8, 0.3, 0.9, 0.2, 0.7)]
    for key, p in packages.PACKAGES.items():
        chosen = packages.transitions_for(key, cuts)
        assert len(chosen) == len(cuts)
        assert all(name == HARD_CUT or name in p["calm"] + p["lively"] for name, _ in chosen), key


def test_place_titles_start_where_a_new_place_lasts():
    board = [{"place": p} for p in ("A", "A", "B", "B", "B", "C", "D", "D", "D", "D")]
    assert packages.place_changes(board, least_gap=2) == [2, 6]  # (C lasts one shot only)


def test_flashes_only_on_big_hits_in_lively_parts():
    from trip_memory.songmap import Section
    beats = np.arange(0, 30, 0.5)
    accents = np.where(np.arange(len(beats)) % 8 == 0, 0.9, 0.2)
    found = packages.flash_times(beats, accents, [Section(0, 10, 0.3, "Calm"), Section(10, 30, 0.8, "Lively")], 30)
    assert found and all(10 <= t < 28 for t in found) and all(b - a >= 1 for a, b in zip(found, found[1:]))


def test_titles_come_on_their_own_way():
    layer = np.zeros((10, 100, 4), np.uint8)
    layer[2:8, 10:90, 3] = 255
    title = {"text": "Kyoto", "anim": "reveal"}
    half = titles.anim_layer(title, layer, titles.anim_length(title) / 2)
    assert half[5, 15, 3] == 255 and half[5, 85, 3] == 0  # left to right
    rl = titles.anim_layer({"text": "Kyoto", "anim": "flicker-rl"}, layer, 0.45)
    assert rl[5, 85, 3] > 0 and rl[5, 15, 3] == 0  # right to left
    assert (titles.anim_layer(title, layer, 5.0) == layer).all()
    assert titles.anim_layer({"text": "x"}, layer, 0.0) is layer


def test_package_effects_filter():
    assert package_fx({}, [1.0], 10) == ""
    graph = package_fx({"flash": [2.0], "hue": True, "glitch": True}, [1.0, 3.0, 5.0], 10)
    assert "eq=brightness" in graph and "hue=h=" in graph and "noise" in graph
