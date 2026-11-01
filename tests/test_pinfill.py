"""Finish my pins: the app learns how you pinned the start of a song and suggests pins for the rest."""

import numpy as np

from trip_memory import pinfill
from trip_memory.songmap import Section, SongMap


def _song():
    beats = int(80 / 0.5)
    accents = [1.0 if k % 4 == 2 else 0.2 for k in range(beats)]  # the third beat of each bar hits hard
    return SongMap(duration=80.0, bpm=120, period=0.5, first_beat=0.0, clear_beat=True, downbeat=0, accents=accents,
                   sections=[Section(0, 40, 0.2, "Calm"), Section(40, 80, 0.8, "Lively")])


STRETCHES = [{"start": 0.0, "end": 40.0, "mood": "Calm"}, {"start": 40.0, "end": 80.0, "mood": "Lively"}]


def test_too_few_pins_teach_nothing():
    assert pinfill.suggest(_song(), [1.0, 3.0], STRETCHES) == []


def test_the_rest_follows_your_way_of_pinning():
    # Calm part: every 2 bars on the third beat; the lively part: every bar, and a pin where the music changes.
    pins = [1.0 + 4.0 * k for k in range(9)] + [40.0, 41.0, 43.0, 45.0]
    suggested = pinfill.suggest(_song(), pins, STRETCHES)
    assert suggested and suggested[0] > 45.0 and suggested[-1] < 80.0
    gaps = np.diff([45.0, *suggested])
    assert np.allclose(gaps, 2.0)  # the lively spacing you used: one bar
    assert all(round(p / 0.5) % 4 == 2 for p in suggested)  # on the beat you like


def test_without_a_steady_beat_only_the_spacing_counts():
    song = _song()
    song.clear_beat = False
    suggested = pinfill.suggest(song, [0.7, 3.7, 6.7, 9.7], [{"start": 0.0, "end": 80.0, "mood": "Calm"}], until=20.0)
    assert suggested == [12.7, 15.7, 18.7]


def test_your_cut_style_carries_to_a_new_song(tmp_path):
    from trip_memory import pinfill
    from trip_memory.songmap import Section, SongMap

    song = SongMap(duration=60.0, bpm=120, period=0.5, first_beat=0.0, clear_beat=True, downbeat=0,
                   sections=[Section(0, 60, 0.5, "Medium")], accents=[0.5] * 120)
    stretches = [{"start": 0.0, "end": 60.0, "mood": "Medium"}]
    pins = [2.0 * k for k in range(1, 15)]  # every 4 beats, on the bar
    file = tmp_path / "learned.json"
    assert not pinfill.remember(file, "a.mp3", song, pins[:5], stretches)  # (too few to learn from)
    assert pinfill.my_style(file) is None
    assert pinfill.remember(file, "a.mp3", song, pins, stretches)
    style = pinfill.my_style(file)
    assert style["songs"] == 1 and abs(style["gap"] - 4) < 0.01
    new = pinfill.suggest_my_way(song, style, stretches)
    gaps = [b - a for a, b in zip(new, new[1:])]
    assert len(new) > 20 and all(abs(g - 2.0) < 0.6 for g in gaps)
    later = pinfill.suggest_my_way(song, style, stretches, pins=[10.0])
    assert later[0] > 10.0
