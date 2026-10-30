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
