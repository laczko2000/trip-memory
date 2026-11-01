from trip_memory import longer
from trip_memory.songmap import Section, SongMap


def _song(duration=90.0):
    return SongMap(duration=duration, bpm=120, period=0.5, first_beat=0.0, clear_beat=True, downbeat=0,
                   bar_energy=[], sections=[Section(0, 12, 0.4, "Calm"), Section(12, 36, 0.8, "Lively"),
                                            Section(36, 70, 0.6, "Medium"), Section(70, 90, 0.4, "Calm")])


def test_a_song_made_longer_keeps_its_intro_and_ending_and_joins_on_bars():
    song = _song()
    pieces = longer.plan(song, 150.0)
    assert pieces[0][0] == 0.0 and pieces[-1][1] == 90.0  # starts at the start, ends with the ending
    total = sum(b - a for a, b in pieces)
    assert abs(total - 150.0) <= 4.0  # (whole bars: about the length you asked for)
    for (a, b), (c, d) in zip(pieces, pieces[1:]):
        assert c < b  # each join goes back to play a part again
        assert c >= 90 * longer.EDGE_SHARE - 2 and b <= 90 * (1 - longer.EDGE_SHARE) + 2  # never the intro/ending
        assert abs(b * 2 - round(b * 2)) < 1e-6 and abs(c * 2 - round(c * 2)) < 1e-6  # on the beat grid (bars)
    assert longer.plan(song, 91.0) == [(0.0, 90.0)]  # not longer: unchanged
