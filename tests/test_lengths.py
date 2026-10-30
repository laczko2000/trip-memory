"""The app reads the length you want from your own words."""

import pytest

from trip_memory.lengths import as_words, read_length


@pytest.mark.parametrize("text, wanted", [
    ("a 1:20 video for Instagram", {"seconds": 80}),
    ("80 seconds of the beach", {"seconds": 80}),
    ("1 min 20 please", {"seconds": 80}),
    ("2 minutes and 30 seconds", {"seconds": 150}),
    ("two and a half minutes, lively", {"seconds": 150}),
    ("a minute and a half", {"seconds": 90}),
    ("1.5 minutes", {"seconds": 90}),
    ("a 3-minute film", {"seconds": 180}),
    ("three minutes", {"seconds": 180}),
    ("half a minute teaser", {"seconds": 30}),
    ("use the whole song", {"song": 1.0}),
    ("as long as the song", {"song": 1.0}),
    ("half the song, calm", {"song": 0.5}),
    ("half of the music", {"song": 0.5}),
    ("the first 2 minutes of the song", {"seconds": 120}),
    ("beach and kids, energetic", None),
])
def test_lengths_are_read_from_your_words(text, wanted):
    assert read_length(text) == wanted


def test_an_unclear_length_is_asked_about():
    assert read_length("a 1.2 minute video") == {"ask": [72, 80], "said": "1.2 minute"}
    assert read_length("1.20 minutes")["ask"] == [72, 80]
    assert read_length("2,3 min")["ask"] == [138, 150]  # 2.3 minutes, or 2:30


def test_lengths_in_words():
    assert as_words({"seconds": 80}) == "1:20" and as_words({"song": 0.5}) == "half the song"
