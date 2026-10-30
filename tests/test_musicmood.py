"""Reading the music: calm and lively stretches lined up with your pins, the song's style, and your changes."""

from trip_memory import musicmood, taps
from trip_memory.songmap import Section, SongMap


def _song():
    return SongMap(duration=60.0, bpm=120, period=0.5, first_beat=0.0, clear_beat=True, downbeat=0,
                   sections=[Section(0, 20, 0.2, "Calm"), Section(20, 24, 0.25, "Calm"),
                             Section(24, 50, 0.8, "Lively"), Section(50, 60, 0.3, "Calm")])


def test_stretches_line_up_with_your_pins():
    stretches = musicmood.bands(_song(), [10.0, 23.0, 51.5])
    assert [(b["start"], b["end"], b["mood"]) for b in stretches] == [
        (0.0, 23.0, "Calm"), (23.0, 51.5, "Lively"), (51.5, 60.0, "Calm")]  # the changes moved onto the pins
    assert musicmood.suggest_style(stretches) == "discovery"
    assert musicmood.suggest_style([{"start": 0, "end": 60, "mood": "Lively"}]) == "action"
    assert musicmood.suggest_style([{"start": 0, "end": 60, "mood": "Calm"}]) == "cinematic"


def test_your_moods_and_style_are_kept_and_used(tmp_path):
    song_path = tmp_path / "song.mp3"
    song_path.write_bytes(b"x")
    folder = tmp_path / "taps"
    taps.save(str(song_path), folder, [10.0, 23.0, 51.5], final=True)
    song = _song()
    reading = musicmood.reading(song, str(song_path), folder)
    assert not reading["mine"] and reading["style"] == "discovery"
    mine = [{"start": 0.0, "end": 23.0, "mood": "Lively"}, {"start": 23.0, "end": 60.0, "mood": "Calm"}]
    taps.set_extra(str(song_path), folder, "moods", mine)
    taps.set_extra(str(song_path), folder, "style", "cinematic")
    taps.save(str(song_path), folder, [10.0, 23.0, 51.5, 52.0], final=True)  # pinning later keeps them
    reading = musicmood.reading(song, str(song_path), folder)
    assert reading["mine"] and reading["style"] == "cinematic" and reading["style_mine"]
    musicmood.apply(song, reading["stretches"])
    assert song.energy_at(5) == musicmood.MOODS["Lively"] and song.sections[1].level == "Calm"
    assert musicmood.thin([1.0, 2.0, 4.5, 5.0, 8.0], 3.0) == [1.0, 4.5, 8.0]
    assert musicmood.words(reading["stretches"]) == "0:00-0:23 lively, 0:23-1:00 calm"
