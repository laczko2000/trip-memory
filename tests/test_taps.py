"""Your own cuts: taps are moved onto the hit you meant and kept per song."""

import numpy as np

from trip_memory import taps
from trip_memory.beats import RATE


def test_taps_snap_onto_the_hits_and_are_saved(tmp_path):
    x = np.zeros(int(6 * RATE), np.float32)
    t = np.arange(int(0.08 * RATE)) / RATE
    kick = (np.sin(2 * np.pi * 60 * t) * np.exp(-t * 30)).astype(np.float32)
    for hit in (1.0, 2.5, 4.0):
        i = int(hit * RATE)
        x[i : i + len(kick)] += kick
    snapped = taps.snap(x, [1.09, 2.46, 3.5])  # a late tap, an early tap, and one with no hit near it
    assert abs(snapped[0] - 1.0) < 0.03 and abs(snapped[1] - 2.5) < 0.03
    assert snapped[2] == 3.5
    song = tmp_path / "song.mp3"
    song.write_bytes(b"x")
    assert taps.final_taps(str(song), tmp_path / "taps") is None  # not tapped yet
    taps.save(str(song), tmp_path / "taps", [2.5, 1.0], final=False)  # older "not final yet" pins count too
    assert taps.final_taps(str(song), tmp_path / "taps") == [1.0, 2.5]
    assert taps.shot_lengths([1.0, 2.5], 4.0) == [1.0, 1.5, 1.5]


def test_your_usual_delay_is_learned_and_taken_off():
    rng = np.random.default_rng(0)
    x = rng.normal(0, 0.005, int(20 * RATE)).astype(np.float32)
    t = np.arange(int(0.08 * RATE)) / RATE
    kick = (np.sin(2 * np.pi * 60 * t) * np.exp(-t * 30)).astype(np.float32)
    hat = (np.diff(rng.normal(0, 0.1, len(t) + 1)) * np.exp(-t * 200)).astype(np.float32)  # bright, like a hi-hat
    beats = np.arange(0.5, 19, 0.5)
    for b in beats:
        x[int(b * RATE) : int(b * RATE) + len(kick)] += kick
        x[int((b + 0.25) * RATE) : int((b + 0.25) * RATE) + len(t)] += hat  # off-beat hi-hats, to be fooled by
    wanted = beats[1::3]
    late = list(wanted + 0.22 + rng.normal(0, 0.03, len(wanted)))  # e.g. Bluetooth headphones, plus wobble
    assert abs(taps.usual_delay(x, late) - 0.22) < 0.03
    snapped = taps.snap(x, late, np.arange(0.5, 19, 0.25))
    assert max(abs(s - w) for s, w in zip(snapped, wanted)) < 0.02


def test_trim_the_song_and_keep_its_pins(tmp_path):
    import subprocess

    from trip_memory.beats import song_length
    from trip_memory.tools import ffmpeg_paths

    song = tmp_path / "song.wav"
    subprocess.run([ffmpeg_paths()[0], "-loglevel", "error", "-f", "lavfi", "-i", "sine=frequency=440:duration=20",
                    str(song)], check=True)
    folder = tmp_path / "taps"
    taps.save(str(song), folder, [3.0, 8.0, 12.5, 19.0], final=True)
    part = taps.trim(str(song), 5.0, 15.0, tmp_path / "trimmed", folder)
    assert abs(song_length(part) - 10.0) < 0.15
    assert taps.final_taps(part, folder) == [3.0, 7.5]  # the pins inside the part, moved with it
    assert taps.trimmed_from(part, folder) == (str(song), 5.0, 15.0)
    taps.save(part, folder, [3.0, 7.5, 9.0], final=True)  # pinning the track later keeps where it came from
    assert taps.trimmed_from(part, folder) == (str(song), 5.0, 15.0)
    assert taps.trimmed_from(str(song), folder) is None
    sound = taps.read_song(part)
    assert np.abs(sound[:200]).max() < 0.2 and np.abs(sound[int(5 * RATE):int(5.1 * RATE)]).max() > 0.1  # fades in


def test_trim_with_the_fades_you_chose(tmp_path):
    import subprocess

    from trip_memory.tools import ffmpeg_paths

    song = tmp_path / "song.wav"
    subprocess.run([ffmpeg_paths()[0], "-loglevel", "error", "-f", "lavfi", "-i", "sine=frequency=440:duration=20",
                    str(song)], check=True)
    folder = tmp_path / "taps"
    part = taps.trim(str(song), 0.0, 20.0, tmp_path / "trimmed", folder, fade_in=3.0, fade_out=4.0)
    assert taps.fades_of(part, folder) == (3.0, 4.0)
    sound = np.abs(taps.read_song(part))
    level = lambda a, b: sound[int(a * RATE):int(b * RATE)].max()
    assert level(0.5, 0.7) < 0.5 * level(8, 9)  # still quiet early in the 3 s fade in
    assert level(18.0, 18.2) < 0.6 * level(8, 9) and level(19.6, 19.8) < 0.2 * level(8, 9)  # fading out over 4 s
    assert taps.fades_of(str(song), folder) is None
