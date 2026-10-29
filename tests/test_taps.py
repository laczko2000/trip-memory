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
    taps.save(str(song), tmp_path / "taps", [2.5, 1.0], final=False)
    assert taps.final_taps(str(song), tmp_path / "taps") is None
    taps.save(str(song), tmp_path / "taps", [2.5, 1.0], final=True)
    assert taps.final_taps(str(song), tmp_path / "taps") == [1.0, 2.5]
    assert taps.shot_lengths([1.0, 2.5], 4.0) == [1.0, 1.5, 1.5]
