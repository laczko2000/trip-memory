"""The song map: finds the calm and lively parts of a song and where it changes."""

import numpy as np

from trip_memory.beats import RATE
from trip_memory.songmap import map_of


def _song(bpm=120, offset=0.2, parts=(("calm", 16), ("lively", 16), ("calm", 8))):
    rng = np.random.default_rng(0)
    period = 60 / bpm
    total = offset + sum(bars for _, bars in parts) * 4 * period + 1
    x = 0.005 * rng.standard_normal(int(total * RATE)).astype(np.float32)
    t = np.arange(int(0.08 * RATE)) / RATE
    kick = (np.sin(2 * np.pi * 55 * t) * np.exp(-t * 30)).astype(np.float32)
    hat = (rng.standard_normal(len(t)) * np.exp(-t * 150)).astype(np.float32)
    when = offset
    for kind, bars in parts:
        for k in range(bars * 4):
            i = int(when * RATE)
            if kind == "calm":
                x[i : i + len(t)] += (0.4 if k % 4 == 0 else 0.15) * kick
                n = len(x[i : i + int(period * RATE)])
                x[i : i + n] += (0.05 * np.sin(2 * np.pi * 220 * np.arange(n) / RATE)).astype(np.float32)
            else:
                x[i : i + len(t)] += (1.0 if k % 4 == 0 else 0.7) * kick + 0.3 * hat
                j = int((when + period / 2) * RATE)
                x[j : j + len(t)] += 0.3 * hat
            when += period
    return x, total


def test_finds_calm_and_lively_parts():
    samples, total = _song()
    song = map_of(samples, total)
    assert abs(song.bpm - 120) < 1
    levels = [(round(s.start), s.level) for s in song.sections]
    assert levels[0][1] == "Calm"
    assert any(abs(start - 32) <= 1 and level == "Lively" for start, level in levels)
    assert any(abs(start - 64) <= 1 and level == "Calm" for start, level in levels)


def test_finds_the_strong_beats():
    samples, total = _song()
    song = map_of(samples, total)
    beats = song.beat_times()
    assert len(song.accents) == len(beats) == len(song.bass) == len(song.highs)
    accents = np.array(song.accents)
    # In the calm part the first beat of each bar has the big kick; the others are soft.
    calm = [(k, t) for k, t in enumerate(beats) if 2 < t < 30]
    on_one = [accents[k] for k, t in calm if round((t - 0.2) / 0.5) % 4 == 0]
    others = [accents[k] for k, t in calm if round((t - 0.2) / 0.5) % 4 != 0]
    assert np.mean(on_one) > np.mean(others) + 0.2
