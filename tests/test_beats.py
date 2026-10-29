"""The beat finder on made-up drum loops."""

import numpy as np

from trip_memory.beats import RATE, beat_of


def _song(bpm, offset, seconds=30, seed=0):
    rng = np.random.default_rng(seed)
    x = 0.02 * rng.standard_normal(RATE * (seconds + 1)).astype(np.float32)
    period, t = 60 / bpm, np.arange(int(0.05 * RATE)) / RATE
    kick = (np.sin(2 * np.pi * 60 * t) * np.exp(-t * 40)).astype(np.float32)
    hat = (rng.standard_normal(len(t)) * np.exp(-t * 200) * 0.3).astype(np.float32)
    for k, when in enumerate(np.arange(offset, seconds, period)):
        i = int(when * RATE)
        x[i : i + len(t)] += kick if k % 2 == 0 else hat * 2 + kick * 0.5
        j = int((when + period / 2) * RATE)
        x[j : j + len(t)] += hat * 0.5
    return x[: RATE * seconds]


def test_finds_tempo_and_first_beat():
    for bpm, offset in [(128, 0.3), (95, 0.1), (110, 0.0)]:
        beat = beat_of(_song(bpm, offset), 30)
        assert beat.clear
        assert abs(beat.bpm - bpm) < 1.5 or abs(beat.bpm - 2 * bpm) < 3
        phase = (beat.first - offset) / beat.period
        assert abs(phase - round(phase)) * beat.period < 0.06


def test_noise_has_no_clear_beat():
    assert not beat_of(0.1 * np.random.default_rng(1).standard_normal(RATE * 30).astype(np.float32), 30).clear
