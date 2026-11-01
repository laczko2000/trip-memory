"""Find the beat of a song, so cuts and photo changes can land on it.

Listens for moments where the sound suddenly gets stronger (drums, plucks), finds the steady rhythm in
them (the tempo) and where the first beat falls. Good enough for pop, dance and most holiday music;
songs without a clear beat get a calm, steady pace instead.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass

import numpy as np

from .tools import ffmpeg_paths

RATE = 11025
HOP = 256  # about 23 ms per step
WINDOW = 1024
MIN_BPM, MAX_BPM = 50, 180


@dataclass
class Beat:
    period: float  # seconds between beats
    first: float  # time of the first beat
    duration: float  # length of the song
    clear: bool  # whether the song has a clear beat

    @property
    def bpm(self) -> float:
        return 60 / self.period

    def times(self, until: float) -> np.ndarray:
        return np.arange(self.first, until, self.period)


def read_song(path: str, seconds: float = 240) -> np.ndarray:
    raw = subprocess.run(
        [ffmpeg_paths()[0], "-loglevel", "error", "-i", path, "-t", str(seconds), "-ac", "1", "-ar", str(RATE),
         "-f", "s16le", "-"],
        capture_output=True,
    ).stdout
    return np.frombuffer(raw, np.int16).astype(np.float32) / 32768


def song_length(path: str) -> float:
    from .tools import probe

    return float(probe(path)["format"]["duration"])


def find_beat(path: str) -> Beat:
    return beat_of(read_song(path), song_length(path))


def beat_of(samples: np.ndarray, duration: float) -> Beat:
    steady = Beat(period=60 / 90, first=0.0, duration=duration, clear=False)
    if len(samples) < RATE * 4:
        return steady
    onset = _onset_strength(samples)
    step = HOP / RATE
    # Tempo: the beat period at which the onsets repeat most (autocorrelation), with a mild preference
    # for tempos around 120 BPM so we don't pick double or half speed.
    onset_c = onset - onset.mean()
    ac = np.correlate(onset_c, onset_c, mode="full")[len(onset_c) - 1 :]
    lags = np.arange(int(np.ceil(60 / MAX_BPM / step)), int(60 / MIN_BPM / step) + 1)
    if lags[-1] * 4 >= len(ac) or ac[0] <= 0:
        return steady
    ac = ac / ac[0]
    bpms = 60 / (lags * step)
    weight = np.exp(-0.5 * (np.log2(bpms / 120) / 0.9) ** 2)
    # A real beat also repeats at 2, 3 and 4 beats; a fraction of it (e.g. 1.5 beats) does not.
    harmonics = sum(ac[lags * k] / k for k in (1, 2, 3, 4)) / sum(1 / k for k in (1, 2, 3, 4))
    scores = np.minimum(ac[lags], harmonics) * weight
    best = int(np.argmax(scores))
    lag = _refine(ac, lags[best])
    # Songs often repeat most strongly every 2 beats (kick, snare, kick, snare); if the halfway point
    # is a beat too, use it, so there are more beats to cut on.
    half = int(round(lag / 2))
    if 60 / (half * step) <= MAX_BPM and ac[half - 1 : half + 2].max() >= 0.15:
        lag = _refine(ac, half - 1 + int(np.argmax(ac[half - 1 : half + 2])))
    if not np.isfinite(lag) or lag < 2:  # (a song with no steady pulse at all: an even grid)
        return steady
    period = lag * step
    # Phase: where to put the beat grid so it sits on the most onsets.
    phases = np.arange(int(lag))
    grid_sums = [onset[np.round(np.arange(p, len(onset) - 1, lag)).astype(int)].sum() for p in phases]
    first = float(phases[int(np.argmax(grid_sums))] * step) + WINDOW / 2 / RATE  # a step is timed by its window's start
    first %= period
    clear = float(ac[lags[best]]) > 0.2
    return Beat(period=period, first=first, duration=duration, clear=clear)


def _onset_strength(samples: np.ndarray) -> np.ndarray:
    window = np.hanning(WINDOW).astype(np.float32)
    frames = np.lib.stride_tricks.sliding_window_view(samples, WINDOW)[::HOP] * window
    spec = np.log1p(10 * np.abs(np.fft.rfft(frames, axis=1)))
    flux = np.maximum(0, np.diff(spec, axis=0)).sum(axis=1)
    flux = np.convolve(flux, np.ones(3) / 3, mode="same")
    return np.concatenate([[0], flux])


def _refine(ac: np.ndarray, lag: int) -> float:
    """Sub-step accuracy for the beat period (a parabola through the peak)."""
    if 0 < lag < len(ac) - 1:
        a, b, c = ac[lag - 1], ac[lag], ac[lag + 1]
        denom = a - 2 * b + c
        if denom < 0:
            return lag + 0.5 * (a - c) / denom
    return float(lag)
