"""A map of a song: its beats, bars, how energetic each bar is, and where the music changes.

This is what the music-driven Auto-edit cuts to. Calm parts get longer shots, lively parts quick ones,
and every big change in the music (intro → verse → chorus...) gets a cut.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from .beats import HOP, RATE, WINDOW, Beat, beat_of, read_song, song_length

BEATS_PER_BAR = 4
MIN_SECTION_BARS = 4
LEVELS = ["Calm", "Medium", "Lively"]


@dataclass
class Section:
    start: float  # seconds
    end: float
    energy: float  # 0 (quiet) .. 1 (the song's loudest, busiest part)
    level: str  # Calm, Medium or Lively


@dataclass
class SongMap:
    duration: float
    bpm: float
    period: float  # seconds per beat
    first_beat: float
    clear_beat: bool
    downbeat: int  # which of the first 4 beats starts a bar
    bar_energy: list[float] = field(default_factory=list)  # 0..1 for each bar
    sections: list[Section] = field(default_factory=list)
    # For each beat, 0..1: how hard it hits overall, in the bass (kick, bass notes) and in the highs (cymbals, claps).
    accents: list[float] = field(default_factory=list)
    bass: list[float] = field(default_factory=list)
    highs: list[float] = field(default_factory=list)

    @property
    def beat(self) -> Beat:
        return Beat(period=self.period, first=self.first_beat, duration=self.duration, clear=self.clear_beat)

    def beat_times(self) -> np.ndarray:
        return np.arange(self.first_beat, self.duration, self.period)

    def bar_starts(self) -> np.ndarray:
        return self.beat_times()[self.downbeat :: BEATS_PER_BAR]

    def energy_at(self, t: float) -> float:
        for s in self.sections:
            if s.start <= t < s.end:
                return s.energy
        return self.sections[-1].energy if self.sections else 0.5


def analyse(path: str, cache_dir: Path | None = None) -> SongMap:
    """Study a song once (the result is remembered)."""
    stat = Path(path).stat()
    key = hashlib.sha1(f"{path}|{stat.st_mtime}|{stat.st_size}|4".encode()).hexdigest()[:16]
    cached = cache_dir / f"{key}.json" if cache_dir else None
    if cached and cached.exists():
        data = json.loads(cached.read_text())
        data["sections"] = [Section(**s) for s in data["sections"]]
        return SongMap(**data)
    result = map_of(read_song(path, seconds=900), song_length(path))
    if cached:
        cached.parent.mkdir(parents=True, exist_ok=True)
        cached.write_text(json.dumps(asdict(result)))
    return result


def map_of(samples: np.ndarray, duration: float) -> SongMap:
    beat = beat_of(samples, duration)
    beats = np.arange(beat.first, duration, beat.period)
    # Loudness and "busyness" (how much is going on) between one beat and the next.
    feats, loud, busy = _beat_features(samples, beats)
    accents, bass, highs = _accents(samples, beats)
    downbeat = _downbeat(samples, beats)
    bars = [slice(i, i + BEATS_PER_BAR) for i in range(downbeat, len(beats) - 1, BEATS_PER_BAR)]
    if not bars:
        return SongMap(duration, beat.bpm, beat.period, beat.first, beat.clear, 0, [0.5],
                       [Section(0.0, duration, 0.5, "Medium")])
    bar_loud = np.array([loud[b].mean() if len(loud[b]) else 0 for b in bars])
    bar_busy = np.array([busy[b].mean() if len(busy[b]) else 0 for b in bars])
    energy = 0.5 * _norm(bar_loud) + 0.5 * _norm(bar_busy)
    bar_feats = np.array([feats[b].mean(axis=0) if len(feats[b]) else np.zeros(feats.shape[1]) for b in bars])
    bar_times = [float(beats[b.start]) for b in bars] + [duration]
    boundaries = _boundaries(bar_feats, energy)
    sections = []
    edges = [0, *boundaries, len(bars)]
    for a, b in zip(edges, edges[1:]):
        e = float(energy[a:b].mean())
        start = 0.0 if a == 0 else bar_times[a]
        sections.append(Section(start=start, end=bar_times[b] if b < len(bars) else duration, energy=e, level=""))
    _name_levels(sections)
    return SongMap(
        duration=duration, bpm=beat.bpm, period=beat.period, first_beat=beat.first, clear_beat=beat.clear,
        downbeat=downbeat, bar_energy=[round(float(e), 3) for e in energy], sections=sections,
        accents=_rounded(accents), bass=_rounded(bass), highs=_rounded(highs),
    )


def _beat_features(samples: np.ndarray, beats: np.ndarray):
    window = np.hanning(WINDOW).astype(np.float32)
    frames = np.lib.stride_tricks.sliding_window_view(samples, WINDOW)[::HOP] * window
    spec = np.abs(np.fft.rfft(frames, axis=1))
    # 12 bands on a rising (roughly musical) scale, for telling the song's parts apart.
    edges = np.unique(np.geomspace(2, spec.shape[1] - 1, 13).astype(int))
    bands = np.log1p(np.stack([spec[:, a:b].mean(axis=1) for a, b in zip(edges, edges[1:])], axis=1))
    power = (frames**2).mean(axis=1)
    flux = np.concatenate([[0], np.maximum(0, np.diff(np.log1p(spec), axis=0)).sum(axis=1)])
    step = HOP / RATE
    idx = np.clip((beats / step).astype(int), 0, len(frames) - 1)
    idx = np.append(idx, len(frames))
    feats, loud, busy = [], [], []
    for a, b in zip(idx, idx[1:]):
        b = max(b, a + 1)
        feats.append(bands[a:b].mean(axis=0))
        loud.append(10 * np.log10(power[a:b].mean() + 1e-12))  # average power, so hits count fully
        busy.append(flux[a:b].mean())
    return np.array(feats), np.array(loud), np.array(busy)


# Frequency bands (Hz): the bass (kick drum, bass notes), the middle (voices, guitars, snare body), the highs (cymbals).
BANDS = {"bass": (30, 150), "mid": (150, 2000), "high": (3500, RATE / 2)}
CONTEXT_BEATS = 16  # a hit counts as strong compared with the beats around it


def _accents(samples: np.ndarray, beats: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """How hard each beat hits: a sudden rise in sound, in each band, right at the beat.

    Each band is judged mostly against its own neighbourhood (and a little against the whole song), so a big
    hit in a quiet part still counts, and a loud chorus doesn't make every beat "strong"."""
    if len(beats) == 0:
        return np.zeros(0), np.zeros(0), np.zeros(0)
    window = np.hanning(WINDOW).astype(np.float32)
    hz = np.fft.rfftfreq(WINDOW, 1 / RATE)
    fine = HOP // 4  # look closely (every 6 ms) around each beat, so a short hit can't fall between two looks
    reach = int(0.08 * RATE)  # the beats found can be a few hundredths of a second off the hit
    padded = np.concatenate([np.zeros(WINDOW + reach, np.float32), samples, np.zeros(WINDOW + reach, np.float32)])
    picks = {name: (hz >= lo) & (hz < hi) for name, (lo, hi) in BANDS.items()}
    at_beat = {name: np.zeros(len(beats)) for name in BANDS}
    for k, t in enumerate(beats):
        centre = int(t * RATE) + WINDOW + reach
        part = padded[centre - reach - WINDOW // 2 - fine : centre + reach + WINDOW // 2]
        frames = np.lib.stride_tricks.sliding_window_view(part, WINDOW)[::fine] * window
        spec = np.log1p(10 * np.abs(np.fft.rfft(frames, axis=1)))
        for name, pick in picks.items():
            band = spec[:, pick].mean(axis=1)
            at_beat[name][k] = max(0.0, float(np.diff(band).max())) if len(band) > 1 else 0.0
    hits = {name: 0.3 * _norm(v) + 0.7 * _local(v) for name, v in at_beat.items()}
    overall = 0.45 * hits["bass"] + 0.3 * hits["mid"] + 0.25 * hits["high"]
    return overall, hits["bass"], hits["high"]


def _local(v: np.ndarray) -> np.ndarray:
    """Each value against its neighbours: 0 = an ordinary beat here, 1 = clearly the strongest around."""
    out = np.zeros(len(v))
    half = CONTEXT_BEATS // 2
    for i in range(len(v)):
        near = v[max(0, i - half) : i + half + 1]
        typical, top = np.median(near), near.max()
        out[i] = np.clip((v[i] - typical) / (top - typical), 0, 1) if top > typical else 0.0
    return out


def _rounded(v: np.ndarray) -> list[float]:
    return [round(float(x), 3) for x in v]


def _downbeat(samples: np.ndarray, beats: np.ndarray) -> int:
    """Which of the first four beats starts a bar: the one that is usually the strongest (the kick on "one")."""
    if len(beats) < 8:
        return 0
    i = np.clip((beats * RATE).astype(int), 0, len(samples) - 1)
    width = int(0.05 * RATE)
    strength = np.array([np.abs(samples[j : j + width]).mean() for j in i])
    scores = [strength[k::BEATS_PER_BAR].mean() for k in range(BEATS_PER_BAR)]
    return int(np.argmax(scores))


def _boundaries(bar_feats: np.ndarray, energy: np.ndarray) -> list[int]:
    """Bars where the music clearly changes: a different sound or a jump in energy compared with before."""
    n = len(bar_feats)
    if n < 2 * MIN_SECTION_BARS:
        return []
    z = (bar_feats - bar_feats.mean(axis=0)) / (bar_feats.std(axis=0) + 1e-6)
    w = MIN_SECTION_BARS
    novelty = np.zeros(n)
    for i in range(w, n - w + 1):
        before, after = z[i - w : i].mean(axis=0), z[i : i + w].mean(axis=0)
        novelty[i] = np.linalg.norm(after - before) / np.sqrt(z.shape[1]) + 2 * abs(
            energy[i : i + w].mean() - energy[i - w : i].mean())
    threshold = novelty.mean() + 0.5 * novelty.std()
    picked: list[int] = []
    for i in np.argsort(novelty)[::-1]:
        if novelty[i] <= threshold or novelty[i] == 0:
            break
        if all(abs(i - j) >= MIN_SECTION_BARS for j in picked):
            picked.append(int(i))
    return sorted(picked)


def _name_levels(sections: list[Section]) -> None:
    for s in sections:
        s.level = "Calm" if s.energy < 0.38 else "Lively" if s.energy > 0.62 else "Medium"


def _norm(v: np.ndarray) -> np.ndarray:
    lo, hi = np.percentile(v, 5), np.percentile(v, 95)
    return np.clip((v - lo) / (hi - lo), 0, 1) if hi > lo else np.full(len(v), 0.5)
