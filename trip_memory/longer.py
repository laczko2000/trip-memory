"""Make a song longer: keep its intro and its ending, and play parts of the middle again, cut on bar lines so
the beat never stumbles. The result is a new track (your song file is never changed), with your pins coming along
(pins in a part that plays twice are there twice)."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from . import songmap, taps
from .beats import song_length

CROSSFADE = 0.06  # seconds: each join is a very short cross-fade on a bar line (no click)
EDGE_SHARE = 0.15  # the intro and the ending (at least this share of the song each) are never repeated


def _bars(song: songmap.SongMap) -> list[float]:
    bars = [float(t) for t in song.bar_starts()]
    if len(bars) < 4:  # (no clear beat: every 2 seconds)
        bars = [k * 2.0 for k in range(int(song.duration // 2) + 1)]
    return bars


def plan(song: songmap.SongMap, target: float) -> list[tuple[float, float]]:
    """The pieces of the song, in order, that make a track of about `target` seconds: [(start, end)]. The first
    piece runs from the start into the middle, then middle sections play again (whole sections where they fit,
    the most lively first, else whole bars in fours), then the rest to the end. The joins are on bar lines."""
    duration = song.duration
    if target <= duration + 1.0:
        return [(0.0, duration)]
    bars = _bars(song)
    lo, hi = duration * EDGE_SHARE, duration * (1 - EDGE_SHARE)
    middle = [b for b in bars if lo <= b <= hi]
    if len(middle) < 2:
        middle = [b for b in bars if 0 < b < duration] or [duration / 3, 2 * duration / 3]
        if len(middle) < 2:
            middle = [duration / 3, 2 * duration / 3]
    # Blocks that can play again: the song's own sections inside the middle (snapped to bars), else 4-bar phrases.
    snap = lambda t: min(middle, key=lambda b: abs(b - t))
    blocks = []
    for s in sorted(song.sections, key=lambda s: -s.energy):
        a, b = snap(s.start), snap(s.end)
        if b - a >= 4.0:
            blocks.append((a, b))
    period = (bars[1] - bars[0]) if len(bars) > 1 else 2.0
    phrase = max(period * 4, 2.0)
    blocks.append((middle[0], min(middle[-1], middle[0] + phrase)) if middle[-1] - middle[0] >= phrase
                  else (middle[0], middle[-1]))
    need = target - duration
    chosen: list[tuple[float, float]] = []
    while need > 0.5 * phrase and len(chosen) < 60:
        fit = [blk for blk in blocks if blk[1] - blk[0] <= need + phrase * 0.5]
        if fit:
            blk = max(fit, key=lambda blk: (blk[1] - blk[0] if (blk[1] - blk[0]) <= need + 0.5 else 0,
                                            -chosen.count(blk)))
        else:  # (every section is longer than what's still missing: the last bars of the liveliest one)
            a, b = blocks[0]
            count = max(1, round(need / period))
            start = max(a, b - count * period)
            start = min((x for x in bars if x >= start - 0.01), default=start)
            blk = (start, b)
        chosen.append(blk)
        need -= blk[1] - blk[0]
    if not chosen:
        return [(0.0, duration)]
    # Each repeat goes right after the place it comes from: the music plays a part, then the same part again.
    chosen.sort(key=lambda blk: blk[1])
    pieces, at = [], 0.0
    for a, b in chosen:
        pieces.append((at, b))
        at = a
    pieces.append((at, duration))
    return [(round(a, 3), round(b, 3)) for a, b in pieces if b - a > 0.05]


def _mmss(seconds: float) -> str:
    return f"{int(seconds // 60)}-{int(round(seconds % 60)):02d}"


def make_longer(song_path: str, target: float, out_dir: Path, pins_folder: Path, maps_folder: Path | None = None) -> str:
    """Make the track and return its path. Pins move along (repeated parts get their pins again)."""
    from .tools import ffmpeg_paths

    song = songmap.analyse(song_path, maps_folder)
    pieces = plan(song, target)
    if len(pieces) == 1:
        raise ValueError("The song is already that long. To make it shorter, trim it on the pin line.")
    out_dir.mkdir(parents=True, exist_ok=True)
    total = sum(b - a for a, b in pieces) - CROSSFADE * (len(pieces) - 1)
    out = out_dir / f"{Path(song_path).stem} (longer, {_mmss(total)}).m4a"
    inputs: list[str] = []
    for a, b in pieces:  # (each piece read on its own: exact, whatever the song's format)
        inputs += ["-ss", f"{a:.3f}", "-t", f"{b - a:.3f}", "-i", song_path]
    graph = [f"[{k}:a]aformat=sample_rates=44100:channel_layouts=stereo,asetpts=PTS-STARTPTS[p{k}]"
             for k in range(len(pieces))]
    prev = "p0"
    for k in range(1, len(pieces)):
        graph.append(f"[{prev}][p{k}]acrossfade=d={CROSSFADE}:c1=tri:c2=tri[x{k}]")
        prev = f"x{k}"
    graph.append(f"[{prev}]anull[out]")
    subprocess.run([ffmpeg_paths()[0], "-y", "-loglevel", "error", *inputs, "-filter_complex", ";".join(graph),
                    "-map", "[out]", "-vn", "-c:a", "aac", "-b:a", "256k", str(out)], check=True, capture_output=True)
    # Your pins, piece by piece (a part that plays twice has its pins twice).
    old = taps.load(song_path, pins_folder)["taps"]
    pins, at = [], 0.0
    for k, (a, b) in enumerate(pieces):
        pins += [at + t - a for t in old if a + 0.05 < t < b - 0.05]
        at += (b - a) - (CROSSFADE if k < len(pieces) - 1 else 0.0)
    taps.save(str(out), pins_folder, pins, True)
    file = taps._file(str(out), pins_folder)
    data = json.loads(file.read_text())
    file.write_text(json.dumps({**data, "source": song_path, "longer": [round(a, 3) for a in [target]],
                                "pieces": pieces}))
    return str(out)


def longer_from(track: str, pins_folder: Path) -> tuple[str, int] | None:
    """(the song it was made from, how many parts play again) for a track made longer here, else None."""
    file = taps._file(track, pins_folder) if Path(track).exists() else None
    try:
        data = json.loads(file.read_text()) if file else {}
    except (OSError, ValueError):
        return None
    if not data.get("longer") or not data.get("source"):
        return None
    return data["source"], max(0, len(data.get("pieces") or []) - 1)


def length_of(path: str) -> float:
    return song_length(path)
