"""Camera clocks: which camera filmed each file, and putting right a camera whose clock was wrong.

A GoPro (or any camera) often has its clock set to another time zone, or reset after a flat battery, so its
shots land in the wrong place in the trip. Shifting that camera's clock moves all its files at once. The app can
also suggest the shift: shots of the same moment filmed by the phone and the GoPro look alike, and the time
between them is the camera's error.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np

OTHER = "Other camera"
GOPRO_NAME = re.compile(r"^(G[HXOPL]\d{6}|GOPR\d{4}|GP\d{6})\b", re.I)
DJI_NAME = re.compile(r"^DJI_\d+", re.I)

MATCH_WINDOW = 120  # seconds: matches whose time differences agree this closely are the same clock error
ROUND_TO = 15 * 60  # time zones are whole quarters of an hour: a shift this close to one is rounded to it
ROUND_WITHIN = 120


def camera_name(make: str | None, model: str | None) -> str | None:
    """'Apple iPhone 15 Pro' style: the maker and model, without the maker twice."""
    make = (make or "").strip().strip("\x00")
    model = (model or "").strip().strip("\x00")
    if not make and not model:
        return None
    if make.lower() in ("apple",) and model:
        return model if model.lower().startswith("iphone") or model.lower().startswith("ipad") else f"Apple {model}"
    if model and make and model.lower().startswith(make.lower().split()[0]):
        return model
    return f"{make} {model}".strip()


def camera_from_probe(info: dict, path: Path) -> str:
    """The camera that filmed a video, from what ffprobe found in it (and its file name)."""
    tags = {k.lower(): str(v) for k, v in info.get("format", {}).get("tags", {}).items()}
    handlers = " ".join(str(s.get("tags", {}).get("handler_name", "")) for s in info.get("streams", []))
    if "gopro" in handlers.lower() or "gopro" in tags.get("firmware", "").lower() or GOPRO_NAME.match(path.name) \
            or tags.get("firmware", "")[:2] in ("HD", "H1", "H2") and len(tags.get("firmware", "")) > 4:
        return "GoPro"
    name = camera_name(tags.get("com.apple.quicktime.make") or tags.get("make") or tags.get("com.android.manufacturer"),
                       tags.get("com.apple.quicktime.model") or tags.get("model") or tags.get("com.android.model"))
    if name:
        return name
    if "dji" in (tags.get("encoder", "") + handlers).lower() or DJI_NAME.match(path.name):
        return "DJI"
    return OTHER


def camera_from_exif(make: str | None, model: str | None, path: Path) -> str:
    if (make or "").lower().startswith("gopro") or GOPRO_NAME.match(path.name):
        return "GoPro"
    return camera_name(make, model) or ("DJI" if DJI_NAME.match(path.name) else OTHER)


def parse_shift(text: str) -> float | None:
    """How much to move a camera's clock, as typed: '+2:00' (hours:minutes), '-1:30:15', '2h', '-90 min',
    '1 h 30 m', '+45s', '-2 days'. None if it can't be read."""
    t = (text or "").strip().lower().replace(" ", "")
    if not t:
        return None
    sign = -1 if t[0] in "-−" else 1
    t = t.lstrip("+-−")
    if re.fullmatch(r"\d+(:\d{1,2}){1,2}", t):
        parts = [int(x) for x in t.split(":")]
        h, m, s = (parts + [0])[:3]
        return sign * (h * 3600 + m * 60 + s)
    units = {"d": 86400, "day": 86400, "days": 86400, "h": 3600, "hr": 3600, "hrs": 3600, "hour": 3600, "hours": 3600,
             "m": 60, "min": 60, "mins": 60, "minute": 60, "minutes": 60,
             "s": 1, "sec": 1, "secs": 1, "second": 1, "seconds": 1}
    found = re.findall(r"(\d+(?:[.,]\d+)?)([a-z]+)", t)
    if not found or "".join(n + u for n, u in found) != t or any(u not in units for _, u in found):
        try:
            return sign * float(t.replace(",", ".")) * 3600  # a plain number: hours
        except ValueError:
            return None
    return sign * sum(float(n.replace(",", ".")) * units[u] for n, u in found)


def fmt_shift(seconds: float) -> str:
    """'+2:00' for two hours later, '-0:05:30' for five and a half minutes earlier, '+1 day 2:00'."""
    s = int(round(seconds))
    sign = "-" if s < 0 else "+"
    s = abs(s)
    days, s = divmod(s, 86400)
    h, s = divmod(s, 3600)
    m, s = divmod(s, 60)
    clock = f"{h}:{m:02d}" + (f":{s:02d}" if s else "")
    if days:
        return f"{sign}{days} day{'s' if days > 1 else ''} {clock}"
    return sign + clock


def shift_words(seconds: float) -> str:
    """'2 hours later', '5 minutes earlier' - for telling you what a shift does."""
    if not seconds:
        return "no change"
    s = abs(seconds)
    if s >= 86400:
        amount = f"{s / 86400:.1f}".rstrip("0").rstrip(".") + " days"
    elif s >= 3600:
        h, m = divmod(round(s / 60), 60)
        amount = f"{h} hour{'s' if h > 1 else ''}" + (f" {m} min" if m else "")
    elif s >= 60:
        amount = f"{round(s / 60)} minutes"
    else:
        amount = f"{round(s)} seconds"
    return f"{amount} {'later' if seconds > 0 else 'earlier'}"


def similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Cosine likeness of every row of a with every row of b."""
    a = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-9)
    b = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-9)
    return a @ b.T


def suggest(mine: list[tuple[float, np.ndarray]], others: list[tuple[float, np.ndarray]],
            top: int = 3, least: float = 0.0) -> dict | None:
    """The clock error of a camera, from its shots (time in seconds, look) and the other cameras' shots.

    Each of its shots is matched with the few other-camera shots that look most like it. Most matches are
    chance, but the true ones all point at the same time difference: the difference most shots agree on
    (within MATCH_WINDOW) is the suggestion. Returns {"seconds", "shots" (how many agree), "of" (shots
    compared), "pairs" [(my index, other index, difference)]}, or None when fewer than two shots agree.
    """
    if not mine or not others:
        return None
    sims = similarity(np.array([v for _, v in mine], dtype=np.float32), np.array([v for _, v in others], dtype=np.float32))
    votes = []  # (difference, likeness, my index, other index)
    for i, (t, _) in enumerate(mine):
        for j in np.argsort(-sims[i])[:top]:
            if sims[i, j] >= least:
                votes.append((others[j][0] - t, float(sims[i, j]), i, int(j)))
    if not votes:
        return None
    votes.sort()
    best, best_score, lo = None, 0.0, 0
    for hi in range(len(votes)):
        while votes[hi][0] - votes[lo][0] > 2 * MATCH_WINDOW:
            lo += 1
        inside = votes[lo:hi + 1]
        per_shot: dict[int, float] = {}
        for _, sim, i, _ in inside:
            per_shot[i] = max(per_shot.get(i, 0.0), sim)
        score = sum(per_shot.values())
        if score > best_score:
            best, best_score = inside, score
    shots = {i for _, _, i, _ in best}
    if len(shots) < 2:
        return None
    pick = {}
    for d, sim, i, j in best:  # each shot's best match inside the agreeing group
        if i not in pick or sim > pick[i][1]:
            pick[i] = (d, sim, j)
    seconds = float(np.median([d for d, _, _ in pick.values()]))
    nearest = round(seconds / ROUND_TO) * ROUND_TO
    if abs(seconds - nearest) <= ROUND_WITHIN:
        seconds = float(nearest)
    else:
        seconds = float(round(seconds))
    return {"seconds": seconds, "shots": len(shots), "of": len(mine),
            "pairs": sorted((i, j, d) for i, (d, _, j) in pick.items())}
