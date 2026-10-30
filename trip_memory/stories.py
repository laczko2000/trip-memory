"""Group a trip into "stories" and spot sets of near-identical shots, then pick the best of each set.

A story is a run of files taken close together in time and place (a new one starts after a long gap,
on a new day, or after moving more than a kilometre). Inside a story, shots taken within a few minutes
of each other that look alike form a "similar set": the same scene or the same people, from slightly
different angles, a bit closer or further, with people moving a little.

Two shots look alike when many of their small details (corners, edges, patterns) match up, even if the
picture has shifted, turned or been zoomed a little; or when their layout and colours match closely.
"""

from __future__ import annotations

import io
import math
from datetime import datetime

import cv2
import numpy as np
from PIL import Image

STORY_GAP_MINUTES = 45
STORY_MOVE_KM = 1.0
SIMILAR_HASH_BITS = 18  # of 64: how different two pictures may look and still count as "the same shot"
SIMILAR_COLOUR = 40.0

# How strict "look alike" is: minutes between shots, share of matching details, layout and colour closeness.
STRICTNESS = {
    "strict": {"minutes": 4, "details": 0.18, "layout": 0.65, "colours": 0.6},
    "normal": {"minutes": 10, "details": 0.12, "layout": 0.5, "colours": 0.5},
    "loose": {"minutes": 25, "details": 0.08, "layout": 0.4, "colours": 0.45},
}
STRICTNESS_NAMES = {"strict": "Only near-identical shots", "normal": "Same people, same setting (normal)",
                    "loose": "Anything alike in the same place"}


def fingerprint(thumb_path: str) -> tuple[str, bytes, float]:
    """A picture's look in a few numbers: a shape hash, its average colours, and how sharp it is."""
    with Image.open(thumb_path) as img:
        gray = img.convert("L")
        small = np.asarray(gray.resize((9, 8), Image.Resampling.LANCZOS), dtype=np.int16)
        bits = (small[:, 1:] > small[:, :-1]).flatten()
        dhash = f"{int(''.join('1' if b else '0' for b in bits), 2):016x}"
        colours = np.asarray(img.convert("RGB").resize((4, 4), Image.Resampling.BOX), dtype=np.uint8)
        sharpness = float(cv2.Laplacian(np.asarray(gray, dtype=np.uint8), cv2.CV_64F).var())
    return dhash, colours.tobytes(), sharpness


def features(thumb_path: str) -> bytes:
    """A picture's details for comparing it with others: its layout (very blurred), its colours, and the
    descriptors of up to 400 small details (corners and edges) that survive shifting, turning and zooming."""
    with Image.open(thumb_path) as img:
        rgb = np.asarray(img.convert("RGB"))
    return _features_of(rgb)


VIDEO_FRAMES = (0.15, 0.5, 0.85)  # where in a video its look is taken: start, middle and end


def video_features(frames: list[str]) -> tuple[bytes, float]:
    """A video's look from a few of its frames (a camera moves, so one frame is not enough), and how sharp
    it is on average."""
    packed, sharp = {}, []
    for k, path in enumerate(frames):
        with Image.open(path) as img:
            rgb = np.asarray(img.convert("RGB"))
        one = _unpack(_features_of(rgb))
        packed.update({f"{name}{k}": value for name, value in one.items()})
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        sharp.append(float(cv2.Laplacian(gray, cv2.CV_64F).var()))
    out = io.BytesIO()
    np.savez(out, frames=np.array(len(frames)), **{k: (v.astype(np.float16) if v.dtype != np.uint8 else v)
                                                    for k, v in packed.items()})
    return out.getvalue(), float(np.mean(sharp)) if sharp else 0.0


def _features_of(rgb: np.ndarray) -> bytes:
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    layout = cv2.resize(cv2.GaussianBlur(gray, (0, 0), 8), (12, 12), interpolation=cv2.INTER_AREA).astype(np.float32).ravel()
    layout = (layout - layout.mean()) / (layout.std() + 1e-6)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, [12, 4, 4], [0, 180, 0, 256, 0, 256]).ravel()
    hist = hist / max(hist.sum(), 1)
    _, details = cv2.ORB_create(400).detectAndCompute(gray, None)
    out = io.BytesIO()
    np.savez(out, layout=layout.astype(np.float16), colours=hist.astype(np.float16),
             details=details if details is not None else np.zeros((0, 32), np.uint8))
    return out.getvalue()


def _unpack(blob: bytes) -> dict:
    """A picture's details, or a video's: {"frames": [the details of each of its frames]}."""
    with np.load(io.BytesIO(blob)) as data:
        if "frames" in data:
            return {"frames": [{"layout": data[f"layout{k}"].astype(np.float32),
                                "colours": data[f"colours{k}"].astype(np.float32), "details": data[f"details{k}"]}
                               for k in range(int(data["frames"]))]}
        return {"layout": data["layout"].astype(np.float32), "colours": data["colours"].astype(np.float32),
                "details": data["details"]}


_MATCHER = cv2.BFMatcher(cv2.NORM_HAMMING)


def likeness(a: dict, b: dict) -> tuple[float, float, float]:
    """(share of matching details, layout match, colour overlap) between two unpacked pictures."""
    details = 0.0
    da, db = a["details"], b["details"]
    if len(da) >= 10 and len(db) >= 10:
        pairs = _MATCHER.knnMatch(da, db, k=2)
        good = sum(1 for p in pairs if len(p) == 2 and p[0].distance < 0.75 * p[1].distance)
        details = good / max(len(da), len(db))
    layout = float(np.mean(a["layout"] * b["layout"]))
    colours = float(np.minimum(a["colours"], b["colours"]).sum())
    return details, layout, colours


def alike(a: dict, b: dict, level: str = "normal") -> bool:
    """Whether two shots look like the same shot. Videos: any of their frames looking alike counts, as
    two takes of the same scene often start or end the same way."""
    if "frames" in a or "frames" in b:
        return any(_alike(x, y, level) for x in a.get("frames", [a]) for y in b.get("frames", [b]))
    return _alike(a, b, level)


def _alike(a: dict, b: dict, level: str) -> bool:
    t = STRICTNESS.get(level, STRICTNESS["normal"])
    colours = float(np.minimum(a["colours"], b["colours"]).sum())
    if colours < 0.3:  # quick check first: clearly different colours are never the same shot
        return False
    if float(np.mean(a["layout"] * b["layout"])) >= t["layout"] and colours >= t["colours"]:
        return True
    details, _, _ = likeness(a, b)
    return details >= t["details"]


def looks_alike(a: dict, b: dict) -> bool:
    if not (a.get("dhash") and b.get("dhash")):
        return False
    bits = bin(int(a["dhash"], 16) ^ int(b["dhash"], 16)).count("1")
    colour = float(np.abs(np.frombuffer(a["colours"], np.uint8).astype(int) - np.frombuffer(b["colours"], np.uint8)).mean())
    return bits <= SIMILAR_HASH_BITS and colour <= SIMILAR_COLOUR


def _km(a: dict, b: dict) -> float:
    if a.get("lat") is None or b.get("lat") is None:
        return 0.0
    la1, lo1, la2, lo2 = map(math.radians, (a["lat"], a["lon"], b["lat"], b["lon"]))
    h = math.sin((la2 - la1) / 2) ** 2 + math.cos(la1) * math.cos(la2) * math.sin((lo2 - lo1) / 2) ** 2
    return 6371 * 2 * math.asin(math.sqrt(h))


def score(row: dict) -> float:
    """How good a shot looks: sharp, well exposed, bigger, and for videos a sensible length."""
    s = math.log1p(row.get("sharpness") or 0)
    if row.get("colours"):
        brightness = float(np.frombuffer(row["colours"], np.uint8).mean())
        s -= abs(brightness - 128) / 64  # very dark or washed-out pictures lose points
    if row["kind"] == "video":
        s += min(row.get("duration") or 0, 20) / 10
        if row.get("motion_shake") is not None:
            s -= 2 * min(1.0, row["motion_shake"])  # a shaky take loses to a steady one
    return s


def group(rows: list[dict], level: str = "normal", links: list[str] | None = None) -> dict[str, tuple[int, int, bool]]:
    """For each file: (story number, similar-set number, is it the best of its set).

    links: places where you joined two stories, each as "taken_at|path" of the later story's first file; no new
    story starts there (still true when that file is later removed: any break just after it is joined).

    A shot joins a set when it looks like one of that set's last few shots, taken within a few minutes, in the
    same story; so a burst stays together even with other shots taken in between."""
    t = STRICTNESS.get(level, STRICTNESS["normal"])
    rows = sorted(rows, key=lambda r: (r["taken_at"], r["path"]))
    result: dict[str, tuple[int, int, bool]] = {}
    story = -1
    sets: list[list[dict]] = []
    open_sets: list[int] = []  # sets of this story that new shots can still join
    unpacked: dict[str, dict | None] = {}
    prev = None

    def look(row: dict) -> dict | None:
        if row["path"] not in unpacked:
            unpacked[row["path"]] = _unpack(row["features"]) if row.get("features") else None
        return unpacked[row["path"]]

    joins = sorted(tuple(link.split("|", 1)) for link in links or [] if "|" in link)
    for row in rows:
        when = datetime.fromisoformat(row["taken_at"])
        if prev is None or ((when - prev[0]).total_seconds() / 60 > STORY_GAP_MINUTES
                            or row["taken_at"][:10] != prev[1]["taken_at"][:10] or _km(row, prev[1]) > STORY_MOVE_KM):
            here, before = (row["taken_at"], row["path"]), prev and (prev[1]["taken_at"], prev[1]["path"])
            if not (before and any(before < j <= here for j in joins)):
                story += 1
                open_sets = []
        best_set = None
        for k in list(reversed(open_sets))[:12]:  # the most recent sets only, so big trips stay quick
            last = sets[k][-1]
            if last["kind"] != row["kind"]:
                continue  # a video's doubles are other videos (a photo of the same scene is still worth keeping)
            ended = datetime.fromisoformat(last["taken_at"]).timestamp() + (last.get("duration") or 0)
            if (when.timestamp() - ended) / 60 > t["minutes"]:
                continue
            for other in sets[k][-2:]:
                a, b = look(row), look(other)
                if (alike(a, b, level) if a is not None and b is not None else looks_alike(row, other)):
                    best_set = k
                    break
            if best_set is not None:
                break
        if best_set is None:
            sets.append([])
            best_set = len(sets) - 1
            open_sets.append(best_set)
        sets[best_set].append(row)
        result[row["path"]] = (story, best_set, False)
        prev = (when, row)
    for members in sets:
        best = max(members, key=score)
        story_no, set_no, _ = result[best["path"]]
        result[best["path"]] = (story_no, set_no, True)
    return result
