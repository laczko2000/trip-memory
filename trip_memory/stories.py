"""Group a trip into "stories" and spot sets of near-identical shots, then pick the best of each set.

A story is a run of files taken close together in time and place (a new one starts after a long gap,
on a new day, or after moving more than a kilometre). Inside a story, shots taken within a few minutes
of each other that look alike form a "similar set": the same scene from slightly different angles.
"""

from __future__ import annotations

import math
from datetime import datetime

import cv2
import numpy as np
from PIL import Image

STORY_GAP_MINUTES = 45
STORY_MOVE_KM = 1.0
SIMILAR_GAP_MINUTES = 3
SIMILAR_HASH_BITS = 18  # of 64: how different two pictures may look and still count as "the same shot"
SIMILAR_COLOUR = 40.0


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
    return s


def group(rows: list[dict]) -> dict[str, tuple[int, int, bool]]:
    """For each file: (story number, similar-set number, is it the best of its set)."""
    rows = sorted(rows, key=lambda r: (r["taken_at"], r["path"]))
    result: dict[str, tuple[int, int, bool]] = {}
    story = similar = 0
    sets: list[list[dict]] = []
    prev = None
    for row in rows:
        when = datetime.fromisoformat(row["taken_at"])
        if prev is not None:
            gap = (when - prev[0]).total_seconds() / 60
            new_story = gap > STORY_GAP_MINUTES or row["taken_at"][:10] != prev[1]["taken_at"][:10] or _km(row, prev[1]) > STORY_MOVE_KM
            if new_story:
                story += 1
            if new_story or gap > SIMILAR_GAP_MINUTES or not looks_alike(row, prev[1]):
                similar += 1
                sets.append([])
        else:
            sets.append([])
        sets[-1].append(row)
        result[row["path"]] = (story, similar, False)
        prev = (when, row)
    for members in sets:
        best = max(members, key=score)
        story_no, set_no, _ = result[best["path"]]
        result[best["path"]] = (story_no, set_no, True)
    return result
