"""The quality check: pictures and videos not good enough for a high-quality video (blurry, too dark, washed out,
almost empty, very shaky, accidental, tiny, screenshots). Measured once from each preview; what counts as bad
depends on how strict you are (gentle, normal, strict). Nothing here touches your files."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image

LEVELS = {"gentle": "Gentle", "normal": "Normal", "strict": "Strict"}

# Why a shot is suggested for removal, in the order the groups are shown.
REASONS = {
    "blurry": ("Blurry", "Nothing in the picture is really sharp."),
    "dark": ("Too dark or black", "Pocket shots, black frames, night pictures where little can be seen."),
    "bright": ("Washed out", "Mostly blown-out white: overexposed."),
    "empty": ("Almost empty", "A plain wall, floor or sky, or a covered lens: nothing to look at."),
    "shaky": ("Very shaky video", "The camera jumps around a lot."),
    "short": ("Accidental video", "Shorter than a moment: started by mistake."),
    "tiny": ("Small, low-quality picture", "Too few pixels for a sharp video (often from a chat app)."),
    "screenshot": ("Screenshot", "A picture of a screen, not taken with a camera."),
}

# The limits per level: (gentle, normal, strict).
BLUR = {"gentle": 25.0, "normal": 45.0, "strict": 80.0}  # sharpest part of the preview (Laplacian variance)
BLUR_SHARE = {"gentle": 0.08, "normal": 0.15, "strict": 0.25}  # ... or this far below the trip's usual sharpness
DARK = {"gentle": 22.0, "normal": 38.0, "strict": 55.0}  # average brightness (0-255)
BRIGHT = {"gentle": 0.7, "normal": 0.5, "strict": 0.38}  # share of blown-out white
EMPTY = {"gentle": 6.0, "normal": 10.0, "strict": 15.0}  # how much the brightness varies at all
SHAKY = {"gentle": 2.2, "normal": 1.6, "strict": 1.2}  # motion's camera jumps (motion.SHAKY = 1.0 is "shaky")
SHORT = {"gentle": 0.6, "normal": 1.0, "strict": 1.5}  # seconds
TINY = {"gentle": 640, "normal": 1000, "strict": 1280}  # pixels on the long side
SCREEN_WORDS = ("screenshot", "screen shot", "screen_shot", "screen recording", "screenrecording", "bildschirmfoto")


def measure(thumb: str, original: str | None = None, kind: str = "photo") -> dict:
    """What the quality check needs from one preview: how sharp its sharpest part is (a sharp face in front of a
    soft background counts as sharp), its brightness, how much of it is blown out, how much it varies at all, and
    (photos) the picture's real size."""
    with Image.open(thumb) as img:
        gray = np.asarray(img.convert("L"), dtype=np.uint8)
    h, w = gray.shape
    tiles = []
    for y in range(4):
        for x in range(4):
            part = gray[y * h // 4:(y + 1) * h // 4, x * w // 4:(x + 1) * w // 4]
            if part.size >= 64:
                tiles.append(float(cv2.Laplacian(part, cv2.CV_64F).var()))
    tiles.sort(reverse=True)
    out = {
        "tile": round(float(np.mean(tiles[:2])) if tiles else 0.0, 1),  # (two best parts: one speck is not sharp)
        "mean": round(float(gray.mean()), 1),
        "std": round(float(gray.std()), 1),
        "p99": round(float(np.percentile(gray, 99)), 1),
        "clip": round(float((gray >= 250).mean()), 3),
    }
    if kind == "photo" and original:
        try:
            with Image.open(original) as img:  # (reads only the file's header)
                out["w"], out["h"] = img.size
        except Exception:  # noqa: BLE001 - an unreadable header: size unknown
            pass
    return out


def is_screenshot(path: str) -> bool:
    name = Path(path).name.lower()
    return Path(path).suffix.lower() == ".png" or any(word in name for word in SCREEN_WORDS)


def flags(rows: list[dict], level: str = "normal") -> dict[str, tuple[str, float]]:
    """{path: (reason, how bad 0-1)} for each shot worth a second look at this level: the worst reason only.
    rows carry "path", "kind", "quality" (from measure), and "duration", "motion_shake" when known."""
    level = level if level in LEVELS else "normal"
    usual = {kind: float(np.median([r["quality"]["tile"] for r in rows if r["kind"] == kind and r.get("quality")]
                                   or [0.0])) for kind in ("photo", "video")}
    out: dict[str, tuple[str, float]] = {}
    for r in rows:
        q = r.get("quality")
        if not q:
            continue
        found: list[tuple[str, float]] = []
        if r["kind"] == "photo" and is_screenshot(r["path"]):
            found.append(("screenshot", 1.0))
        if q["p99"] < 40 or q["mean"] < DARK[level]:
            found.append(("dark", 1.0 - q["mean"] / max(DARK[level], 1) * 0.5))
        elif q["clip"] >= BRIGHT[level]:
            found.append(("bright", min(1.0, q["clip"])))
        lit = not found or found[-1][0] == "screenshot"  # (a black or white picture is empty anyway: said once)
        if lit and q["std"] < EMPTY[level] and q["tile"] < 2 * BLUR[level]:
            found.append(("empty", 1.0 - q["std"] / EMPTY[level] * 0.5))
        else:
            # Videos are softer than photos (moving, compressed): judged against a lower limit.
            limit = BLUR[level] * (0.7 if r["kind"] == "video" else 1.0)
            soft = q["tile"] < limit or (q["tile"] < 2 * limit and q["tile"] < BLUR_SHARE[level] * usual[r["kind"]])
            if soft and lit:  # (a dark or washed-out picture is soft anyway)
                found.append(("blurry", 1.0 - min(q["tile"] / (2 * limit), 1.0) * 0.5))
        if r["kind"] == "video":
            if (r.get("duration") or 0) and r["duration"] < SHORT[level]:
                found.append(("short", 1.0 - r["duration"] / SHORT[level] * 0.5))
            if (r.get("motion_shake") or 0) >= SHAKY[level]:
                found.append(("shaky", min(1.0, r["motion_shake"] / (2 * SHAKY[level]))))
        elif max(q.get("w") or 1e9, q.get("h") or 0) < TINY[level]:
            found.append(("tiny", 1.0 - max(q["w"], q["h"]) / TINY[level] * 0.5))
        if found:
            order = list(REASONS)
            out[r["path"]] = max(found, key=lambda f: (f[1], -order.index(f[0])))
    return out
