"""Text shown over a shot: the trip's opening title, a story's name, your own words, the end card.

A title is drawn once as a transparent picture and laid over the shot while it plays, fading in and out.
It uses fonts that come with every Mac (with stand-ins on other computers).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

MAC = "/System/Library/Fonts/Supplemental/"
FALLBACK = ["/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"]

# Each style: its name in the app, fonts to try (first found wins), size (share of the video's height),
# capitals or not, and where it goes unless you choose otherwise.
STYLES = {
    "clean": {"label": "Clean", "fonts": [MAC + "Futura.ttc", MAC + "Arial Bold.ttf",
                                          "/usr/share/fonts/opentype/inter/Inter-SemiBold.otf"],
              "size": 0.085, "upper": False, "where": "lower"},
    "bold": {"label": "Bold", "fonts": [MAC + "Impact.ttf", MAC + "Arial Black.ttf",
                                        "/usr/share/fonts/opentype/inter/Inter-ExtraBold.otf"],
             "size": 0.13, "upper": True, "where": "center"},
    "handwritten": {"label": "Handwritten", "fonts": [MAC + "Bradley Hand Bold.ttf", "/System/Library/Fonts/Noteworthy.ttc",
                                                      "/System/Library/Fonts/MarkerFelt.ttc",
                                                      "/usr/share/fonts/truetype/dejavu/DejaVuSerif-BoldItalic.ttf"],
                    "size": 0.1, "upper": False, "where": "lower"},
    "elegant": {"label": "Elegant", "fonts": [MAC + "Georgia Italic.ttf", MAC + "Didot.ttc",
                                              "/usr/share/fonts/truetype/dejavu/DejaVuSerif-Italic.ttf"],
                "size": 0.09, "upper": False, "where": "center"},
    "caption": {"label": "Place and date caption", "fonts": [MAC + "Arial Bold.ttf",
                                                             "/usr/share/fonts/opentype/inter/Inter-Bold.otf"],
                "size": 0.05, "upper": False, "where": "lower-left"},
}
WHERE = {"center": "Middle", "lower": "Bottom", "upper": "Top", "lower-left": "Bottom left"}

FADE_IN, FADE_OUT, DELAY = 0.4, 0.5, 0.25  # seconds
LONGEST = 4.0  # a title stays at most this long on a long shot


def _font(style: str, size: int) -> ImageFont.FreeTypeFont:
    for path in STYLES.get(style, STYLES["clean"])["fonts"] + FALLBACK:
        if Path(path).exists():
            try:
                return ImageFont.truetype(path, size)
            except OSError:
                continue
    return ImageFont.load_default(size)


def _wrap(draw: ImageDraw.ImageDraw, text: str, font, width: float) -> list[str]:
    lines: list[str] = []
    for paragraph in text.split("\n"):
        line = ""
        for word in paragraph.split():
            test = f"{line} {word}".strip()
            if line and draw.textlength(test, font=font) > width:
                lines.append(line)
                line = word
            else:
                line = test
        lines.append(line)
    return [x for x in lines if x] or [""]


def draw(title: dict, w: int, h: int) -> Image.Image:
    """The title as a transparent picture the size of the video. title: {"text", "sub", "style", "where"}."""
    style = title.get("style") if title.get("style") in STYLES else "clean"
    spec = STYLES[style]
    where = title.get("where") if title.get("where") in WHERE else spec["where"]
    text = (title.get("text") or "").strip()
    sub = (title.get("sub") or "").strip()
    if spec["upper"]:
        text = text.upper()
    big = _font(style, round(h * spec["size"]))
    small = _font("caption" if style != "caption" else "clean", round(h * (0.04 if style != "caption" else 0.035)))
    layer = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    pen = ImageDraw.Draw(layer)
    margin = w * 0.06
    lines = _wrap(pen, text, big, w - 2 * margin) if text else []
    sub_lines = _wrap(pen, sub, small, w - 2 * margin) if sub else []
    gap = h * 0.012
    big_h = big.getmetrics()[0] + big.getmetrics()[1] * 0.4
    small_h = small.getmetrics()[0] + small.getmetrics()[1] * 0.4
    block = len(lines) * big_h + len(sub_lines) * small_h + (gap * 2 if lines and sub_lines else 0)
    top = {"center": (h - block) / 2, "upper": h * 0.08, "lower": h * 0.9 - block, "lower-left": h * 0.9 - block}[where]
    left = where == "lower-left"
    x = margin if left else w / 2
    anchor = "la" if left else "ma"
    y = top
    placed = []
    for line in lines:
        placed.append((x, y, line, big, (255, 255, 255, 255)))
        y += big_h
    if lines and sub_lines:
        y += gap * 2
    for line in sub_lines:
        placed.append((x, y, line, small, (240, 240, 240, 255)))
        y += small_h
    # A soft shadow under the letters so they read on bright and busy pictures.
    shadow = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    shade = ImageDraw.Draw(shadow)
    for px, py, line, font, _ in placed:
        shade.text((px, py + h * 0.004), line, font=font, fill=(0, 0, 0, 170), anchor=anchor)
    if left and placed:  # a thin bar beside a caption
        shade.rectangle([margin - w * 0.016, top, margin - w * 0.011, y], fill=(0, 0, 0, 170))
    shadow = shadow.filter(ImageFilter.GaussianBlur(h * 0.006))
    layer = Image.alpha_composite(layer, shadow)
    pen = ImageDraw.Draw(layer)
    if left and placed:
        pen.rectangle([margin - w * 0.016, top, margin - w * 0.011, y], fill=(255, 255, 255, 230))
    for px, py, line, font, fill in placed:
        pen.text((px, py), line, font=font, fill=fill, anchor=anchor)
    return layer


def timing(length: float) -> tuple[float, float]:
    """When the title shows on a shot of this length: (start, end) in seconds, fades included."""
    start = min(DELAY, length * 0.1)
    end = max(start + 0.3, min(length - 0.2, start + LONGEST))
    return start, end


def strength(t: float, length: float) -> float:
    """How visible the title is at time t (0..1): fading in, staying, fading out."""
    start, end = timing(length)
    if t < start or t > end:
        return 0.0
    fade_in = min(FADE_IN, (end - start) / 3)
    fade_out = min(FADE_OUT, (end - start) / 3)
    return float(min(1.0, (t - start) / fade_in if fade_in else 1.0, (end - t) / fade_out if fade_out else 1.0))


def blend(frame: np.ndarray, layer: np.ndarray, amount: float) -> np.ndarray:
    """Lay the title (RGBA array) over an RGB frame at this strength."""
    if amount <= 0:
        return frame
    alpha = layer[..., 3:4].astype(np.float32) / 255 * amount
    return (frame * (1 - alpha) + layer[..., :3] * alpha).astype(np.uint8)


def ffmpeg_fades(length: float) -> str:
    """The same fades as an FFmpeg filter on the title picture (for videos)."""
    start, end = timing(length)
    fade_in = min(FADE_IN, (end - start) / 3)
    fade_out = min(FADE_OUT, (end - start) / 3)
    return (f"fade=t=in:st={start:.3f}:d={fade_in:.3f}:alpha=1,"
            f"fade=t=out:st={end - fade_out:.3f}:d={fade_out:.3f}:alpha=1")
