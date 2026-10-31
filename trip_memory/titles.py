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
    # (the looks of the opening and ending cards: see CARDS)
    "cinematic": {"label": "Cinematic", "fonts": [MAC + "Futura.ttc", "/System/Library/Fonts/Avenir Next.ttc",
                                                  "/usr/share/fonts/opentype/inter/Inter-Medium.otf"],
                  "size": 0.065, "upper": True, "where": "center", "spacing": 0.32},
    "minimal": {"label": "Minimal", "fonts": ["/System/Library/Fonts/HelveticaNeue.ttc", MAC + "Arial.ttf",
                                              "/usr/share/fonts/opentype/inter/Inter-Light.otf"],
                "size": 0.055, "upper": False, "where": "center", "spacing": 0.06, "color": (235, 235, 235)},
    "typewriter": {"label": "Typewriter", "fonts": [MAC + "American Typewriter.ttc", MAC + "Courier New Bold.ttf",
                                                    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf"],
                   "size": 0.075, "upper": False, "where": "center", "color": (250, 236, 205)},
    "summer": {"label": "Summer", "fonts": ["/System/Library/Fonts/MarkerFelt.ttc", MAC + "Chalkboard.ttc",
                                            "/usr/share/fonts/opentype/inter/InterDisplay-Black.otf"],
               "size": 0.12, "upper": False, "where": "center", "color": (255, 214, 92)},
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


def _length(draw: ImageDraw.ImageDraw, text: str, font, spacing: float = 0.0) -> float:
    """How wide the words are, with spacing (a share of the letters' size) between the letters."""
    if not spacing:
        return draw.textlength(text, font=font)
    gap = spacing * font.size
    return sum(draw.textlength(ch, font=font) for ch in text) + gap * max(0, len(text) - 1)


def _text(draw: ImageDraw.ImageDraw, x: float, y: float, text: str, font, fill, anchor: str, spacing: float = 0.0) -> None:
    if not spacing:
        draw.text((x, y), text, font=font, fill=fill, anchor=anchor)
        return
    if anchor == "ma":
        x -= _length(draw, text, font, spacing) / 2
    for ch in text:
        draw.text((x, y), ch, font=font, fill=fill, anchor="la")
        x += draw.textlength(ch, font=font) + spacing * font.size


def _wrap(draw: ImageDraw.ImageDraw, text: str, font, width: float, spacing: float = 0.0) -> list[str]:
    lines: list[str] = []
    for paragraph in text.split("\n"):
        line = ""
        for word in paragraph.split():
            test = f"{line} {word}".strip()
            if line and _length(draw, test, font, spacing) > width:
                lines.append(line)
                line = word
            else:
                line = test
        lines.append(line)
    return [x for x in lines if x] or [""]


def draw(title: dict, w: int, h: int) -> Image.Image:
    """The title as a transparent picture the size of the video. title: {"text", "sub", "style", "where"}
    (and "color": [r, g, b] to change the style's colour)."""
    style = title.get("style") if title.get("style") in STYLES else "clean"
    spec = STYLES[style]
    where = title.get("where") if title.get("where") in WHERE else spec["where"]
    text = (title.get("text") or "").strip()
    sub = (title.get("sub") or "").strip()
    if spec["upper"]:
        text = text.upper()
    ref = min(w, h)  # an upright phone video gets letters sized to its width, not its height
    big = _font(style, round(ref * spec["size"]))
    small = _font("caption" if style != "caption" else "clean", round(ref * (0.04 if style != "caption" else 0.035)))
    layer = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    pen = ImageDraw.Draw(layer)
    margin = w * 0.06
    spacing = spec.get("spacing", 0.0)
    color = tuple(title.get("color") or spec.get("color") or (255, 255, 255))
    soft = tuple(min(255, c + 15) for c in color) if color != (255, 255, 255) else (240, 240, 240)
    lines = _wrap(pen, text, big, w - 2 * margin, spacing) if text else []
    sub_lines = _wrap(pen, sub, small, w - 2 * margin) if sub else []
    gap = ref * 0.012
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
        placed.append((x, y, line, big, (*color, 255)))
        y += big_h
    if lines and sub_lines:
        y += gap * 2
    for line in sub_lines:
        placed.append((x, y, line, small, (*soft, 255)))
        y += small_h
    # A soft shadow under the letters so they read on bright and busy pictures.
    shadow = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    shade = ImageDraw.Draw(shadow)
    for px, py, line, font, _ in placed:
        _text(shade, px, py + h * 0.004, line, font, (0, 0, 0, 170), anchor, spacing if font is big else 0.0)
    if left and placed:  # a thin bar beside a caption
        shade.rectangle([margin - w * 0.016, top, margin - w * 0.011, y], fill=(0, 0, 0, 170))
    shadow = shadow.filter(ImageFilter.GaussianBlur(h * 0.006))
    layer = Image.alpha_composite(layer, shadow)
    pen = ImageDraw.Draw(layer)
    if left and placed:
        pen.rectangle([margin - w * 0.016, top, margin - w * 0.011, y], fill=(255, 255, 255, 230))
    for px, py, line, font, fill in placed:
        _text(pen, px, py, line, font, fill, anchor, spacing if font is big else 0.0)
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


# The opening and ending cards: words on black, or on the shot itself blurred and darkened.
# bg: "black" or "blur"; dim: how much darker the blurred picture gets (0..1); tone: "" or "sepia" or "warm";
# bars: black bars above and below, like a cinema film.
CARDS = {
    "cinematic": {"label": "Cinematic", "bg": "blur", "dim": 0.45, "tone": "", "bars": True, "style": "cinematic"},
    "journal": {"label": "Travel journal", "bg": "blur", "dim": 0.3, "tone": "warm", "bars": False,
                "style": "handwritten", "color": (255, 246, 226)},
    "bold": {"label": "Bold", "bg": "black", "dim": 0, "tone": "", "bars": False, "style": "bold"},
    "minimal": {"label": "Minimal", "bg": "black", "dim": 0, "tone": "", "bars": False, "style": "minimal"},
    "retro": {"label": "Retro film", "bg": "blur", "dim": 0.35, "tone": "sepia", "bars": False, "style": "typewriter"},
    "summer": {"label": "Summer", "bg": "blur", "dim": 0.15, "tone": "warm", "bars": False, "style": "summer"},
}
CARD_FADE = 0.8  # seconds for the opening card to give way to the shot
BARS = 0.11  # share of the height each cinema bar covers


def card_words(card: str, title: dict) -> dict:
    """The card's words drawn in its own style (and colour)."""
    spec = CARDS.get(card) or CARDS["bold"]
    return {"text": title.get("text") or "", "sub": title.get("sub") or "", "style": spec["style"], "where": "center",
            **({"color": spec["color"]} if spec.get("color") else {})}


def card_layer(card: str, title: dict, w: int, h: int) -> Image.Image:
    """What goes over the card's background: its words (and the cinema bars), see-through elsewhere.
    On a black card the background is part of it."""
    spec = CARDS.get(card) or CARDS["bold"]
    layer = Image.new("RGBA", (w, h), (0, 0, 0, 255 if spec["bg"] == "black" else 0))
    if spec["bars"]:
        bar = round(h * BARS)
        pen = ImageDraw.Draw(layer)
        pen.rectangle([0, 0, w, bar], fill=(0, 0, 0, 255))
        pen.rectangle([0, h - bar, w, h], fill=(0, 0, 0, 255))
    if title.get("text") or title.get("sub"):
        layer = Image.alpha_composite(layer, draw(card_words(card, title), w, h))
    return layer


def card_filter(card: str, h: int) -> str:
    """The FFmpeg filter that turns the shot into the card's background (blurred, darker, toned)."""
    spec = CARDS.get(card) or CARDS["bold"]
    radius = max(2, min(40, h // 30))
    out = f"boxblur={radius}:2,eq=brightness={-0.35 * spec['dim']:.3f}:saturation={1 - 0.4 * spec['dim']:.3f}"
    if spec["tone"] == "sepia":
        out += ",colorchannelmixer=.393:.769:.189:0:.349:.686:.168:0:.272:.534:.131"
    elif spec["tone"] == "warm":
        out += ",colorbalance=rs=.08:bs=-.08:rm=.05:bm=-.05"
    return out


def card_css(card: str) -> str:
    """The same background in the app's quick player (a CSS filter)."""
    spec = CARDS.get(card) or CARDS["bold"]
    if spec["bg"] == "black":
        return ""
    tone = {"sepia": " sepia(0.85)", "warm": " sepia(0.25) saturate(1.2)"}.get(spec["tone"], "")
    return f"blur(12px) brightness({1 - 0.55 * spec['dim']:.2f}){tone}"


def card_picture(card: str, title: dict, picture: Image.Image | None, w: int, h: int) -> Image.Image:
    """A still of the card (to choose a look): the words over the picture blurred, or over black."""
    spec = CARDS.get(card) or CARDS["bold"]
    back = Image.new("RGB", (w, h), (0, 0, 0))
    if spec["bg"] == "blur" and picture is not None:
        pic = picture.convert("RGB")
        scale = max(w / pic.width, h / pic.height)
        pic = pic.resize((max(1, round(pic.width * scale)), max(1, round(pic.height * scale))))
        left, top = (pic.width - w) // 2, (pic.height - h) // 2
        pic = pic.crop((left, top, left + w, top + h)).filter(ImageFilter.GaussianBlur(h / 40))
        arr = np.asarray(pic).astype(np.float32) * (1 - 0.55 * spec["dim"])
        if spec["tone"] == "sepia":
            arr = arr @ np.array([[.393, .349, .272], [.769, .686, .534], [.189, .168, .131]], dtype=np.float32)
        elif spec["tone"] == "warm":
            arr = arr * np.array([1.08, 1.0, 0.9], dtype=np.float32)
        back = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    return Image.alpha_composite(back.convert("RGBA"), card_layer(card, title, w, h)).convert("RGB")
