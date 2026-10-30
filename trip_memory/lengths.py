"""Reading how long you want the video, from your own words: "1:20", "80 seconds", "1 min 20",
"two and a half minutes", "the whole song", "half the song". The app reads this itself (the AI is not trusted with
numbers), and asks when your words could mean two lengths ("1.2 minutes": 1:12 or 1:20?)."""

from __future__ import annotations

import re

WORDS = {"a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8,
         "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "fifteen": 15, "twenty": 20, "thirty": 30, "forty": 40,
         "forty-five": 45, "fifty": 50, "sixty": 60, "ninety": 90}
PARTS = {"whole": 1.0, "full": 1.0, "entire": 1.0, "all": 1.0, "half": 0.5, "third": 1 / 3, "quarter": 0.25,
         "two thirds": 2 / 3, "three quarters": 0.75}
NUM = r"(\d+(?:[.,]\d+)?)"
MIN = r"(?:minutes?|mins?|m)\b"
SEC = r"(?:seconds?|secs?|s)\b"


def _words_to_digits(text: str) -> str:
    text = re.sub(r"\ban? (minute|min) and a half\b", "1.5 minutes", text)
    text = re.sub(r"\b(\w+) and a half\b", lambda m: f"{WORDS.get(m.group(1), m.group(1))}.5", text)
    text = re.sub(r"\bhalf an? (minute|min)\b", "30 seconds", text)
    text = re.sub(r"\b(" + "|".join(sorted(WORDS, key=len, reverse=True)) + r")[- ](minutes?|mins?|seconds?|secs?)\b",
                  lambda m: f"{WORDS[m.group(1)]} {m.group(2)}", text)
    return re.sub(r"(\d)-(minutes?|mins?|seconds?|secs?)\b", r"\1 \2", text)  # "2-minute video"


def read_length(text: str) -> dict | None:
    """What your words say about the length:
    {"seconds": 80} · {"song": 0.5} (a share of the song) · {"ask": [72, 80], "said": "1.2 minutes"} · None."""
    t = _words_to_digits(" ".join((text or "").lower().split()))
    if re.search(r"\b(as long as|length of|the whole length of|all of) the (song|music|track)\b", t):
        return {"song": 1.0}
    m = re.search(r"\b(two thirds|three quarters|whole|full|entire|half|third|quarter)\b(?: of)? (?:the )?"
                  r"(song|music|track)\b", t)
    if m:
        return {"song": PARTS[m.group(1)]}
    m = re.search(r"\b(\d{1,2}):([0-5]\d)\b", t)  # 1:20
    if m:
        return {"seconds": int(m.group(1)) * 60 + int(m.group(2))}
    m = re.search(rf"\b{NUM} ?{MIN}(?: and)? ?(\d{{1,2}}) ?(?:{SEC})?", t)  # 1 min 20 (s), 2 minutes and 30 seconds
    if m and "." not in m.group(1) and "," not in m.group(1):
        return {"seconds": int(m.group(1)) * 60 + int(m.group(2))}
    m = re.search(rf"\b{NUM} ?{MIN}", t)
    if m:
        whole, _, frac = m.group(1).replace(",", ".").partition(".")
        if not frac or frac.rstrip("0") in ("", "5", "25", "75"):
            return {"seconds": round(float(f"{whole}.{frac or 0}") * 60)}
        decimal = round(float(f"{whole}.{frac}") * 60)
        clock = int(whole) * 60 + int(frac.ljust(2, "0")) if int(frac.ljust(2, "0")) < 60 else None
        if clock is None or clock == decimal:
            return {"seconds": decimal}
        return {"ask": sorted({decimal, clock}), "said": m.group(0).strip()}
    m = re.search(rf"\b{NUM} ?{SEC}", t)
    if m:
        return {"seconds": round(float(m.group(1).replace(",", ".")))}
    return None


def as_words(length: dict | None) -> str:
    if not length:
        return ""
    if "song" in length:
        share = length["song"]
        return {1.0: "the whole song", 0.5: "half the song", 0.25: "a quarter of the song",
                0.75: "three quarters of the song"}.get(round(share, 2), f"{share:.0%} of the song")
    s = round(length["seconds"])
    return f"{s // 60}:{s % 60:02d}"
