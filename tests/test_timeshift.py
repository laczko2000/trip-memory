"""Camera clocks: reading a shift, and finding a camera's clock error from shots of the same moment."""

import numpy as np

from trip_memory import timeshift


def test_reading_a_shift():
    p = timeshift.parse_shift
    assert p("+2:00") == 7200
    assert p("-1:30:15") == -(5400 + 15)
    assert p("2h") == 7200 and p("-90 min") == -5400 and p("1 h 30 m") == 5400 and p("45s") == 45
    assert p("-2 days") == -2 * 86400 and p("1.5") == 5400
    assert p("") is None and p("soon") is None
    assert timeshift.fmt_shift(7200) == "+2:00" and timeshift.fmt_shift(-330) == "-0:05:30"
    assert timeshift.fmt_shift(86400 + 3600) == "+1 day 1:00"
    assert timeshift.shift_words(-7200) == "2 hours earlier"


def test_camera_names():
    assert timeshift.camera_name("Apple", "iPhone 15 Pro") == "iPhone 15 Pro"
    assert timeshift.camera_name("SAMSUNG", "SAMSUNG SM-G991B") == "SAMSUNG SM-G991B"
    info = {"format": {"tags": {}}, "streams": [{"tags": {"handler_name": "\tGoPro AVC  "}}]}
    from pathlib import Path
    assert timeshift.camera_from_probe(info, Path("a.mp4")) == "GoPro"
    assert timeshift.camera_from_probe({"format": {"tags": {}}, "streams": []}, Path("GX010042.MP4")) == "GoPro"
    apple = {"format": {"tags": {"com.apple.quicktime.make": "Apple", "com.apple.quicktime.model": "iPhone 13"}}}
    assert timeshift.camera_from_probe(apple, Path("IMG_1.MOV")) == "iPhone 13"
    assert timeshift.camera_from_probe({}, Path("clip.mp4")) == timeshift.OTHER


def test_the_shift_most_shots_agree_on_wins():
    rng = np.random.default_rng(1)
    scenes = rng.normal(size=(8, 32))
    error = 2 * 3600 + 40  # the GoPro is 2 hours (and a bit) behind
    phone = [(1000.0 * k, scenes[k] + rng.normal(scale=0.1, size=32)) for k in range(8)]
    gopro = [(1000.0 * k + 30 - error, scenes[k] + rng.normal(scale=0.1, size=32)) for k in (1, 3, 4, 6)]
    gopro.append((50.0, rng.normal(size=32)))  # a shot nobody else filmed
    found = timeshift.suggest(gopro, phone)
    assert found["seconds"] == 2 * 3600  # rounded to the time zone
    assert found["shots"] == 4 and found["of"] == 5
    assert timeshift.suggest(gopro[:1], phone) is None
