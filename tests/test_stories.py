"""Stories and sets of similar shots."""

from PIL import Image, ImageFilter

from trip_memory import stories



def _row(tmp_path, name, when, img, lat=38.72, lon=-9.14, kind="photo"):
    path = tmp_path / f"{name}.jpg"
    img.save(path)
    dhash, colours, sharp = stories.fingerprint(str(path))
    return {"path": str(path), "kind": kind, "taken_at": when, "duration": 0, "lat": lat, "lon": lon,
            "dhash": dhash, "colours": colours, "sharpness": sharp}


def _picture(colour, size=(480, 320)):
    """A simple scene: a coloured sky over a darker ground with a sun."""
    img = Image.new("RGB", size, colour)
    ground = Image.new("RGB", (size[0], size[1] // 3), (40, 90, 40))
    img.paste(ground, (0, size[1] * 2 // 3))
    for x in range(60, 100):
        for y in range(40, 80):
            img.putpixel((x, y), (250, 240, 120))
    return img


def test_bursts_become_one_set_and_the_sharpest_is_best(tmp_path):
    scene = _picture((120, 170, 230))
    rows = [
        _row(tmp_path, "a1", "2025-07-03T10:00:00+02:00", scene.filter(ImageFilter.GaussianBlur(4))),
        _row(tmp_path, "a2", "2025-07-03T10:00:20+02:00", scene),
        _row(tmp_path, "a3", "2025-07-03T10:00:40+02:00", scene.crop((10, 5, 470, 315)).resize((480, 320))),
        # 20 minutes later, a different scene: same story, new set
        _row(tmp_path, "b1", "2025-07-03T10:20:00+02:00", _picture((230, 120, 60))),
        # Two hours later: a new story
        _row(tmp_path, "c1", "2025-07-03T12:30:00+02:00", _picture((120, 170, 230))),
        # Far away the same afternoon: also a new story
        _row(tmp_path, "d1", "2025-07-03T12:40:00+02:00", _picture((120, 170, 230)), lat=38.80),
    ]
    g = {k.split("/")[-1][:-4]: v for k, v in stories.group(rows).items()}
    assert g["a1"][:2] == g["a2"][:2] == g["a3"][:2]  # one story, one set
    assert g["a2"][2] and not g["a1"][2]  # the sharp one wins over the blurred one
    assert g["b1"][0] == g["a1"][0] and g["b1"][1] != g["a1"][1]
    assert g["c1"][0] != g["a1"][0]
    assert g["d1"][0] != g["c1"][0]
    assert all(v[2] for k, v in g.items() if k[0] in "bcd")  # single shots are their own best


def test_a_burst_stays_together_with_people_moving_and_other_shots_in_between(tmp_path):
    import numpy as np

    rng = np.random.default_rng(0)
    scene = Image.fromarray(np.kron(rng.integers(0, 255, (8, 12, 3)), np.ones((40, 40, 1))).astype(np.uint8))
    other = _picture((230, 120, 60))

    def moved(dx, zoom):
        w, h = scene.size
        crop = scene.crop((dx, dx // 2, int(w * zoom) + dx, int(h * zoom) + dx // 2)).resize((w, h))
        a = np.array(crop)
        a[100:220, 150:250] = np.roll(a[100:220, 150:250], 60, axis=1)  # someone moved
        return Image.fromarray(a)

    def row(name, minute, img):
        r = _row(tmp_path, name, f"2025-07-03T10:{minute:02d}:00+02:00", img)
        r["features"] = stories.features(r["path"])
        return r

    rows = [row("a1", 0, scene), row("b1", 2, other), row("a2", 4, moved(12, 0.9)), row("a3", 9, moved(30, 0.85))]
    g = {k.split("/")[-1][:-4]: v[1] for k, v in stories.group(rows, "normal").items()}
    assert g["a1"] == g["a2"] == g["a3"] != g["b1"]  # level 2: the same scene within 10 minutes
    burst = {k.split("/")[-1][:-4]: v[1] for k, v in stories.group(rows).items()}
    assert burst["a3"] != burst["a1"]  # level 1 (bursts): 5 minutes after the last one is another moment
    late = row("a4", 12, moved(12, 0.9))
    g = {k.split("/")[-1][:-4]: v[1] for k, v in stories.group([*rows, late], "normal").items()}
    assert g["a4"] != g["a1"]  # never more than 10 minutes from the set's first shot
