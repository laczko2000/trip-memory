import numpy as np
from PIL import Image, ImageFilter

from trip_memory import quality


def _save(path, arr):
    Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8)).save(path)
    return str(path)


def test_the_quality_check_finds_bad_shots_and_keeps_good_ones(tmp_path):
    rng = np.random.default_rng(1)
    blocks = np.kron(rng.integers(30, 220, (9, 12)), np.ones((40, 40)))  # shapes in many shades ...
    scene = (blocks + rng.normal(0, 25, (360, 480))).astype(np.float32)  # ... with lots of sharp detail
    sharp = _save(tmp_path / "sharp.jpg", scene)
    soft = tmp_path / "soft.jpg"
    Image.open(sharp).filter(ImageFilter.GaussianBlur(6)).save(soft)
    # A sharp subject in one corner, the rest soft (a portrait with a blurred background): still sharp.
    portrait = np.asarray(Image.open(soft), dtype=np.float32).copy()
    portrait[:120, :160] = scene[:120, :160]
    portrait = _save(tmp_path / "portrait.jpg", portrait)
    black = _save(tmp_path / "black.jpg", np.full((360, 480), 8.0))
    white = _save(tmp_path / "white.jpg", np.full((360, 480), 254.0) + rng.normal(0, 1, (360, 480)))
    wall = _save(tmp_path / "wall.jpg", np.full((360, 480), 140.0) + rng.normal(0, 2, (360, 480)))
    small = tmp_path / "small.jpg"
    Image.open(sharp).resize((400, 300)).save(small)
    shot = tmp_path / "Screenshot 2026-04-02.png"
    Image.open(sharp).save(shot)

    def row(path, kind="photo", **extra):
        return {"path": str(path), "kind": kind, "quality": quality.measure(str(path), str(path), kind), **extra}

    big = {"w": 4000, "h": 3000}
    rows = [row(sharp), row(soft), row(portrait), row(black), row(white), row(wall), row(small), row(shot),
            row(sharp, "video", duration=0.4), row(tmp_path / "sharp.jpg", "video", duration=8.0, motion_shake=3.0)]
    rows[-1]["path"] = "shaky.mp4"
    rows[-2]["path"] = "oops.mp4"
    for r in rows[:6]:
        r["quality"].update(big)
    found = quality.flags(rows, "normal")
    reason = {k.split("/")[-1]: v[0] for k, v in found.items()}
    assert "sharp.jpg" not in reason and "portrait.jpg" not in reason
    assert reason["soft.jpg"] == "blurry"
    assert reason["black.jpg"] == "dark"
    assert reason["white.jpg"] == "bright"
    assert reason["wall.jpg"] == "empty"
    assert reason["small.jpg"] == "tiny"
    assert reason["Screenshot 2026-04-02.png"] == "screenshot"
    assert reason["oops.mp4"] == "short" and reason["shaky.mp4"] == "shaky"
    gentle = quality.flags(rows, "gentle")
    assert len(gentle) <= len(found)
