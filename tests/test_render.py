"""End-to-end check: make a tiny trip folder, build a video, confirm its length and size."""

import subprocess

from PIL import Image

from trip_memory.media import scan_folder
from trip_memory.render import RenderSettings, render
from trip_memory.tools import ffmpeg_paths, probe


def make_clip(path, size, seconds, when, audio=True):
    ffmpeg = ffmpeg_paths()[0]
    args = [ffmpeg, "-loglevel", "error", "-y", "-f", "lavfi", "-i", f"testsrc2=s={size}:r=30:d={seconds}"]
    if audio:
        args += ["-f", "lavfi", "-i", f"sine=d={seconds}", "-shortest"]
    subprocess.run([*args, "-metadata", f"creation_time={when}", "-c:v", "libx264", str(path)], check=True)


def test_builds_video_in_date_order(tmp_path):
    trip = tmp_path / "trip"
    (trip / "day2").mkdir(parents=True)
    make_clip(trip / "late.mp4", "640x360", 3, "2025-07-04T10:00:00Z")
    make_clip(trip / "early.mov", "360x640", 4, "2025-07-03T10:00:00Z", audio=False)
    exif = Image.Exif()
    exif[306] = "2025:07:03 12:00:00"
    exif[274] = 6  # taken sideways
    Image.new("RGB", (600, 800), "orange").save(trip / "day2" / "photo.jpg", exif=exif)

    items, problems = scan_folder(trip)
    assert problems == []
    assert items[0].name == "early.mov"
    assert {i.kind for i in items} == {"video", "photo"}

    settings = RenderSettings(width=640, height=360, photo_seconds=2, transition="Mixed", transition_seconds=0.5)
    out = render(items, settings, tmp_path / "out.mp4")
    info = probe(str(out))
    video = next(s for s in info["streams"] if s["codec_type"] == "video")
    assert (video["width"], video["height"]) == (640, 360)
    assert any(s["codec_type"] == "audio" for s in info["streams"])
    expected = 4 + 3 + 2 - 2 * 0.5
    assert abs(float(info["format"]["duration"]) - expected) < 0.3
