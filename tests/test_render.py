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


def test_prepared_clips_are_reused(tmp_path):
    trip = tmp_path / "trip"
    trip.mkdir()
    make_clip(trip / "a.mp4", "640x360", 2, "2025-07-03T10:00:00Z")
    Image.new("RGB", (400, 300), "blue").save(trip / "b.jpg")
    items, _ = scan_folder(trip)
    cache = tmp_path / "cache"
    settings = RenderSettings(width=640, height=360, photo_seconds=2)

    render(items, settings, tmp_path / "one.mp4", cache_dir=cache)
    first = {p.name: p.stat().st_mtime_ns for p in cache.glob("*.mp4")}
    assert len(first) == 2
    render(items, settings, tmp_path / "two.mp4", cache_dir=cache)
    assert {p.name for p in cache.glob("*.mp4")} == set(first)

    # A different photo length needs a new photo clip, but the video clip is reused.
    render(items, RenderSettings(width=640, height=360, photo_seconds=3), tmp_path / "three.mp4", cache_dir=cache)
    assert len(list(cache.glob("*.mp4"))) == 3


def test_turned_clips_get_their_own_prepared_clip(tmp_path):
    trip = tmp_path / "trip"
    trip.mkdir()
    make_clip(trip / "a.mp4", "640x360", 2, "2025-07-03T10:00:00Z")
    Image.new("RGB", (400, 300), "blue").save(trip / "b.jpg")
    items, _ = scan_folder(trip)
    cache = tmp_path / "cache"
    settings = RenderSettings(width=640, height=360, photo_seconds=2)
    render(items, settings, tmp_path / "one.mp4", cache_dir=cache)
    for item in items:
        item.rotation = 90
    out = render(items, settings, tmp_path / "two.mp4", cache_dir=cache)
    assert len(list(cache.glob("*.mp4"))) == 4
    assert abs(float(probe(str(out))["format"]["duration"]) - (4 - settings.transition_seconds)) < 0.3


def _loudness(path):
    out = subprocess.run([ffmpeg_paths()[0], "-hide_banner", "-i", str(path), "-af", "volumedetect", "-f", "null", "-"],
                         capture_output=True, text=True).stderr
    return float(out.split("mean_volume:")[1].split("dB")[0])


def test_clip_sound_can_be_switched_off_and_made_louder(tmp_path):
    trip = tmp_path / "trip"
    trip.mkdir()
    make_clip(trip / "a.mp4", "640x360", 2, "2025-07-03T10:00:00Z")
    items, _ = scan_folder(trip)
    settings = RenderSettings(width=320, height=180)
    items[0].sound = False
    silent = _loudness(render(items, settings, tmp_path / "silent.mp4"))
    items[0].sound, items[0].volume = True, 1.0
    normal = _loudness(render(items, settings, tmp_path / "normal.mp4"))
    items[0].volume = 2.0
    louder = _loudness(render(items, settings, tmp_path / "louder.mp4"))
    assert silent < -80 and normal > -40 and louder > normal + 4


def test_joining_survives_clips_that_start_late_or_use_another_clock(tmp_path):
    """Clips from the Mac's video chip can start slightly after 0 or use another clock; joining must not stop."""
    from trip_memory.render import _join, _segment_length

    ffmpeg = ffmpeg_paths()[0]
    base = [ffmpeg, "-loglevel", "error", "-y", "-f", "lavfi", "-i", "testsrc2=s=320x180:r=30:d=3",
            "-f", "lavfi", "-i", "sine=d=3", "-c:v", "libx264", "-c:a", "aac", "-ar", "48000", "-ac", "2"]
    subprocess.run([*base, str(tmp_path / "a.mp4")], check=True)
    subprocess.run([*base, "-video_track_timescale", "30", str(tmp_path / "b.mp4")], check=True)
    subprocess.run([*base, "-output_ts_offset", "2", str(tmp_path / "c.mp4")], check=True)
    segments = [tmp_path / n for n in ("a.mp4", "b.mp4", "c.mp4", "a.mp4")]
    out = tmp_path / "out.mp4"
    settings = RenderSettings(width=320, height=180, transition_seconds=0.8)
    _join(segments, [_segment_length(s) for s in segments], [None, "fade", None], settings, out, lambda _f: None)
    assert abs(float(probe(str(out))["format"]["duration"]) - (12 - 0.8 - 2 / 30)) < 0.3
