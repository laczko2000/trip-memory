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
    _join(segments, [_segment_length(s) for s in segments], [(None, 0.8), ("fade", 0.8), (None, 0.8)], settings, out, lambda _f: None)
    assert abs(float(probe(str(out))["format"]["duration"]) - (12 - 0.8 - 2 / 30)) < 0.3


def test_slow_motion_and_speed_up_change_the_length(tmp_path):
    trip = tmp_path / "trip"
    trip.mkdir()
    make_clip(trip / "a.mp4", "640x360", 2, "2025-07-03T10:00:00Z")
    items, _ = scan_folder(trip)
    settings = RenderSettings(width=320, height=180)
    for speed, expected in ((0.5, 4.0), (1.5, 2 / 1.5)):
        items[0].speed, items[0].sound = speed, True
        out = render(items, settings, tmp_path / f"s{speed}.mp4")
        assert abs(float(probe(str(out))["format"]["duration"]) - expected) < 0.15


def test_a_short_join_is_tried_again_and_reported(tmp_path, monkeypatch):
    from trip_memory import render as r

    calls = []

    def fake_join(segments, lengths, styles, s, out, progress):
        calls.append(r._hardware["ok"])
        return 2.0 if len(calls) == 1 else sum(lengths) - 0.1  # the first try stops early

    monkeypatch.setattr(r, "_join_once", fake_join)
    monkeypatch.setattr(r, "probe", lambda _p: {"streams": []})
    seg = tmp_path / "a.mp4"
    seg.write_bytes(b"")
    r._join([seg, seg], [5.0, 5.0], [(None, 0.8)], RenderSettings(), tmp_path / "v.mp4", lambda _f, _m="": None)
    assert len(calls) == 2 and (tmp_path / "v.report.txt").exists()


def test_quick_transitions_join_to_full_length(tmp_path):
    """A real 38-clip cut with short effects (0.25 s, 0.35 s: not whole frames) once stopped at 26 of 61 s."""
    from trip_memory.render import _join_once, _planned_length, _segment_length

    frames = [71, 66, 54, 53, 51, 52, 47, 43, 43, 48, 66, 47, 46, 51, 47, 61, 47, 57, 62, 52, 47, 60, 51, 79, 66, 51,
              47, 48, 50, 47, 75, 76, 43, 46, 47, 57, 44, 42]
    sound = [2.346, 2.2, 1.8, 1.749, 1.7, 1.733, 1.557, 1.433, 1.433, 1.6, 2.2, 1.557, 1.533, 1.7, 1.557, 2.033, 1.557,
             1.9, 2.048, 1.733, 1.557, 2.0, 1.7, 2.633, 2.2, 1.7, 1.557, 1.6, 1.664, 1.557, 2.5, 2.533, 1.433, 1.533,
             1.557, 1.9, 1.45, 1.4]
    cut = (None, 1 / 30)
    styles = [("fadewhite", 0.3), ("hblur", 0.4), ("hblur", 0.4), ("fadewhite", 0.3), ("fadewhite", 0.3),
              ("fadewhite", 0.3), cut, cut, cut, ("zoomin", 0.35), ("fadewhite", 0.3), cut, ("slideleft", 0.25),
              ("fadewhite", 0.3), cut, ("slideup", 0.25), cut, cut, ("zoomin", 0.35), ("fadewhite", 0.3), cut,
              ("slideleft", 0.25), ("fadewhite", 0.3), ("slideup", 0.25), ("zoomin", 0.35), ("fadewhite", 0.3), cut,
              ("fadewhite", 0.3), ("slideleft", 0.25), cut, ("fadewhite", 0.3), cut, cut, ("slideup", 0.25), cut, cut,
              cut]
    ffmpeg = ffmpeg_paths()[0]
    segments = []
    for k, (n, audio) in enumerate(zip(frames, sound)):
        seg = tmp_path / f"{k}.mp4"
        subprocess.run([ffmpeg, "-v", "error", "-y", "-f", "lavfi", "-i", "testsrc2=s=96x54:r=30", "-f", "lavfi",
                        "-i", "sine=sample_rate=48000", "-frames:v", str(n), "-af", f"atrim=0:{audio}",
                        "-c:v", "libx264", "-video_track_timescale", "15360", "-c:a", "aac", "-ac", "2", str(seg)],
                       check=True)
        segments.append(seg)
    lengths = [_segment_length(s) for s in segments]
    settings = RenderSettings(width=96, height=54)
    made = _join_once(segments, lengths, styles, settings, tmp_path / "out.mp4", lambda _f: None)
    assert abs(made - _planned_length(lengths, styles, settings)) < 0.1


def test_zooms_punch_ins_and_speed_ramps_keep_the_length(tmp_path):
    trip = tmp_path / "trip"
    trip.mkdir()
    make_clip(trip / "a.mp4", "640x360", 4, "2025-07-03T10:00:00Z")
    make_clip(trip / "b.mp4", "640x360", 4, "2025-07-03T10:01:00Z")
    make_clip(trip / "c.mp4", "640x360", 4, "2025-07-03T10:02:00Z")
    items, _ = scan_folder(trip)
    a, b, c = items
    a.transition, a.effects = "Zoom through", {"push": True}
    b.trim_start, b.trim_end, b.effects, b.sound = 0.0, 2.8, {"ramp": True, "punch": [0.5]}, True  # 2 s on screen
    b.transition = "Zoom out reveal"
    c.speed = 0.5  # smooth slow motion
    c.trim_start, c.trim_end = 0.0, 1.5
    out = render(items, RenderSettings(width=320, height=180, transition_seconds=0.5), tmp_path / "fx.mp4")
    assert abs(float(probe(str(out))["format"]["duration"]) - (4 + 2 + 3)) < 0.2


def _frame(path, t, w=640, h=360):
    raw = subprocess.run([ffmpeg_paths()[0], "-loglevel", "error", "-ss", str(t), "-i", str(path), "-frames:v", "1",
                          "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], capture_output=True, check=True).stdout
    import numpy as np

    return np.frombuffer(raw, np.uint8).reshape(h, w, 3)


def test_titles_fade_in_over_videos_and_photos(tmp_path):
    import dataclasses

    trip = tmp_path / "trip"
    trip.mkdir()
    make_clip(trip / "a.mp4", "640x360", 3, "2025-07-03T10:00:00Z")
    Image.new("RGB", (400, 300), (0, 0, 120)).save(trip / "b.jpg")
    title = {"text": "PORTUGAL", "sub": "July", "style": "bold", "where": "center"}
    settings = RenderSettings(width=640, height=360, photo_seconds=3)
    for item in scan_folder(trip)[0]:
        out = render([dataclasses.replace(item, overlay=title)], settings, tmp_path / f"{item.kind}.mp4")
        assert abs(float(probe(str(out))["format"]["duration"]) - 3.0) < 0.2
        plain, titled = _frame(out, 0.05), _frame(out, 1.5)
        centre = (slice(150, 210), slice(200, 440))
        assert titled[centre].min(axis=2).max() > 220 > plain[centre].min(axis=2).max(), item.kind  # letters fade in


def test_the_video_ends_on_black_with_the_end_card_and_the_music_fading(tmp_path):
    trip = tmp_path / "trip"
    trip.mkdir()
    Image.new("RGB", (640, 360), (200, 200, 200)).save(trip / "a.jpg")
    music = tmp_path / "song.wav"
    subprocess.run([ffmpeg_paths()[0], "-loglevel", "error", "-y", "-f", "lavfi", "-i", "sine=f=440:d=20", str(music)],
                   check=True)
    settings = RenderSettings(width=640, height=360, photo_seconds=3, music_path=str(music), music_volume=1.0,
                              end_black=2.0, end_title={"text": "THE END", "style": "bold", "where": "center"})
    out = render(scan_folder(trip)[0], settings, tmp_path / "out.mp4")
    length = float(probe(str(out))["format"]["duration"])
    assert abs(length - (3 + 2.0 + 1.2 - 1.2)) < 0.2  # the photo, then black (the fade overlaps the two)
    black = _frame(out, 3.8)
    assert black[:100].max() < 30  # dark above the title
    assert black[150:210, 200:440].max() > 200  # the end card's letters
    loud = subprocess.run([ffmpeg_paths()[0], "-ss", f"{length - 0.4:.2f}", "-i", str(out), "-af", "volumedetect",
                           "-f", "null", "-"], capture_output=True, text=True).stderr
    assert float(loud.split("max_volume:")[1].split("dB")[0]) < -12  # the music has faded out


def test_shake_on_a_hit_and_a_freeze_frame_with_a_name(tmp_path):
    import numpy as np

    trip = tmp_path / "trip"
    trip.mkdir()
    make_clip(trip / "a.mp4", "640x360", 4, "2025-07-03T10:00:00Z")
    make_clip(trip / "b.mp4", "640x360", 4, "2025-07-03T10:01:00Z")
    a, b = scan_folder(trip)[0]
    a.trim_start, a.trim_end, a.effects = 0.0, 2.0, {"shake": [1.0]}
    b.trim_start, b.trim_end, b.effects = 0.0, 2.0, {"freeze": 1.0, "label": "Anna"}  # 2 s playing + 1 s frozen
    settings = RenderSettings(width=640, height=360)
    shaken = render([a], settings, tmp_path / "shake.mp4")
    assert abs(float(probe(str(shaken))["format"]["duration"]) - 2.0) < 0.1
    assert _frame(shaken, 1.03).mean() > _frame(shaken, 0.9).mean() + 15  # the flash on the hit
    a.effects = None
    cut = RenderSettings(width=640, height=360, transition="None (hard cut)")
    frozen = render([b, a], cut, tmp_path / "freeze.mp4")  # (the video's own fade-out is on the last shot)
    assert abs(float(probe(str(frozen))["format"]["duration"]) - 5.0) < 0.1
    late, later = _frame(frozen, 2.5).astype(int), _frame(frozen, 2.9).astype(int)
    assert np.abs(late - later).mean() < 2  # the picture holds still
    corner = (slice(260, 340), slice(30, 200))  # the name, bottom left
    assert late[corner].min(axis=2).max() > 220 > _frame(frozen, 1.0)[corner].min(axis=2).max()


def test_a_look_colours_the_whole_video(tmp_path):
    from trip_memory import looks

    trip = tmp_path / "trip"
    trip.mkdir()
    make_clip(trip / "a.mp4", "640x360", 2, "2025-07-03T10:00:00Z")
    Image.new("RGB", (400, 300), (200, 60, 30)).save(trip / "b.jpg")
    items = scan_folder(trip)[0]
    out = render(items, RenderSettings(width=640, height=360, photo_seconds=2, look="mono"), tmp_path / "bw.mp4")
    for t in (1.0, 2.8):
        rgb = _frame(out, t).astype(int)
        assert abs(rgb[..., 0] - rgb[..., 2]).mean() < 6  # no colour left
    warm = looks.preview(trip / "b.jpg", "summer")
    assert warm.startswith("data:image/jpeg;base64,") and warm != looks.preview(trip / "b.jpg", "natural")


def test_a_phone_video_fills_the_upright_frame_around_the_people(tmp_path):
    import dataclasses

    trip = tmp_path / "trip"
    trip.mkdir()
    picture = Image.new("RGB", (800, 450), (0, 0, 150))
    picture.paste((220, 0, 0), (620, 0, 720, 450))  # "the people": a red band near the right
    picture.save(trip / "a.jpg")
    subprocess.run([ffmpeg_paths()[0], "-loglevel", "error", "-y", "-loop", "1", "-i", str(trip / "a.jpg"), "-t", "2",
                    "-pix_fmt", "yuv420p", "-c:v", "libx264", str(trip / "b.mp4")], check=True)
    settings = RenderSettings(width=180, height=320, photo_seconds=2)
    for item in scan_folder(trip)[0]:
        out = render([dataclasses.replace(item, focus=670 / 800)], settings, tmp_path / f"{item.kind}.mp4")
        stream = next(s for s in probe(str(out))["streams"] if s["codec_type"] == "video")
        assert (stream["width"], stream["height"]) == (180, 320)
        middle = _frame(out, 1.0, 180, 320)[100:220, 70:110].astype(int)
        assert middle[..., 0].mean() > 150 and middle[..., 2].mean() < 80, item.kind  # the red band is in view


def test_a_collage_pops_its_photos_in_one_after_another(tmp_path):
    from trip_memory.render import collage_tiles

    trip = tmp_path / "trip"
    trip.mkdir()
    colours = [(220, 0, 0), (0, 200, 0), (0, 0, 220), (230, 230, 0)]
    for n, colour in enumerate(colours):
        Image.new("RGB", (400, 300), colour).save(trip / f"{n}.jpg")
    first = next(i for i in scan_folder(trip)[0] if i.name == "0.jpg")
    first.seconds = 2.0
    first.effects = {"collage": [str(trip / f"{n}.jpg") for n in (1, 2, 3)], "appear": [0.0, 0.5, 1.0, 1.5]}
    out = render([first], RenderSettings(width=640, height=360), tmp_path / "c.mp4")
    assert abs(float(probe(str(out))["format"]["duration"]) - 2.0) < 0.1
    tiles = collage_tiles(4, 640, 360)
    centre = lambda t, k: _frame(out, t)[tiles[k][1] + tiles[k][3] // 2, tiles[k][0] + tiles[k][2] // 2].astype(int)  # noqa: E731
    assert centre(0.8, 1)[1] > 150  # the second photo (green) is in by 0.8 s
    assert centre(0.8, 3).max() < 60  # the fourth is not yet
    yellow = centre(1.75, 3)  # ... and is by the end (as the video fades out)
    assert yellow[0] > yellow[2] + 40 and yellow[1] > yellow[2] + 40


def test_timeline_titles_show_at_their_time_and_follow_their_shot(tmp_path):
    import json

    import numpy as np

    from trip_memory.render import place_titles

    trip = tmp_path / "trip"
    trip.mkdir()
    Image.new("RGB", (400, 300), (20, 20, 20)).save(trip / "a.jpg")
    Image.new("RGB", (400, 300), (20, 20, 20)).save(trip / "b.jpg")
    items, _ = scan_folder(trip)
    times = [{"start": 0.0, "length": 2.0}, {"start": 1.5, "length": 2.0}]
    placed = place_titles([{"text": "Hi", "shot": 1, "offset": 0.5, "length": 1}, {"text": "", "start": 1, "length": 1},
                           {"text": "Lost", "shot": 7, "offset": 0, "length": 1}], times)
    assert placed == [{"text": "Hi", "shot": 1, "offset": 0.5, "length": 1, "start": 2.0}]
    settings = RenderSettings(width=320, height=180, photo_seconds=2, transition_seconds=0.5,
                              titles=[{"text": "BIG WHITE", "style": "bold", "where": "center", "start": 1.0,
                                       "length": 1.5, "fade_in": 0, "fade_out": 0}])
    timing = tmp_path / "times.json"
    out = render(items, settings, tmp_path / "t.mp4", timing=timing)
    assert [round(t["start"], 1) for t in json.loads(timing.read_text())] == [0.0, 1.5]

    def brightness(at):
        frame = subprocess.run([ffmpeg_paths()[0], "-loglevel", "error", "-ss", str(at), "-i", str(out), "-frames:v", "1",
                                "-f", "rawvideo", "-pix_fmt", "gray", "-"], capture_output=True).stdout
        return np.frombuffer(frame, np.uint8).mean()

    assert brightness(1.8) > brightness(0.6) + 5  # the white title is on screen only from 1 s to 2.5 s
    assert abs(brightness(3.0) - brightness(0.6)) < 3


def test_a_zoomed_in_photo_shows_only_that_part():
    from PIL import Image
    from trip_memory.render import crop_photo

    img = Image.new("RGB", (400, 200), "red")
    assert crop_photo(img, None).size == (400, 200)
    part = crop_photo(img, {"zoom": 2.0, "x": 0.9, "y": 0.5})  # too far right: kept inside the photo
    assert part.size == (200, 100)
    img.paste("blue", (200, 0, 400, 200))
    assert crop_photo(img, {"zoom": 2.0, "x": 0.9, "y": 0.5}).getpixel((10, 50)) == (0, 0, 255)


def test_a_clip_can_play_backwards(tmp_path):
    import subprocess

    from PIL import Image
    from trip_memory.tools import ffmpeg_paths

    trip = tmp_path / "trip"
    trip.mkdir()
    subprocess.run([ffmpeg_paths()[0], "-loglevel", "error", "-y", "-f", "lavfi", "-i",
                    "color=red:s=320x180:r=30:d=1[a];color=blue:s=320x180:r=30:d=1[b];[a][b]concat=n=2",
                    "-metadata", "creation_time=2025-07-03T10:00:00Z", "-c:v", "libx264", "-pix_fmt", "yuv420p",
                    str(trip / "a.mp4")], check=True)
    items, _ = scan_folder(trip)
    items[0].reverse = True
    out = render(items, RenderSettings(width=320, height=180), tmp_path / "back.mp4")
    frame = tmp_path / "first.png"
    subprocess.run([ffmpeg_paths()[0], "-loglevel", "error", "-y", "-ss", "0.7", "-i", str(out), "-frames:v", "1",
                    str(frame)], check=True)
    with Image.open(frame) as img:
        r, g, b = img.convert("RGB").getpixel((160, 90))
    assert b > 150 and r < 100, (r, g, b)  # it starts with the end of the clip (blue)


def test_every_transition_in_the_picker_joins(tmp_path):
    """All the picker's transitions (FFmpeg's own and the hand-drawn spin, bounce and drop) make a video."""
    from trip_memory.render import _join_once, _planned_length, _segment_length
    from trip_memory.transitions import GROUPS, STYLES

    names = [n for names in GROUPS.values() for n in names if STYLES[n].xfade]
    ffmpeg = ffmpeg_paths()[0]
    segments = []
    for k in range(len(names) + 1):
        seg = tmp_path / f"{k}.mp4"
        subprocess.run([ffmpeg, "-v", "error", "-y", "-f", "lavfi", "-i", "testsrc2=s=96x54:r=30", "-f", "lavfi",
                        "-i", "sine=sample_rate=48000", "-frames:v", "30", "-af", "atrim=0:1",
                        "-c:v", "libx264", "-c:a", "aac", "-ac", "2", str(seg)], check=True)
        segments.append(seg)
    lengths = [_segment_length(s) for s in segments]
    styles = [(STYLES[n].xfade, 0.3) for n in names]
    settings = RenderSettings(width=96, height=54)
    made = _join_once(segments, lengths, styles, settings, tmp_path / "out.mp4", lambda _f: None)
    assert abs(made - _planned_length(lengths, styles, settings)) < 0.1
