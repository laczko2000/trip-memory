"""The catalogue: places from GPS, scene suggestions, faces grouped into people, and your edits."""

import json
import subprocess
from fractions import Fraction

from pathlib import Path

import numpy as np
from PIL import Image

from trip_memory import catalog
from trip_memory.tools import ffmpeg_paths
from trip_memory.faces import Face, group_faces
from trip_memory.scenes import SCENES, classify


def _gps_photo(path, when, lat, lon, colour):
    def dms(v):
        d = int(v)
        m = int((v - d) * 60)
        return tuple(Fraction(x).limit_denominator() for x in (d, m, round(((v - d) * 60 - m) * 60, 2)))

    exif = Image.Exif()
    exif[306] = when
    exif[0x8825] = {1: "N" if lat >= 0 else "S", 2: dms(abs(lat)), 3: "E" if lon >= 0 else "W", 4: dms(abs(lon))}
    Image.new("RGB", (400, 300), colour).save(path, exif=exif)


class FakeFaces:
    """Pretends red pictures show Anna and blue ones show Ben."""

    def __init__(self, _model_dir):
        self.anna, self.ben = np.eye(512, dtype=np.float32)[:2]

    def find(self, rgb):
        who = self.anna if rgb[0, 0, 0] > rgb[0, 0, 2] else self.ben
        return [Face(box=(10, 10, 110, 110), score=0.9, embedding=who)]


class FakeScenes:
    def __init__(self, _cache_dir):
        pass

    def embed(self, _frames):
        return np.ones(4, dtype=np.float32)

    def embed_each(self, frames):
        return np.tile(np.float32([1, 0, 0, 0]), (len(frames), 1))

    def describe(self, sentences):
        # Anything about a beach matches the pictures; everything else points elsewhere.
        return np.array([[1, 0, 0, 0] if "sandy beach" in s else [0, 1, 0, 0] for s in sentences], dtype=np.float32)

    def classify(self, _vector):
        return [("Beach", 0.8), ("Sea & boats", 0.1), ("Sunset", 0.05)]


def test_catalogue(tmp_path, monkeypatch):
    monkeypatch.setattr(catalog, "app_dir", lambda: tmp_path / "app")
    monkeypatch.setattr(catalog, "FaceFinder", FakeFaces)
    monkeypatch.setattr(catalog, "SceneTagger", FakeScenes)
    monkeypatch.setattr(catalog, "MIN_APPEARANCES", 2)
    (tmp_path / "app").mkdir()
    trip_dir = tmp_path / "trip"
    trip_dir.mkdir()
    _gps_photo(trip_dir / "a.jpg", "2025:07:03 10:00:00", 38.7223, -9.1393, "red")
    _gps_photo(trip_dir / "b.jpg", "2025:07:04 10:00:00", 47.4979, 19.0402, "blue")
    _gps_photo(trip_dir / "c.jpg", "2025:07:05 10:00:00", 47.4979, 19.0402, "red")

    trip = catalog.Trip(trip_dir)
    trip.analyse()
    assert trip.progress.notes == []
    items = trip.items()
    assert [(i["place"], i["country"]) for i in items] == [
        ("Lisbon", "Portugal"), ("Budapest", "Hungary"), ("Budapest", "Hungary")]
    assert {i["scene_final"] for i in items} == {"Beach"}
    assert items[0]["people"] == items[2]["people"] != items[1]["people"]
    assert all("Beach" in i["tags_final"] for i in items)
    assert {i["people_tag"] for i in items} == {"Solo"}
    assert {i["look_name"] for i in items} == {"Beach (3) · group 1"}  # they all look alike
    trip.set_tags(items[0]["path"], ["Sunset & sunrise"])
    assert trip.items()[0]["tags_final"] == ["Sunset & sunrise"]

    people = trip.people()
    # Seen twice: worth naming. Seen once: listed separately for review.
    assert [(p["count"], p["status"]) for p in people] == [(2, "new"), (1, "rare")]
    anna, single = people
    trip.name_person(anna["id"], "Anna")
    assert trip.person_names() == ["Anna"]

    trip.mark_unknown([single["id"]])
    assert trip.people()[1]["status"] == "unknown"
    assert trip.items()[1]["people"] == []  # strangers vanish from the Library
    trip.restore_person(single["id"])
    trip.name_person(single["id"], "Anna")  # picking an existing name joins the two
    assert [(p["label"], p["count"]) for p in trip.people()] == [("Anna", 3)]

    trip.set_scene(items[0]["path"], "City")
    trip.set_included([items[1]["path"]], False)
    assert trip.items()[0]["scene_final"] == "City"
    assert [m.path.name for m in trip.media_items()] == ["a.jpg", "c.jpg"]

    # Re-running does not redo finished work or lose your edits.
    trip.analyse()
    assert trip.items()[0]["scene_final"] == "City"


def test_group_faces_keeps_known_people():
    a, b = np.eye(4, dtype=np.float32)[:2]
    groups = group_faces(np.array([a, b, a * 0.9 + b * 0.1]), known={7: a})
    assert groups[0] == 7 and groups[2] == 7 and groups[1] < 0


def test_classify_picks_closest_scene():
    vectors = np.eye(len(SCENES), dtype=np.float32)
    assert classify(vectors[3], vectors)[0][0] == SCENES[3]


def test_missing_places_are_guessed_from_nearby_photos():
    def row(when, place=None, country=None):
        return {"taken_at": when, "day": when[:10], "place": place, "country": country, "place_guessed": False}

    rows = [
        row("2025-07-03T10:00:00+02:00", "Lisbon", "Portugal"),
        row("2025-07-03T11:30:00+02:00"),  # 1.5 h later: same place
        row("2025-07-03T21:00:00+02:00"),  # same day, much later: country only
        row("2025-07-09T12:00:00+02:00"),  # another day with no GPS at all: unknown
    ]
    catalog._guess_places(rows)
    assert [(r["place"], r["country"], r["place_guessed"]) for r in rows] == [
        ("Lisbon", "Portugal", False),
        ("Lisbon", "Portugal", True),
        (None, "Portugal", True),
        (None, None, False),
    ]


def test_only_chosen_files_are_analysed_and_used(tmp_path, monkeypatch):
    monkeypatch.setattr(catalog, "app_dir", lambda: tmp_path / "app")
    monkeypatch.setattr(catalog, "FaceFinder", FakeFaces)
    monkeypatch.setattr(catalog, "SceneTagger", FakeScenes)
    (tmp_path / "app").mkdir()
    trip_dir = tmp_path / "trip"
    trip_dir.mkdir()
    for n, colour in enumerate(["red", "blue", "red"]):
        _gps_photo(trip_dir / f"{n}.jpg", f"2025:07:0{n + 1} 10:00:00", 38.7, -9.1, colour)

    trip = catalog.Trip(trip_dir)
    trip.prepare()  # quick look: dates, places and previews, no AI
    items = trip.items()
    assert all(i["thumb"] and i["place"] == "Lisbon" and not i["faces_done"] for i in items)

    trip.set_chosen([items[1]["path"]], False)
    trip.analyse()
    done = {i["path"].split("/")[-1]: bool(i["faces_done"]) for i in trip.items()}
    assert done == {"0.jpg": True, "1.jpg": False, "2.jpg": True}
    assert [m.path.name for m in trip.media_items()] == ["0.jpg", "2.jpg"]


def test_storyboard_order_trims_and_day_titles(tmp_path, monkeypatch):
    monkeypatch.setattr(catalog, "app_dir", lambda: tmp_path / "app")
    (tmp_path / "app").mkdir()
    trip_dir = tmp_path / "trip"
    trip_dir.mkdir()
    for n, when in enumerate(["2025:07:03 10:00:00", "2025:07:03 12:00:00", "2025:07:04 09:00:00"]):
        _gps_photo(trip_dir / f"{n}.jpg", when, 38.7, -9.1, "red")
    trip = catalog.Trip(trip_dir)
    trip.prepare()

    names = lambda: [m.path.name for m in trip.media_items()]
    assert names() == ["0.jpg", "1.jpg", "2.jpg"]
    trip.move(str(trip_dir / "2.jpg"), -2)
    assert names() == ["2.jpg", "0.jpg", "1.jpg"]

    with_titles = trip.media_items(chapters=True)
    assert [m.kind for m in with_titles] == ["title", "photo", "title", "photo", "photo"]
    assert with_titles[0].title == "Day 2\nLisbon · Fri 4 Jul"

    trip.set_transition(str(trip_dir / "0.jpg"), "Circle")
    trip.set_trim(str(trip_dir / "1.jpg"), 1.0, 2.0)
    by_name = {m.path.name: m for m in trip.media_items()}
    assert by_name["0.jpg"].transition == "Circle" and by_name["1.jpg"].trim_end == 2.0

    trip.reset_order()
    assert names() == ["0.jpg", "1.jpg", "2.jpg"]


def test_rotate_turns_preview_and_video(tmp_path, monkeypatch):
    monkeypatch.setattr(catalog, "app_dir", lambda: tmp_path / "app")
    (tmp_path / "app").mkdir()
    trip_dir = tmp_path / "trip"
    trip_dir.mkdir()
    _gps_photo(trip_dir / "a.jpg", "2025:07:03 10:00:00", 38.7223, -9.1393, "red")
    trip = catalog.Trip(trip_dir)
    trip.prepare()
    path = str(trip_dir / "a.jpg")

    trip.rotate(path)
    item = trip.items()[0]
    assert item["rotation"] == 90
    with Image.open(item["thumb"]) as thumb:
        assert thumb.height > thumb.width
    assert trip.media_items()[0].rotation == 90
    for _ in range(3):
        trip.rotate(path)
    assert trip.items()[0]["rotation"] == 0


class SidewaysFaces:
    """Finds a clear face only when the top-left corner is red."""

    def __init__(self, _model_dir):
        pass

    def find(self, rgb):
        score = 0.9 if rgb[0, 0, 0] > rgb[0, 0, 2] else 0.1
        return [Face(box=(10, 10, 110, 110), score=score, embedding=np.eye(512, dtype=np.float32)[0])]


def test_sideways_photos_with_faces_are_turned_upright(tmp_path, monkeypatch):
    monkeypatch.setattr(catalog, "app_dir", lambda: tmp_path / "app")
    monkeypatch.setattr(catalog, "FaceFinder", SidewaysFaces)
    monkeypatch.setattr(catalog, "SceneTagger", FakeScenes)
    (tmp_path / "app").mkdir()
    trip_dir = tmp_path / "trip"
    trip_dir.mkdir()
    picture = Image.new("RGB", (400, 300), "blue")
    picture.paste((255, 0, 0), (0, 150, 200, 300))  # red bottom-left: top-left once turned clockwise
    exif = Image.Exif()
    exif[306] = "2025:07:03 10:00:00"
    picture.save(trip_dir / "a.jpg", exif=exif)

    trip = catalog.Trip(trip_dir)
    trip.analyse()
    assert trip.items()[0]["rotation"] == 90
    assert any("upright" in note for note in trip.progress.notes)


def test_video_sound_is_off_until_you_switch_it_on(tmp_path, monkeypatch):
    monkeypatch.setattr(catalog, "app_dir", lambda: tmp_path / "app")
    (tmp_path / "app").mkdir()
    trip_dir = tmp_path / "trip"
    trip_dir.mkdir()
    _gps_photo(trip_dir / "a.jpg", "2025:07:03 10:00:00", 38.7223, -9.1393, "red")
    trip = catalog.Trip(trip_dir)
    trip.prepare()
    path = str(trip_dir / "a.jpg")
    assert trip.media_items()[0].sound is False
    trip.set_clip_sound(path, True, 1.5)
    item = trip.media_items()[0]
    assert item.sound is True and item.volume == 1.5


def test_auto_edit_uses_only_your_selection_and_keeps_it(tmp_path, monkeypatch):
    from trip_memory import autoedit

    monkeypatch.setattr(catalog, "app_dir", lambda: tmp_path / "app")
    (tmp_path / "app").mkdir()
    trip_dir = tmp_path / "trip"
    trip_dir.mkdir()
    for n in range(12):
        _gps_photo(trip_dir / f"p{n:02d}.jpg", f"2025:07:03 10:{n:02d}:00", 38.7, -9.1, (n * 20, 80, 160))
    trip = catalog.Trip(trip_dir)
    trip.prepare()
    paths = [r["path"] for r in trip.items()]
    trip.set_chosen(paths, True)
    trip.set_included(paths[4:], False)  # you selected only the first four
    limits = autoedit.Limits(max_video=4, max_photo=2, min_shot=1)
    trip.auto_edit_steady(limits, 4.0, 0.8, "Fade")  # room for two of them
    board = [r["path"] for r in trip.storyboard()]
    assert set(board) <= set(paths[:4]) and 1 <= len(board) < 4
    # The ones that didn't fit stay selected, so a longer Auto-edit uses them again.
    assert [r["path"] for r in trip.auto_edit_pool()] == paths[:4]
    trip.auto_edit_steady(limits, 60.0, 0.8, "Fade")
    assert [r["path"] for r in trip.storyboard()] == paths[:4]


def test_stories_are_named_ordered_and_switched(tmp_path, monkeypatch):
    monkeypatch.setattr(catalog, "app_dir", lambda: tmp_path / "app")
    monkeypatch.setattr(catalog, "FaceFinder", FakeFaces)
    monkeypatch.setattr(catalog, "SceneTagger", FakeScenes)
    (tmp_path / "app").mkdir()
    trip_dir = tmp_path / "trip"
    trip_dir.mkdir()
    _gps_photo(trip_dir / "a.jpg", "2025:07:03 09:00:00", 38.7223, -9.1393, "red")
    _gps_photo(trip_dir / "b.jpg", "2025:07:04 15:00:00", 47.4979, 19.0402, "blue")
    trip = catalog.Trip(trip_dir)
    trip.analyse()
    stories = trip.stories()
    assert [s["name"] for s in stories] == ["Beach · Lisbon · Thu 3 Jul, morning", "Beach · Budapest · Fri 4 Jul, afternoon"]

    trip.rename_story(stories[1]["key"], "Thermal baths")
    trip.move_story(stories[1]["key"], -1)
    assert [s["name"] for s in trip.stories()] == ["Thermal baths", "Beach · Lisbon · Thu 3 Jul, morning"]
    assert [r["path"][-5:] for r in trip.storyboard()] == ["b.jpg", "a.jpg"]  # the video follows your story order
    assert [r["path"][-5:] for r in trip.auto_edit_pool()] == ["b.jpg", "a.jpg"]

    trip.set_story_enabled(stories[0]["key"], False)
    assert [r["path"][-5:] for r in trip.storyboard()] == ["b.jpg"]
    assert [r["path"][-5:] for r in trip.auto_edit_pool()] == ["b.jpg"]


def test_previews_made_before_a_turn_are_made_again(tmp_path):
    from PIL import Image
    from trip_memory.catalog import Trip

    trip = tmp_path / "trip"
    trip.mkdir()
    Image.new("RGB", (60, 40), "red").save(trip / "a.jpg")
    t = Trip(trip)
    t.prepare()
    path = t.items()[0]["path"]
    t.execute("UPDATE items SET rotation = 90 WHERE path = ?", (path,))  # turned, but the preview is still the old one
    t.prepare()
    item = t.items()[0]
    assert item["thumb"].endswith("_90.jpg")
    with Image.open(item["thumb"]) as img:
        assert img.height > img.width


def test_unticked_files_can_be_removed_from_the_trip_and_brought_back(tmp_path, monkeypatch):
    monkeypatch.setattr(catalog, "app_dir", lambda: tmp_path / "app")
    (tmp_path / "app").mkdir()
    trip_dir = tmp_path / "trip"
    trip_dir.mkdir()
    for n, colour in enumerate(("red", "red", "blue")):
        _gps_photo(trip_dir / f"{n}.jpg", f"2025:07:03 10:0{n}:00", 38.7223, -9.1393, colour)
    trip = catalog.Trip(trip_dir)
    trip.prepare()
    gone = trip_dir / "1.jpg"
    trip.set_chosen([str(gone)], False)
    assert trip.remove_unchosen() == 1
    assert str(gone) not in [i["path"] for i in trip.items()] and gone.exists()  # out of the trip, not off the disk
    assert [i["path"] for i in trip.items(removed=True)] == [str(gone)]
    assert all(str(gone) not in [i["path"] for i in s["items"]] for s in trip.stories())
    trip.suggest_selection()
    trip.prepare()  # opening the folder again doesn't bring it back
    assert str(gone) not in [i["path"] for i in trip.items()]
    trip.restore([str(gone)])
    back = [i for i in trip.items() if i["path"] == str(gone)]
    assert back and not back[0]["chosen"] and trip.removed_count() == 0


def test_stories_can_be_linked_and_split_again(tmp_path, monkeypatch):
    monkeypatch.setattr(catalog, "app_dir", lambda: tmp_path / "app")
    (tmp_path / "app").mkdir()
    trip_dir = tmp_path / "trip"
    trip_dir.mkdir()
    for n, when in enumerate(["2025:07:03 10:00:00", "2025:07:03 12:00:00", "2025:07:03 14:00:00", "2025:07:03 16:00:00"]):
        _gps_photo(trip_dir / f"{n}.jpg", when, 38.7, -9.1, ["red", "blue", "green", "white"][n])
    trip = catalog.Trip(trip_dir)
    trip.prepare()
    story = lambda: [it["story"] for it in sorted(trip.items(), key=lambda r: r["path"])]
    assert story() == [0, 1, 2, 3]

    trip.link_story(2, to_next=False)  # story 3 onto story 2
    assert story() == [0, 1, 1, 2]
    trip.link_story(0, to_next=True)  # story 1 with story 2
    assert story() == [0, 0, 0, 1]
    assert trip.joined_stories() == {0}

    trip.set_chosen([str(trip_dir / "2.jpg")], False)  # the link survives removing the file it started at
    trip.remove_unchosen()
    assert [it["story"] for it in trip.items()] == [0, 0, 1]

    trip.unlink_story(0)
    assert [it["story"] for it in trip.items()] == [0, 1, 2]
    assert trip.joined_stories() == set()


def test_titles_opening_story_end_and_your_own(tmp_path, monkeypatch):
    monkeypatch.setattr(catalog, "app_dir", lambda: tmp_path / "app")
    (tmp_path / "app").mkdir()
    trip_dir = tmp_path / "Portugal_2025"
    trip_dir.mkdir()
    for n, when in enumerate(["2025:07:03 10:00:00", "2025:07:03 10:05:00", "2025:07:04 12:00:00", "2025:07:04 12:05:00"]):
        _gps_photo(trip_dir / f"{n}.jpg", when, 38.7, -9.1, "red")
    trip = catalog.Trip(trip_dir)
    trip.prepare()
    shown = lambda: {Path(p).name: (t["auto"], t["text"]) for p, t in trip.shot_titles().items()}
    assert shown() == {"0.jpg": ("opening", "Portugal 2025"), "2.jpg": ("story", trip.stories()[1]["name"])}
    assert trip.end_card() == (3.0, {"text": "Until next time", "sub": "Portugal 2025", "style": "elegant", "where": "center"})
    trip.set_title_settings(end_black=0)  # no black ending: the end card goes on the last shot
    assert shown()["3.jpg"] == ("ending", "Until next time")
    assert trip.title_settings()["trip_sub"] == "3 – 4 July 2025"

    trip.set_shot_title(str(trip_dir / "2.jpg"), "Beach day", "", "bold", "center")
    trip.set_title_settings(opening=False, ending=False)
    assert shown() == {"2.jpg": (None, "Beach day")}
    overlays = {m.path.name: m.overlay for m in trip.media_items()}
    assert overlays["2.jpg"]["style"] == "bold" and overlays["0.jpg"] is None

    trip.set_shot_title(str(trip_dir / "2.jpg"), "", "")
    assert shown()["2.jpg"][0] == "story"


def test_video_doubles_are_found_on_several_frames(tmp_path, monkeypatch):
    from tests.test_render import make_clip

    monkeypatch.setattr(catalog, "app_dir", lambda: tmp_path / "app")
    (tmp_path / "app").mkdir()
    trip_dir = tmp_path / "trip"
    trip_dir.mkdir()
    make_clip(trip_dir / "take1.mp4", "640x360", 4, "2025-07-03T10:00:00Z", audio=False)
    make_clip(trip_dir / "take2.mp4", "640x360", 6, "2025-07-03T10:06:00Z", audio=False)  # 6 min after take 1 ended
    subprocess.run([ffmpeg_paths()[0], "-loglevel", "error", "-y", "-f", "lavfi", "-i", "mandelbrot=s=640x360:r=30",
                    "-t", "4", "-metadata", "creation_time=2025-07-03T10:08:00Z", "-c:v", "libx264",
                    str(trip_dir / "other.mp4")], check=True)
    Image.new("RGB", (640, 360), "red").save(trip_dir / "photo.jpg")
    trip = catalog.Trip(trip_dir)
    trip.prepare()
    by_name = {Path(i["path"]).name: i for i in trip.items()}
    assert by_name["take1.mp4"]["similar_set"] == by_name["take2.mp4"]["similar_set"]
    assert by_name["other.mp4"]["similar_set"] != by_name["take1.mp4"]["similar_set"]
    assert by_name["take2.mp4"]["best"] and not by_name["take1.mp4"]["best"]  # the longer take wins (same sharpness)
    trip.untick_doubles()
    assert not {Path(i["path"]).name: i for i in trip.items()}["take1.mp4"]["chosen"]


def test_the_look_is_kept_per_trip(tmp_path, monkeypatch):
    monkeypatch.setattr(catalog, "app_dir", lambda: tmp_path / "app")
    (tmp_path / "app").mkdir()
    (tmp_path / "trip").mkdir()
    trip = catalog.Trip(tmp_path / "trip")
    assert trip.look() == "natural"
    trip.set_look("film")
    assert catalog.Trip(tmp_path / "trip").look() == "film"
    trip.set_look("nonsense")
    assert trip.look() == "natural"


def test_where_the_people_are_is_found_once(tmp_path, monkeypatch):
    monkeypatch.setattr(catalog, "app_dir", lambda: tmp_path / "app")
    monkeypatch.setattr(catalog, "FaceFinder", FakeFaces)  # a face at x 10..110
    (tmp_path / "app").mkdir()
    (tmp_path / "trip").mkdir()
    _gps_photo(tmp_path / "trip" / "a.jpg", "2025:07:03 09:00:00", 38.7, -9.1, "red")
    trip = catalog.Trip(tmp_path / "trip")
    trip.prepare()
    (item,) = trip.media_items(focus=True)
    width = Image.open(trip.items()[0]["thumb"]).width
    assert abs(item.focus - 60 / width) < 1e-6
    monkeypatch.setattr(catalog, "FaceFinder", None)  # not needed again
    assert trip.media_items(focus=True)[0].focus == item.focus and trip.media_items()[0].focus is None


def test_a_collage_shows_as_one_shot_and_can_be_split(tmp_path, monkeypatch):
    import json

    monkeypatch.setattr(catalog, "app_dir", lambda: tmp_path / "app")
    (tmp_path / "app").mkdir()
    folder = tmp_path / "trip"
    folder.mkdir()
    for n in range(4):
        _gps_photo(folder / f"{n}.jpg", f"2025:07:03 10:0{n}:00", 38.7, -9.1, "red")
    trip = catalog.Trip(folder)
    trip.prepare()
    parts = [str(folder / f"{n}.jpg") for n in (1, 2)]
    trip.execute("UPDATE items SET position = 0, seconds = 1.5, effects = ? WHERE path = ?",
                 (json.dumps({"collage": parts, "appear": [0.0, 0.5, 1.0]}), str(folder / "0.jpg")))
    trip.execute("UPDATE items SET rotation = 90 WHERE path = ?", (parts[1],))
    assert [Path(r["path"]).name for r in trip.storyboard()] == ["0.jpg", "3.jpg"]
    lead = trip.media_items()[0]
    assert lead.effects["turns"] == [0, 90]
    trip.split_collage(str(folder / "0.jpg"))
    board = trip.storyboard()
    assert [Path(r["path"]).name for r in board] == ["0.jpg", "1.jpg", "2.jpg", "3.jpg"]
    assert [r["seconds"] for r in board[:3]] == [0.5, 0.5, 0.5]


def test_a_clip_you_cut_only_ever_uses_the_part_you_kept(tmp_path, monkeypatch):
    from trip_memory import autoedit

    monkeypatch.setattr(catalog, "app_dir", lambda: tmp_path / "app")
    (tmp_path / "app").mkdir()
    trip_dir = tmp_path / "trip"
    trip_dir.mkdir()
    subprocess.run([ffmpeg_paths()[0], "-loglevel", "error", "-y", "-f", "lavfi", "-i", "testsrc2=s=160x90:r=30:d=10",
                    "-pix_fmt", "yuv420p", str(trip_dir / "clip.mp4")], check=True)
    trip = catalog.Trip(trip_dir)
    trip.prepare()
    path = str(trip_dir / "clip.mp4")
    trip.set_keep(path, 4.0, 7.0)
    row = trip.auto_edit_pool()[0]
    assert row["duration"] == 3.0 and row["keep_offset"] == 4.0  # Auto-edit sees a 3-second clip
    item = trip.media_items()[0]
    assert (item.trim_start, item.trim_end) == (4.0, 7.0)  # no trim of its own: the part you kept
    trip.auto_edit_steady(autoedit.Limits(max_video=2, max_photo=2, min_shot=1), 2.0, 0.5, "Fade")
    shot = trip.storyboard()[0]
    assert 4.0 <= shot["trim_start"] < shot["trim_end"] <= 7.0 + 1e-6
    trip.set_keep(path, None, None)  # Undo cut: the whole clip again
    assert "keep_offset" not in trip.auto_edit_pool()[0]
    assert trip.undo_label() == "bringing a whole clip back"


def test_a_long_clip_gives_two_shots_from_different_parts_never_the_same_moment(tmp_path, monkeypatch):
    from trip_memory import autoedit

    monkeypatch.setattr(catalog, "app_dir", lambda: tmp_path / "app")
    (tmp_path / "app").mkdir()
    trip_dir = tmp_path / "trip"
    trip_dir.mkdir()
    subprocess.run([ffmpeg_paths()[0], "-loglevel", "error", "-y", "-f", "lavfi", "-i", "testsrc2=s=160x90:r=30:d=20",
                    "-pix_fmt", "yuv420p", str(trip_dir / "long.mp4")], check=True)
    trip = catalog.Trip(trip_dir)
    trip.prepare()
    trip.auto_edit_steady(autoedit.Limits(max_video=3, max_photo=2, min_shot=1), 6.0, 0.0, "Fade")
    board = trip.storyboard()
    assert [Path(r["path"]).name for r in board] == ["long.mp4", "long.mp4#2"]
    (a0, a1), (b0, b1) = [(r["trim_start"], r["trim_end"]) for r in board]
    assert a1 + autoedit.PART_GAP - 0.01 <= b0  # different moments of the clip, with a gap between them
    assert {str(i.path) for i in trip.media_items()} == {str(trip_dir / "long.mp4")}  # both play from the file
    assert [r["path"] for r in trip.items()] == [str(trip_dir / "long.mp4")]  # Choose files and Library: one clip
    trip.auto_edit_steady(autoedit.Limits(max_video=3, max_photo=2, min_shot=1), 2.0, 0.0, "Fade")
    assert len(trip.storyboard()) == 1  # a new cut that needs one shot: no parts left over


def test_the_same_moment_in_two_files_is_used_once():
    rows = [
        {"path": "/t/IMG_1.HEIC", "kind": "photo", "taken_at": "2025-07-01T10:00:00", "duration": 0, "similar_set": None},
        {"path": "/t/IMG_1.MOV", "kind": "video", "taken_at": "2025-07-01T10:00:00", "duration": 3.0, "similar_set": None},
        {"path": "/t/a.mov", "kind": "video", "taken_at": "2025-07-01T11:00:00", "duration": 20.0, "similar_set": 4},
        {"path": "/t/a copy.mov", "kind": "video", "taken_at": "2025-07-01T11:00:05", "duration": 8.0, "similar_set": 4},
        {"path": "/t/b.mov", "kind": "video", "taken_at": "2025-07-01T11:00:00", "duration": 20.2, "similar_set": None},
        {"path": "/t/other phone.mov", "kind": "video", "taken_at": "2025-07-01T11:00:03", "duration": 9.0, "similar_set": 9},
    ]
    assert catalog.Trip._same_moments(rows) == {"/t/IMG_1.MOV", "/t/a copy.mov", "/t/a.mov"}


def test_the_storyline_follows_your_themes(tmp_path, monkeypatch):
    from trip_memory import autoedit, themes

    monkeypatch.setattr(catalog, "app_dir", lambda: tmp_path / "app")
    (tmp_path / "app").mkdir()
    trip_dir = tmp_path / "trip"
    trip_dir.mkdir()
    for n, colour in enumerate(["red", "green", "blue", "yellow"]):
        Image.new("RGB", (160, 90), colour).save(trip_dir / f"p{n}.jpg")
    trip = catalog.Trip(trip_dir)
    trip.prepare()
    paths = [r["path"] for r in trip.items()]
    trip.execute("UPDATE items SET tags_user = ? WHERE path = ?", (json.dumps(["Eating & drinking"]), paths[0]))
    trip.suggest_themes()  # no picture fingerprints here: a first guess from the tags
    rows = {Path(r["path"]).name: r for r in trip.theme_rows()}
    assert rows["p0.jpg"]["theme"] == "Food & drink" and not rows["p0.jpg"]["theme_ok"]
    trip.set_theme([paths[2]], "Boat")
    trip.set_theme([paths[0]], None)  # confirmed as suggested
    rows = {Path(r["path"]).name: r for r in trip.theme_rows()}
    assert rows["p2.jpg"]["theme"] == "Boat" and rows["p2.jpg"]["theme_ok"] and rows["p0.jpg"]["theme_ok"]
    # By theme: boat first, then food; the two "Other" shots switched off.
    blocks = [b | {"on": b["activity"] != themes.OTHER} for b in trip.storyline_blocks()]
    blocks.sort(key=lambda b: b["activity"] != "Boat")
    trip.set_storyline(order=themes.THEME, blocks=[{k: b[k] for k in ("key", "activity", "person", "on")} for b in blocks],
                       finale=False)
    plan = trip.auto_edit_steady(autoedit.Limits(max_video=3, max_photo=2, min_shot=1), None, 0.0, "Fade")
    assert [Path(s.path).name for s in plan.shots] == ["p2.jpg", "p0.jpg"]  # the boat first, though taken later
    assert [Path(r["path"]).name for r in trip.storyboard()] == ["p2.jpg", "p0.jpg"]
    trip.set_storyline(order=themes.TIME)
    plan = trip.auto_edit_steady(autoedit.Limits(max_video=3, max_photo=2, min_shot=1), None, 0.0, "Fade")
    assert [Path(s.path).name for s in plan.shots] == ["p0.jpg", "p2.jpg"]  # as it happened (Other still off)
