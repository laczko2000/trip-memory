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
    limits = autoedit.Limits(max_video=4, max_photo=2, min_shot=1, pack="normal")
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
    trip.set_title_settings(end_room=0, open_room=3.0)  # (no room kept: the old black ending)
    shown = lambda: {Path(p).name: (t["auto"], t["text"]) for p, t in trip.shot_titles().items()}
    assert shown() == {"0.jpg": ("opening", "Portugal 2025"), "2.jpg": ("story", trip.stories()[1]["name"])}
    assert trip.end_card() == (3.0, {"text": "Until next time", "sub": "Portugal 2025", "style": "elegant", "where": "center"})
    trip.set_title_settings(end_black=0)  # no black ending: the end card goes on the last shot
    assert shown()["3.jpg"] == ("ending", "Until next time")
    assert trip.title_settings()["trip_sub"] == "3 – 4 July 2025"

    # An opening card (one of six looks) takes the place of the title over the first shot.
    assert trip.opening_card() is None
    trip.set_title_settings(card="cinematic", end_black=3.0)
    assert trip.opening_card() == {"text": "Portugal 2025", "sub": "3 – 4 July 2025", "card": "cinematic", "seconds": 3.0}
    assert "0.jpg" not in shown()
    assert trip.end_card()[1]["card"] == "cinematic" and trip.end_card()[1]["style"] == "cinematic"
    trip.set_title_settings(card="")

    trip.set_shot_title(str(trip_dir / "2.jpg"), "Beach day", "", "bold", "center")
    trip.set_title_settings(opening=False, ending=False)
    assert shown() == {"2.jpg": (None, "Beach day")}
    overlays = {m.path.name: m.overlay for m in trip.media_items()}
    assert overlays["2.jpg"]["style"] == "bold" and overlays["0.jpg"] is None

    trip.set_shot_title(str(trip_dir / "2.jpg"), "", "")
    assert shown()["2.jpg"][0] == "story"
    trip.set_storyline(day_titles=True)  # a title as each day starts, instead of the story titles
    assert shown() == {"0.jpg": ("day", "Day 1"), "2.jpg": ("day", "Day 2")}
    assert trip.shot_titles()[str(trip_dir / "2.jpg")]["sub"].endswith("Fri 4 Jul")


def test_video_doubles_are_found_on_several_frames(tmp_path, monkeypatch):
    from tests.test_render import make_clip

    monkeypatch.setattr(catalog, "app_dir", lambda: tmp_path / "app")
    (tmp_path / "app").mkdir()
    trip_dir = tmp_path / "trip"
    trip_dir.mkdir()
    make_clip(trip_dir / "take1.mp4", "640x360", 4, "2025-07-03T10:00:00Z", audio=False)
    make_clip(trip_dir / "take2.mp4", "640x360", 6, "2025-07-03T10:00:20Z", audio=False)  # just after take 1
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


def test_keep_marked_sorts_out_a_set_for_good(tmp_path, monkeypatch):
    from tests.test_render import make_clip

    monkeypatch.setattr(catalog, "app_dir", lambda: tmp_path / "app")
    (tmp_path / "app").mkdir()
    trip_dir = tmp_path / "trip"
    trip_dir.mkdir()
    make_clip(trip_dir / "take1.mp4", "640x360", 4, "2025-07-03T10:00:00Z", audio=False)
    make_clip(trip_dir / "take2.mp4", "640x360", 6, "2025-07-03T10:00:20Z", audio=False)
    trip = catalog.Trip(trip_dir)
    trip.prepare()
    names = lambda: {Path(i["path"]).name: i for i in trip.items()}
    paths = [i["path"] for i in trip.items()]
    assert names()["take1.mp4"]["similar_set"] == names()["take2.mp4"]["similar_set"]
    trip.set_chosen(paths, True)  # you keep both
    trip.keep_marked(paths)
    for _ in range(2):  # still apart after the app looks for doubles again
        by = names()
        assert by["take1.mp4"]["similar_set"] != by["take2.mp4"]["similar_set"]
        assert by["take1.mp4"]["best"] and by["take2.mp4"]["best"]
        trip._group_stories()
    trip.untick_doubles()
    assert all(i["chosen"] for i in trip.items())  # Keep only the star leaves a sorted-out set alone
    trip.regroup_sets()
    assert names()["take1.mp4"]["similar_set"] == names()["take2.mp4"]["similar_set"]


def test_good_parts_of_a_video_become_shots_of_their_own(tmp_path, monkeypatch):
    from tests.test_render import make_clip

    monkeypatch.setattr(catalog, "app_dir", lambda: tmp_path / "app")
    (tmp_path / "app").mkdir()
    trip_dir = tmp_path / "trip"
    trip_dir.mkdir()
    make_clip(trip_dir / "kite.mp4", "640x360", 6, "2025-07-03T10:00:00Z")
    trip = catalog.Trip(trip_dir)
    trip.prepare()
    video = str(trip_dir / "kite.mp4")
    trip.set_locks(video, [[1.0, 2.0], [4.5, 5.5]])
    one = trip.make_shot(video, 0.5, 2.5)
    two = trip.make_shot(video, 4.0, 5.8)
    trip.prepare()  # looking through the folder again keeps them
    by = {i["path"]: i for i in trip.items()}
    assert by[one]["shot_of"] == video and by[one]["chosen"] and abs(by[one]["duration"] - 2.0) < 0.2
    assert by[one]["thumb"] and Path(one).exists() and Path(one).parent == trip.shots_dir
    assert json.loads(by[one]["locks"]) == [[0.5, 1.5]] and json.loads(by[two]["locks"]) == [[0.5, 1.5]]
    assert by[one]["taken_at"] > by[video]["taken_at"]
    assert len({by[p]["similar_set"] for p in (video, one, two)}) == 3  # never doubles of each other
    assert [round(s["start"], 1) for s in trip.shots_from(video)] == [0.5, 4.0]
    trip.drop_shot(one)
    assert not Path(one).exists() and [s["path"] for s in trip.shots_from(video)] == [two]


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


def test_the_first_picture_of_a_collage_can_be_swapped_and_doubles_are_never_offered(tmp_path, monkeypatch):
    import json

    monkeypatch.setattr(catalog, "app_dir", lambda: tmp_path / "app")
    (tmp_path / "app").mkdir()
    folder = tmp_path / "trip"
    folder.mkdir()
    for n in range(5):
        _gps_photo(folder / f"{n}.jpg", f"2025:07:03 10:0{n}:00", 38.7, -9.1, "red")
    trip = catalog.Trip(folder)
    trip.prepare()
    p = [str(folder / f"{n}.jpg") for n in range(5)]
    trip.execute("UPDATE items SET position = 0, seconds = 3, effects = ? WHERE path = ?",
                 (json.dumps({"collage": p[1:3]}), p[0]))
    trip.execute("UPDATE items SET similar_set = 9 WHERE path IN (?, ?)", (p[1], p[3]))  # 3 is a double of 1
    trip.execute("UPDATE items SET position = 5 WHERE path = ?", (p[4],))
    shown = trip.in_video_family()
    assert catalog.Trip.family(dict(trip.query("SELECT * FROM items WHERE path = ?", (p[3],))[0])) & shown
    assert catalog.Trip.family({"path": str(folder / "4.HEIC")}) & shown  # a Live Photo twin of one in the video
    assert all(r["path"] != p[3] for r in trip.alternatives(p[4], 3.0))
    lead = trip.replace_first(p[0], p[4])
    assert lead == p[4] and trip.together(lead) == ("collage", p[1:3])
    assert p[0] not in [r["path"] for r in trip.storyboard()]


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
    clip = str(trip_dir / "long.mp4")
    assert trip.used_parts(clip, but=clip + "#2") == [(a0, a1, 1)]  # what the other shot uses (shown red)
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
    plan = trip.auto_edit_steady(autoedit.Limits(max_video=3, max_photo=2, min_shot=1, pack="normal"), None, 0.0, "Fade")
    assert [Path(s.path).name for s in plan.shots] == ["p2.jpg", "p0.jpg"]  # the boat first, though taken later
    assert [Path(r["path"]).name for r in trip.storyboard()] == ["p2.jpg", "p0.jpg"]
    trip.set_storyline(order=themes.TIME)
    plan = trip.auto_edit_steady(autoedit.Limits(max_video=3, max_photo=2, min_shot=1, pack="normal"), None, 0.0, "Fade")
    assert [Path(s.path).name for s in plan.shots] == ["p0.jpg", "p2.jpg"]  # as it happened (Other still off)
    # Renaming a theme takes its shots, its first guesses from tags and the storyline along.
    trip.set_storyline(order=themes.THEME, blocks=[{k: b[k] for k in ("key", "activity", "person", "on")} for b in blocks])
    assert trip.rename_activity("Food & drink", "Dinners")
    assert not trip.rename_activity("Boat", "dinners")  # taken
    trip.unconfirm_theme([paths[0]])
    rows = {Path(r["path"]).name: r for r in trip.theme_rows()}
    assert rows["p0.jpg"]["theme"] == "Dinners"  # the tag guess finds the new name
    assert [b["key"] for b in trip.storyline_blocks()][:2] == ["Boat|", "Dinners|"]


def test_a_moment_inside_a_clip_is_found_and_kept(tmp_path, monkeypatch):
    from tests.test_render import make_clip
    from trip_memory import autoedit, themes

    monkeypatch.setattr(catalog, "app_dir", lambda: tmp_path / "app")
    (tmp_path / "app").mkdir()
    trip_dir = tmp_path / "trip"
    trip_dir.mkdir()
    make_clip(trip_dir / "kite.mp4", "320x180", 12, "2025-07-03T10:00:00Z")
    trip = catalog.Trip(trip_dir)
    trip.prepare()
    path = trip.items()[0]["path"]
    trip.set_theme([path], "Water sports")
    jump = np.eye(4, dtype=np.float32)[0]
    calm = np.eye(4, dtype=np.float32)[1]
    # A stand-in for the picture model: the frame at 8.5 s is the jump (this test clip has few key frames, so the
    # app reads a frame every 2 seconds: 0.5, 2.5, 4.5, 6.5, 8.5...).
    trip.frame_embedder = lambda pictures: np.array([jump if n == 4 else calm for n, _ in enumerate(pictures)])
    found = trip.moment_matches("Water sports", "big jumps", jump)
    assert found[0]["path"] == path and found[0]["t"] == 8.5 and found[0]["likely"] and not found[0]["ticked"]
    trip.add_moment("Water sports", "big jumps")
    trip.tick_moment("Water sports", "big jumps", path, 8.5, True)
    again = trip.moment_matches("Water sports", "big jumps", None)  # the frames are kept: no second look
    assert again[0]["ticked"] and again[0]["t"] == 8.5
    story = {"parts": [{"name": "Jumps", "pace": "lively", "seconds": 10,
                        "entries": [{"key": "Water sports|", "pick": "moment", "moment": "Big jumps"}]}]}
    trip.set_storyline(order=themes.STORY, story=story, finale=False)
    pool = trip.auto_edit_pool()
    assert pool[0]["keep_offset"] == 7.0 and pool[0]["duration"] == 3.0
    trip.auto_edit_steady(autoedit.Limits(max_video=3, max_photo=2, min_shot=1), 3.0, 0.0, "Fade")
    row = next(r for r in trip.items() if r["path"] == path)
    assert 7.0 <= row["trim_start"] and row["trim_end"] <= 10.01


def test_the_trip_remembers_its_song(tmp_path, monkeypatch):
    monkeypatch.setattr(catalog, "app_dir", lambda: tmp_path / "app")
    (tmp_path / "app").mkdir()
    (tmp_path / "trip").mkdir()
    trip = catalog.Trip(tmp_path / "trip")
    assert trip.song() is None
    trip.set_song("/music/summer.m4a")
    assert catalog.Trip(tmp_path / "trip").song() == "/music/summer.m4a"


def test_a_locked_moment_is_always_in_the_cut_exactly(tmp_path, monkeypatch):
    from trip_memory import autoedit

    monkeypatch.setattr(catalog, "app_dir", lambda: tmp_path / "app")
    (tmp_path / "app").mkdir()
    trip_dir = tmp_path / "trip"
    trip_dir.mkdir()
    subprocess.run([ffmpeg_paths()[0], "-loglevel", "error", "-y", "-f", "lavfi", "-i", "testsrc2=s=160x90:r=30:d=20",
                    "-pix_fmt", "yuv420p", str(trip_dir / "kite.mp4")], check=True)
    for n in range(4):
        Image.new("RGB", (160, 90), ["red", "green", "blue", "yellow"][n]).save(trip_dir / f"p{n}.jpg")
    trip = catalog.Trip(trip_dir)
    trip.prepare()
    path = str(trip_dir / "kite.mp4")
    trip.add_lock(path, 12.0, 14.0)
    trip.add_lock(path, 13.5, 15.0)  # overlapping: one moment
    assert trip.locks(path) == [[12.0, 15.0]]
    trip.auto_edit_steady(autoedit.Limits(max_video=3, max_photo=2, min_shot=1), 12.0, 0.0, "Fade")
    board = trip.storyboard()
    mine = [(r["trim_start"], r["trim_end"]) for r in board if (r["part_of"] or r["path"]) == path]
    assert any(abs(a - 12.0) < 0.15 and abs(b - 15.0) < 0.15 for a, b in mine), mine  # the whole moment
    for a, b in mine:  # other parts of the clip never overlap it
        assert b <= 12.0 + 1e-6 or a >= 15.0 - 1e-6 or (abs(a - 12.0) < 0.15 and abs(b - 15.0) < 0.15)
    trip.change_lock(path, 0)
    assert trip.locks(path) == []


def test_shifting_a_cameras_clock_moves_all_its_files(tmp_path, monkeypatch):
    monkeypatch.setattr(catalog, "app_dir", lambda: tmp_path / "app")
    (tmp_path / "app").mkdir()
    trip_dir = tmp_path / "trip"
    trip_dir.mkdir()
    for name, when, make, model in [("IMG_1.jpg", "2025:07:03 10:00:00", "Apple", "iPhone 15 Pro"),
                                    ("IMG_2.jpg", "2025:07:03 12:00:00", "Apple", "iPhone 15 Pro"),
                                    ("GOPR0001.jpg", "2025:07:03 09:00:00", "GoPro", "HERO11 Black")]:
        exif = Image.Exif()
        exif[306], exif[271], exif[272] = when, make, model
        Image.new("RGB", (400, 300), "red").save(trip_dir / name, exif=exif)
    trip = catalog.Trip(trip_dir)
    trip.prepare()
    cams = {c["camera"]: c["count"] for c in trip.cameras()}
    assert cams == {"iPhone 15 Pro": 2, "GoPro": 1}
    order = lambda: [Path(i["path"]).name for i in trip.items()]
    assert order()[0] == "GOPR0001.jpg"
    assert trip.set_time_shift("GoPro", 2 * 3600 + 30 * 60) == 1
    assert order() == ["IMG_1.jpg", "GOPR0001.jpg", "IMG_2.jpg"]
    trip.prepare()  # opening the trip again keeps the shift, from the file's own time
    gopro = next(i for i in trip.items() if i["path"].endswith("GOPR0001.jpg"))
    assert gopro["taken_at"].startswith("2025-07-03T11:30") and gopro["taken_orig"].startswith("2025-07-03T09:00")
    trip.set_time_shift("GoPro", 0)
    assert order()[0] == "GOPR0001.jpg"


def test_your_themes_and_what_they_look_like_carry_over_to_the_next_trip(tmp_path, monkeypatch):
    monkeypatch.setattr(catalog, "app_dir", lambda: tmp_path / "app")
    (tmp_path / "app").mkdir()
    kite, food = np.eye(512, dtype=np.float32)[:2]
    trips = []
    for name in ("tarifa", "crete"):
        folder = tmp_path / name
        folder.mkdir()
        for n, colour in enumerate(["red", "green"]):
            Image.new("RGB", (160, 90), colour).save(folder / f"p{n}.jpg")
        trip = catalog.Trip(folder)
        trip.prepare()
        for path, vector in zip([r["path"] for r in trip.items()], (kite, food)):
            trip.execute("UPDATE items SET clip = ? WHERE path = ?", ((vector + 0.01).tobytes(), path))
        trips.append(trip)
    tarifa, crete = trips
    tarifa.add_activity("Kitesurfing", None)
    tarifa.set_theme([tarifa.items()[0]["path"]], "Kitesurfing")
    assert "Kitesurfing" in [a["name"] for a in catalog.Trip(tmp_path / "crete").activities()]
    crete.suggest_themes()
    first = next(r for r in crete.theme_rows() if r["path"].endswith("p0.jpg"))
    assert first["theme"] == "Kitesurfing" and not first["theme_ok"]  # learned in Tarifa, suggested in Crete


def test_moment_names_are_offered_again_in_the_next_trip(tmp_path, monkeypatch):
    monkeypatch.setattr(catalog, "app_dir", lambda: tmp_path / "app")
    (tmp_path / "app").mkdir()
    for name in ("a", "b"):
        (tmp_path / name).mkdir()
    a = catalog.Trip(tmp_path / "a")
    a.add_moment("Kitesurfing", "kite jump")
    a.add_moment("Food & drink", "cheers")
    b = catalog.Trip(tmp_path / "b")
    assert b.moment_names("Kitesurfing") == ["kite jump", "cheers"]
    assert b.moment_names("Food & drink")[0] == "cheers"


def test_editing_under_the_video_keeps_the_cuts_where_they_are(tmp_path, monkeypatch):
    monkeypatch.setattr(catalog, "app_dir", lambda: tmp_path / "app")
    (tmp_path / "app").mkdir()
    trip_dir = tmp_path / "trip"
    trip_dir.mkdir()
    for n, colour in enumerate(["red", "green", "blue", "yellow"]):
        exif = Image.Exif()
        exif[306] = f"2025:07:03 1{n}:00:00"
        Image.new("RGB", (160, 90), colour).save(trip_dir / f"p{n}.jpg", exif=exif)
    trip = catalog.Trip(trip_dir)
    trip.prepare()
    paths = [r["path"] for r in trip.items()]
    for path, seconds in zip(paths, (1.0, 2.0, 3.0, 4.0)):
        trip.execute("UPDATE items SET seconds = ? WHERE path = ?", (seconds, path))
    board = lambda: [(Path(r["path"]).name, r["seconds"]) for r in trip.storyboard()]
    trip.swap_shots(paths[0], paths[2])
    assert board() == [("p2.jpg", 1.0), ("p1.jpg", 2.0), ("p0.jpg", 3.0), ("p3.jpg", 4.0)]
    trip.take_out(paths[1])  # p2 before it plays on for its 2 seconds
    assert board() == [("p2.jpg", 3.0), ("p0.jpg", 3.0), ("p3.jpg", 4.0)]
    trip.replace_shot(paths[0], paths[1])
    assert board() == [("p2.jpg", 3.0), ("p1.jpg", 3.0), ("p3.jpg", 4.0)]


def test_post_its_mark_the_best_bits_and_become_the_videos_picture(tmp_path, monkeypatch):
    monkeypatch.setattr(catalog, "app_dir", lambda: tmp_path / "app")
    (tmp_path / "app").mkdir()
    trip_dir = tmp_path / "trip"
    trip_dir.mkdir()
    subprocess.run([ffmpeg_paths()[0], "-loglevel", "error", "-y", "-f", "lavfi", "-i", "testsrc2=s=320x180:r=30:d=6",
                    "-c:v", "libx264", str(trip_dir / "kite.mp4")], check=True)
    trip = catalog.Trip(trip_dir)
    trip.prepare()
    path = trip.items()[0]["path"]
    trip.add_postit(path, 4.0, "jump")
    trip.add_postit(path, 1.5)
    trip.add_postit(path, 1.6)  # the same moment again
    found = trip.postits(path)
    assert [(p["t"], p["name"]) for p in found] == [(1.5, ""), (4.0, "jump")]
    assert trip.items()[0]["thumb"] == found[0]["thumb"] and Path(found[0]["thumb"]).exists()
    trip.change_postit(path, 0, remove=True)
    assert [p["name"] for p in trip.postits(path)] == ["jump"]
    trip.set_keep(path, 1.0, 6.0)  # Auto-edit sees the bookmark in the time of the part you kept
    assert next(r for r in trip.auto_edit_pool() if r["path"] == path)["bookmarks"] == [3.0]


def test_pasting_a_shot_in_front_keeps_every_cut_in_place(tmp_path, monkeypatch):
    monkeypatch.setattr(catalog, "app_dir", lambda: tmp_path / "app")
    (tmp_path / "app").mkdir()
    trip_dir = tmp_path / "trip"
    trip_dir.mkdir()
    for n, colour in enumerate(["red", "green", "blue", "yellow"]):
        exif = Image.Exif()
        exif[306] = f"2025:07:03 1{n}:00:00"
        Image.new("RGB", (160, 90), colour).save(trip_dir / f"p{n}.jpg", exif=exif)
    trip = catalog.Trip(trip_dir)
    trip.prepare()
    paths = [r["path"] for r in trip.items()]
    for path, seconds in zip(paths, (1.0, 2.0, 3.0, 4.0)):
        trip.execute("UPDATE items SET seconds = ? WHERE path = ?", (seconds, path))
    board = lambda: [(Path(r["path"]).name, r["seconds"]) for r in trip.storyboard()]
    trip.insert_shot(paths[3], 1)  # the last one pasted in front of the second
    assert board() == [("p0.jpg", 1.0), ("p3.jpg", 2.0), ("p1.jpg", 3.0), ("p2.jpg", 4.0)]
    trip.insert_shot(paths[0], 4)  # the first one to the end
    assert board() == [("p3.jpg", 1.0), ("p1.jpg", 2.0), ("p2.jpg", 3.0), ("p0.jpg", 4.0)]
    trip.move(paths[2], -1)  # one up in the Storyboard: the cuts stay too
    assert board() == [("p3.jpg", 1.0), ("p2.jpg", 2.0), ("p1.jpg", 3.0), ("p0.jpg", 4.0)]
    trip.move(paths[3], 2)  # two down
    assert board() == [("p2.jpg", 1.0), ("p1.jpg", 2.0), ("p3.jpg", 3.0), ("p0.jpg", 4.0)]


def test_crop_saves_a_cropped_copy_in_place_of_the_photo(tmp_path, monkeypatch):
    monkeypatch.setattr(catalog, "app_dir", lambda: tmp_path / "app")
    (tmp_path / "app").mkdir()
    trip_dir = tmp_path / "trip"
    trip_dir.mkdir()
    img = Image.new("RGB", (400, 200), "red")
    img.paste("blue", (200, 0, 400, 200))
    exif = Image.Exif()
    exif[306] = "2025:07:03 10:00:00"
    img.save(trip_dir / "p.jpg", exif=exif)
    trip = catalog.Trip(trip_dir)
    trip.prepare()
    photo = str(trip_dir / "p.jpg")
    trip.set_crop(photo, 2.0, 0.75, 0.5)
    copy = trip.crop_to_copy(photo)
    by = {i["path"]: i for i in trip.items()}
    assert by[copy]["chosen"] and not by[photo]["chosen"] and by[copy]["shot_of"] == photo
    assert by[copy]["taken_at"] == by[photo]["taken_at"] and by[copy]["thumb"]
    with Image.open(copy) as c:
        assert c.size == (200, 100) and c.getpixel((100, 50))[2] > 200  # the blue half, zoomed in
    with Image.open(photo) as p:
        assert p.size == (400, 200)  # your own photo is not changed
    trip.prepare()
    assert copy in {i["path"] for i in trip.items()}


def test_highlights_from_one_trip_are_suggested_on_the_next(tmp_path, monkeypatch):
    from tests.test_render import make_clip

    monkeypatch.setattr(catalog, "app_dir", lambda: tmp_path / "app")
    (tmp_path / "app").mkdir()
    jump = np.eye(4, dtype=np.float32)[0]
    calm = np.eye(4, dtype=np.float32)[1]
    # The picture model stand-in: the frame at 8.5 s of the test clip is the jump (frames every 2 s from 0.5).
    embed = lambda pictures: np.array([jump if n == 4 else calm for n, _ in enumerate(pictures)])  # noqa: E731
    first = tmp_path / "first"
    first.mkdir()
    make_clip(first / "kite.mp4", "320x180", 12, "2025-07-03T10:00:00Z")
    trip = catalog.Trip(first)
    trip.prepare()
    trip.frame_embedder = embed
    path = trip.items()[0]["path"]
    for t in (7.6, 8.0, 8.5, 9.0):
        trip.add_postit(path, t)
    assert trip.suggest_highlights() == []  # four is too few to learn from
    trip.add_postit(path, 9.4)
    assert trip.highlights_learned() == (5, 0)
    trip.change_postit(path, 0, remove=True)  # a highlight you take off is forgotten
    trip.learn_highlights()
    assert trip.highlights_learned() == (4, 0)
    trip.add_postit(path, 7.6)
    trip.learn_highlights()  # (the app does this when you close the window)

    second = tmp_path / "second"
    second.mkdir()
    make_clip(second / "surf.mp4", "256x144", 12, "2025-08-03T10:00:00Z")
    make_clip(second / "beach.mp4", "320x180", 12, "2025-08-04T10:00:00Z", audio=False)
    new = catalog.Trip(second)
    new.prepare()
    new.frame_embedder = embed
    ideas = new.suggest_highlights()
    assert len(ideas) == 2 and all(i["t"] == 8.5 for i in ideas)
    keep, drop = ideas[0]["path"], ideas[1]["path"]
    new.keep_idea(keep, 8.5)
    assert [p["t"] for p in new.postits(keep)] == [8.5] and new.highlights_learned() == (6, 0)
    new.drop_idea(drop, 8.5)
    assert new.highlight_ideas() == [] and new.highlights_learned() == (6, 1)
    assert new.suggest_highlights() == []  # the dropped second isn't suggested again


def test_reflow_puts_a_longer_shot_on_the_next_pins_at_normal_speed(tmp_path, monkeypatch):
    from trip_memory import autoedit

    monkeypatch.setattr(catalog, "app_dir", lambda: tmp_path / "app")
    (tmp_path / "app").mkdir()
    trip_dir = tmp_path / "trip"
    trip_dir.mkdir()
    for n, colour in enumerate(["red", "green", "blue", "yellow"]):
        exif = Image.Exif()
        exif[306] = f"2025:07:03 1{n}:00:00"
        Image.new("RGB", (160, 90), colour).save(trip_dir / f"p{n}.jpg", exif=exif)
    trip = catalog.Trip(trip_dir)
    trip.prepare()
    paths = [r["path"] for r in trip.items()]
    for path in paths:
        trip.execute("UPDATE items SET seconds = 2.0, transition = ? WHERE path = ?", (autoedit.HARD_CUT, path))
    # The first shot wants 3.7 s: it ends on the pin at 4, the others follow pin by pin, the last no longer fits.
    gone = trip.reflow([2.0, 4.0, 6.0, 8.0], {paths[0]: 3.7})
    board = [(Path(r["path"]).name, round(r["seconds"], 1)) for r in trip.storyboard()]
    assert gone == 1 and board == [("p0.jpg", 4.0), ("p1.jpg", 2.0), ("p2.jpg", 2.0)]
    assert all((r["speed"] or 1.0) == 1.0 for r in trip.storyboard())


def test_rough_cut_snaps_to_the_pins_only_when_fitted_and_the_fit_can_be_undone(tmp_path, monkeypatch):
    from trip_memory import autoedit

    monkeypatch.setattr(catalog, "app_dir", lambda: tmp_path / "app")
    (tmp_path / "app").mkdir()
    trip_dir = tmp_path / "trip"
    trip_dir.mkdir()
    for n, colour in enumerate(["red", "green", "blue"]):
        exif = Image.Exif()
        exif[306] = f"2025:07:03 1{n}:00:00"
        Image.new("RGB", (160, 90), colour).save(trip_dir / f"p{n}.jpg", exif=exif)
    trip = catalog.Trip(trip_dir)
    trip.prepare()
    for r in trip.items():
        trip.execute("UPDATE items SET seconds = 2.6, transition = ? WHERE path = ?", (autoedit.HARD_CUT, r["path"]))
    assert trip.edit_stage() == "rough"
    result = trip.fit_to_music([2.0, 4.0, 6.0, 8.0])
    assert trip.edit_stage() == "fitted" and result["moved"] == 3 and result["gone"] == 0
    assert [round(r["seconds"], 1) for r in trip.storyboard()] == [2.0, 2.0, 2.0]
    assert trip.undo() == "fitting to the music"
    assert trip.edit_stage() == "rough" and [round(r["seconds"], 1) for r in trip.storyboard()] == [2.6, 2.6, 2.6]


def test_a_new_trip_replaces_the_last_trips_themes_with_its_own(tmp_path, monkeypatch):
    from trip_memory import themefinder

    monkeypatch.setattr(catalog, "app_dir", lambda: tmp_path / "app")
    (tmp_path / "app").mkdir()
    trip_dir = tmp_path / "trip"
    trip_dir.mkdir()
    table = themefinder.phrases()
    rng = np.random.default_rng(2)
    sentences = rng.normal(size=(len(table), 512)).astype(np.float32)
    sentences /= np.linalg.norm(sentences, axis=1, keepdims=True)
    gates = next(i for i, p in enumerate(table) if p[1] == "Torii gates")
    ramen = next(i for i, p in enumerate(table) if p[1] == "Ramen & noodles")
    for n in range(8):
        exif = Image.Exif()
        exif[306] = f"2025:07:03 1{n}:00:00"
        Image.new("RGB", (160 + n, 90), "red").save(trip_dir / f"p{n}.jpg", exif=exif)
    trip = catalog.Trip(trip_dir)
    trip.prepare()
    for n, r in enumerate(sorted(trip.items(), key=lambda r: r["path"])):
        v = sentences[gates if n < 4 else ramen] + 0.02 * rng.normal(size=512)
        trip.execute("UPDATE items SET clip = ?, chosen = 1 WHERE path = ?",
                     ((v / np.linalg.norm(v)).astype(np.float32).tobytes(), r["path"]))
    assert trip.missing_fingerprints() == 0
    found = trip.propose_themes(sentences)
    assert {p["name"] for p in found} == {"Temples & shrines", "Food"} and trip.theme_proposals() == found
    food = next(p for p in found if p["name"] == "Food")
    trip.use_theme_proposals([food | {"name": "Ramen", "vector": None,
                                      "subs": [{"name": "Noodles", "paths": food["paths"][:2]}]}])
    assert [a["name"] for a in trip.activities()] == ["Ramen", "Other"]
    confirmed = [r for r in trip.items() if r["theme"] == "Ramen"]
    assert len(confirmed) == 2 and trip.theme_proposals() == []
    assert [len(m["shots"]) for m in trip.moments() if m["activity"] == "Ramen"] == [2]


def test_fit_to_music_puts_every_cut_exactly_on_a_pin_whatever_the_transitions(tmp_path, monkeypatch):
    from trip_memory import autoedit
    from trip_memory.render import overlaps, shot_times, style_between

    monkeypatch.setattr(catalog, "app_dir", lambda: tmp_path / "app")
    (tmp_path / "app").mkdir()
    trip_dir = tmp_path / "trip"
    trip_dir.mkdir()
    for n in range(7):
        exif = Image.Exif()
        exif[306] = f"2025:07:03 1{n}:00:00"
        Image.new("RGB", (160 + n, 90), "red").save(trip_dir / f"p{n}.jpg", exif=exif)
    trip = catalog.Trip(trip_dir)
    trip.prepare()
    kinds = [autoedit.HARD_CUT, None, "Dissolve", autoedit.HARD_CUT, None, "Zoom punch", None]
    for r, kind, seconds in zip(sorted(trip.items(), key=lambda r: r["path"]), kinds, [1.3, 2.9, 1.1, 2.2, 1.7, 3.1, 2.0]):
        trip.execute("UPDATE items SET seconds = ?, transition = ? WHERE path = ?", (seconds, kind, r["path"]))
    pins = [1.2, 2.0, 2.6, 3.5, 4.1, 5.0, 6.2, 7.0, 8.3, 9.0, 10.4, 12.0, 14.0]
    trip.fit_to_music(pins, 4.0, 0.8, "Fade")
    board = trip.storyboard()
    lengths = [trip.screen_length(r) for r in board]
    styles = [style_between(board[k]["transition"], k + 1, "Fade", 0.8) for k in range(len(board) - 1)]
    times, over = shot_times(lengths, styles), overlaps(lengths, styles)
    cuts = [times[k]["start"] + times[k]["length"] - over[k] / 2 for k in range(len(board) - 1)]
    assert all(min(abs(c - p) for p in pins) < 0.02 for c in cuts), cuts


def test_theme_folders_move_like_folders_and_sub_theme_ticks_confirm_the_theme(tmp_path, monkeypatch):
    monkeypatch.setattr(catalog, "app_dir", lambda: tmp_path / "app")
    (tmp_path / "app").mkdir()
    trip_dir = tmp_path / "trip"
    trip_dir.mkdir()
    for n, colour in enumerate(["red", "green", "blue", "yellow"]):
        exif = Image.Exif()
        exif[306] = f"2025:07:03 1{n}:00:00"
        Image.new("RGB", (160, 90), colour).save(trip_dir / f"p{n}.jpg", exif=exif)
    trip = catalog.Trip(trip_dir)
    trip.prepare()
    paths = [r["path"] for r in trip.items()]
    trip.set_chosen(paths, True)
    trip.add_activity("Deer and fish", None)
    trip.add_activity("Animals", None)
    trip.execute("UPDATE items SET theme = NULL, theme_auto = 'Deer and fish'")
    # A tick in a sub-theme confirms the shot in its theme: no question mark there.
    trip.add_moment("Deer and fish", "deer")
    trip.tick_moment("Deer and fish", "deer", paths[0], 0.0, True)
    ok = {r["path"]: r["theme_ok"] for r in trip.theme_rows()}
    assert ok[paths[0]] and not ok[paths[1]]
    # Sub-theme dragged onto another theme: it and its shots move there.
    trip.move_moment("Deer and fish", "deer", "Animals")
    assert trip.moment("Animals", "deer") and not trip.moment("Deer and fish", "deer")
    assert {r["path"]: r["theme"] for r in trip.theme_rows()}[paths[0]] == "Animals"
    # Dragged out: a theme of its own.
    assert trip.moment_to_theme("Animals", "deer") == "deer"
    assert "deer" in [a["name"] for a in trip.activities()] and not trip.moments()
    assert {r["path"]: r["theme"] for r in trip.theme_rows()}[paths[0]] == "deer"
    # A theme dropped onto another: a sub-theme there, all its shots moved in and ticked.
    trip.theme_to_moment("deer", "Animals")
    assert "deer" not in [a["name"] for a in trip.activities()]
    assert [s["path"] for s in trip.moment("Animals", "deer")["shots"]] == [paths[0]]
    assert {r["path"]: r["theme"] for r in trip.theme_rows()}[paths[0]] == "Animals"


def test_a_starred_photo_is_always_in_the_video(tmp_path, monkeypatch):
    monkeypatch.setattr(catalog, "app_dir", lambda: tmp_path / "app")
    (tmp_path / "app").mkdir()
    trip_dir = tmp_path / "trip"
    trip_dir.mkdir()
    Image.new("RGB", (160, 90), "red").save(trip_dir / "p0.jpg")
    trip = catalog.Trip(trip_dir)
    trip.prepare()
    path = trip.items()[0]["path"]
    assert not trip.starred(path)
    trip.set_star(path, True)
    assert trip.starred(path)
    assert catalog.Trip._with_locks([trip.items()[0]])[0]["forced"]
    trip.set_star(path, False)
    assert not trip.starred(path)


def test_the_opening_and_the_ending_have_their_own_look_and_length(tmp_path, monkeypatch):
    monkeypatch.setattr(catalog, "app_dir", lambda: tmp_path / "app")
    (tmp_path / "app").mkdir()
    (tmp_path / "trip").mkdir()
    trip = catalog.Trip(tmp_path / "trip")
    trip.set_title_settings(card="bold", open_room=2.0, end_room=0)  # (the old black ending)
    assert trip.end_card()[1]["card"] == "bold"  # (until you choose: the ending looks like the opening)
    trip.set_title_settings(end_card="minimal", end_black=6.0)
    assert trip.opening_card()["card"] == "bold" and trip.opening_card()["seconds"] == 2.0
    seconds, words = trip.end_card()
    assert seconds == 6.0 and words["card"] == "minimal"


def test_the_ending_card_has_its_own_room_at_the_end(tmp_path):
    trip_dir = tmp_path / "Portugal_2025"
    trip_dir.mkdir()
    _gps_photo(trip_dir / "0.jpg", "2025:07:03 10:00:00", 38.7, -9.1, "red")
    trip = catalog.Trip(trip_dir)
    trip.prepare()
    trip.set_title_settings(end_room=8.0, end_card="minimal")
    assert trip.closing_card()["seconds"] == 8.0 and trip.closing_card()["card"] == "minimal"
    assert trip.end_card() == (0.0, None)  # (no black after the last shot: the card goes over it)
    assert trip.card_rooms()["end_room"] == 8.0
    trip.set_title_settings(end_room=0, end_black=3.0)
    assert trip.closing_card() is None and trip.end_card()[0] == 3.0


def test_doubles_level_one_bursts_then_a_second_cleaning(tmp_path, monkeypatch):
    monkeypatch.setattr(catalog, "app_dir", lambda: tmp_path / "app")
    (tmp_path / "app").mkdir()
    trip_dir = tmp_path / "trip"
    trip_dir.mkdir()
    import numpy as np

    rng = np.random.default_rng(0)
    scene = Image.fromarray(np.kron(rng.integers(0, 255, (8, 12, 3)), np.ones((40, 40, 1))).astype(np.uint8))
    for name, when in [("a", "2025:07:03 10:00:00"), ("b", "2025:07:03 10:00:20"), ("c", "2025:07:03 10:02:30"),
                       ("d", "2025:07:03 10:20:00")]:
        exif = Image.Exif()
        exif[306] = when
        scene.save(trip_dir / f"{name}.jpg", exif=exif)
    trip = catalog.Trip(trip_dir)
    trip.prepare()
    sets = lambda: {Path(i["path"]).stem: i["similar_set"] for i in trip.items()}
    assert trip.look_alike_level() == "l1"
    s = sets()
    assert s["a"] == s["b"] != s["c"] != s["d"]  # level 1: only the burst
    both = [str(trip_dir / "a.jpg"), str(trip_dir / "b.jpg")]
    trip.set_chosen(both, True)
    trip.keep_marked(both)  # sorted out, both kept
    trip.set_look_alike_level("l3")  # a later cleaning
    s = sets()
    assert s["a"] == s["b"] == s["c"] != s["d"]  # a few minutes apart; 20 minutes later is another picture
    assert trip.accept_sets() == (1, 2)  # a set you didn't touch (all ticked): only its ⭐ stays
    assert sum(i["chosen"] for i in trip.items()) == 2  # the ⭐ of the set, and d


def test_mark_upright_without_preview():
    from trip_memory.catalog import Trip

    rows = [{"thumb": None}, {}, {"thumb": "/nowhere.jpg"}]
    Trip._mark_upright(rows)
    assert [r["upright"] for r in rows] == [False, False, False]


def test_remove_one_shot_from_themes(tmp_path, monkeypatch):
    monkeypatch.setattr(catalog, "app_dir", lambda: tmp_path / "app")
    (tmp_path / "app").mkdir()
    trip_dir = tmp_path / "trip"
    trip_dir.mkdir()
    for n, colour in enumerate(("red", "red", "blue")):
        _gps_photo(trip_dir / f"{n}.jpg", f"2025:07:03 10:0{n}:00", 38.7223, -9.1393, colour)
    trip = catalog.Trip(trip_dir)
    trip.prepare()
    gone = str(trip_dir / "0.jpg")
    trip.remove([gone])  # even while ticked
    assert gone not in [i["path"] for i in trip.items()]
    assert [i["path"] for i in trip.items(removed=True)] == [gone]
    assert (trip_dir / "0.jpg").exists()


def test_alternatives_suggest_shots_not_in_the_video(tmp_path, monkeypatch):
    monkeypatch.setattr(catalog, "app_dir", lambda: tmp_path / "app")
    (tmp_path / "app").mkdir()
    trip_dir = tmp_path / "trip"
    trip_dir.mkdir()
    for n, colour in enumerate(("red", "red", "blue")):
        _gps_photo(trip_dir / f"{n}.jpg", f"2025:07:03 10:0{n}:00", 38.7223, -9.1393, colour)
    trip = catalog.Trip(trip_dir)
    trip.prepare()
    for n in range(3):
        trip.execute("UPDATE items SET thumb = path, position = ?, included = ? WHERE path = ?",
                     (n if n < 2 else None, int(n < 2), str(trip_dir / f"{n}.jpg")))
    found = trip.alternatives(str(trip_dir / "0.jpg"), 3.0)
    assert [Path(r["path"]).name for r in found] == ["2.jpg"]
    assert found[0]["why"]


def test_a_moved_trip_keeps_its_work(tmp_path, monkeypatch):
    import shutil
    monkeypatch.setattr(catalog, "app_dir", lambda: tmp_path / "app")
    (tmp_path / "app").mkdir()
    trip_dir = tmp_path / "Japán"
    trip_dir.mkdir()
    for n, colour in enumerate(("red", "red", "blue")):
        _gps_photo(trip_dir / f"{n}.jpg", f"2025:07:03 10:0{n}:00", 38.7223, -9.1393, colour)
    trip = catalog.Trip(trip_dir)
    trip.prepare()
    trip.set_tile_frame(str(trip_dir / "1.jpg"), 0.2, 0.3, 0.5)
    trip.set_chosen([str(trip_dir / "2.jpg")], False)
    trip.db.close()
    assert [t["name"] for t in catalog.known_trips()] == ["Japán"]
    moved = tmp_path / "Trips" / "Japan 2025"
    moved.parent.mkdir()
    shutil.move(str(trip_dir), str(moved))
    found = catalog.find_moved_trip(moved)
    assert found and found["folder"] == str(trip_dir.resolve()) and not found["found"]
    catalog.move_trip(found["folder"], moved)
    again = catalog.Trip(moved)
    paths = {i["path"]: i for i in again.items()}
    assert set(paths) == {str(moved.resolve() / f"{n}.jpg") for n in range(3)}
    assert not paths[str(moved.resolve() / "2.jpg")]["chosen"]
    assert str(moved.resolve() / "1.jpg") in again.tile_frames()
    assert all(Path(i["thumb"]).exists() for i in again.items() if i["thumb"])
    assert catalog.find_moved_trip(moved) is None


def test_damaged_files_leave_the_trip_by_themselves(tmp_path, monkeypatch):
    monkeypatch.setattr(catalog, "app_dir", lambda: tmp_path / "app")
    monkeypatch.setattr(catalog, "FaceFinder", FakeFaces)
    monkeypatch.setattr(catalog, "SceneTagger", FakeScenes)
    (tmp_path / "app").mkdir()
    trip_dir = tmp_path / "trip"
    trip_dir.mkdir()
    _gps_photo(trip_dir / "a.jpg", "2025:07:03 10:00:00", 38.7223, -9.1393, "red")
    _gps_photo(trip_dir / "b.jpg", "2025:07:03 10:05:00", 38.7223, -9.1393, "blue")
    whole = (trip_dir / "b.jpg").read_bytes()
    (trip_dir / "b.jpg").write_bytes(whole[: len(whole) // 3])  # a copy cut short

    trip = catalog.Trip(trip_dir)
    trip.analyse()
    assert [Path(i["path"]).name for i in trip.items()] == ["a.jpg"]
    assert [Path(i["path"]).name for i in trip.items(removed=True)] == ["b.jpg"]  # (See removed brings it back)
    assert "damaged" in trip.progress.notes[0] and "b.jpg" in trip.progress.notes[0]
    assert (trip_dir / "b.jpg").exists()  # your file stays


def test_quality_check_suggests_and_remembers_what_you_keep(tmp_path, monkeypatch):
    monkeypatch.setattr(catalog, "app_dir", lambda: tmp_path / "app")
    monkeypatch.setattr(catalog, "FaceFinder", FakeFaces)
    monkeypatch.setattr(catalog, "SceneTagger", FakeScenes)
    (tmp_path / "app").mkdir()
    trip_dir = tmp_path / "trip"
    trip_dir.mkdir()
    _gps_photo(trip_dir / "a.jpg", "2025:07:03 10:00:00", 38.7223, -9.1393, "red")  # one plain colour: empty
    trip = catalog.Trip(trip_dir)
    trip.analyse()
    found = trip.quality_suggestions()
    assert [Path(r["path"]).name for r in found] == ["a.jpg"] and found[0]["reason"]
    trip.keep_quality([found[0]["path"]])
    assert trip.quality_suggestions() == []  # kept: never suggested again


def test_a_big_picture_is_made_the_right_way_round(tmp_path, monkeypatch):
    monkeypatch.setattr(catalog, "app_dir", lambda: tmp_path / "app")
    monkeypatch.setattr(catalog, "FaceFinder", FakeFaces)
    monkeypatch.setattr(catalog, "SceneTagger", FakeScenes)
    (tmp_path / "app").mkdir()
    trip_dir = tmp_path / "trip"
    trip_dir.mkdir()
    from PIL import Image
    Image.new("RGB", (3000, 2000), "red").save(trip_dir / "wide.jpg")
    trip = catalog.Trip(trip_dir)
    trip.analyse()
    path = str(trip_dir / "wide.jpg")
    with Image.open(trip.big_picture(path)) as img:
        assert img.size == (1800, 1200)
    trip.rotate(path, 90)
    with Image.open(trip.big_picture(path)) as img:
        assert img.size == (1200, 1800)  # turned: a new big picture, upright


def test_the_app_sorts_the_other_shots_into_real_themes_when_asked(tmp_path, monkeypatch):
    monkeypatch.setattr(catalog, "app_dir", lambda: tmp_path / "app")
    (tmp_path / "app").mkdir()
    folder = tmp_path / "trip"
    folder.mkdir()
    for n in range(3):
        Image.new("RGB", (160, 90), "red").save(folder / f"p{n}.jpg")
    trip = catalog.Trip(folder)
    trip.prepare()
    rng = np.random.default_rng(3)
    for r in trip.items():  # pictures like none of the themes' words: Other
        v = rng.normal(size=512).astype(np.float32)
        trip.execute("UPDATE items SET clip = ? WHERE path = ?", ((v / np.linalg.norm(v)).tobytes(), r["path"]))
    trip.execute("UPDATE items SET theme_auto = 'Other'")  # (as the app had them)
    trip.set_sort_other(True)
    assert all(r["theme"] != "Other" and not r["theme_ok"] for r in trip.theme_rows())  # each in a real theme
    assert trip.sort_other_on()
