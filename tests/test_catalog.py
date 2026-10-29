"""The catalogue: places from GPS, scene suggestions, faces grouped into people, and your edits."""

from fractions import Fraction

import numpy as np
from PIL import Image

from trip_memory import catalog
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
