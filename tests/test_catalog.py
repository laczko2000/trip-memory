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
