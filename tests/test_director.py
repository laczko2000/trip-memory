"""The AI director: clip descriptions, storyline ideas, and putting the stories in the chosen order."""

import pytest

from trip_memory import catalog, director
from tests.test_catalog import _gps_photo


class FakeDirector:
    """Calls a red picture a beach and a blue one a pool; suggests both orders of the three stories."""

    batch = 2

    def __init__(self):
        self.prompts = []

    def describe(self, clips):
        from io import BytesIO

        from PIL import Image

        result = []
        for frames in clips:
            r, _, b = Image.open(BytesIO(frames[0])).convert("RGB").getpixel((5, 5))
            result.append(director.Caption("Waves on a sandy beach", 5) if r > b else director.Caption("Empty pool", 2))
        return result

    def storylines(self, prompt):
        self.prompts.append(prompt)
        return {
            "storylines": [
                {"title": "Pools first", "idea": "Start calm.", "stories": [2, 2, 9, 1]},  # a repeat and a made-up number
                {"title": "Nothing", "idea": "", "stories": []},
                {"title": "Beach only", "idea": "The best bits.", "stories": [3, 1]},
            ],
            "names": [{"story": 1, "name": "Morning at the beach"}, {"story": 2, "name": "Hotel pool"}],
        }


@pytest.fixture
def trip(tmp_path, monkeypatch):
    monkeypatch.setattr(catalog, "app_dir", lambda: tmp_path / "app")
    (tmp_path / "app").mkdir()
    folder = tmp_path / "trip"
    folder.mkdir()
    _gps_photo(folder / "a.jpg", "2025:07:03 09:00:00", 38.7223, -9.1393, "red")
    _gps_photo(folder / "b.jpg", "2025:07:04 15:00:00", 47.4979, 19.0402, "blue")
    _gps_photo(folder / "c.jpg", "2025:07:05 15:00:00", 47.4979, 19.0402, "red")
    trip = catalog.Trip(folder)
    trip.prepare()
    return trip


def test_clips_are_described_and_scored(trip):
    trip.describe_clips(FakeDirector())
    items = trip.items()
    assert [(i["ai_caption"], i["ai_score"]) for i in items] == [
        ("Waves on a sandy beach", 5), ("Empty pool", 2), ("Waves on a sandy beach", 5)]
    from trip_memory.autoedit import shot_value

    assert shot_value(items[0]) > shot_value(items[1])  # Auto-edit prefers what the AI found memorable


def test_storylines_are_suggested_and_applied(trip):
    trip.describe_clips(FakeDirector())
    ai = FakeDirector()
    trip.rename_story(trip.stories()[1]["key"], "Our own name")
    saved = trip.suggest_storylines(ai)
    assert "Waves on a sandy beach" in ai.prompts[0]  # the AI sees what each story shows
    assert [s["title"] for s in saved["storylines"]] == ["Pools first", "Beach only"]  # the empty one is dropped

    trip.apply_storyline(0)
    stories = trip.stories()
    assert [s["name"] for s in stories] == ["Our own name", "Morning at the beach", stories[2]["auto_name"]]
    assert [s["enabled"] for s in stories] == [True, True, False]  # story 3 is left out
    assert trip.storyline_used() == 0
    assert [r["path"][-5:] for r in trip.auto_edit_pool()] == ["b.jpg", "a.jpg"]

    trip.apply_storyline(1)
    assert [r["path"][-5:] for r in trip.auto_edit_pool()] == ["c.jpg", "a.jpg"]


def test_a_single_clip_answer_with_another_number_still_counts():
    assert director._captions({"clips": [{"clip": 0, "caption": " A  dog ", "highlight": 9}]}, 1) == [director.Caption("A dog", 5)]


def test_missing_setup_explains_what_to_do(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(director.DirectorError, match="API key"):
        director.make({"provider": "claude"})
    with pytest.raises(director.DirectorError, match="ollama pull"):
        director.make({"provider": "ollama"})
