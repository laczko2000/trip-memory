"""The AI director: clip descriptions, storyline ideas, and putting the stories in the chosen order."""

from pathlib import Path

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


def test_look_alikes_are_described_once(trip, tmp_path):
    folder = trip.folder
    _gps_photo(folder / "a2.jpg", "2025:07:03 09:00:20", 38.7223, -9.1393, "red")  # a double of a.jpg
    trip.prepare()
    trip.set_chosen([str(folder / "a2.jpg")], True)
    by_name = {Path(i["path"]).name: i for i in trip.items()}
    trip.execute("UPDATE items SET similar_set = ?, best = 0 WHERE path = ?", (by_name["a.jpg"]["similar_set"], by_name["a2.jpg"]["path"]))

    ai = FakeDirector()
    looked = []
    describe = ai.describe
    ai.describe = lambda clips: (looked.append(len(clips)), describe(clips))[1]
    trip.describe_clips(ai)
    assert sum(looked) == 3  # four files, but the double shares its set's description
    assert {i["ai_caption"] for i in trip.items() if Path(i["path"]).name.startswith("a")} == {"Waves on a sandy beach"}


class FakePlanner(FakeDirector):
    def plan(self, prompt):
        self.prompts.append(prompt)
        return {"theme": "Beach and pool", "idea": "Start calm, end on the waves.",
                "parts": [{"role": "opening", "shots": [2], "note": "wide"},
                          {"role": "highlight", "shots": [1, 2, 7], "note": "the waves"}]}  # a repeat, a made-up shot


def test_stories_are_planned_and_auto_edit_follows_the_plan(trip):
    from trip_memory import autoedit

    trip.link_story(0, to_next=True)  # a and b: one story of two shots
    trip.describe_clips(FakeDirector())
    ai = FakePlanner()
    trip.plan_stories(ai)
    assert "Waves on a sandy beach" in ai.prompts[0]
    story = trip.stories()[0]
    assert story["theme"] == "Beach and pool" and story["name"] == "Beach and pool" and story["planned"]
    roles = {Path(it["path"]).name: (it["plan_role"], it["plan_pos"]) for it in story["items"]}
    assert roles == {"b.jpg": ("opening", 0), "a.jpg": ("highlight", 1)}

    order = [Path(r["path"]).name for r in autoedit.choose(trip.auto_edit_pool())]
    assert order[:2] == ["b.jpg", "a.jpg"]  # the plan's order, not the time order

    trip.move_in_plan(str(trip.folder / "a.jpg"), -1)
    assert [Path(r["path"]).name for r in autoedit.choose(trip.auto_edit_pool())][:2] == ["a.jpg", "b.jpg"]
    trip.forget_plan(story["key"])
    assert not trip.stories()[0]["planned"] and trip.stories()[0]["theme"] is None


def test_montage_and_slow_moments_set_the_pace():
    from trip_memory import autoedit

    photo = {"kind": "photo"}
    assert autoedit.pace_of({**photo, "plan_role": "montage"})[0] > 0.8
    assert autoedit.pace_of({**photo, "plan_role": "slow moment"}) == (0.15, True)


class FakeLeaver(FakeDirector):
    """Plans story 1 (a, b, c linked) as c then a: b is left out."""

    def plan(self, prompt):
        return {"theme": "Water", "idea": "", "parts": [{"role": "opening", "shots": [3], "note": ""},
                                                       {"role": "ending", "shots": [1], "note": ""}]}


def test_stories_storyboard_and_video_agree(trip):
    trip.link_story(0, to_next=True)
    trip.link_story(0, to_next=True)  # a, b and c: one story
    trip.plan_stories(FakeLeaver())
    names = lambda rows: [Path(r["path"]).name for r in rows]
    # A shot the plan leaves out does not creep into the storyboard, Auto-edit or the video.
    assert names(trip.storyboard()) == ["c.jpg", "a.jpg"]
    assert sorted(names(trip.auto_edit_pool())) == ["a.jpg", "c.jpg"]
    assert [m.path.name for m in trip.media_items()] == ["c.jpg", "a.jpg"]

    # After a first cut, changing the plan reorders the storyboard to match.
    trip.execute("UPDATE items SET position = CASE WHEN path LIKE '%c.jpg' THEN 0 ELSE 1 END")
    trip.move_in_plan(str(trip.folder / "a.jpg"), -1)
    assert names(trip.storyboard()) == ["a.jpg", "c.jpg"]
    trip.set_plan_role(str(trip.folder / "b.jpg"), "build-up")  # put back in by you
    assert "b.jpg" in names(trip.storyboard())


def test_a_new_storyline_renames_and_resets_the_storyboard(trip):
    ai = FakeDirector()
    trip.suggest_storylines(ai)
    trip.apply_storyline(0)
    assert trip.stories()[1]["name"] == "Morning at the beach"
    trip.execute("UPDATE items SET position = 5, cut_out = 1 WHERE path LIKE '%a.jpg'")  # an older first cut

    ai.storylines = lambda prompt: {"storylines": [{"title": "Again", "idea": "", "stories": [1, 2]}],
                                    "names": [{"story": 1, "name": "Sunrise swim"}]}
    trip.suggest_storylines(ai)
    trip.apply_storyline(0)
    stories = {s["key"]: s for s in trip.stories()}
    beach = next(s for s in stories.values() if s["items"][0]["path"].endswith("a.jpg"))
    assert beach["name"] == "Sunrise swim" and not beach["renamed"]  # the new storyline's name, not the old one
    assert [Path(r["path"]).name for r in trip.storyboard()] == ["a.jpg", "b.jpg"]  # old first cut gone
    titles = {Path(p).name: t["text"] for p, t in trip.shot_titles().items()}
    assert titles["b.jpg"] == stories[trip.stories()[1]["key"]]["name"]  # the video's story titles match too
    trip.rename_story(beach["key"], "Our swim")
    assert next(s for s in trip.stories() if s["key"] == beach["key"])["name"] == "Our swim"


def test_storyline_names_saved_by_older_versions_become_the_ais(trip):
    ai = FakeDirector()
    trip.suggest_storylines(ai)
    key = trip.stories()[0]["key"]
    trip.execute("ALTER TABLE story_info DROP COLUMN ai_name")
    trip.execute("INSERT OR REPLACE INTO story_info (key, name) VALUES (?, 'Morning at the beach')", (key,))
    again = catalog.Trip(trip.folder)
    story = next(s for s in again.stories() if s["key"] == key)
    assert story["name"] == "Morning at the beach" and not story["renamed"]


class FakeWisher(FakeDirector):
    def wish(self, prompt):
        self.prompts.append(prompt)
        return {"summary": "A lively minute at the beach.", "length_seconds": 60, "format": "phone",
                "people": ["anna", "Zoe"], "focus": ["Beaches", ""], "title": "Sun and sand",
                "versions": [
                    {"name": "Beach only", "idea": "The sea.", "stories": [3, 1, 3, 8], "pace": "lively",
                     "teaser": False, "slow_motion": False, "camera_effects": True, "look": "summer"},
                    {"name": "Day by day", "idea": "In order.", "stories": [1, 2, 3], "pace": "calm",
                     "teaser": True, "slow_motion": True, "camera_effects": False, "look": ""},
                ],
                "names": [{"story": 1, "name": "Morning swim"}]}


def test_type_the_video_you_want(trip):
    trip.describe_clips(FakeDirector())
    trip.execute("INSERT INTO people (name) VALUES ('Anna')")
    ai = FakeWisher()
    wish = trip.make_wish(ai, "  a lively minute,\n beach only ", 95.0)
    assert '"a lively minute, beach only"' in ai.prompts[0] and "Anna" in ai.prompts[0] and "1.6 minutes" in ai.prompts[0]
    assert [Path(k).name for k in wish["stories"]] == ["c.jpg", "a.jpg"] and wish["name"] == "Beach only"
    assert wish["people"] == ["Anna"] and wish["focus"] == ["beaches"] and len(wish["versions"]) == 2
    assert wish["length_seconds"] == 60 and wish["pace"] == "lively" and not wish["slow_motion"]
    assert trip.title_settings()["trip_title"] == "Sun and sand" and trip.look() == "summer"
    assert trip.video_size() == "phone"  # "Instagram" means an upright video
    stories = trip.stories()
    assert [Path(s["items"][0]["path"]).name for s in stories if s["enabled"]] == ["c.jpg", "a.jpg"]
    assert stories[1]["name"] == "Morning swim"
    assert trip.wish()["summary"] == "A lively minute at the beach."

    # The beach shots (the AI called them sandy beaches) are favoured; the pool shot would lose points.
    assert {Path(r["path"]).name: r["wish"] for r in trip.auto_edit_pool()} == {"c.jpg": 1.0, "a.jpg": 1.0}
    pool_row = next(r for r in trip.items() if r["path"].endswith("b.jpg"))
    assert trip.wish_match(pool_row, trip.wish()) == -0.5

    # Another version; then undo goes back to the first.
    trip.use_version(1)
    assert trip.wish()["name"] == "Day by day" and trip.wish()["pace"] == "calm"
    assert [Path(s["items"][0]["path"]).name for s in trip.stories() if s["enabled"]] == ["a.jpg", "b.jpg", "c.jpg"]
    assert trip.undo_label() == "version 1 of your wish"
    trip.undo()
    assert trip.wish()["name"] == "Beach only"
    trip.forget_wish()
    assert trip.wish() is None and trip.wish_match(pool_row, trip.wish()) == 0.0
    trip.undo()
    assert trip.wish()["name"] == "Beach only"


def test_your_own_choices_win_over_the_ai(trip):
    trip.describe_clips(FakeDirector())
    first, second, third = trip.stories()
    trip.set_story_enabled(second["key"], True)  # the AI leaves this one out: you keep it in
    trip.set_story_enabled(third["key"], False)  # the AI wants it: you keep it out
    trip.rename_story(first["key"], "Our beach")
    trip.set_title_settings(trip_title="Summer 25")
    trip.make_wish(FakeWisher(), "beach only", 60.0)
    stories = {s["key"]: s for s in trip.stories()}
    assert stories[second["key"]]["enabled"] and not stories[third["key"]]["enabled"]
    assert stories[first["key"]]["name"] == "Our beach" and trip.title_settings()["trip_title"] == "Summer 25"
    assert set(trip.hand_changes()) == {"1 story name", "1 story switched off", "1 story switched on", "the opening title"}
    trip.forget_my_story_switches()
    trip.use_version(0)
    assert not {s["key"]: s for s in trip.stories()}[second["key"]]["enabled"]


def test_nothing_typed_lets_the_ai_surprise_you():
    prompt = director.wish_prompt("  ", [{"name": "Beach", "when": "Thu", "count": 3, "captions": []}], [], None)
    assert director.SURPRISE in prompt


def test_a_first_cut_can_be_undone(trip):
    from trip_memory import autoedit

    trip.execute("UPDATE items SET position = 7")
    trip.auto_edit_steady(autoedit.Limits(), 10.0, 0.5, "Fade")
    assert trip.undo_label() == "the first cut"
    assert trip.undo() == "the first cut"
    assert {r["position"] for r in trip.items()} == {7} and trip.undo_label() is None


def test_a_wish_with_nothing_usable_keeps_things_as_they_are():
    w = director.clean_wish({"length_seconds": -5, "pace": "wild", "stories": ["1", 0], "format": "square"}, 2, [])
    v = w["versions"][0]
    assert w["length_seconds"] == 0 and v["pace"] == "medium" and v["stories"] == [] and v["slow_motion"]
    assert w["format"] == "" and len(w["versions"]) == 1


class FakeStopper(FakeDirector):
    """You press Stop while the AI is on its first batch."""

    def __init__(self, trip):
        super().__init__()
        self.trip = trip

    def describe(self, clips):
        self.trip.stop_ai()
        return super().describe(clips)

    def storylines(self, prompt):
        self.trip.stop_ai()
        return super().storylines(prompt)

    def wish(self, prompt):
        self.trip.stop_ai()
        return FakeWisher.wish(self, prompt)


def test_stop_keeps_what_the_ai_already_described(trip):
    trip.describe_clips(FakeStopper(trip))
    captions = [i["ai_caption"] for i in trip.items()]
    assert captions[:2] == ["Waves on a sandy beach", "Empty pool"] and captions[2] is None  # one batch of 2 kept
    assert "Stopped" in trip.progress.notes[-1]
    trip.describe_clips(FakeDirector())  # carries on later with the rest
    assert all(i["ai_caption"] for i in trip.items())


def test_a_stopped_storyline_or_wish_changes_nothing(trip):
    trip.describe_clips(FakeDirector())
    before = [(s["name"], s["enabled"]) for s in trip.stories()]
    with pytest.raises(director.Stopped):
        trip.suggest_storylines(FakeStopper(trip))
    with pytest.raises(director.Stopped):
        trip.make_wish(FakeStopper(trip), "beach only", 60.0)
    assert [(s["name"], s["enabled"]) for s in trip.stories()] == before and trip.wish() is None
    assert trip.suggest_storylines(FakeDirector())["storylines"]  # a new request works again


def test_the_length_you_wrote_wins_over_the_ais(trip):
    trip.describe_clips(FakeDirector())
    wish = trip.make_wish(FakeWisher(), "a 1:20 video, beach only", 200.0)  # the AI said 60 s
    assert wish["length"] == {"seconds": 80} and wish["length_seconds"] == 80
    trip.make_wish(FakeWisher(), "half the song, calm", 200.0)
    assert trip.wish()["length"] == {"song": 0.5} and trip.wish()["length_seconds"] == 0
    trip.make_wish(FakeWisher(), "a 1.2 minute video", 200.0, length={"seconds": 72})  # you chose 1:12 when asked
    assert trip.wish()["length_seconds"] == 72
    trip.set_wish_length({"seconds": 95})
    assert trip.wish()["length"] == {"seconds": 95}
    trip.make_wish(FakeWisher(), "beach only", 200.0)  # nothing said: the AI's length
    assert trip.wish()["length"] == {"seconds": 60}
