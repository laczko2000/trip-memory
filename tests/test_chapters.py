"""Chapters: a note about the order becomes chapters, shots are matched to them, and the video follows them."""

import numpy as np

from trip_memory import autoedit, chapters, director
from trip_memory.songmap import Section, SongMap


def _kinds(found):
    return [(c["kind"], c["what"]) for c in found]


def test_a_note_becomes_chapters_in_the_order_asked():
    note = ("I want at the beginning more the kiting videos, then separate the people part, put all the food at the "
            "end, and then after close it with something action at the end")
    assert _kinds(chapters.read_note(note, [])) == [
        ("words", "kiting"), ("people", "People"), ("rest", "Everything else"), ("words", "food"),
        ("finale", "Finale: the most action")]


def test_beginning_and_end_win_over_the_order_you_said_them_in():
    found = chapters.read_note("Sunsets at the end. Anna. Start with the beach, everything else in between", ["Anna"])
    assert _kinds(found) == [("words", "beach"), ("person", "Anna"), ("rest", "Everything else"), ("words", "sunsets")]


def test_kiting_looks_for_kitesurfing():
    search, tags = chapters.hint_for("kiting")
    assert "kite" in search
    assert chapters.stem("kiting") == "kit" and chapters.stem("beaches") == "beach"


def _shot(n, tags=(), caption="", people=(), people_tag=None, taken=None):
    return {"path": f"/t/{n:02d}.jpg", "kind": "photo", "duration": 0, "taken_at": taken or f"2025-07-03T10:{n:02d}:00",
            "tags_final": list(tags), "scene_final": None, "ai_caption": caption, "place": None,
            "people": list(people), "people_tag": people_tag, "story": 0, "similar_set": n, "sharpness": 100.0}


def test_shots_go_to_the_chapter_they_fit_best():
    found = [chapters.chapter("words", "kiting", "kitesurfing"), chapters.chapter("people"),
             chapters.chapter("rest"), chapters.chapter("words", "food"), chapters.chapter("finale")]
    rows = [_shot(0, caption="a kitesurfer jumps a wave", people_tag="Solo"),
            _shot(1, people=["Anna"], people_tag="Couple"),
            _shot(2, tags=["Eating & drinking"], people_tag="Group"),  # food with people: food is the closer match
            _shot(3, tags=["Old town"])]
    assert [w for w, _fit in chapters.assign(rows, found, {})] == [0, 1, 3, 2]
    without_rest = [c for c in found if c["kind"] != "rest"]
    assert chapters.assign(rows, without_rest, {})[3][0] is None  # fits nothing: left out


def test_the_picture_model_finds_shots_the_words_miss():
    found = [chapters.chapter("words", "kiting", "kitesurfing")]
    found[0]["vector"] = [1.0, 0.0]
    rows = [_shot(n) for n in range(6)]
    vectors = {r["path"]: np.array([0.1, 1.0], np.float32) for r in rows}
    vectors[rows[4]["path"]] = np.array([1.0, 0.1], np.float32)  # this one looks like kiting
    assert [w for w, _ in chapters.assign(rows, found, vectors)] == [None, None, None, None, 0, None]


def test_the_ai_s_chapters_are_cleaned():
    answer = {"chapters": [{"kind": "words", "what": "Kiting"}, {"kind": "person", "what": "bob"},
                           {"kind": "person", "what": "Nobody"}, {"kind": "finale", "what": ""},
                           {"kind": "people", "what": ""}]}
    assert _kinds(director.clean_chapters(answer, ["Bob"])) == [
        ("words", "kiting"), ("person", "Bob"), ("words", "nobody"), ("rest", "Everything else"),
        ("people", "People"), ("finale", "Finale: the most action")]
    assert "kiting first" in director.chapter_prompt("kiting   first", ["Bob"])


def test_the_video_tells_the_chapters_in_order_and_ends_on_action():
    song = SongMap(duration=60.0, bpm=120, period=0.5, first_beat=0.0, clear_beat=True, downbeat=0,
                   sections=[Section(0, 60, 0.5, "Medium")])
    found = [chapters.chapter("words", "food"), chapters.chapter("words", "kiting"), chapters.chapter("finale")]
    rows = []
    for n in range(40):  # taken in time order: kiting in the morning, food later
        kind = 1 if n < 20 else 0
        r = _shot(n)
        r.update(kind="video", duration=10.0, motion_action=(20.0 if n in (3, 7, 11) else 0.5), motion_camera=0.0,
                 motion_shake=0.1, chapter=kind, chapter_fit=2.0)
        rows.append(r)
    plan = autoedit.plan_to_music(rows, {}, song, autoedit.Limits(storyline="date"), None, 0.5, "Fade", found)
    shots = [int(s.path[3:5]) for s in plan.shots]
    assert abs(plan.length - 60.0) < 0.1
    food = [k for k, n in enumerate(shots) if n >= 20]
    kiting = [k for k, n in enumerate(shots) if n < 20 and n not in (3, 7, 11)]
    assert food and kiting and max(food) < min(kiting)  # food first, as asked, though it was filmed later
    assert shots[-1] in (3, 7, 11)  # the finale ends on one of the action clips
    # Both chapters get a fair share of the video.
    assert 0.25 < len(food) / len(shots) < 0.75
