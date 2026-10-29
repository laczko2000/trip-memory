"""Auto-edit: picks shots up to a length, finds the best moment, and lands the cuts on the beat."""

import numpy as np

from trip_memory import autoedit
from trip_memory.songmap import Section, SongMap


def _row(n, kind="video", duration=12.0, story=0, similar=None, best=False, tags=()):
    return {
        "path": f"/trip/{n:02d}.mov", "kind": kind, "duration": duration if kind == "video" else 0,
        "taken_at": f"2025-07-03T10:{n:02d}:00+00:00", "story": story, "similar_set": similar if similar is not None else n,
        "best": best, "tags_final": list(tags), "people_tag": None, "people": [], "sharpness": 100.0,
    }


def test_best_moment_is_found():
    score = np.zeros(60)  # 12 s at five steps a second
    score[35:45] = 1.0  # the action is at 7-9 s
    start = autoedit.best_start(score, 12.0, 3.0)
    assert 6.0 <= start <= 7.0


def test_everything_you_ticked_is_used_in_time_order():
    rows = [_row(2, similar=7), _row(0, similar=7, best=True), _row(1, similar=7)]
    assert [r["path"] for r in autoedit.choose(rows)] == ["/trip/00.mov", "/trip/01.mov", "/trip/02.mov"]

def _song():
    # 120 BPM: calm for 16 s, lively for 16 s, calm again for 8 s.
    return SongMap(
        duration=40.0, bpm=120, period=0.5, first_beat=0.0, clear_beat=True, downbeat=0,
        sections=[Section(0, 16, 0.1, "Calm"), Section(16, 32, 0.9, "Lively"), Section(32, 40, 0.1, "Calm")],
    )


def _clip(n, action, duration=10.0, camera=0.0, kind="video"):
    r = _row(n, kind=kind, duration=duration)
    r.update(motion_action=action, motion_camera=camera, motion_shake=0.1)
    return r


def test_music_slots_follow_the_music_and_your_limits():
    limits = autoedit.Limits(max_video=4.0, max_photo=2.0, min_shot=1.0)
    slots = autoedit.music_slots(_song(), limits, end=40.0)
    assert all(s.seconds <= 4.0 + 1e-6 and s.seconds >= 1.0 - 1e-6 for s in slots)
    calm = [s.seconds for s in slots if s.end <= 16]
    lively = [s.seconds for s in slots if 16 <= s.start < 32]
    assert min(calm) >= 4.0 - 1e-6 and max(lively) <= 1.0 + 1e-6  # long shots when calm, quick when lively
    assert any(abs(s.start - 16) < 1e-6 for s in slots) and any(abs(s.start - 32) < 1e-6 for s in slots)
    for s in slots:
        assert abs(s.start / 0.5 - round(s.start / 0.5)) < 1e-6  # every cut on a beat


def test_action_goes_to_lively_music_and_cuts_stay_on_the_beat():
    rows = []
    for n in range(60):  # alternating calm and action clips through the day
        rows.append(_clip(n, action=20.0 if n % 2 else 0.5, camera=1.5 if n % 2 == 0 else 0.0))
    limits = autoedit.Limits(max_video=4.0, max_photo=2.0, min_shot=1.0)
    plan = autoedit.plan_to_music(rows, {}, _song(), limits, None, 0.8, "Fade")
    assert abs(plan.length - 40.0) < 0.05
    by_path = {r["path"]: r for r in rows}
    begin, overlap = 0.0, 0.0
    lively_actions, calm_actions = [], []
    for k, shot in enumerate(plan.shots):
        start = begin
        assert shot.seconds <= 4.0 + 0.8 + 0.05
        end_mid = begin + shot.seconds - (0 if k == len(plan.shots) - 1 else (0.8 if shot.transition == "Fade" else 1 / 30)) / 2
        assert abs(end_mid / 0.5 - round(end_mid / 0.5)) * 0.5 < 0.02  # each cut on a beat
        (lively_actions if 16 <= start + 0.1 < 32 else calm_actions).append(by_path[shot.path]["motion_action"])
        begin += shot.seconds - (0.8 if shot.transition == "Fade" else 1 / 30)
    assert np.mean(lively_actions) > 15 and np.mean(calm_actions) < 5
    times = [by_path[s.path]["taken_at"] for s in plan.shots]
    assert times == sorted(times)  # still in time order


def test_photos_respect_their_maximum():
    rows = [_clip(n, 0.0, kind="photo") for n in range(40)]
    limits = autoedit.Limits(max_video=4.0, max_photo=2.0, min_shot=1.0)
    plan = autoedit.plan_to_music(rows, {}, _song(), limits, 20.0, 0.8, "Fade")
    assert plan.shots and all(s.seconds <= 2.0 + 0.8 for s in plan.shots)
    assert 18 <= plan.length <= 22


def test_a_trip_of_photos_fills_a_minute():
    # Lots of photos (and a few videos) taken after the videos: the photos must be used, not just the videos.
    rows = [_clip(n, 5.0) for n in range(4)] + [_clip(10 + n, 0.0, kind="photo") for n in range(40)]
    for n, r in enumerate(rows):
        r["similar_set"] = 1  # even if they all look alike
    limits = autoedit.Limits(max_video=4.0, max_photo=2.5, min_shot=1.0)
    long_song = SongMap(duration=90.0, bpm=120, period=0.5, first_beat=0.0, clear_beat=True, downbeat=0,
                        sections=[Section(0, 32, 0.1, "Calm"), Section(32, 90, 0.9, "Lively")])
    plan = autoedit.plan_to_music(rows, {}, long_song, limits, 60.0, 0.8, "Fade")
    assert 58 <= plan.length <= 62
    assert sum(s.kind == "photo" for s in plan.shots) >= 25


def test_few_shots_stretch_further():
    rows = [_clip(n, 5.0) for n in range(6)]
    limits = autoedit.Limits(max_video=4.0, max_photo=2.5, min_shot=1.0)
    plan = autoedit.plan_to_music(rows, {}, _song(), limits, None, 0.8, "Fade")
    assert len(plan.shots) == 6 and plan.length > 6 * 2.0  # longer pieces, not 6 quick ones


def test_cuts_go_on_the_strong_beats():
    song = _song()
    beats = song.beat_times()
    # Big hits on beats 3 and 11 of every 16 (off the bar lines), the rest are soft.
    song.accents = [0.9 if k % 16 in (3, 11) else 0.1 for k in range(len(beats))]
    song.bass = list(song.accents)
    limits = autoedit.Limits(max_video=6.0, max_photo=6.0, min_shot=1.0, cut_on="strong")
    slots = autoedit.music_slots(song, limits, end=16.0)
    inner = [s.start for s in slots[1:]]
    assert inner and all(round(t / 0.5) % 16 in (3, 11) for t in inner)
    every = autoedit.music_slots(song, autoedit.Limits(max_video=6.0, max_photo=6.0, min_shot=1.0, cut_on="beats"), end=16.0)
    assert all(round(s.start / 0.5) % 8 == 0 for s in every)  # the steady count: every bar pair
    few = autoedit.music_slots(song, autoedit.Limits(max_video=6.0, max_photo=6.0, min_shot=1.0, cut_on="changes"), end=16.0)
    assert len(few) <= len(slots)


def test_cuts_follow_your_taps_then_the_music():
    song = _song()
    song.taps = [1.5, 2.75, 5.0, 9.0]
    limits = autoedit.Limits(max_video=6.0, max_photo=6.0, min_shot=1.0, cut_on="taps")
    slots = autoedit.music_slots(song, limits, end=40.0)
    assert [s.start for s in slots[:5]] == [0.0, 1.5, 2.75, 5.0, 9.0]
    assert all(s.seconds <= 6.0 * 1.03 for s in slots)
    assert abs(slots[-1].end - 40.0) < 1e-6
    assert any(abs(s.start - 16) < 1e-6 for s in slots)  # after your last tap, the music's own changes again
