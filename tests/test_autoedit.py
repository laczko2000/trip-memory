"""Auto-edit: picks shots up to a length, finds the best moment, and lands the cuts on the beat."""

import numpy as np

from trip_memory import autoedit, transitions
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
    limits = autoedit.Limits(max_video=4.0, max_photo=2.0, min_shot=1.0, storyline="date")
    plan = autoedit.plan_to_music(rows, {}, _song(), limits, None, 0.8, "Fade")
    assert abs(plan.length - 40.0) < 0.05
    by_path = {r["path"]: r for r in rows}
    begin, overlap = 0.0, 0.0
    lively_actions, calm_actions = [], []
    for k, shot in enumerate(plan.shots):
        start = begin
        assert shot.seconds <= 4.0 + 0.8 + 0.05
        lap = 0 if k == len(plan.shots) - 1 else transitions.seconds_of(shot.transition, 0.8)
        end_mid = begin + shot.seconds - lap / 2
        assert abs(end_mid / 0.5 - round(end_mid / 0.5)) * 0.5 < 0.02  # each cut on a beat
        (lively_actions if 16 <= start + 0.1 < 32 else calm_actions).append(by_path[shot.path]["motion_action"])
        begin += shot.seconds - lap
    assert np.mean(lively_actions) > 15 and np.mean(calm_actions) < 5
    times = [by_path[s.path]["taken_at"] for s in plan.shots]
    assert times == sorted(times)  # still in time order


def test_photos_respect_their_maximum():
    rows = [_clip(n, 0.0, kind="photo") for n in range(40)]
    limits = autoedit.Limits(max_video=4.0, max_photo=2.0, min_shot=1.0)
    plan = autoedit.plan_to_music(rows, {}, _song(), limits, 20.0, 0.8, "Fade")
    assert plan.shots and all(s.seconds <= 2.0 + 0.8 for s in plan.shots if not s.effects.get("collage"))
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


def test_every_tap_is_kept_even_with_fewer_shots_than_taps():
    song = _song()
    song.taps = [1.25, 2.25, 3.75, 4.75, 6.25, 7.25, 8.25, 10.25, 11.25, 12.75, 13.75, 15.25]
    rows = [_clip(n, action=1.0, duration=3.0 if n % 3 else 6.0) for n in range(10)]  # short clips, only ten
    limits = autoedit.Limits(max_video=4.0, max_photo=2.5, min_shot=1.0, cut_on="taps")
    plan = autoedit.plan_to_music(rows, {}, song, limits, 20.0, 0.8, "Fade")
    cuts, begin = [], 0.0  # where each change-over is centred
    for shot in plan.shots[:-1]:
        overlap = transitions.seconds_of(shot.transition, 0.8)
        cuts.append(begin + shot.seconds - overlap / 2)
        begin += shot.seconds - overlap
    assert len(plan.shots) == 10  # the shots run out...
    assert max(abs(c - t) for c, t in zip(cuts, song.taps)) <= 1 / 30  # ...but no tap was joined away or moved


def test_pins_are_the_only_cuts_however_long_or_short():
    song = _song()
    song.taps = [0.25, 0.5, 0.75, 1.0, 12.0, 12.25]  # a burst of quick cuts, a long 11 s shot, two more quick ones
    rows = [_clip(n, action=1.0, duration=8.0 if n == 4 else 3.0) for n in range(10)]
    limits = autoedit.Limits(max_video=2.0, max_photo=2.0, min_shot=1.0, cut_on="taps", storyline="date")
    plan = autoedit.plan_to_music(rows, {}, song, limits, 20.0, 0.8, "Fade")
    cuts, begin = [], 0.0
    for shot in plan.shots[:-1]:
        overlap = transitions.seconds_of(shot.transition, 0.8)
        cuts.append(begin + shot.seconds - overlap / 2)
        begin += shot.seconds - overlap
    assert plan.pins == 6
    assert max(abs(c - t) for c, t in zip(cuts, song.taps)) <= 1 / 30
    long_shot = plan.shots[4]  # 1.0 -> 12.0: the only clip long enough, slowed down to fill it
    assert long_shot.path.endswith("04.mov") and long_shot.speed < 1


def test_transitions_are_suggested_with_a_reason():
    rows = [_clip(n, action=20.0 if 8 <= n < 24 else 0.5, camera=1.5) for n in range(40)]
    for r in rows:
        r["story"] = r["story_rank"] = 0 if int(r["path"][-6:-4]) < 20 else 1
        r["story_name"] = "Deer" if r["story"] else "Running in the park"
    plan = autoedit.plan_to_music(rows, {}, _song(), autoedit.Limits(max_video=4.0, max_photo=2.0, min_shot=1.0),
                                  None, 0.8, "Fade")
    cuts = plan.shots[:-1]
    assert all(s.transition in transitions.STYLES and s.why for s in cuts)
    assert any(s.why == "Next story: Deer" for s in cuts)
    assert any(s.transition == transitions.HARD_CUT for s in cuts)  # mostly clean cuts on the beat
    assert len({s.transition for s in cuts}) >= 3  # but not one and the same everywhere
    assert plan.shots[-1].transition is None


def test_storyline_has_a_teaser_openers_and_a_calm_ending():
    rows = []
    for n in range(30):  # three stories of ten shots; one action shot in each, look-alikes in pairs
        r = _clip(n, action=25.0 if n % 10 == 6 else 2.0)
        r.update(story=n // 10, story_rank=n // 10, look_group=n // 2)
        rows.append(r)
    rows[12]["motion_action"], rows[12]["motion_camera"] = 0.2, 1.5  # a calm, cinematic view early in story 2
    rows[29]["motion_action"] = 0.1  # a still shot near the end
    shaped = autoedit.arrange(autoedit.choose(rows), "teaser")
    paths = [r["path"] for r in shaped]
    assert paths[:3] == ["/trip/06.mov", "/trip/16.mov", "/trip/26.mov"]  # the teaser: each story's action shot
    assert paths.index("/trip/12.mov") == paths.index("/trip/10.mov") - 1 or paths[3 + 9] == "/trip/12.mov"
    assert paths[-1] == "/trip/29.mov" and shaped[-1]["must"]
    for a, b in zip(shaped, shaped[1:]):
        assert a["look_group"] != b["look_group"] or a["story"] != b["story"]
    assert sorted(paths) == sorted(r["path"] for r in rows)  # every shot is still there
    assert [r["path"] for r in autoedit.arrange(autoedit.choose(rows), "date")] == sorted(paths)


def test_speed_effects():
    rows = [_clip(n, action=0.2, camera=1.5) for n in range(20)]  # calm, cinematic views
    rows += [_clip(20 + n, action=0.5, camera=0.0) for n in range(30)]  # calm shots, also used in lively music
    limits = autoedit.Limits(max_video=4.0, max_photo=2.0, min_shot=1.0, storyline="date", speed_effects=True)
    plan = autoedit.plan_to_music(rows, {}, _song(), limits, None, 0.8, "Fade")
    calm = [s for s in plan.shots if s.speed < 1]
    fast = [s for s in plan.shots if s.speed > 1]
    assert calm and all(s.speed_why for s in calm)
    assert fast and all(s.speed_why == "Sped up to match the lively music" for s in fast)
    off = autoedit.plan_to_music(rows, {}, _song(), autoedit.Limits(max_video=4.0, max_photo=2.0, min_shot=1.0,
                                                                     storyline="date", speed_effects=False),
                                 None, 0.8, "Fade")
    assert all(s.speed == 1.0 for s in off.shots)


def test_the_shaky_blurred_ends_of_a_clip_are_never_used():
    n = 60  # 12 s at five steps a second
    camera = [12.0] * 8 + [1.0] * 40 + [15.0] * 12  # raising the phone, a steady stretch, swinging it down
    sharp = [20.0] * 8 + [200.0] * 40 + [30.0] * 12
    action = [3.0] * n
    action[50:56] = [25.0] * 6  # the "action" is in the blurred, swinging end: must not win
    score = autoedit.interest(action, [0.2] * n, [], n, sharp, camera, [120.0] * n)
    assert (score[:8] == autoedit.BAD).all() and (score[48:] == autoedit.BAD).all()
    start = autoedit.best_start(score, 12.0, 3.0)
    assert 1.6 <= start and start + 3.0 <= 9.6
    assert autoedit.quality(score) >= 0  # its best stretch is a clean one


def test_camera_effects_suit_the_music():
    song = _song()
    beats = song.beat_times()
    song.accents = [0.9 if k % 8 == 3 else 0.2 for k in range(len(beats))]
    rows = [_clip(n, action=0.3, camera=0.0) for n in range(20)]
    rows += [_clip(20 + n, action=20.0, camera=0.5, duration=12.0) for n in range(30)]
    for r in rows[:20]:
        r["tags_final"] = ["Static camera", "Still"]
    limits = autoedit.Limits(max_video=4.0, max_photo=2.0, min_shot=1.0, storyline="date", cut_on="beats")
    plan = autoedit.plan_to_music(rows, {}, song, limits, None, 0.8, "Fade")
    kinds = {k for s in plan.shots for k in s.effects}
    assert "push" in kinds and kinds & {"punch", "shake"}  # (action clips shake on the hardest hits)
    assert any(s.transition in ("Zoom through", "Zoom out reveal") for s in plan.shots)
    for s in plan.shots:
        for t in s.effects.get("punch", []) + s.effects.get("shake", []):
            assert 0 < t < s.seconds
    plain = autoedit.plan_to_music(rows, {}, song, autoedit.Limits(max_video=4.0, max_photo=2.0, min_shot=1.0,
                                                                   storyline="date", camera_effects=False),
                                   None, 0.8, "Fade")
    assert not any(s.effects for s in plain.shots)
    assert not any(s.transition in ("Zoom through", "Zoom out reveal") for s in plain.shots)


def test_a_planned_montage_is_cut_quickly_in_the_lively_music():
    rows = [_clip(n, 0.0, kind="photo") for n in range(30)]
    roles = ["opening"] + ["build-up"] * 7 + ["montage"] * 8 + ["build-up"] * 13 + ["ending"]
    for n, (r, role) in enumerate(zip(rows, roles)):
        r.update(plan_role=role, plan_pos=n, similar_set=n)
    limits = autoedit.Limits(max_video=4.0, max_photo=4.0, min_shot=1.0, storyline="stories")
    plan = autoedit.plan_to_music(rows, {}, _song(), limits, None, 0.8, "Fade")
    begin, montage = 0.0, []
    by_path = {r["path"]: r for r in rows}
    for k, shot in enumerate(plan.shots):
        lap = 0 if k == len(plan.shots) - 1 else transitions.seconds_of(shot.transition, 0.8)
        if by_path[shot.path]["plan_role"] == "montage":
            montage.append((begin, shot.seconds))
        begin += shot.seconds - lap
    assert len(montage) >= 6
    assert all(16 <= start + 0.1 < 32 for start, _ in montage)  # in the lively part
    assert np.mean([s for _, s in montage]) < 2.5  # quick
    assert by_path[plan.shots[0].path]["plan_role"] == "opening"


def test_freeze_frames_introduce_each_named_person_once():
    slot = autoedit.Slot(0.0, 3.0, 0.5)
    introduced: set[str] = set()
    anna = {"kind": "video", "people": ["Anna", "Person 4"], "people_tag": "Solo"}
    assert autoedit.freeze_for(anna, slot, 3.0, introduced, 99) == (
        {"freeze": autoedit.FREEZE, "label": "Anna"}, "Freeze-frame introducing Anna")
    assert autoedit.freeze_for(anna, slot, 3.0, introduced, 99) == ({}, None)  # already introduced
    ben = {"kind": "video", "people": ["Ben"], "people_tag": "Solo"}
    assert autoedit.freeze_for(ben, slot, 3.0, introduced, 5) == ({}, None)  # too soon after the last one
    assert autoedit.freeze_for(ben, autoedit.Slot(0.0, 3.0, 0.9), 3.0, introduced, 99) == ({}, None)  # busy music
    assert autoedit.freeze_for(ben, slot, 3.0, introduced, 99)[0]["label"] == "Ben"


def test_the_hardest_hit_shakes_an_action_shot():
    slot = autoedit.Slot(10.0, 13.0, 0.8)
    row = {"kind": "video", "tags_final": ["Fast action"], "duration": 8}
    effects, why = autoedit.effects_for(row, slot, 3.0, 1.0, [(11.5, 0.95)], 5)
    assert effects == {"shake": [1.5]} and "shake" in why
    effects, _ = autoedit.effects_for(row, slot, 3.0, 1.0, [(11.5, 0.8)], 5)
    assert effects == {"punch": [1.5]}


def test_a_quick_run_of_photos_becomes_a_collage():
    slots = [autoedit.Slot(k * 0.5, k * 0.5 + 0.5, 0.8) for k in range(6)]
    shots = [autoedit.Shot(path=f"p{k}.jpg", kind="photo", seconds=0.5 + 1 / 30, transition=transitions.HARD_CUT)
             for k in range(6)]
    rows = {s.path: {"story": 1} for s in shots}
    rows["p5.jpg"] = {"story": 2}
    merged = autoedit.collages(shots, slots, rows, 0.8)
    assert [s.path for s in merged] == ["p0.jpg", "p4.jpg", "p5.jpg"]  # p0..p3 together, then the rest
    lead = merged[0]
    assert lead.effects["collage"] == ["p1.jpg", "p2.jpg", "p3.jpg"]
    assert lead.effects["appear"] == [0.0, 0.5, 1.0, 1.5]  # each pops in where its cut was
    assert abs(lead.seconds - (2.0 + 1 / 30)) < 1e-3  # the same time on screen as the four shots
    calm = [autoedit.Slot(s.start, s.end, 0.3) for s in slots]
    assert len(autoedit.collages(shots, calm, rows, 0.8)) == 6  # not in calm music


def test_a_shorter_video_ends_on_your_pin_nearest_the_length():
    song = _song()
    song.taps = [2.0, 4.0, 6.0, 8.0, 10.5, 13.0, 15.5, 19.0]
    rows = [_clip(n, action=1.0, duration=8.0) for n in range(12)]
    limits = autoedit.Limits(max_video=4.0, max_photo=2.5, min_shot=1.0, cut_on="taps")
    plan = autoedit.plan_to_music(rows, {}, song, limits, 14.0, 0.8, "Fade")
    assert abs(plan.length - 13.0) < 0.05  # the pin at 13 s, not the bar line after 14 s


def test_too_few_shots_thin_your_pins_but_keep_their_pattern():
    from trip_memory.songmap import Section, SongMap

    song = SongMap(duration=60.0, bpm=120, period=0.5, first_beat=0.0, clear_beat=True, downbeat=0,
                   accents=[0.2] * 120, sections=[Section(0, 30, 0.3, "Calm"), Section(30, 60, 0.8, "Lively")])
    pins = [2.0 * k for k in range(1, 29)]  # a pin every 2 s, one of them where the music changes (30 s)
    kept = autoedit.fit_pins(song, pins, 15, longest=4.0)
    assert len(kept) == 15 and kept[0] == 2.0 and kept[-1] == 56.0 and 30.0 in kept
    assert kept[:4] == [2.0, 6.0, 10.0, 14.0] and max(np.diff(kept)) <= 4.0 + 1e-9  # every other pin, no shot too long
    assert autoedit.fit_pins(song, pins, 5, longest=4.0) == kept  # can't go further without too long shots


def test_a_long_song_with_many_pins_is_filled_to_the_end_with_the_shots_you_have():
    from trip_memory.songmap import Section, SongMap

    rng = np.random.default_rng(1)
    song = SongMap(duration=180.0, bpm=120, period=0.5, first_beat=0.0, clear_beat=True, downbeat=0,
                   accents=list(rng.random(360)), sections=[Section(0, 90, 0.3, "Calm"), Section(90, 180, 0.8, "Lively")])
    song.taps = [1.2 * k for k in range(1, 150)]  # 149 pins
    rows = [_clip(n, action=rng.random(), duration=float(rng.uniform(1.0, 3.0))) for n in range(60)]  # short clips
    rows += [_row(100 + n, kind="photo") for n in range(10)]
    for r in rows:
        r.setdefault("taken_at", "2025-07-01T10:00:00")
    plan = autoedit.plan_to_music(rows, {}, song, autoedit.Limits(cut_on="taps", storyline="date"), None, 0.8, "Fade")
    assert plan.length >= 175 and len(plan.shots) >= 50 and plan.kept_pins and len(plan.kept_pins) < 149


def test_a_locked_moment_plays_whole_and_the_rest_of_the_clip_stays_free():
    rows, parts = autoedit.lock_parts(
        [{"path": "/k.mov", "kind": "video", "duration": 30.0, "taken_at": "2025-07-03T10:00:00",
          "locks": [(12.0, 15.0)]},
         {"path": "/p.jpg", "kind": "photo", "duration": None, "taken_at": "2025-07-03T10:01:00"}], {})
    by = {r["path"]: r for r in rows}
    assert by["/k.mov"]["locked"] and by["/k.mov"]["forced"] and by["/k.mov"]["duration"] == 3.0
    assert parts["/k.mov"] == ("/k.mov", 12.0)
    free = sorted((parts[p][1], parts[p][1] + by[p]["duration"]) for p in by if p.startswith("/k.mov#F"))
    assert free == [(0.0, 11.5), (15.5, 30.0)]  # never overlapping the moment
    assert all(by[p]["part_of"] == "/k.mov" for p in by if "#" in p)
    assert "/p.jpg" in by and "/p.jpg" not in parts


def test_every_bookmark_is_inside_the_part_used():
    rows, parts = autoedit.bookmark_parts(
        [{"path": "/k.mov", "kind": "video", "duration": 60.0, "taken_at": "2025-07-03T10:00:00",
          "bookmarks": [10.0, 12.0, 40.0]},
         {"path": "/s.mov", "kind": "video", "duration": 8.0, "taken_at": "2025-07-03T10:02:00", "bookmarks": [6.5]}],
        {})
    by = {r["path"]: r for r in rows}
    # bookmarks far apart: a part of the clip each, split halfway between them; close ones stay together
    assert parts["/k.mov"] == ("/k.mov", 0.0) and parts["/k.mov#B2"] == ("/k.mov", 26.0)
    assert by["/k.mov"]["bookmarks"] == [10.0, 12.0] and by["/k.mov#B2"]["bookmarks"] == [14.0]
    assert by["/k.mov"]["forced"] and by["/k.mov#B2"]["forced"] and "/s.mov" not in parts
    # whatever length the music gives the shot, the bookmark is in it
    for used in (1.0, 2.5, 6.0):
        start = autoedit.cover_bookmark(0.0, used, 8.0, [6.5])
        assert start <= 6.5 <= start + used and 0 <= start <= 8.0 - used
    assert autoedit.cover_bookmark(5.0, 3.0, 8.0, [6.5]) == 5.0  # already inside: left alone
