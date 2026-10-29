"""The trip catalogue: everything the app has learned about a trip, saved so it is only worked out once.

Each trip folder gets its own small database (SQLite) plus thumbnail pictures, stored in
~/Library/Application Support/TripMemory, never inside your photo folders.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import subprocess
import sys
import tempfile
import threading
import traceback
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from .faces import FaceFinder, group_faces
from .media import MediaItem, scan_folder
from . import autoedit, motion, stories
from .scenes import SceneTagger, mean_vector
from .tags import TAG_GROUP, Tagger, group_name, look_alike_groups, people_tag, story_subject
from .tools import ffmpeg_paths, probe


def app_dir() -> Path:
    if sys.platform == "darwin":
        base = Path.home() / "Library" / "Application Support"
    else:
        base = Path.home() / ".local" / "share"
    path = base / "TripMemory"
    path.mkdir(parents=True, exist_ok=True)
    return path


SCHEMA = """
CREATE TABLE IF NOT EXISTS items (
    path TEXT PRIMARY KEY,
    kind TEXT NOT NULL,
    taken_at TEXT NOT NULL,
    duration REAL DEFAULT 0,
    has_audio INTEGER DEFAULT 0,
    hdr INTEGER DEFAULT 0,
    mtime REAL,
    lat REAL,
    lon REAL,
    place TEXT,
    country TEXT,
    thumb TEXT,
    scene TEXT,
    scene_top TEXT,
    scene_user TEXT,
    clip BLOB,
    included INTEGER DEFAULT 1,
    chosen INTEGER DEFAULT 1,
    position REAL,
    trim_start REAL,
    trim_end REAL,
    transition TEXT,
    dhash TEXT,
    colours BLOB,
    sharpness REAL,
    story INTEGER,
    similar_set INTEGER,
    best INTEGER DEFAULT 0,
    rotation INTEGER DEFAULT 0,
    rotation_checked INTEGER DEFAULT 0,
    tags TEXT,
    tags_user TEXT,
    motion_action REAL,
    motion_camera REAL,
    motion_shake REAL,
    look_group INTEGER,
    seconds REAL,
    curves TEXT,
    clip_sound INTEGER DEFAULT 0,
    clip_volume REAL DEFAULT 1.0,
    cut_out INTEGER DEFAULT 0,
    subjects TEXT,
    faces_done INTEGER DEFAULT 0,
    scene_done INTEGER DEFAULT 0,
    tags_done INTEGER DEFAULT 0,
    transition_why TEXT,
    camera_moves TEXT,
    speed REAL DEFAULT 1.0,
    speed_why TEXT,
    fps REAL
);
CREATE TABLE IF NOT EXISTS story_info (
    key TEXT PRIMARY KEY,  -- the story's first file
    name TEXT,
    position REAL,
    enabled INTEGER DEFAULT 1
);
CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value TEXT
);
CREATE TABLE IF NOT EXISTS people (
    id INTEGER PRIMARY KEY,
    name TEXT,
    hidden INTEGER DEFAULT 0  -- marked "Unknown" by you
);
CREATE TABLE IF NOT EXISTS faces (
    id INTEGER PRIMARY KEY,
    path TEXT NOT NULL,
    score REAL,
    crop TEXT,
    embedding BLOB,
    person_id INTEGER REFERENCES people(id)
);
CREATE INDEX IF NOT EXISTS faces_path ON faces(path);
CREATE INDEX IF NOT EXISTS faces_person ON faces(person_id);
"""


# A photo without GPS borrows the place of the nearest photo taken within this many hours,
# and at least the country of one taken the same day.
PLACE_GUESS_HOURS = 3

# Unnamed people seen in fewer photos/videos than this are listed separately for review (often strangers).
MIN_APPEARANCES = 3


@dataclass
class Progress:
    fraction: float = 0.0
    message: str = ""
    running: bool = False
    notes: list[str] = field(default_factory=list)


class Trip:
    """One trip folder and its catalogue."""

    def __init__(self, folder: str | Path):
        self.folder = Path(folder).expanduser().resolve()
        key = hashlib.sha1(str(self.folder).encode()).hexdigest()[:12]
        self.dir = app_dir() / "trips" / key
        (self.dir / "thumbs").mkdir(parents=True, exist_ok=True)
        (self.dir / "faces").mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self.db = sqlite3.connect(self.dir / "catalog.db", check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.db.executescript(SCHEMA)
        for column in (
            "chosen INTEGER DEFAULT 1", "position REAL", "trim_start REAL", "trim_end REAL", "transition TEXT",
            "dhash TEXT", "colours BLOB", "sharpness REAL", "story INTEGER", "similar_set INTEGER", "best INTEGER DEFAULT 0",
            "rotation INTEGER DEFAULT 0", "rotation_checked INTEGER DEFAULT 0",
            "tags TEXT", "tags_user TEXT", "motion_action REAL", "motion_camera REAL", "motion_shake REAL",
            "look_group INTEGER", "tags_done INTEGER DEFAULT 0", "seconds REAL", "curves TEXT",
            "clip_sound INTEGER DEFAULT 0", "clip_volume REAL DEFAULT 1.0", "cut_out INTEGER DEFAULT 0", "subjects TEXT",
            "transition_why TEXT", "camera_moves TEXT", "speed REAL DEFAULT 1.0", "speed_why TEXT", "fps REAL",
        ):
            try:  # catalogues made by older versions lack these columns
                self.db.execute(f"ALTER TABLE items ADD COLUMN {column}")
            except sqlite3.OperationalError:
                pass
        (self.dir / "folder.txt").write_text(str(self.folder))
        self.progress = Progress()

    # ---------- reading ----------

    def query(self, sql: str, args: tuple = ()) -> list[sqlite3.Row]:
        with self._lock:
            return self.db.execute(sql, args).fetchall()

    def execute(self, sql: str, args: tuple = ()) -> None:
        with self._lock:
            self.db.execute(sql, args)
            self.db.commit()

    def items(self) -> list[dict]:
        """All items in date order, with the names of the (known) people in them."""
        rows = [dict(r) for r in self.query("SELECT * FROM items ORDER BY taken_at, path")]
        people = self._people_status()
        by_path: dict[str, set[int]] = {}
        for f in self.query("SELECT path, person_id FROM faces WHERE person_id IS NOT NULL"):
            if people.get(f["person_id"], {}).get("status") not in (None, "unknown"):
                by_path.setdefault(f["path"], set()).add(f["person_id"])
        face_counts = {r["path"]: r["n"] for r in self.query("SELECT path, COUNT(*) AS n FROM faces GROUP BY path")}
        for row in rows:
            ids = sorted(by_path.get(row["path"], set()))
            row["person_ids"] = ids
            row["people"] = [people[i]["label"] for i in ids]
            row["scene_final"] = row["scene_user"] or row["scene"]
            row["tags_auto"] = json.loads(row["tags"] or "[]")
            row["tags_final"] = json.loads(row["tags_user"]) if row["tags_user"] is not None else row["tags_auto"]
            row["people_tag"] = people_tag(face_counts.get(row["path"], 0)) if row["faces_done"] else None
            row["day"] = row["taken_at"][:10]
            row["place_guessed"] = False
            row.pop("clip", None)
            row.pop("colours", None)
        _guess_places(rows)
        groups: dict[int, list[dict]] = {}
        for row in rows:
            if row["look_group"] is not None:
                groups.setdefault(row["look_group"], []).append(row)
        for number, members in groups.items():
            name = f"{group_name([m['tags_final'] for m in members])} ({len(members)}) · group {number}"
            for m in members:
                m["look_name"] = name
        for row in rows:
            row.setdefault("look_name", None)
        return rows

    def _people_status(self) -> dict[int, dict]:
        """For every person: label, how many photos/videos they are in, and their status.

        Status is "named" (you named them), "new" (not named yet), "rare" (not named, seen only once
        or twice: often a stranger, sometimes someone the app did not recognise) or "unknown" (you marked them).
        """
        counts = {
            r["person_id"]: r["n"]
            for r in self.query(
                "SELECT f.person_id, COUNT(DISTINCT f.path) AS n FROM faces f "
                "JOIN items i ON i.path = f.path WHERE i.chosen = 1 GROUP BY f.person_id"
            )
        }
        result = {}
        for p in self.query("SELECT * FROM people"):
            count = counts.get(p["id"], 0)
            if p["hidden"]:
                status = "unknown"
            elif p["name"]:
                status = "named"
            elif count < MIN_APPEARANCES:
                status = "rare"
            else:
                status = "new"
            result[p["id"]] = {"label": p["name"] or f"Person {p['id']}", "count": count, "status": status}
        return result

    def person_names(self) -> list[str]:
        """Names you have already given, for the drop-down."""
        return sorted({r["name"] for r in self.query("SELECT name FROM people WHERE name IS NOT NULL AND hidden = 0")})

    def people(self) -> list[dict]:
        """People with their status, how often they appear and a few example face pictures."""
        result = []
        for pid, info in self._people_status().items():
            if not info["count"]:
                continue
            faces = self.query(
                "SELECT f.crop FROM faces f JOIN items i ON i.path = f.path "
                "WHERE f.person_id = ? AND i.chosen = 1 ORDER BY f.score DESC LIMIT 6",
                (pid,),
            )
            name = info["label"] if info["status"] == "named" else ""
            result.append({"id": pid, "name": name, **info, "faces": [f["crop"] for f in faces]})
        order = {"named": 0, "new": 1, "rare": 2, "unknown": 3}
        result.sort(key=lambda p: (order[p["status"]], -p["count"], p["label"]))
        return result

    def storyboard(self) -> list[dict]:
        """The files going into the video, in storyboard order.

        Files you moved keep their place; anything not placed yet follows story by story (your story order).
        """
        items = self.items()
        stories = self.stories(items)
        off = {r["path"] for s in stories if not s["enabled"] for r in s["items"]}
        rank = {r["path"]: n for n, s in enumerate(stories) for r in s["items"]}
        rows = [it for it in items if it["chosen"] and it["included"] and not it["cut_out"] and it["path"] not in off]
        placed = sorted((r for r in rows if r["position"] is not None), key=lambda r: r["position"])
        rest = sorted((r for r in rows if r["position"] is None), key=lambda r: (rank.get(r["path"], 0), r["taken_at"]))
        # Anything not placed yet goes in by story and time: after the last placed file that comes before it.
        for r in rest:
            key = (rank.get(r["path"], 0), r["taken_at"])
            at = 0
            for i, p in enumerate(placed):
                if (rank.get(p["path"], 0), p["taken_at"]) <= key:
                    at = i + 1
            placed.insert(at, r)
        return placed

    def media_items(self, chapters: bool = False) -> list[MediaItem]:
        """What the video is made from, in storyboard order, optionally with a title card for each day."""
        board = self.storyboard()
        days = sorted({r["day"] for r in self.items() if r["chosen"]})
        result: list[MediaItem] = []
        previous_day = None
        for r in board:
            if chapters and r["day"] != previous_day:
                result.append(self._title_card(r, board, days))
            previous_day = r["day"]
            result.append(
                MediaItem(
                    path=Path(r["path"]),
                    kind=r["kind"],
                    taken_at=datetime.fromisoformat(r["taken_at"]),
                    duration=r["duration"],
                    has_audio=bool(r["has_audio"]),
                    hdr=bool(r["hdr"]),
                    rotation=r["rotation"] or 0,
                    trim_start=r["trim_start"],
                    trim_end=r["trim_end"],
                    transition=r["transition"],
                    seconds=r["seconds"],
                    sound=bool(r["clip_sound"]),  # videos are silent unless you switch their sound on
                    volume=1.0 if r["clip_volume"] is None else r["clip_volume"],
                    speed=r["speed"] or 1.0,
                )
            )
        return result

    def _title_card(self, first: dict, board: list[dict], days: list[str]) -> MediaItem:
        """A chapter card such as "Day 2" / "Lisbon · Sat 5 Jul" over a blurred picture of that day."""
        same_day = [r for r in board if r["day"] == first["day"]]
        places = [r["place"] or r["country"] for r in same_day if r["place"] or r["country"]]
        place = max(set(places), key=places.count) if places else None
        when = datetime.fromisoformat(first["day"]).strftime("%a %-d %b")
        number = days.index(first["day"]) + 1 if first["day"] in days else 1
        subtitle = f"{place} · {when}" if place else when
        return MediaItem(
            path=Path(first["thumb"] or first["path"]),
            kind="title",
            taken_at=datetime.fromisoformat(first["taken_at"]),
            title=f"Day {number}\n{subtitle}",
        )

    # ---------- changes made by you ----------

    def set_included(self, paths: list[str], included: bool) -> None:
        with self._lock:
            # Your own choice also overrides an earlier Auto-edit leaving the file out.
            # Put back in: it slots into the storyboard by time (see storyboard()).
            self.db.executemany("UPDATE items SET included = ?, cut_out = 0, position = NULL WHERE path = ?",
                                [(int(included), p) for p in paths])
            self.db.commit()

    def set_chosen(self, paths: list[str], chosen: bool) -> None:
        """Choose which files of the folder belong to this trip (only those are analysed and used)."""
        with self._lock:
            self.db.executemany("UPDATE items SET chosen = ? WHERE path = ?", [(int(chosen), p) for p in paths])
            self.db.commit()

    def move(self, path: str, steps: int) -> None:
        """Move a file earlier (negative steps) or later in the storyboard."""
        order = [r["path"] for r in self.storyboard()]
        if path not in order:
            return
        i = order.index(path)
        j = max(0, min(len(order) - 1, i + steps))
        order.insert(j, order.pop(i))
        with self._lock:
            self.db.executemany("UPDATE items SET position = ? WHERE path = ?", list(enumerate(order)))
            self.db.commit()

    def set_tags(self, path: str, tags: list[str] | None) -> None:
        """Your own tags for a file (None goes back to the app's suggestion)."""
        self.execute("UPDATE items SET tags_user = ? WHERE path = ?", (None if tags is None else json.dumps(tags), path))

    def rotate(self, path: str, degrees: int = 90) -> None:
        """Turn a photo or video clockwise. Its people and scene are worked out again at the next analysis."""
        row = self.query("SELECT rotation FROM items WHERE path = ?", (path,))
        if not row:
            return
        self._set_rotation(path, ((row[0]["rotation"] or 0) + degrees) % 360)
        self._group_stories()

    def _set_rotation(self, path: str, rotation: int) -> None:
        with self._lock:
            self.db.execute("DELETE FROM faces WHERE path = ?", (path,))
            self.db.execute(
                "UPDATE items SET rotation = ?, rotation_checked = 1, faces_done = 0, scene_done = 0, tags_done = 0, dhash = NULL "
                "WHERE path = ?",
                (rotation, path),
            )
            self.db.commit()
        row = self.query("SELECT path, kind, duration, rotation FROM items WHERE path = ?", (path,))[0]
        self.execute("UPDATE items SET thumb = ? WHERE path = ?", (str(self._make_thumb(row)), path))

    def reset_order(self) -> None:
        self.execute("UPDATE items SET position = NULL")

    def auto_edit_pool(self) -> list[dict]:
        """What Auto-edit picks from: your selection (ticked in Choose files and "In the video" in the Library),
        story by story in your story order, leaving out stories you switched off."""
        items = self.items()
        pool = []
        for rank, story in enumerate(self.stories(items)):
            if story["enabled"]:
                for r in story["items"]:
                    if r["chosen"] and r["included"]:
                        pool.append(r | {"story_rank": rank, "story_name": story["name"]})
        return pool

    def auto_edit_to_music(
        self,
        song_path: str,
        limits: autoedit.Limits,
        target: float | None,
        transition_seconds: float,
        soft_transition: str,
        on_progress: Callable[[float, str], None] | None = None,
    ) -> autoedit.Plan:
        """The music-driven first cut: the song's beats, bars and energy decide the cuts; shots are matched to it."""
        from . import songmap, taps

        report = on_progress or (lambda _f, _m: None)
        rows = self.auto_edit_pool()
        self._frame_rates(rows)
        scores = self._video_scores(rows, report)
        report(0.95, "Listening to the music")
        song = songmap.analyse(song_path, app_dir() / "music-maps")
        song.taps = taps.final_taps(song_path, app_dir() / "music-taps")  # your own cuts, if you tapped this song
        result = autoedit.plan_to_music(rows, scores, song, limits, target, transition_seconds, soft_transition)
        self._apply_plan(result)
        report(1.0, "Done")
        return result

    def set_speed(self, path: str, speed: float) -> None:
        """Play a video slower or faster; the same part of it is used, so it lasts longer or shorter."""
        self.execute("UPDATE items SET speed = ?, speed_why = NULL WHERE path = ?", (speed, path))

    def _frame_rates(self, rows: list[dict]) -> None:
        """How many pictures a second each video has (high = smooth slow motion), looked up once."""
        for r in rows:
            if r["kind"] == "video" and r.get("fps") is None:
                try:
                    stream = next(st for st in probe(r["path"])["streams"] if st.get("codec_type") == "video")
                    num, _, den = (stream.get("avg_frame_rate") or "30/1").partition("/")
                    fps = float(num) / float(den or 1) if float(den or 1) else 30.0
                except Exception:
                    fps = 30.0
                self.execute("UPDATE items SET fps = ? WHERE path = ?", (fps, r["path"]))
                r["fps"] = fps

    def _video_scores(self, rows: list[dict], report) -> dict:
        videos = [r for r in rows if r["kind"] == "video"]
        for n, r in enumerate(v for v in videos if not v["curves"]):
            report(n / max(1, len(videos)), f"Finding the best moments: {Path(r['path']).name}")
            moved = motion.measure(r["path"])
            if moved:
                curves = _curves_json(moved, Path(r["path"]), r["has_audio"])
                self.execute(
                    "UPDATE items SET curves = ?, motion_action = COALESCE(motion_action, ?), "
                    "motion_camera = COALESCE(motion_camera, ?), motion_shake = COALESCE(motion_shake, ?) WHERE path = ?",
                    (curves, moved.action, moved.camera, moved.shake, r["path"]),
                )
                r.update(curves=curves, motion_action=r["motion_action"] if r["motion_action"] is not None else moved.action,
                         motion_camera=r["motion_camera"] if r["motion_camera"] is not None else moved.camera,
                         motion_shake=r["motion_shake"] if r["motion_shake"] is not None else moved.shake)
        scores = {}
        for r in videos:
            if r["curves"]:
                c = json.loads(r["curves"])
                n = max(len(c["action"]) + 1, int((r["duration"] or 0) * motion.FPS))
                scores[r["path"]] = autoedit.interest(c["action"], c["shake"], c["loud"], n)
        return scores

    def _apply_plan(self, result: autoedit.Plan) -> None:
        with self._lock:
            for shot_no, shot in enumerate(result.shots):
                trim = (shot.start, shot.start + shot.seconds * shot.speed) if shot.kind == "video" else (None, None)
                self.db.execute(
                    "UPDATE items SET cut_out = 0, position = ?, trim_start = ?, trim_end = ?, transition = ?, "
                    "transition_why = ?, seconds = ?, speed = ?, speed_why = ? WHERE path = ?",
                    (shot_no, *trim, shot.transition, shot.why, shot.seconds if shot.kind == "photo" else None,
                     shot.speed, shot.speed_why, shot.path),
                )
            # Left out of this cut, but still in your selection, so the next Auto-edit can use it again.
            self.db.executemany("UPDATE items SET cut_out = 1, position = NULL WHERE path = ?", [(p,) for p in result.left_out])
            self.db.commit()

    def auto_edit_steady(
        self,
        limits: autoedit.Limits,
        target: float | None,
        transition_seconds: float,
        soft_transition: str,
        on_progress: Callable[[float, str], None] | None = None,
    ) -> autoedit.Plan:
        """A first cut without music: a steady pace of about two seconds a shot, within your limits."""
        from .songmap import Section, SongMap

        report = on_progress or (lambda _f, _m: None)
        rows = self.auto_edit_pool()
        self._frame_rates(rows)
        scores = self._video_scores(rows, report)
        length = target or 3600.0  # "use everything": it stops when the shots run out
        steady = SongMap(duration=length, bpm=120, period=0.5, first_beat=0.0, clear_beat=False, downbeat=0,
                         sections=[Section(0.0, length, 0.5, "Medium")])
        result = autoedit.plan_to_music(rows, scores, steady, limits, None, transition_seconds, soft_transition)
        self._apply_plan(result)
        report(1.0, "Done")
        return result

    def set_trim(self, path: str, start: float | None, end: float | None) -> None:
        self.execute("UPDATE items SET trim_start = ?, trim_end = ? WHERE path = ?", (start, end, path))

    def set_clip_sound(self, path: str, on: bool, volume: float | None = None) -> None:
        """Use a video's own sound in the finished video (off by default, so the music plays alone)."""
        if volume is None:
            self.execute("UPDATE items SET clip_sound = ? WHERE path = ?", (int(on), path))
        else:
            self.execute("UPDATE items SET clip_sound = ?, clip_volume = ? WHERE path = ?", (int(on), volume, path))

    def set_transition(self, path: str, transition: str | None) -> None:
        self.execute("UPDATE items SET transition = ?, transition_why = NULL WHERE path = ?", (transition, path))

    def set_scene(self, path: str, scene: str | None) -> None:
        self.execute("UPDATE items SET scene_user = ? WHERE path = ?", (scene, path))

    def name_person(self, person_id: int, name: str) -> None:
        """Name a person. Picking a name that already exists merges the two."""
        name = (name or "").strip()
        if not name:
            self.execute("UPDATE people SET name = NULL WHERE id = ?", (person_id,))
            return
        same = self.query("SELECT id FROM people WHERE name = ? AND id != ?", (name, person_id))
        with self._lock:
            if same:
                self.db.execute("UPDATE faces SET person_id = ? WHERE person_id = ?", (same[0]["id"], person_id))
                self.db.execute("DELETE FROM people WHERE id = ?", (person_id,))
                self.db.execute("UPDATE people SET hidden = 0 WHERE id = ?", (same[0]["id"],))
            else:
                self.db.execute("UPDATE people SET name = ?, hidden = 0 WHERE id = ?", (name, person_id))
            self.db.commit()

    def mark_unknown(self, person_ids: list[int]) -> None:
        with self._lock:
            self.db.executemany("UPDATE people SET hidden = 1, name = NULL WHERE id = ?", [(i,) for i in person_ids])
            self.db.commit()

    def restore_person(self, person_id: int) -> None:
        """Bring someone back from "Unknown" so you can name them."""
        self.execute("UPDATE people SET hidden = 0 WHERE id = ?", (person_id,))

    # ---------- analysis ----------

    def prepare(self, on_change: Callable[[], None] | None = None) -> None:
        """The quick first look: find the files, their dates and places, and make small previews."""
        self._run(self._prepare, on_change)

    def analyse(self, on_change: Callable[[], None] | None = None) -> None:
        """Work out scenes and people for the chosen files not analysed yet. Safe to re-run."""
        self._run(self._analyse, on_change)

    def _run(self, step, on_change) -> None:
        p = self.progress
        p.running, p.notes, p.fraction = True, [], 0.0
        notify = on_change or (lambda: None)
        try:
            step(notify)
            p.fraction, p.message = 1.0, "Done"
        finally:
            p.running = False
            notify()

    def _prepare(self, notify: Callable[[], None]) -> None:
        p = self.progress
        p.message = "Looking through the folder…"
        notify()
        found, problems = scan_folder(self.folder)
        p.notes.extend(problems)
        self._store_found(found)

        p.message = "Finding place names…"
        notify()
        self._name_places()

        # A preview made before its file was turned shows it the old way round (the video itself is turned):
        # make those again, so the preview always matches what plays and what goes into the video.
        stale = [r["path"] for r in self.query("SELECT path, thumb, rotation FROM items WHERE thumb IS NOT NULL")
                 if not r["thumb"].endswith(f"_{r['rotation'] or 0}.jpg") or not Path(r["thumb"]).exists()]
        if stale:
            with self._lock:
                self.db.executemany("UPDATE items SET thumb = NULL WHERE path = ?", [(path,) for path in stale])
                self.db.commit()
        todo = self.query("SELECT path, kind, duration, rotation FROM items WHERE thumb IS NULL ORDER BY taken_at")
        for i, row in enumerate(todo):
            p.fraction = i / len(todo)
            p.message = f"Making previews: {i + 1} of {len(todo)}"
            if i % 10 == 0:
                notify()
            try:
                self.execute("UPDATE items SET thumb = ? WHERE path = ?", (str(self._make_thumb(row)), row["path"]))
            except Exception as exc:
                p.notes.append(f"No preview for {Path(row['path']).name}: {exc}")

        p.message = "Grouping into stories and spotting similar shots…"
        notify()
        self._group_stories()
        if not self.query("SELECT 1 FROM meta WHERE key = 'suggested'") and not self.query(
            "SELECT 1 FROM items WHERE faces_done = 1 OR scene_done = 1 LIMIT 1"
        ):
            self.suggest_selection()  # a new trip starts with the best shot of each similar set ticked

    def _group_stories(self) -> None:
        for row in self.query("SELECT path, thumb FROM items WHERE thumb IS NOT NULL AND dhash IS NULL"):
            try:
                dhash, colours, sharp = stories.fingerprint(row["thumb"])
            except OSError:
                continue
            self.execute("UPDATE items SET dhash = ?, colours = ?, sharpness = ? WHERE path = ?", (dhash, colours, sharp, row["path"]))
        rows = [dict(r) for r in self.query("SELECT path, kind, taken_at, duration, lat, lon, dhash, colours, sharpness FROM items")]
        groups = stories.group(rows)
        with self._lock:
            self.db.executemany(
                "UPDATE items SET story = ?, similar_set = ?, best = ? WHERE path = ?",
                [(st, sim, int(best), path) for path, (st, sim, best) in groups.items()],
            )
            self.db.commit()

    def suggest_selection(self) -> None:
        """Tick only the best shot of each set of similar shots (single shots stay ticked)."""
        with self._lock:
            self.db.execute("UPDATE items SET chosen = best")
            self.db.execute("INSERT OR REPLACE INTO meta (key, value) VALUES ('suggested', '1')")
            self.db.commit()

    def _analyse(self, notify: Callable[[], None]) -> None:
        self._prepare(notify)
        p = self.progress
        todo = self.query(
            "SELECT * FROM items WHERE chosen = 1 AND (faces_done = 0 OR scene_done = 0 OR tags_done = 0) ORDER BY taken_at"
        )
        faces = scenes = tagger = None
        if todo:
            p.message = "Loading the AI models (the first time this downloads about 1 GB)…"
            notify()
            faces = self._load(lambda: FaceFinder(app_dir() / "models"), "people")
            scenes = self._load(lambda: SceneTagger(app_dir() / "models" / "clip"), "scenes")
            if scenes is not None:
                tagger = self._load(lambda: Tagger(scenes), "tags")

        for i, row in enumerate(todo):
            p.fraction = i / len(todo)
            p.message = f"Analysing {i + 1} of {len(todo)}: {Path(row['path']).name}"
            notify()
            try:
                self._analyse_item(row, faces, scenes, tagger)
            except Exception as exc:
                traceback.print_exc()
                p.notes.append(f"Could not analyse {Path(row['path']).name}: {exc}")
        self._find_camera_moves(notify)
        self._group_stories()  # pictures turned upright get a fresh look-alike fingerprint
        self._group_look_alike()
        self._find_subjects(tagger, notify)

        p.message = "Grouping faces into people…"
        notify()
        self._group_people()

    def _find_camera_moves(self, notify: Callable[[], None]) -> None:
        """How the camera moves in each video (pans, zooms, whips...), for videos analysed before this was added."""
        rows = self.query("SELECT * FROM items WHERE chosen = 1 AND kind = 'video' AND camera_moves IS NULL")
        for i, row in enumerate(rows):
            self.progress.fraction = i / len(rows)
            self.progress.message = f"Reading camera moves {i + 1} of {len(rows)}: {Path(row['path']).name}"
            notify()
            try:
                moved = motion.measure(row["path"])
            except Exception as exc:
                traceback.print_exc()
                self.progress.notes.append(f"Could not read the camera moves of {Path(row['path']).name}: {exc}")
                continue
            camera = moved.camera_tags() if moved else []
            tags = json.loads(row["tags"]) if row["tags"] else []
            self.execute("UPDATE items SET camera_moves = ?, tags = ? WHERE path = ?",
                         (json.dumps(camera), json.dumps(_with_camera(tags, camera)), row["path"]))

    def _find_subjects(self, tagger, notify: Callable[[], None]) -> None:
        """What each photo or video shows (fish, deer, running...), for naming the stories."""
        rows = self.query("SELECT path, clip FROM items WHERE clip IS NOT NULL AND subjects IS NULL")
        if not rows:
            return
        if tagger is None:
            self.progress.message = "Loading the AI model for naming the stories…"
            notify()
            scenes = self._load(lambda: SceneTagger(app_dir() / "models" / "clip"), "story names")
            tagger = scenes and self._load(lambda: Tagger(scenes), "story names")
            if tagger is None:
                return
        with self._lock:
            self.db.executemany(
                "UPDATE items SET subjects = ? WHERE path = ?",
                [(json.dumps(tagger.subjects(np.frombuffer(r["clip"], dtype=np.float32))), r["path"]) for r in rows],
            )
            self.db.commit()

    # ---------- stories ----------

    def stories(self, items: list[dict] | None = None) -> list[dict]:
        """The trip's stories in your order, each with a name: what it shows, where and when."""
        items = items if items is not None else self.items()
        groups: dict[int, list[dict]] = {}
        for it in items:
            if it["story"] is not None:
                groups.setdefault(it["story"], []).append(it)
        info = {r["key"]: dict(r) for r in self.query("SELECT * FROM story_info")}
        result = []
        for number, members in sorted(groups.items()):
            members.sort(key=lambda r: (r["taken_at"], r["path"]))
            key = members[0]["path"]
            saved = info.get(key, {})
            auto = story_name(members)
            result.append({
                "key": key, "number": number, "items": members, "auto_name": auto,
                "name": saved.get("name") or auto, "renamed": bool(saved.get("name")),
                "enabled": saved.get("enabled", 1) != 0, "position": saved.get("position"),
                "start": members[0]["taken_at"], "end": members[-1]["taken_at"],
            })
        placed = sorted((s for s in result if s["position"] is not None), key=lambda s: s["position"])
        return placed + [s for s in result if s["position"] is None]

    def _story_info(self, key: str) -> None:
        self.execute("INSERT OR IGNORE INTO story_info (key) VALUES (?)", (key,))

    def rename_story(self, key: str, name: str) -> None:
        self._story_info(key)
        self.execute("UPDATE story_info SET name = ? WHERE key = ?", ((name or "").strip() or None, key))

    def set_story_enabled(self, key: str, enabled: bool) -> None:
        self._story_info(key)
        self.execute("UPDATE story_info SET enabled = ? WHERE key = ?", (int(enabled), key))

    def move_story(self, key: str, steps: int) -> None:
        """Move a story earlier (negative) or later; Auto-edit follows this order."""
        order = [s["key"] for s in self.stories()]
        if key not in order:
            return
        i = order.index(key)
        order.insert(max(0, min(len(order) - 1, i + steps)), order.pop(i))
        with self._lock:
            for n, k in enumerate(order):
                self.db.execute("INSERT OR IGNORE INTO story_info (key) VALUES (?)", (k,))
                self.db.execute("UPDATE story_info SET position = ? WHERE key = ?", (n, k))
            self.db.commit()

    def reset_story_order(self) -> None:
        self.execute("UPDATE story_info SET position = NULL")

    def _thumb_path(self, path: str, rotation: int = 0) -> Path:
        # The rotation is part of the name so the browser shows the turned picture straight away.
        return self.dir / "thumbs" / f"{hashlib.sha1(path.encode()).hexdigest()[:16]}_{rotation}.jpg"

    def _make_thumb(self, row) -> Path:
        rotation = row["rotation"] or 0
        out = self._thumb_path(row["path"], rotation)
        if row["kind"] == "photo":
            with Image.open(row["path"]) as img:
                img.draft("RGB", (960, 960))  # fast for JPEGs
                img = ImageOps.exif_transpose(img).convert("RGB").rotate(-rotation, expand=True)
                img.thumbnail((480, 480))
                img.save(out, quality=85)
        else:
            subprocess.run(
                [ffmpeg_paths()[0], "-loglevel", "error", "-y", "-ss", f"{(row['duration'] or 0) / 2:.2f}",
                 "-i", row["path"], "-frames:v", "1", "-vf", rotate_filter(rotation) + "scale=480:-2", "-q:v", "4", str(out)],
                capture_output=True,
            )
            if not out.exists():
                raise RuntimeError("could not read a frame")
        return out

    def _load(self, make, what: str):
        try:
            return make()
        except Exception as exc:
            traceback.print_exc()
            self.progress.notes.append(f"Could not load the AI model for {what}: {exc}")
            return None

    def _store_found(self, found: list[MediaItem]) -> None:
        with self._lock:
            known = {r["path"]: r["mtime"] for r in self.db.execute("SELECT path, mtime FROM items")}
            present = set()
            for item in found:
                path = str(item.path)
                present.add(path)
                mtime = item.path.stat().st_mtime
                if path in known and known[path] == mtime:
                    continue
                if path in known:  # the file changed: forget what we knew about it
                    self.db.execute("DELETE FROM faces WHERE path = ?", (path,))
                    self.db.execute("DELETE FROM items WHERE path = ?", (path,))
                self.db.execute(
                    "INSERT INTO items (path, kind, taken_at, duration, has_audio, hdr, mtime, lat, lon) VALUES (?,?,?,?,?,?,?,?,?)",
                    (path, item.kind, item.taken_at.isoformat(), item.duration, int(item.has_audio), int(item.hdr), mtime, item.lat, item.lon),
                )
            for path in set(known) - present:
                self.db.execute("DELETE FROM faces WHERE path = ?", (path,))
                self.db.execute("DELETE FROM items WHERE path = ?", (path,))
            self.db.commit()

    def _name_places(self) -> None:
        rows = self.query("SELECT path, lat, lon FROM items WHERE lat IS NOT NULL AND place IS NULL")
        if not rows:
            return
        import pycountry
        import reverse_geocoder

        found = reverse_geocoder.search([(r["lat"], r["lon"]) for r in rows], mode=1, verbose=False)
        with self._lock:
            for row, hit in zip(rows, found):
                country = pycountry.countries.get(alpha_2=hit["cc"])
                self.db.execute(
                    "UPDATE items SET place = ?, country = ? WHERE path = ?",
                    (hit["name"], country.name if country else hit["cc"], row["path"]),
                )
            self.db.commit()

    def _analyse_item(self, row: sqlite3.Row, faces, scenes, tagger=None) -> None:
        path = Path(row["path"])
        key = hashlib.sha1(str(path).encode()).hexdigest()[:16]
        with tempfile.TemporaryDirectory() as tmp:
            frames = _frames(path, row["kind"], row["duration"], Path(tmp), row["rotation"] or 0)
            if not frames:
                raise RuntimeError("no picture could be read")
            updates: dict = {}

            if faces is not None and not row["rotation_checked"]:
                turn = _upright_rotation(faces, frames[0])
                updates["rotation_checked"] = 1
                if turn:
                    self._set_rotation(str(path), turn)
                    self.progress.notes.append(f"Turned {path.name} upright ({turn}°). Use the rotate button if it's wrong.")
                    frames = _frames(path, row["kind"], row["duration"], Path(tmp), turn)

            if faces is not None and not row["faces_done"]:
                found = []
                for frame in frames:
                    with Image.open(frame) as img:
                        rgb = np.array(img.convert("RGB"))
                    for face in faces.find(rgb):
                        found.append((face, rgb))
                found = _one_per_person(found)
                with self._lock:
                    self.db.execute("DELETE FROM faces WHERE path = ?", (str(path),))
                    for n, (face, rgb) in enumerate(found):
                        crop_path = self.dir / "faces" / f"{key}_{n}.jpg"
                        _save_face(rgb, face.box, crop_path)
                        self.db.execute(
                            "INSERT INTO faces (path, score, crop, embedding) VALUES (?,?,?,?)",
                            (str(path), face.score, str(crop_path), face.embedding.tobytes()),
                        )
                    self.db.commit()
                updates["faces_done"] = 1

            if row["kind"] == "video" and row["motion_action"] is None:
                moved = motion.measure(str(path))
                if moved:
                    updates.update(motion_action=moved.action, motion_camera=moved.camera, motion_shake=moved.shake,
                                   curves=_curves_json(moved, path, row["has_audio"]),
                                   camera_moves=json.dumps(moved.camera_tags()))

            if scenes is not None and not (row["scene_done"] and row["tags_done"]):
                each = scenes.embed_each(frames)
                vector = mean_vector(each)
                if not row["scene_done"]:
                    top = scenes.classify(vector)
                    updates.update(scene=top[0][0], scene_top=json.dumps(top), clip=vector.tobytes(), scene_done=1)
                if tagger is not None:
                    found_tags = tagger.tag(each)
                    action = updates.get("motion_action", row["motion_action"])
                    if action is not None:
                        found_tags = motion.Motion(
                            action, updates.get("motion_camera", row["motion_camera"]),
                            updates.get("motion_shake", row["motion_shake"]),
                        ).tags() + found_tags
                    camera = updates.get("camera_moves", row["camera_moves"])
                    found_tags = _with_camera(found_tags, json.loads(camera) if camera else [])
                    updates.update(tags=json.dumps(found_tags), tags_done=1)

            if "tags" not in updates and "motion_action" in updates:  # no AI tags this time, but movement is known
                moved_tags = motion.Motion(updates["motion_action"], updates["motion_camera"], updates["motion_shake"]).tags()
                updates["tags"] = json.dumps(_with_camera(moved_tags, json.loads(updates["camera_moves"])))

        if not updates:
            return
        sets = ", ".join(f"{k} = ?" for k in updates)
        self.execute(f"UPDATE items SET {sets} WHERE path = ?", (*updates.values(), str(path)))

    def _group_look_alike(self) -> None:
        """Group files that look alike across the whole trip (all the snorkelling, all the sunsets...)."""
        rows = self.query("SELECT path, clip FROM items WHERE chosen = 1 AND clip IS NOT NULL ORDER BY taken_at")
        groups = look_alike_groups({r["path"]: np.frombuffer(r["clip"], dtype=np.float32) for r in rows})
        with self._lock:
            self.db.execute("UPDATE items SET look_group = NULL")
            self.db.executemany("UPDATE items SET look_group = ? WHERE path = ?", [(g, p) for p, g in groups.items()])
            self.db.commit()

    def _group_people(self) -> None:
        with self._lock:
            known: dict[int, np.ndarray] = {}
            for p in self.db.execute("SELECT id FROM people"):
                embs = [np.frombuffer(r[0], dtype=np.float32) for r in self.db.execute(
                    "SELECT embedding FROM faces WHERE person_id = ?", (p["id"],))]
                if embs:
                    known[p["id"]] = np.mean(embs, axis=0)
            new = self.db.execute("SELECT id, embedding FROM faces WHERE person_id IS NULL ORDER BY score DESC").fetchall()
            if not new:
                return
            groups = group_faces(np.array([np.frombuffer(r["embedding"], dtype=np.float32) for r in new]), known)
            created: dict[int, int] = {}
            for row, group in zip(new, groups):
                if group < 0:
                    if group not in created:
                        created[group] = self.db.execute("INSERT INTO people (name) VALUES (NULL)").lastrowid
                    group = created[group]
                self.db.execute("UPDATE faces SET person_id = ? WHERE id = ?", (group, row["id"]))
            self.db.commit()


def _guess_places(rows: list[dict]) -> None:
    """Fill in missing places from photos taken around the same time: nobody changes country in a few hours."""
    located = [(datetime.fromisoformat(r["taken_at"]), r) for r in rows if r["place"]]
    if not located:
        return
    for row in rows:
        if row["place"]:
            continue
        when = datetime.fromisoformat(row["taken_at"])
        gap, nearest = min(((abs((t - when).total_seconds()), r) for t, r in located), key=lambda x: x[0])
        if gap <= PLACE_GUESS_HOURS * 3600:
            row["place"], row["country"] = nearest["place"], nearest["country"]
        elif nearest["day"] == row["day"]:
            row["country"] = nearest["country"]
        else:
            same_day = [r for _, r in located if r["day"] == row["day"]]
            if not same_day:
                continue
            row["country"] = same_day[0]["country"]
        row["place_guessed"] = True


def _with_camera(tags: list[str], camera: list[str]) -> list[str]:
    """The tags with the camera-move tags brought up to date."""
    return [t for t in tags if t not in motion.CAMERA_TAGS] + list(camera)


def _curves_json(moved: motion.Motion, path: Path, has_audio) -> str:
    """Movement and loudness five times a second, for finding a video's best moment."""
    loud = autoedit.loudness_steps(str(path)) if has_audio else []
    return json.dumps({"action": moved.action_steps, "shake": moved.shake_steps, "loud": loud})


def story_name(members: list[dict]) -> str:
    """A story's name from what it shows, where and when, e.g. "Fish & aquarium · Lisbon · Sat 5 Jul, morning"."""
    subject = story_subject([json.loads(m["subjects"]) if m.get("subjects") else None for m in members])
    if subject is None:
        tags = [t for m in members for t in m.get("tags_final") or [] if TAG_GROUP.get(t) in ("Activity", "Place")]
        subject = max(set(tags), key=tags.count) if tags else None
    places = [m.get("place") or m.get("country") for m in members if m.get("place") or m.get("country")]
    place = max(set(places), key=places.count) if places else None
    when = datetime.fromisoformat(members[0]["taken_at"]).astimezone()
    part = "morning" if 5 <= when.hour < 12 else "afternoon" if when.hour < 17 else "evening" if when.hour < 21 else "night"
    return " · ".join(x for x in (subject, place, f"{when:%a} {when.day} {when:%b}, {part}") if x)


def rotate_filter(rotation: int) -> str:
    """FFmpeg filter that turns a video clockwise by 90, 180 or 270 degrees."""
    return {90: "transpose=1,", 180: "hflip,vflip,", 270: "transpose=2,"}.get(rotation % 360, "")


def _upright_rotation(faces, frame: Path) -> int:
    """If a picture has no upright faces but clear faces when turned, it was saved sideways or upside down."""
    with Image.open(frame) as img:
        picture = img.convert("RGB")
    best = max((f.score for f in faces.find(np.array(picture))), default=0.0)
    if best >= 0.5:
        return 0
    turns = {}
    for turn in (90, 180, 270):
        found = faces.find(np.array(picture.rotate(-turn, expand=True)))
        turns[turn] = max((f.score for f in found), default=0.0)
    turn = max(turns, key=turns.get)
    return turn if turns[turn] >= 0.8 else 0


def _frames(path: Path, kind: str, duration: float, out_dir: Path, rotation: int = 0) -> list[Path]:
    """Pictures to look at: the photo itself, or a few frames spread through a video."""
    if kind == "photo":
        out = out_dir / f"photo_{rotation}.jpg"
        with Image.open(path) as img:
            img = ImageOps.exif_transpose(img).convert("RGB").rotate(-rotation, expand=True)
            img.thumbnail((1600, 1600))
            img.save(out, quality=90)
        return [out]
    count = 1 if duration < 4 else min(5, int(duration // 3))
    times = [duration / 2] if count == 1 else [duration * (0.1 + 0.8 * i / (count - 1)) for i in range(count)]
    frames = []
    for i, t in enumerate(times):
        out = out_dir / f"frame_{i}_{rotation}.jpg"
        subprocess.run(
            [ffmpeg_paths()[0], "-loglevel", "error", "-y", "-ss", f"{t:.2f}", "-i", str(path),
             "-frames:v", "1", "-vf", rotate_filter(rotation) + "scale='min(1600,iw)':-2", "-q:v", "3", str(out)],
            capture_output=True,
        )
        if out.exists():
            frames.append(out)
    return frames


def _one_per_person(found: list) -> list:
    """In a video the same face appears in several frames; keep the clearest one of each person."""
    kept: list = []
    for face, rgb in sorted(found, key=lambda f: -f[0].score):
        if all(float(face.embedding @ k[0].embedding) < 0.5 for k in kept):
            kept.append((face, rgb))
    return kept


def _save_face(rgb: np.ndarray, box: tuple, out: Path) -> None:
    x1, y1, x2, y2 = box
    pad = 0.3 * max(x2 - x1, y2 - y1)
    h, w = rgb.shape[:2]
    crop = rgb[max(0, int(y1 - pad)) : min(h, int(y2 + pad)), max(0, int(x1 - pad)) : min(w, int(x2 + pad))]
    img = Image.fromarray(crop)
    img.thumbnail((160, 160))
    img.save(out, quality=85)
