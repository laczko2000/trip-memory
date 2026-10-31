"""The trip catalogue: everything the app has learned about a trip, saved so it is only worked out once.

Each trip folder gets its own small database (SQLite) plus thumbnail pictures, stored in
~/Library/Application Support/TripMemory, never inside your photo folders.
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import re
import sqlite3
import subprocess
import sys
import tempfile
import time
import threading
import traceback
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

from .faces import FaceFinder, group_faces
from .media import MediaItem, scan_folder
from . import autoedit, director, highlights, transitions, moments, motion, musicmood, stories, themes, timeshift, titles
from .scenes import SceneTagger, mean_vector
from .tags import TAG_GROUP, Tagger, group_name, look_alike_groups, people_tag, story_subject
from .tools import ffmpeg_paths, probe
from .looks import LOOKS
from .render import FPS, RAMP_SOURCE, held, overlaps, source_factor, style_between


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
    fps REAL,
    effects TEXT,
    effects_why TEXT,
    ai_caption TEXT,
    ai_score INTEGER,
    removed INTEGER DEFAULT 0,
    features BLOB
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
CREATE TABLE IF NOT EXISTS frames (  -- a CLIP vector for a frame every second inside a video (moments)
    path TEXT NOT NULL,
    t REAL NOT NULL,
    vec BLOB,
    PRIMARY KEY (path, t)
);
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
    stop: bool = False  # you pressed Stop: the AI stops after what it is doing now, keeping what it finished
    turn: int = 0  # counts single AI requests, so a late answer to a stopped one is never used


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
            "effects TEXT", "effects_why TEXT", "ai_caption TEXT", "ai_score INTEGER", "removed INTEGER DEFAULT 0", "features BLOB",
            "plan_role TEXT", "plan_pos REAL", "title_text TEXT", "title_sub TEXT", "title_style TEXT", "title_where TEXT",
            "focus_x REAL", "keep_start REAL", "keep_end REAL", "part_of TEXT",
            "theme TEXT", "theme_auto TEXT", "theme_sure REAL", "locks TEXT", "camera TEXT", "taken_orig TEXT", "postits TEXT",
            "shot_of TEXT", "crop TEXT", "reverse INTEGER DEFAULT 0", "transition_len REAL",
        ):
            try:  # catalogues made by older versions lack these columns
                self.db.execute(f"ALTER TABLE items ADD COLUMN {column}")
            except sqlite3.OperationalError:
                pass
        for column in ("theme TEXT", "idea TEXT", "plan_notes TEXT", "ai_name TEXT", "mine_enabled INTEGER"):
            try:
                self.db.execute(f"ALTER TABLE story_info ADD COLUMN {column}")
                if column == "ai_name TEXT":
                    self._split_ai_names()
            except sqlite3.OperationalError:
                pass
        (self.dir / "folder.txt").write_text(str(self.folder))
        self.progress = Progress()
        self._repeats: set[str] = set()  # the last pool's files that repeat another file
        self._moment_starts: dict[str, float] = {}  # the last pool's clips used for a moment: where their part starts
        self.frame_embedder = None  # pictures -> CLIP vectors (the picture model is loaded the first time)
        self._chapters: list[dict] | None = None  # the last pool's storyline, as chapters for Auto-edit

    # ---------- reading ----------

    def query(self, sql: str, args: tuple = ()) -> list[sqlite3.Row]:
        with self._lock:
            return self.db.execute(sql, args).fetchall()

    def db_write_many(self, sql: str, rows: list[tuple]) -> None:
        with self._lock:
            self.db.executemany(sql, rows)
            self.db.commit()

    def execute(self, sql: str, args: tuple = ()) -> None:
        with self._lock:
            self.db.execute(sql, args)
            self.db.commit()

    def items(self, removed: bool = False, parts: bool = False) -> list[dict]:
        """All items in date order, with the names of the (known) people in them. Files you removed from the trip
        are left out (or, with removed=True, only those are given). parts: also the second and third parts of long
        clips that Auto-edit used as shots of their own (only the storyboard shows those)."""
        rows = [dict(r) for r in self.query("SELECT * FROM items WHERE removed = ? AND (part_of IS NULL OR ?) "
                                            "ORDER BY taken_at, path", (int(removed), int(parts)))]
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
            row["postits"] = json.loads(row["postits"]) if row.get("postits") else []
            if row["postits"] and row["postits"][0].get("thumb") and Path(row["postits"][0]["thumb"]).exists():
                row["thumb"] = row["postits"][0]["thumb"]  # your first post-it is the video's picture everywhere
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
        items = self.items(parts=True)
        stories = self.stories([it for it in items if not it["part_of"]])
        off = {r["path"] for s in stories if not s["enabled"] for r in s["items"]}
        rank = {r["path"]: n for n, s in enumerate(stories) for r in s["items"]}
        left_out = self._left_out_by_plan(stories)
        if self.storyline():  # your storyline chooses the shots: only stories you switched off yourself stay out
            mine = {r["key"] for r in self.query("SELECT key FROM story_info WHERE mine_enabled = 0")}
            off = {r["path"] for s in stories if s["key"] in mine for r in s["items"]}
            left_out = set()
        source = {it["path"]: it["part_of"] or it["path"] for it in items}
        clips = {it["path"]: it for it in items if not it["part_of"]}
        rows = [it for it in items if it["chosen"] and it["included"] and not it["cut_out"]
                and source[it["path"]] not in off and source[it["path"]] not in left_out
                and (not it["part_of"] or (it["part_of"] in clips and clips[it["part_of"]]["chosen"]
                                           and clips[it["part_of"]]["included"]))]
        # The photos inside a collage show as part of its first photo, not on their own.
        inside = {p for r in rows if r["effects"] for key in ("collage", "side")
                  for p in json.loads(r["effects"]).get(key, [])}
        rows = [r for r in rows if r["path"] not in inside]

        def order(r: dict) -> tuple:  # by story, then the story's plan, then time
            return rank.get(source[r["path"]], 0), r["plan_pos"] is None, r["plan_pos"] or 0, r["taken_at"]

        placed = sorted((r for r in rows if r["position"] is not None), key=lambda r: r["position"])
        rest = sorted((r for r in rows if r["position"] is None), key=order)
        # Anything not placed yet goes in by story and plan: after the last placed file that comes before it.
        for r in rest:
            key = order(r)
            at = 0
            for i, p in enumerate(placed):
                if order(p) <= key:
                    at = i + 1
            placed.insert(at, r)
        return placed

    @staticmethod
    def _left_out_by_plan(stories: list[dict]) -> set[str]:
        """Shots a story's plan leaves out: they stay out of the storyboard and Auto-edit (your selection keeps them)."""
        return {it["path"] for s in stories if s["planned"] for it in s["items"] if not it["plan_role"]}

    def media_items(self, chapters: bool = False, focus: bool = False) -> list[MediaItem]:
        """What the video is made from, in storyboard order, optionally with a title card for each day.
        focus: also say where the people are in each shot (for a phone video, which shows a narrow slice)."""
        board = self.storyboard()
        where = self.focus_points(board) if focus else {}
        turned = {r["path"]: r["rotation"] or 0 for r in self.items()}
        days = sorted({r["day"] for r in self.items() if r["chosen"]})
        shown = self.shot_titles(board)
        result: list[MediaItem] = []
        previous_day = None
        for r in board:
            if chapters and r["day"] != previous_day:
                result.append(self._title_card(r, board, days))
            previous_day = r["day"]
            result.append(
                MediaItem(
                    path=Path(r["part_of"] or r["path"]),  # a part of a long clip plays from the clip itself
                    kind=r["kind"],
                    taken_at=datetime.fromisoformat(r["taken_at"]),
                    duration=r["duration"],
                    has_audio=bool(r["has_audio"]),
                    hdr=bool(r["hdr"]),
                    rotation=r["rotation"] or 0,
                    trim_start=r["trim_start"] if r["trim_start"] is not None else r["keep_start"],
                    trim_end=r["trim_end"] if r["trim_start"] is not None else r["keep_end"],
                    transition=r["transition"],
                    transition_seconds=r.get("transition_len"),
                    seconds=r["seconds"],
                    sound=bool(r["clip_sound"]),  # videos are silent unless you switch their sound on
                    volume=1.0 if r["clip_volume"] is None else r["clip_volume"],
                    speed=r["speed"] or 1.0,
                    effects=self._with_parts(r["effects"], turned),
                    overlay=shown.get(r["path"]),
                    focus=where.get(r["path"]),
                    crop=json.loads(r["crop"]) if r.get("crop") else None,
                    reverse=bool(r.get("reverse")),
                )
            )
        return result

    @staticmethod
    def _with_parts(effects: str | None, turned: dict[str, int]) -> dict | None:
        """A shot's effects; a collage also gets how each of its photos is turned."""
        if not effects:
            return None
        fx = json.loads(effects)
        for key in ("collage", "side"):
            if fx.get(key):
                fx["turns"] = [turned.get(p, 0) for p in fx[key]]
        return fx

    def split_collage(self, path: str) -> None:
        """Show a collage's photos one after another again, each for the time it popped in for."""
        row = self.query("SELECT effects, position, seconds FROM items WHERE path = ?", (path,))
        fx = json.loads(row[0]["effects"]) if row and row[0]["effects"] else {}
        if not fx.get("collage"):
            return
        appear = [*fx.get("appear", []), row[0]["seconds"] or 0.0]
        lengths = [max(0.5, b - a) for a, b in zip(appear, appear[1:])] if len(appear) == len(fx["collage"]) + 2 \
            else [None] * (len(fx["collage"]) + 1)
        position = row[0]["position"]
        with self._lock:
            self.db.execute("UPDATE items SET effects = NULL, effects_why = NULL, seconds = ? WHERE path = ?",
                            (lengths[0], path))
            for n, (part, length) in enumerate(zip(fx["collage"], lengths[1:]), 1):
                self.db.execute("UPDATE items SET seconds = ?, cut_out = 0, position = ? WHERE path = ?",
                                (length, None if position is None else position + n / 100, part))
            self.db.commit()

    def focus_points(self, rows: list[dict]) -> dict[str, float]:
        """Where the faces are across each shot's picture (0 left .. 1 right; bigger faces count more), looked
        up once from its preview. Shots without faces are left out (a phone video then shows their middle)."""
        todo = [r for r in rows if r.get("focus_x") is None and r.get("thumb") and Path(r["thumb"]).exists()]
        if todo:
            finder = self._load(lambda: FaceFinder(app_dir() / "models"), "people")
            found = []
            for r in todo:
                x = -1.0  # looked, no faces
                if finder is not None:
                    try:
                        with Image.open(r["thumb"]) as img:
                            rgb = np.asarray(img.convert("RGB"))
                        boxes = [f.box for f in finder.find(rgb)]
                    except Exception:
                        boxes = []
                    if boxes:
                        sizes = [(b[2] - b[0]) * (b[3] - b[1]) for b in boxes]
                        x = sum((b[0] + b[2]) / 2 * a for b, a in zip(boxes, sizes)) / sum(sizes) / rgb.shape[1]
                found.append((x, r["path"]))
                r["focus_x"] = x
            if finder is not None:
                self.db_write_many("UPDATE items SET focus_x = ? WHERE path = ?", found)
        return {r["path"]: float(r["focus_x"]) for r in rows if r.get("focus_x") is not None and r["focus_x"] >= 0}

    # ---------- titles ----------

    def title_settings(self) -> dict:
        """The automatic titles: opening title, a title for each story, end card (and their words)."""
        rows = self.query("SELECT value FROM meta WHERE key = 'titles'")
        saved = json.loads(rows[0]["value"]) if rows else {}
        days = sorted({r["day"] for r in self.items() if r["chosen"]})
        span = ""
        if days:
            a, b = (datetime.fromisoformat(d) for d in (days[0], days[-1]))
            span = (f"{a:%-d %B %Y}" if a == b else f"{a:%-d} – {b:%-d %B %Y}" if (a.year, a.month) == (b.year, b.month)
                    else f"{a:%-d %b} – {b:%-d %b %Y}")
        return {"opening": True, "trip_title": saved.get("ai_title") or self.folder.name.replace("_", " "), "trip_sub": span,
                "stories": True, "story_style": "caption", "ending": True, "end_text": "Until next time",
                "end_sub": self.folder.name.replace("_", " "), "style": "bold", "end_black": 3.0,
                "card": "", "open_seconds": 3.0, "open_room": 8.0, "end_room": 8.0, **saved} | (
                    {} if "end_card" in saved else {"end_card": saved.get("card", "")})  # (the ending's own look)

    def opening_card(self) -> dict | None:
        """The opening card (the trip's title on black or on the first shot blurred), or None: then the
        opening title shows over the first shot. {"text", "sub", "card", "seconds"}."""
        t = self.title_settings()
        if not (t["opening"] and t["card"] in titles.CARDS and (t["trip_title"] or t["trip_sub"])):
            return None
        return {"text": t["trip_title"], "sub": t["trip_sub"], "card": t["card"],
                "seconds": float(t["open_room"] or t["open_seconds"] or 3)}

    def card_rooms(self) -> dict:
        """The seconds Auto-edit keeps at the start and the end of the music for the opening and ending cards."""
        t = self.title_settings()
        return {"open_room": float(t["open_room"] or 0) if t["opening"] else 0.0,
                "end_room": float(t["end_room"] or 0) if t["ending"] else 0.0}

    def closing_card(self) -> dict | None:
        """The ending card over the last seconds of the video, as the music ends: {"text", "sub", "card" (a key of
        titles.CARDS, or "" for words on black), "seconds"}; None without one (or with the old black ending)."""
        t = self.title_settings()
        if not (t["ending"] and float(t["end_room"] or 0) > 0):
            return None
        card = t["end_card"] if t["end_card"] in titles.CARDS else ""
        return {"text": t["end_text"], "sub": t["end_sub"], "card": card, "seconds": float(t["end_room"])}

    def end_card(self) -> tuple[float, dict | None]:
        """(seconds of black at the end, the words on it) for the video's ending. With room kept for the ending
        card (closing_card) there is no black after the last shot."""
        t = self.title_settings()
        if t["ending"] and float(t["end_room"] or 0) > 0:
            return 0.0, None
        words = ({"text": t["end_text"], "sub": t["end_sub"], "style": "elegant", "where": "center"}
                 if t["ending"] and (t["end_text"] or t["end_sub"]) else None)
        if t["end_card"] in titles.CARDS and t["end_black"]:
            words = {**titles.card_words(t["end_card"], words or {}), "card": t["end_card"]}
        return float(t["end_black"] or 0), words

    def set_title_settings(self, **values) -> None:
        rows = self.query("SELECT value FROM meta WHERE key = 'titles'")
        saved = {**(json.loads(rows[0]["value"]) if rows else {}), **values}
        self.execute("INSERT OR REPLACE INTO meta (key, value) VALUES ('titles', ?)", (json.dumps(saved),))

    def look(self) -> str:
        """The colour mood of the whole video (a key of looks.LOOKS)."""
        rows = self.query("SELECT value FROM meta WHERE key = 'look'")
        return rows[0]["value"] if rows and rows[0]["value"] in LOOKS else "natural"

    def set_look(self, name: str) -> None:
        self.execute("INSERT OR REPLACE INTO meta (key, value) VALUES ('look', ?)", (name if name in LOOKS else "natural",))

    def set_shot_title(self, path: str, text: str, sub: str = "", style: str = "clean", where: str = "") -> None:
        """Your own words over a shot (empty: none, the automatic title shows if there is one)."""
        text, sub = (text or "").strip(), (sub or "").strip()
        self._set_muted_titles([p for p in self.muted_titles() if p != path])  # yours shows again
        if not text and not sub:
            self.execute("UPDATE items SET title_text = NULL, title_sub = NULL, title_style = NULL, title_where = NULL "
                         "WHERE path = ?", (path,))
            return
        self.execute("UPDATE items SET title_text = ?, title_sub = ?, title_style = ?, title_where = ? WHERE path = ?",
                     (text, sub, style, where or None, path))

    def shot_titles(self, board: list[dict] | None = None) -> dict[str, dict]:
        """The text over each shot of the video: yours, else the automatic opening, story and end titles.
        Each: {"text", "sub", "style", "where", "auto": "opening" | "story" | "ending" | None}."""
        board = self.storyboard() if board is None else board
        if not board:
            return {}
        t = self.title_settings()
        names = {s["number"]: s["name"] for s in self.stories()}
        # By theme or day by day, the shots don't follow the stories: no story titles then.
        line = self.storyline()
        chapters_on = line["order"] != themes.TIME or line["by_day"]
        result: dict[str, dict] = {}
        if line["order"] == themes.TIME and line["day_titles"]:  # "Day 2 · Lisbon · Tue 15 Jul" as each day starts
            days = sorted({x["day"] for x in board})
            seen: set[str] = set()
            for r in board:
                if r["day"] in seen:
                    continue
                seen.add(r["day"])
                places = [x["place"] or x["country"] for x in board
                          if x["day"] == r["day"] and (x["place"] or x["country"])]
                when = datetime.fromisoformat(r["day"]).strftime("%a %-d %b")
                result[r["path"]] = {"text": f"Day {days.index(r['day']) + 1}",
                                     "sub": (max(set(places), key=places.count) + " · " if places else "") + when,
                                     "style": t["story_style"], "where": "", "auto": "day"}
            chapters_on = True  # day titles instead of story titles
        previous = object()
        for k, r in enumerate(board):
            if r["story"] != previous and t["stories"] and k > 0 and not chapters_on:
                places = [x["place"] or x["country"] for x in board if x["story"] == r["story"] and (x["place"] or x["country"])]
                when = datetime.fromisoformat(r["taken_at"]).astimezone()
                sub = (max(set(places), key=places.count) + " · " if places else "") + f"{when:%a %-d %b}"
                result[r["path"]] = {"text": names.get(r["story"], ""), "sub": sub, "style": t["story_style"],
                                     "where": "", "auto": "story"}
            previous = r["story"]
        if t["opening"] and (t["trip_title"] or t["trip_sub"]) and t["card"] not in titles.CARDS:
            result[board[0]["path"]] = {"text": t["trip_title"], "sub": t["trip_sub"], "style": t["style"],
                                        "where": "center", "auto": "opening"}
        if t["ending"] and not t["end_black"] and len(board) > 1 and (t["end_text"] or t["end_sub"]):
            # (with a black ending, the end card shows on the black instead: see end_card)
            result[board[-1]["path"]] = {"text": t["end_text"], "sub": t["end_sub"], "style": "elegant",
                                         "where": "center", "auto": "ending"}
        for r in board:
            if r["title_text"] or r["title_sub"]:
                result[r["path"]] = {"text": r["title_text"] or "", "sub": r["title_sub"] or "",
                                     "style": r["title_style"] or "clean", "where": r["title_where"] or "", "auto": None}
        for path in self.muted_titles():  # moved onto the video's timeline (or taken away there)
            result.pop(path, None)
        return result

    # Titles on the video's own timeline (the editor under the video). Each: {"id", "text", "sub", "style",
    # "where", "length", "fade_in", "fade_out", and "link" (a shot's path) + "offset", or "start" (seconds)}.

    def timeline_titles(self) -> list[dict]:
        rows = self.query("SELECT value FROM meta WHERE key = 'timeline_titles'")
        return json.loads(rows[0]["value"]) if rows else []

    def save_timeline_titles(self, found: list[dict]) -> None:
        self.execute("INSERT OR REPLACE INTO meta (key, value) VALUES ('timeline_titles', ?)", (json.dumps(found),))

    def muted_titles(self) -> list[str]:
        """Shots whose own or automatic title is not drawn on the shot any more (it lives on the timeline)."""
        rows = self.query("SELECT value FROM meta WHERE key = 'muted_titles'")
        return json.loads(rows[0]["value"]) if rows else []

    def _set_muted_titles(self, paths: list[str]) -> None:
        self.execute("INSERT OR REPLACE INTO meta (key, value) VALUES ('muted_titles', ?)", (json.dumps(paths),))

    def title_to_timeline(self, path: str, shot_length: float) -> dict | None:
        """A shot's title (yours or automatic) becomes a timeline title linked to that shot, at the same time."""
        title = self.shot_titles().get(path)
        if not title:
            return None
        a, b = titles.timing(shot_length)
        new = {"id": f"t{time.time_ns()}", "text": title["text"], "sub": title["sub"], "style": title["style"],
               "where": title["where"], "link": path, "offset": round(a, 2), "length": round(b - a, 2),
               "fade_in": titles.FADE_IN, "fade_out": titles.FADE_OUT}
        self.save_timeline_titles(self.timeline_titles() + [new])
        self._set_muted_titles(self.muted_titles() + [path])
        return new

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

    def remove_unchosen(self) -> int:
        """Take every file that isn't ticked out of the trip: it disappears from every tab and is never analysed.
        The file itself stays where it is on your computer. Returns how many were removed."""
        self._move_links_off_removed()
        with self._lock:
            n = self.db.execute("UPDATE items SET removed = 1, included = 0, position = NULL "
                                "WHERE chosen = 0 AND removed = 0").rowcount
            self.db.execute("DELETE FROM faces WHERE path IN (SELECT path FROM items WHERE removed = 1)")
            self.db.commit()
        self._group_stories()
        return n

    def _move_links_off_removed(self) -> None:
        """Before files are removed: a link that starts at one of them moves to the next file of the same story,
        or goes if that whole story is removed (so the stories around it stay apart)."""
        links = self.story_links()
        if not links:
            return
        rows = self.query("SELECT path, taken_at, story, chosen FROM items WHERE part_of IS NULL AND removed = 0 ORDER BY taken_at, path")
        keep = []
        for link in links:
            at = next((i for i, r in enumerate(rows) if f"{r['taken_at']}|{r['path']}" >= link), None)
            if at is None:
                continue
            story = rows[at]["story"]
            nxt = next((r for r in rows[at:] if r["chosen"]), None)
            if nxt is not None and nxt["story"] == story:
                keep.append(f"{nxt['taken_at']}|{nxt['path']}")
        self.execute("INSERT OR REPLACE INTO meta (key, value) VALUES ('story_links', ?)", (json.dumps(sorted(set(keep))),))

    def restore(self, paths: list[str]) -> None:
        """Bring removed files back into the trip (unticked, so you can tick the ones you want)."""
        with self._lock:
            self.db.executemany("UPDATE items SET removed = 0, chosen = 0, included = 1, faces_done = 0 WHERE path = ?",
                                [(p,) for p in paths])
            self.db.commit()
        self._group_stories()

    def removed_count(self) -> int:
        return self.query("SELECT COUNT(*) AS n FROM items WHERE removed = 1")[0]["n"]

    def set_chosen(self, paths: list[str], chosen: bool) -> None:
        """Choose which files of the folder belong to this trip (only those are analysed and used)."""
        with self._lock:
            self.db.executemany("UPDATE items SET chosen = ? WHERE path = ?", [(int(chosen), p) for p in paths])
            self.db.commit()

    def move(self, path: str, steps: int, photo_seconds: float = 4.0) -> None:
        """Move a file earlier (negative steps) or later in the storyboard. Every place keeps its length, so the
        cuts stay on your pins (the shots that move along take the length of their new place)."""
        order = [r["path"] for r in self.storyboard()]
        if path not in order:
            return
        i = order.index(path)
        j = max(0, min(len(order) - 1, i + steps))
        if j != i:
            self.insert_shot(path, j + 1 if j > i else j, photo_seconds)

    # ---------- the editor under the video: cuts stay where they are ----------

    @staticmethod
    def screen_length(r: dict, photo_seconds: float = 4.0) -> float:
        """How long a storyboard shot is on screen (before the transitions overlap it)."""
        fx = json.loads(r["effects"]) if isinstance(r.get("effects"), str) and r["effects"] else (r.get("effects") or {})
        if r["kind"] != "video":
            return float(r["seconds"] or photo_seconds)
        start = r["trim_start"] if r["trim_start"] is not None else (r["keep_start"] or 0.0)
        end = r["trim_end"] if r["trim_start"] is not None else (r["keep_end"] or r["duration"] or 0.0)
        return max(0.1, (end - start) / source_factor(fx) / (r["speed"] or 1.0)) + held(fx)

    def fit_length(self, path: str, seconds: float) -> None:
        """Make a shot last this long on screen, keeping the middle of its part (and its locked moment), at its
        own speed. A video too short for it stays shorter (reflow then puts the cuts back on the pins)."""
        rows = [r for r in self.items(parts=True) if r["path"] == path]
        if not rows:
            return
        r = rows[0]
        if r["kind"] != "video":
            self.execute("UPDATE items SET seconds = ? WHERE path = ?", (round(seconds, 3), path))
            return
        fx = json.loads(r["effects"]) if r["effects"] else {}
        low, high = (r["keep_start"] or 0.0), (r["keep_end"] or r["duration"] or 0.0)
        start = r["trim_start"] if r["trim_start"] is not None else low
        end = r["trim_end"] if r["trim_start"] is not None else high
        middle = (start + end) / 2
        locks = [(a, b) for a, b in (json.loads(r["locks"]) if r.get("locks") else []) if a < end and b > start]
        if locks:  # keep the locked moment it shows in, whole
            middle = (locks[0][0] + locks[0][1]) / 2
        speed = r["speed"] or 1.0  # (never changed here: a clip plays at the speed you gave it)
        wanted = max(0.1, seconds - held(fx)) * source_factor(fx)
        span = min(wanted * speed, high - low)  # too short a video: it is simply shorter
        a = min(max(low, middle - span / 2), high - span)
        self.execute("UPDATE items SET trim_start = ?, trim_end = ?, speed = ? WHERE path = ?",
                     (round(a, 3), round(a + span, 3), round(speed, 3), path))

    def most_length(self, r: dict) -> float:
        """The longest a shot can be on screen at its speed (a photo: as long as you like)."""
        if r["kind"] != "video":
            return 1e9
        fx = json.loads(r["effects"]) if r.get("effects") else {}
        low, high = (r["keep_start"] or 0.0), (r["keep_end"] or r["duration"] or 0.0)
        return max(0.1, (high - low) / source_factor(fx) / (r["speed"] or 1.0)) + held(fx)

    def reflow(self, pins: list[float], wants: dict[str, float] | None = None, photo_seconds: float = 4.0,
               transition_seconds: float = 0.8, transition: str | None = None) -> int:
        """Lay the shots on the music's pins again, in their order, at their own speed (never sped up or slowed
        down to fit): each shot ends on the pin nearest to the length it wants (wants: path -> seconds, else the
        length it has), as far as its video reaches. A longer shot takes more pins, so the ones after it move on;
        shots past the end of the music leave the video. Returns how many left.

        A cut is the middle of the transition between two shots; the transitions are counted exactly as the
        video is made (each shot's own, else the video's transition, shorter between short shots)."""
        board = self._fix_order()
        cuts = sorted(p for p in pins or [] if p > 0.2)
        if not board:
            return 0
        wants = wants or {}
        if not cuts:  # no music: just the lengths you asked for
            for r in board:
                if r["path"] in wants:
                    self.fit_length(r["path"], wants[r["path"]])
            return 0
        end = cuts[-1]
        n = len(board)
        styles = [style_between(board[k].get("transition"), k + 1, transition, transition_seconds,
                                board[k].get("transition_len")) for k in range(n - 1)]
        screens = [self.screen_length(r, photo_seconds) for r in board]
        most = [self.most_length(r) for r in board]
        before = overlaps(screens, styles) + [0.0]
        want = [wants.get(r["path"], screens[k] - (before[k - 1] if k else 0) / 2 - before[k] / 2)
                for k, r in enumerate(board)]  # the time between its two cuts
        lengths, kept = list(screens), n
        for _ in range(4):  # (how much two shots overlap depends on their lengths: a few rounds settle it)
            over = [round(o * FPS) for o in overlaps(lengths, styles)] + [0]  # in frames, as the video counts
            t, half_before, kept, start = 0.0, 0.0, n, 0
            for k in range(n):
                if t >= end - 0.25:
                    kept = k
                    break
                half_after = over[k] / FPS / 2
                room = most[k] - half_before - half_after
                near = [p for p in cuts if t + 0.4 < p <= t + room + 0.05]
                e = min(near, key=lambda p: (abs(p - (t + want[k])), p)) if near else min(end, t + room)
                after = round(e * FPS - over[k] / 2)  # where the next shot starts (whole frames: no drift)
                lengths[k] = (after + over[k] - start) / FPS
                t, half_before, start = e, half_after, after
        for k in range(kept):
            if abs(lengths[k] - screens[k]) > 0.01:
                self.fit_length(board[k]["path"], lengths[k])
        gone = [r["path"] for r in board[kept:]]
        if gone:
            with self._lock:
                self.db.executemany("UPDATE items SET cut_out = 1, position = NULL WHERE path = ?", [(p,) for p in gone])
                self.db.commit()
        return len(gone)

    def _fix_order(self) -> list[dict]:
        """The storyboard with every shot's place written down (so one change can't reshuffle the rest)."""
        board = self.storyboard()
        with self._lock:
            self.db.executemany("UPDATE items SET position = ? WHERE path = ?", [(n, r["path"]) for n, r in enumerate(board)])
            self.db.commit()
        return board

    def swap_shots(self, a: str, b: str, photo_seconds: float = 4.0) -> None:
        """Swap two shots in the storyboard. Each takes the other's length, so every cut stays where it was
        (on your pins)."""
        board = {r["path"]: (n, r) for n, r in enumerate(self._fix_order())}
        if a not in board or b not in board or a == b:
            return
        self.checkpoint("swapping two shots")
        (na, ra), (nb, rb) = board[a], board[b]
        la, lb = self.screen_length(ra, photo_seconds), self.screen_length(rb, photo_seconds)
        with self._lock:
            self.db.executemany("UPDATE items SET position = ? WHERE path = ?", [(nb, a), (na, b)])
            self.db.commit()
        self.fit_length(a, lb)
        self.fit_length(b, la)

    def insert_shot(self, path: str, before: int, photo_seconds: float = 4.0) -> None:
        """Move a shot in front of the shot now at place `before` (len(board): to the end). Every place keeps its
        length, so the cuts stay on your pins: the shots that move along take the length of their new place."""
        board = self._fix_order()
        order = [r["path"] for r in board]
        if path not in order:
            return
        slots = [self.screen_length(r, photo_seconds) for r in board]
        rows = {r["path"]: r for r in board}
        i = order.index(path)
        j = before - 1 if before > i else before
        j = max(0, min(len(order) - 1, j))
        if j == i:
            return
        self.checkpoint("moving a shot")
        order.insert(j, order.pop(i))
        with self._lock:
            self.db.executemany("UPDATE items SET position = ? WHERE path = ?", list(enumerate(order)))
            self.db.commit()
        for k in range(min(i, j), max(i, j) + 1):
            if abs(self.screen_length(rows[order[k]], photo_seconds) - slots[k]) > 0.01:
                self.fit_length(order[k], slots[k])

    def replace_shot(self, old: str, new: str, photo_seconds: float = 4.0) -> None:
        """Put another shot in the place of one, as long as it was."""
        board = {r["path"]: (n, r) for n, r in enumerate(self._fix_order())}
        if old not in board:
            return
        self.checkpoint("swapping a shot for another")
        n, r = board[old]
        length = self.screen_length(r, photo_seconds)
        with self._lock:
            self.db.execute("UPDATE items SET cut_out = 1, position = NULL WHERE path = ?", (old,))
            self.db.execute("UPDATE items SET included = 1, cut_out = 0, chosen = 1, position = ?, trim_start = NULL, "
                            "trim_end = NULL, speed = 1.0, effects = NULL WHERE path = ?", (n, new))
            self.db.commit()
        self.fit_length(new, length)

    def together(self, path: str) -> tuple[str | None, list[str]]:
        """The shots that play on this cut with it: ("collage", photos) or ("side", upright videos), or (None, [])."""
        row = self.query("SELECT effects FROM items WHERE path = ?", (path,))
        fx = json.loads(row[0]["effects"]) if row and row[0]["effects"] else {}
        for key in ("collage", "side"):
            if fx.get(key):
                return key, list(fx[key])
        return None, []

    def set_together(self, path: str, key: str, members: list[str]) -> None:
        """Change the shots that play on this cut with it (a collage's photos or the videos next to it). The ones
        taken off leave the video; with none left it plays alone."""
        row = self.query("SELECT effects, seconds, trim_start, trim_end FROM items WHERE path = ?", (path,))
        if not row:
            return
        fx = json.loads(row[0]["effects"]) if row[0]["effects"] else {}
        old = list(fx.get(key) or [])
        members = [m for m in dict.fromkeys(members) if m != path][: 3 if key == "collage" else 2]
        self.checkpoint("changing a collage" if key == "collage" else "changing the videos side by side")
        if key == "collage":
            seconds = row[0]["seconds"] or 2.0
            step = min(0.6, max(autoedit.POP_GAP, seconds * 0.6 / max(1, len(members))))
            fx = {"collage": members, "appear": [round(k * step, 3) for k in range(len(members) + 1)]} if members else {}
        else:
            length = (row[0]["trim_end"] or 0) - (row[0]["trim_start"] or 0)
            was = dict(zip(old, fx.get("starts") or []))
            starts = []
            for m in members:
                if m in was:
                    starts.append(was[m])
                    continue
                r = self.query("SELECT duration, keep_start, keep_end FROM items WHERE path = ?", (m,))
                lo, hi = kept_part(dict(r[0])) or (0.0, r[0]["duration"] or 0.0) if r else (0.0, 0.0)
                starts.append(round(max(lo, lo + (hi - lo - length) / 2), 2))
            fx = {"side": members, "starts": starts} if members else {}
        why = (f"Collage of {len(members) + 1} photos" if key == "collage" else f"{len(members) + 1} videos side by side") \
            if members else None
        with self._lock:
            self.db.execute("UPDATE items SET effects = ?, effects_why = ? WHERE path = ?",
                            (json.dumps(fx) if fx else None, why, path))
            self.db.executemany("UPDATE items SET cut_out = 1, position = NULL WHERE path = ?",
                                [(m,) for m in old if m not in members])
            self.db.executemany("UPDATE items SET included = 1, chosen = 1, cut_out = 0, position = NULL WHERE path = ?",
                                [(m,) for m in members if m not in old])
            self.db.commit()

    def take_out(self, path: str, photo_seconds: float = 4.0) -> None:
        """Take a shot out of the video; the shot before it plays on for its time, so the later cuts stay put."""
        board = self._fix_order()
        at = next((n for n, r in enumerate(board) if r["path"] == path), None)
        if at is None:
            return
        self.checkpoint("taking a shot out")
        gone = self.screen_length(board[at], photo_seconds)
        self.execute("UPDATE items SET cut_out = 1, position = NULL WHERE path = ?", (path,))
        neighbour = board[at - 1] if at else (board[1] if len(board) > 1 else None)
        if neighbour:
            self.fit_length(neighbour["path"], self.screen_length(neighbour, photo_seconds) + gone)

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
                "UPDATE items SET rotation = ?, rotation_checked = 1, crop = NULL, faces_done = 0, scene_done = 0, tags_done = 0, dhash = NULL, features = NULL "
                "WHERE path = ?",
                (rotation, path),
            )
            self.db.commit()
        row = self.query("SELECT * FROM items WHERE path = ?", (path,))[0]
        self.execute("UPDATE items SET thumb = ? WHERE path = ?", (str(self._make_thumb(row)), path))
        self.execute("UPDATE items SET rotation = ?, crop = NULL WHERE part_of = ?", (rotation, path))  # (parts too)

    def reset_order(self) -> None:
        self.execute("UPDATE items SET position = NULL")

    def auto_edit_pool(self) -> list[dict]:
        """What Auto-edit picks from: your selection (ticked in Choose files and "In the video" in the Library),
        story by story in your story order, leaving out stories you switched off and shots a story's plan
        leaves out. With chapters, each shot carries its chapter ("chapter", "chapter_fit"); shots in no chapter
        are left out, and only the stories you switched off yourself are (the plans and the wish don't count)."""
        storyline = self.storyline()
        if storyline:
            return self._storyline_pool(storyline)
        self._chapters = None
        items = self.items()
        stories = self.stories(items)
        left_out = self._left_out_by_plan(stories)
        wish = self.wish()
        pool = []
        repeats = self._same_moments([r for s in stories if s["enabled"] for r in s["items"]
                                      if r["chosen"] and r["included"] and r["path"] not in left_out])
        self._repeats = repeats
        for rank, story in enumerate(stories):
            if story["enabled"]:
                for r in story["items"]:
                    if r["chosen"] and r["included"] and r["path"] not in left_out and r["path"] not in repeats:
                        r = r | {"wish": self.wish_match(r, wish)}
                        if kept_part(r):  # a clip you cut: Auto-edit only sees the part you kept
                            start, end = kept_part(r)
                            r = r | {"duration": end - start, "keep_offset": start, "full_duration": r["duration"]}
                        pool.append(r | {"story_rank": rank, "story_name": story["name"]})
        return self._with_locks(pool)

    @staticmethod
    def _with_locks(pool: list[dict]) -> list[dict]:
        """Each clip's locked moments, in the time Auto-edit sees it (from the start of the part you kept)."""
        for r in pool:
            locks = json.loads(r["locks"]) if isinstance(r.get("locks"), str) and r["locks"] else []
            offset, length = r.get("keep_offset") or 0.0, r["duration"] or 0.0
            r["locks"] = [(max(0.0, a - offset), min(length, b - offset)) for a, b in locks
                          if min(length, b - offset) - max(0.0, a - offset) >= 0.3] if r["kind"] == "video" else []
            # Your bookmarks (post-its): each is always inside the part of the clip the video uses.
            marks = r.get("postits") or []
            if isinstance(marks, str):
                marks = json.loads(marks)
            r["bookmarks"] = sorted(p["t"] - offset for p in marks if 0.0 <= p["t"] - offset <= length) \
                if r["kind"] == "video" and not r["locks"] else []
            if r["kind"] != "video" and marks:  # a photo you starred: always in the video
                r["forced"] = True
        return pool

    def locks(self, path: str) -> list[list[float]]:
        """The moments you locked in a clip: [[start, end]] in seconds from the start of the file."""
        rows = self.query("SELECT locks FROM items WHERE path = ?", (path,))
        return json.loads(rows[0]["locks"]) if rows and rows[0]["locks"] else []

    def set_locks(self, path: str, locks: list, checkpoint: bool = True) -> None:
        """Save a clip's locked moments (overlapping ones become one). Every view shows them; the next cut uses
        each exactly, and only the rest of the clip is free for other shots."""
        merged: list[list[float]] = []
        for a, b in sorted((round(float(a), 2), round(float(b), 2)) for a, b in locks if float(b) - float(a) >= 0.3):
            if merged and a <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], b)
            else:
                merged.append([a, b])
        if checkpoint:
            self.checkpoint("locking a moment")
        self.execute("UPDATE items SET locks = ? WHERE path = ?", (json.dumps(merged) if merged else None, path))

    def used_parts(self, clip: str, but: str | None = None) -> list[tuple[float, float, int]]:
        """The seconds of a clip the other shots in the Storyboard already use: (start, end, shot number)."""
        out = []
        for n, r in enumerate(self.storyboard(), 1):
            if (r["part_of"] or r["path"]) == clip and r["path"] != but and r["kind"] == "video" \
                    and r["trim_start"] is not None and r["trim_end"] is not None:
                out.append((float(r["trim_start"]), float(r["trim_end"]), n))
        return out

    def locked_count(self) -> int:
        """How many moments are locked in your chosen clips."""
        return sum(len(json.loads(r["locks"])) for r in self.query(
            "SELECT locks FROM items WHERE locks IS NOT NULL AND chosen = 1 AND removed = 0 AND part_of IS NULL"))

    # ---------- post-its: the best bits of a video, marked while you watch ----------

    def postits(self, path: str) -> list[dict]:
        """[{"t" (seconds into the file), "name", "thumb"}], in time order."""
        rows = self.query("SELECT postits FROM items WHERE path = ?", (path,))
        return json.loads(rows[0]["postits"]) if rows and rows[0]["postits"] else []

    def set_postits(self, path: str, found: list[dict]) -> None:
        rows = self.query("SELECT path, rotation, thumb FROM items WHERE path = ?", (path,))
        if not rows:
            return
        clean = []
        for p in sorted(found, key=lambda p: p["t"]):
            if clean and abs(p["t"] - clean[-1]["t"]) < 0.3:
                continue  # the same moment twice
            thumb = p.get("thumb") if p.get("thumb") and Path(p["thumb"]).exists() else None
            clean.append({"t": round(float(p["t"]), 2), "name": (p.get("name") or "").strip(),
                          "thumb": thumb or self.moment_picture(dict(rows[0]), float(p["t"]))})
        self.execute("UPDATE items SET postits = ? WHERE path = ?", (json.dumps(clean) if clean else None, path))

    def starred(self, path: str) -> bool:
        """A photo you starred: it is always in the video."""
        return bool(self.postits(path))

    def set_star(self, path: str, on: bool) -> None:
        """⭐ on a photo: it is always in the video (kept as one post-it at 0 s, like a video's highlight)."""
        self.execute("UPDATE items SET postits = ? WHERE path = ? AND kind != 'video'",
                     (json.dumps([{"t": 0.0, "name": "", "thumb": None}]) if on else None, path))

    def add_postit(self, path: str, t: float, name: str = "") -> None:
        self.set_postits(path, self.postits(path) + [{"t": t, "name": name}])

    def change_postit(self, path: str, k: int, name: str | None = None, remove: bool = False) -> None:
        found = self.postits(path)
        if 0 <= k < len(found):
            if remove:
                found.pop(k)
            elif name is not None:
                found[k]["name"] = name.strip()
        self.set_postits(path, found)

    # ---------- learning your highlights (kept in the app's folder, for every trip) ----------

    def _meta_json(self, key: str, default):
        rows = self.query("SELECT value FROM meta WHERE key = ?", (key,))
        return json.loads(rows[0]["value"]) if rows and rows[0]["value"] else default

    def _set_meta_json(self, key: str, value) -> None:
        self.execute("INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)", (key, json.dumps(value)))

    def edit_stage(self) -> str:
        """"rough": you arrange the clips freely (each as long as you made it), nothing snaps to the music yet.
        "fitted": the cuts sit on your pins, and every change keeps them there (see reflow)."""
        return "fitted" if self._meta_json("edit_stage", "rough") == "fitted" else "rough"

    def set_edit_stage(self, stage: str) -> None:
        self._set_meta_json("edit_stage", "fitted" if stage == "fitted" else "rough")

    def fit_to_music(self, pins: list[float], photo_seconds: float = 4.0,
                     transition_seconds: float = 0.8, transition: str | None = None) -> dict:
        """The rough cut onto the music: every cut moves to the nearest pin it can reach (see reflow).
        Returns {"moved": cuts that moved, "gone": clips past the end of the music, "longer"/"shorter": clips}."""
        self.checkpoint("fitting to the music")
        before = {r["path"]: self.screen_length(r, photo_seconds) for r in self.storyboard()}
        gone = self.reflow(pins, None, photo_seconds, transition_seconds, transition)
        after = {r["path"]: self.screen_length(r, photo_seconds) for r in self.storyboard()}
        diffs = [after[p] - before[p] for p in after if p in before]
        self.set_edit_stage("fitted")
        return {"moved": sum(1 for d in diffs if abs(d) > 0.05), "gone": gone,
                "longer": sum(1 for d in diffs if d > 0.05), "shorter": sum(1 for d in diffs if d < -0.05)}

    @staticmethod
    def _frame_at(frames: dict, path: str, t: float) -> np.ndarray | None:
        times, vectors = frames.get(path, ([], None))
        if not len(times):
            return None
        return np.asarray(vectors[int(np.argmin([abs(x - t) for x in times]))], np.float32)

    def learn_highlights(self, notify: Callable[[float, str], None] | None = None) -> list[dict]:
        """What the app learned from your highlights (all trips), brought up to date with this trip's ⭐s:
        new ones are learned, ones you took off are forgotten."""
        trip = self.dir.name
        examples = highlights.load(app_dir())
        rows = [dict(r) for r in self.query(
            "SELECT * FROM items WHERE kind = 'video' AND postits IS NOT NULL AND removed = 0")]
        marks = {f"{trip}|{r['path']}|{p['t']:.1f}": (r, p["t"]) for r in rows for p in json.loads(r["postits"])}
        before = len(examples)
        examples = [e for e in examples if not (e["trip"] == trip and e["keep"] and e["key"] not in marks)]
        have = {e["key"] for e in examples}
        todo = {k: v for k, v in marks.items() if k not in have}
        if todo:
            frames = self.frame_vectors(list({r["path"]: r for r, _ in todo.values()}.values()), notify)
            for key, (r, t) in todo.items():
                vec = self._frame_at(frames, r["path"], t)
                if vec is not None:
                    examples.append(highlights.example(key, trip, True, vec, highlights.motion_at(
                        json.loads(r["curves"]) if r.get("curves") else None, t), r.get("theme") or r.get("theme_auto")))
        if todo or len(examples) != before:
            highlights.save(app_dir(), examples)
        return examples

    def highlights_learned(self) -> tuple[int, int]:
        """(highlights the app learns from, suggestions you dropped), all trips: this trip's ⭐s as they are now
        (learned when it next looks for highlights) and the ones of earlier trips."""
        examples = highlights.load(app_dir())
        here = sum(len(json.loads(r["postits"])) for r in self.query(
            "SELECT postits FROM items WHERE kind = 'video' AND postits IS NOT NULL AND removed = 0"))
        return (here + sum(1 for e in examples if e["keep"] and e["trip"] != self.dir.name),
                sum(1 for e in examples if not e["keep"]))

    def suggest_highlights(self, notify: Callable[[float, str], None] | None = None) -> list[dict]:
        """Seconds in your chosen videos without a ⭐ yet that look like your highlights (all trips):
        [{"path", "t", "score"}], kept until you keep or drop them."""
        examples = self.learn_highlights(notify)
        if highlights.counts(examples)[0] < highlights.START_FROM:
            self._set_meta_json("highlight_ideas", [])
            return []
        rows = [dict(r) for r in self.query(
            "SELECT * FROM items WHERE kind = 'video' AND chosen = 1 AND removed = 0 AND part_of IS NULL "
            "AND postits IS NULL")]
        frames = self.frame_vectors(rows, notify)
        for r in rows:  # only inside the part you kept of a clip you cut
            part = kept_part(r)
            if part and r["path"] in frames:
                times, vectors = frames[r["path"]]
                inside = [k for k, t in enumerate(times) if part[0] <= t <= part[1]]
                frames[r["path"]] = ([times[k] for k in inside], np.asarray(vectors)[inside] if inside else vectors[:0])
        motions = {r["path"]: json.loads(r["curves"]) if r.get("curves") else None for r in rows}
        ideas = highlights.suggest(examples, frames, motions, self._meta_json("highlight_drops", {}))
        self._set_meta_json("highlight_ideas", ideas)
        return ideas

    def highlight_ideas(self, path: str | None = None) -> list[dict]:
        """The app's suggested highlights still waiting for you (one video's, or all)."""
        ideas = self._meta_json("highlight_ideas", [])
        return [i for i in ideas if path is None or i["path"] == path]

    def _forget_idea(self, path: str, t: float) -> None:
        self._set_meta_json("highlight_ideas", [i for i in self.highlight_ideas()
                                                if not (i["path"] == path and abs(i["t"] - t) < 0.05)])

    def keep_idea(self, path: str, t: float, suggested: float | None = None) -> None:
        """A suggestion was right: it becomes a ⭐ highlight (and the app learns from it). suggested: the second
        the app suggested, when you put the highlight a little before or after it."""
        self._forget_idea(path, t if suggested is None else suggested)
        self.add_postit(path, t)
        try:
            self.learn_highlights()
        except Exception:  # (without the picture model now: it learns the next time it looks)
            traceback.print_exc()

    def drop_idea(self, path: str, t: float) -> None:
        """A suggestion was wrong: it goes, and the app learns what is not a highlight for you."""
        self._forget_idea(path, t)
        drops = self._meta_json("highlight_drops", {})
        drops.setdefault(path, []).append(round(t, 2))
        self._set_meta_json("highlight_drops", drops)
        rows = [dict(r) for r in self.query("SELECT * FROM items WHERE path = ?", (path,))]
        if not rows:
            return
        try:
            vec = self._frame_at(self.frame_vectors(rows), path, t)
        except Exception:
            traceback.print_exc()
            vec = None
        if vec is not None:
            examples = highlights.load(app_dir())
            examples.append(highlights.example(f"drop|{self.dir.name}|{path}|{t:.1f}", self.dir.name, False, vec,
                                               highlights.motion_at(json.loads(rows[0]["curves"]) if rows[0].get(
                                                   "curves") else None, t), rows[0].get("theme")))
            highlights.save(app_dir(), examples)

    def add_lock(self, path: str, start: float, end: float) -> None:
        self.set_locks(path, self.locks(path) + [[start, end]])

    def change_lock(self, path: str, k: int, start: float | None = None, end: float | None = None,
                    checkpoint: bool = True) -> None:
        """Move lock k to (start, end), or unlock it (no start)."""
        locks = self.locks(path)
        if 0 <= k < len(locks):
            if start is None:
                locks.pop(k)
            else:
                locks[k] = [start, end]
        self.set_locks(path, locks, checkpoint)

    def _storyline_pool(self, storyline: dict) -> list[dict]:
        """The shots for your storyline, each with its chapter; self._chapters gets the chapters for Auto-edit."""
        off = {r["key"] for r in self.query("SELECT key FROM story_info WHERE mine_enabled = 0")}
        stories = self.stories(self.items())
        rows = []
        for rank, story in enumerate(stories):
            if story["key"] in off:
                continue
            for r in story["items"]:
                if r["chosen"] and r["included"]:
                    rows.append(r | {"story_rank": rank, "story_name": story["name"],
                                     "theme": r["theme"] or r["theme_auto"]})
        repeats = self._repeats = self._same_moments(rows)
        rows = [r for r in rows if r["path"] not in repeats]
        saved = {(m["activity"], m["name"].lower()): m["shots"] for m in self.moments()}
        for part in (storyline.get("story") or {}).get("parts") or []:
            for e in part["entries"]:
                if e.get("pick") == "moment":
                    e["shots"] = saved.get((e["key"].partition("|")[0], (e.get("moment") or "").lower())) or []
        chapters, where = themes.chapters_for(storyline, rows)
        self._chapters = chapters
        pool = []
        self._moment_starts = {}
        for r, chapter in zip(rows, where):
            if chapter is None:
                continue
            if r.get("moment_at") is not None and r["kind"] == "video" and r["duration"] and not r.get("locks"):
                # a moment you picked (a big jump...): only the seconds around it, inside the part you kept
                start, end = kept_part(r) or (0.0, r["duration"])
                a, b = moments.window(r["moment_at"], r["duration"], start, end)
                self._moment_starts[r["path"]] = a
                r = r | {"duration": b - a, "keep_offset": a, "full_duration": r["duration"]}
            elif kept_part(r):  # a clip you cut: Auto-edit only sees the part you kept
                start, end = kept_part(r)
                r = r | {"duration": end - start, "keep_offset": start, "full_duration": r["duration"]}
            pool.append(r | {"chapter": chapter, "pace_want": chapters[chapter].get("pace")})
        return self._with_locks(pool)

    # ---------- themes ----------

    def activities(self) -> list[dict]:
        """The activities a shot can have (the app's own and the ones you added), with their CLIP vectors.
        A new trip starts with the ones of your last trip (your own, your new names)."""
        rows = self.query("SELECT value FROM meta WHERE key = 'activities'")
        if rows:
            return json.loads(rows[0]["value"])
        return _learned_themes().get("activities") or themes.default_activities()

    def save_activities(self, activities: list[dict]) -> None:
        self.execute("INSERT OR REPLACE INTO meta (key, value) VALUES ('activities', ?)", (json.dumps(activities),))
        self._remember_themes()

    LEARNED_TRIPS = 12  # earlier trips whose confirmed shots teach a new trip's themes
    LEARNED_PER_THEME = 40

    def _remember_themes(self) -> None:
        """Keep this trip's activities and a few shots of each you confirmed in the app (not only in the trip), so
        the next trip starts with them and already recognises your kitesurfing."""
        rows = self.query("SELECT clip, theme FROM items WHERE theme IS NOT NULL AND clip IS NOT NULL "
                          "AND removed = 0 AND part_of IS NULL ORDER BY taken_at DESC")
        shots: dict[str, list[str]] = {}
        for r in rows:
            if r["theme"] != themes.OTHER and len(shots.setdefault(r["theme"], [])) < self.LEARNED_PER_THEME:
                vector = np.frombuffer(r["clip"], dtype=np.float32).astype(np.float16)
                shots[r["theme"]].append(base64.b64encode(vector.tobytes()).decode())
        learned = _learned_themes()
        trips = {k: v for k, v in (learned.get("trips") or {}).items() if k != self.dir.name}
        if shots:
            trips[self.dir.name] = shots  # the latest trip last; the oldest go first
        rows = self.query("SELECT value FROM meta WHERE key = 'activities'")
        learned |= {"activities": json.loads(rows[0]["value"]) if rows else learned.get("activities"),
                    "trips": dict(list(trips.items())[-self.LEARNED_TRIPS:])}
        _save_learned(learned)

    def earlier_trips_learned(self) -> int:
        return sum(1 for k in (_learned_themes().get("trips") or {}) if k != self.dir.name)

    def _earlier_examples(self, activities: list[dict]) -> list[tuple[np.ndarray, str]]:
        """(vector, activity) for the shots you confirmed in other trips, under this trip's names."""
        names = {a["name"] for a in activities}
        renamed = {a["was"]: a["name"] for a in activities if a.get("was")}
        out = []
        for trip, shots in (_learned_themes().get("trips") or {}).items():
            if trip == self.dir.name:
                continue
            for name, vectors in shots.items():
                name = name if name in names else renamed.get(name)
                if name:
                    out += [(np.frombuffer(base64.b64decode(v), dtype=np.float16).astype(np.float32), name)
                            for v in vectors]
        return out

    # ---------- a new trip's own themes (themefinder) ----------

    def missing_fingerprints(self) -> int:
        """Chosen shots the picture model hasn't looked at yet (Analyse does it)."""
        return self.query("SELECT COUNT(*) AS n FROM items WHERE chosen = 1 AND removed = 0 AND part_of IS NULL "
                          "AND clip IS NULL")[0]["n"]

    def propose_themes(self, sentence_vectors: np.ndarray, ai=None,
                       notify: Callable[[float, str], None] | None = None) -> list[dict]:
        """Themes for this trip from its own shots (see themefinder), kept in meta 'theme_proposals' until you
        answer: [{"name", "emoji", "describe", "paths", "sample", "subs": [{"name", "paths"}], "from"}]."""
        from . import themefinder

        rows = [dict(r) for r in self.query(
            "SELECT path, clip, kind, duration, keep_start, keep_end, postits, ai_caption, place, taken_at FROM items "
            "WHERE chosen = 1 AND removed = 0 AND part_of IS NULL AND clip IS NOT NULL ORDER BY taken_at, path")]
        if not rows:
            return []
        (notify or (lambda f, m: None))(0.2, "grouping the shots that look alike")
        vectors = np.array([np.frombuffer(r["clip"], dtype=np.float32) for r in rows])
        marks = {a["name"]: a.get("emoji") or "⭐" for a in (_learned_themes().get("activities") or [])}
        earlier: dict[str, tuple[str, list[np.ndarray]]] = {}
        for trip, shots in (_learned_themes().get("trips") or {}).items():
            if trip == self.dir.name:
                continue
            for name, found in shots.items():
                if name != themes.OTHER:
                    earlier.setdefault(name, (marks.get(name, "⭐"), []))[1].extend(
                        np.frombuffer(base64.b64decode(v), dtype=np.float16).astype(np.float32) for v in found)
        found = themefinder.propose(vectors, sentence_vectors, earlier)
        if ai is not None and found:
            (notify or (lambda f, m: None))(0.6, "asking the AI for better names")
            facts = []
            for p in found:
                members = [rows[i] for i in p["sample"]]
                days = sorted({r["taken_at"][:10] for r in (rows[i] for i in p["rows"])})
                facts.append({"captions": [r["ai_caption"] for r in members if r.get("ai_caption")],
                              "places": sorted({r["place"] for r in members if r.get("place")}),
                              "days": f"{days[0]} to {days[-1]}" if len(days) > 1 else (days[0] if days else "")})
            try:
                from .director import name_themes
                found = themefinder.apply_names(found, name_themes(ai, themefinder.names_prompt(found, facts)))
            except Exception as exc:  # (the picture model's names stay)
                print("Themes: the AI could not name them:", exc)
        out = [{"name": p["name"], "emoji": p["emoji"], "describe": p["describe"], "from": p["from"],
                "paths": [rows[i]["path"] for i in p["rows"]], "sample": [rows[i]["path"] for i in p["sample"]],
                "subs": [{"name": s["name"], "paths": [rows[i]["path"] for i in s["rows"]]} for s in p["subs"]]}
               for p in found]
        self._set_meta_json("theme_proposals", out)
        return out

    def theme_proposals(self) -> list[dict]:
        return self._meta_json("theme_proposals", [])

    def drop_theme_proposals(self) -> None:
        self._set_meta_json("theme_proposals", [])

    def move_in_proposals(self, path: str, to: int) -> None:
        """You moved a shot into another proposed theme (before using them): it is confirmed there."""
        proposals = self.theme_proposals()
        if not 0 <= to < len(proposals):
            return
        for p in proposals:
            for key in ("paths", "sample", "moved_in"):
                if key in p:
                    p[key] = [q for q in p[key] if q != path]
            for sub in p.get("subs") or []:
                sub["paths"] = [q for q in sub["paths"] if q != path]
        target = proposals[to]
        target["paths"] = [path, *target["paths"]]
        target["sample"] = [path, *target.get("sample", [])]
        target["moved_in"] = [path, *target.get("moved_in", [])]
        self._set_meta_json("theme_proposals", proposals)

    CORE_SHARE = 0.5  # the shots of a proposed theme most like the rest of it count as confirmed

    def use_theme_proposals(self, chosen: list[dict]) -> None:
        """The themes you kept become this trip's themes (instead of the last trip's): the clearest shots of each
        are confirmed, the app suggests the rest from them, sub-themes become moments.
        chosen: proposals (renamed, merged) with "vector" (what the name looks like to CLIP, or None)."""
        self.checkpoint("using the suggested themes")
        activities = [{"name": p["name"], "emoji": p.get("emoji") or "⭐", "describe": p.get("describe") or
                       [p["name"].lower()], "vector": p.get("vector")} for p in chosen]
        activities.append({"name": themes.OTHER, "emoji": "❔", "describe": [], "vector": None})
        names = {a["name"] for a in activities}
        clips = {r["path"]: np.frombuffer(r["clip"], dtype=np.float32) for r in self.query(
            "SELECT path, clip FROM items WHERE clip IS NOT NULL AND part_of IS NULL")}
        sure = []
        for p in chosen:
            have = [q for q in p["paths"] if q in clips]
            if not have:
                continue
            x = np.array([clips[q] for q in have])
            centre = x.mean(axis=0)
            order = np.argsort(-(x @ centre))
            sure += [(p["name"], have[i]) for i in order[:max(1, int(round(len(have) * self.CORE_SHARE)))]]
        for p in chosen:  # the shots you moved yourself are confirmed where you put them
            sure += [(p["name"], q) for q in p.get("moved_in") or []]
        with self._lock:
            self.db.execute("UPDATE items SET theme = NULL WHERE theme IS NOT NULL AND theme NOT IN (%s)"
                            % ",".join("?" * len(names)), tuple(names))
            self.db.executemany("UPDATE items SET theme = ? WHERE path = ?", sure)
            self.db.commit()
        self.save_activities(activities)
        self._save_moments([m for m in self.moments() if m["activity"] in names])
        rows = {r["path"]: dict(r) for r in self.query(
            "SELECT path, kind, duration, keep_start, keep_end, postits FROM items WHERE part_of IS NULL")}
        for p in chosen:
            for sub in p.get("subs") or []:
                self.add_moment(p["name"], sub["name"])
                for q in sub["paths"]:
                    r = rows.get(q)
                    if not r:
                        continue
                    t = 0.0
                    if r["kind"] == "video":
                        notes = json.loads(r["postits"]) if r.get("postits") else []
                        start, end = kept_part(r) or (0.0, r["duration"] or 0.0)
                        t = notes[0]["t"] if notes else round((start + end) / 2, 2)
                    self.tick_moment(p["name"], sub["name"], q, t, True)
        self.drop_theme_proposals()
        self._set_meta_json("own_themes", True)
        self.suggest_themes()

    def theme_rows(self) -> list[dict]:
        """Your selected shots with their theme: activity (yours if you confirmed it, else the app's suggestion), who,
        where and when."""
        rows = [r for r in self.items() if r["chosen"]]
        for r in rows:
            r["theme_ok"] = r["theme"] is not None
            r["theme"] = r["theme"] or r["theme_auto"] or themes.OTHER
            r["day_part"] = themes.day_part(r["taken_at"])
        return rows

    def has_fingerprints(self) -> bool:
        return bool(self.query("SELECT 1 FROM items WHERE clip IS NOT NULL LIMIT 1"))

    def suggest_themes(self) -> None:
        """Suggest an activity for every selected shot you haven't confirmed, learning from the ones you have."""
        rows = self.query("SELECT path, clip, theme, tags, tags_user, scene, scene_user FROM items "
                          "WHERE chosen = 1 AND removed = 0 AND part_of IS NULL ORDER BY taken_at, path")
        activities = self.activities()
        names = {a["name"] for a in activities}
        renamed = {a["was"]: a["name"] for a in activities if a.get("was")}
        seen = [r for r in rows if r["clip"] is not None]
        vectors = np.array([np.frombuffer(r["clip"], dtype=np.float32) for r in seen]) if seen else np.zeros((0, 512))
        confirmed = [(i, r["theme"]) for i, r in enumerate(seen) if r["theme"] in names]
        guesses = {r["path"]: g for r, g in zip(seen, themes.suggest(vectors, activities, confirmed,
                                                                     self._earlier_examples(activities)))}
        updates = []
        for r in rows:
            if r["path"] in guesses:
                name, sure = guesses[r["path"]]
            else:  # no picture fingerprint yet: a first guess from its tags
                tags = json.loads(r["tags_user"]) if r["tags_user"] is not None else json.loads(r["tags"] or "[]")
                name, sure = themes.by_tags(tags, r["scene_user"] or r["scene"]), 0.3
                name = renamed.get(name, name) if renamed.get(name, name) in names else themes.OTHER
            updates.append((name, round(sure, 3), r["path"]))
        with self._lock:
            self.db.executemany("UPDATE items SET theme_auto = ?, theme_sure = ? WHERE path = ?", updates)
            self.db.commit()

    def theme_check(self) -> dict:
        """The second check, run now: {"agree" (0..1), "shots": [{"path", "now", "better", "why", "yours"}],
        "total"}. The shots you answered "keep mine" for are not asked again."""
        rows = [r for r in self.theme_rows() if not r.get("part_of") and not r.get("removed")]
        clips = {r["path"]: r["clip"] for r in self.query("SELECT path, clip FROM items WHERE clip IS NOT NULL")}
        seen = [r | {"clip": clips[r["path"]]} for r in rows if r["path"] in clips]
        keep = set(self._meta_json("theme_check_keep", []))
        if not seen:
            result = {"agree": None, "shots": [], "total": len(rows)}
        else:
            vectors = np.array([np.frombuffer(r["clip"], dtype=np.float32) for r in seen])
            found = themes.second_check(vectors, self.activities(), [r["theme"] for r in seen],
                                        [bool(r["theme_ok"]) for r in seen],
                                        {i for i, r in enumerate(seen) if r["path"] in keep})
            shots = [{"path": seen[f["row"]]["path"], "now": f["now"], "better": f["better"], "why": f["why"],
                      "yours": bool(seen[f["row"]]["theme_ok"])} for f in found]
            result = {"agree": round(1 - len(shots) / len(seen), 3), "shots": shots, "total": len(seen)}
        self.execute("INSERT OR REPLACE INTO meta (key, value) VALUES ('theme_check', ?)", (json.dumps(result),))
        return result

    def last_theme_check(self) -> dict | None:
        return self._meta_json("theme_check", None)

    def answer_theme_check(self, path: str, move: bool) -> None:
        """Move it (to the theme the check suggests: confirmed) or keep mine (confirmed as it is, not asked again)."""
        check = self.last_theme_check() or {"shots": []}
        shot = next((x for x in check["shots"] if x["path"] == path), None)
        if not shot:
            return
        check["shots"] = [x for x in check["shots"] if x["path"] != path]
        if move:
            check["agree"] = round(min(1.0, (check.get("agree") or 0) + 1 / max(1, check.get("total") or 1)), 3)
        else:
            keep = self._meta_json("theme_check_keep", [])
            self.execute("INSERT OR REPLACE INTO meta (key, value) VALUES ('theme_check_keep', ?)",
                         (json.dumps(sorted({*keep, path})),))
        self.execute("INSERT OR REPLACE INTO meta (key, value) VALUES ('theme_check', ?)", (json.dumps(check),))
        self.set_theme([path], shot["better"] if move else shot["now"])

    def set_theme(self, paths: list[str], activity: str | None) -> None:
        """Confirm these shots' activity (activity None: the one suggested now), then suggest the rest again."""
        with self._lock:
            if activity is None:
                self.db.executemany("UPDATE items SET theme = COALESCE(theme, theme_auto, ?) WHERE path = ?",
                                    [(themes.OTHER, p) for p in paths])
            else:
                self.db.executemany("UPDATE items SET theme = ? WHERE path = ?", [(activity, p) for p in paths])
            self.db.commit()
        self.suggest_themes()
        self._remember_themes()

    def unconfirm_theme(self, paths: list[str]) -> None:
        with self._lock:
            self.db.executemany("UPDATE items SET theme = NULL WHERE path = ?", [(p,) for p in paths])
            self.db.commit()
        self.suggest_themes()
        self._remember_themes()

    def add_activity(self, name: str, vector: list[float] | None) -> None:
        activities = self.activities()
        if any(a["name"].lower() == name.lower() for a in activities):
            return
        new = themes.new_activity(name) | {"vector": vector}
        at = next((k for k, a in enumerate(activities) if a["name"] == themes.OTHER), len(activities))
        activities.insert(at, new)
        self.save_activities(activities)
        self.suggest_themes()

    def rename_activity(self, old: str, new: str) -> bool:
        """Give an activity another name ("Water sports" -> "Kite action"): its shots, the ones you confirmed and the
        storyline go along. False if the name is taken."""
        activities = self.activities()
        if old == themes.OTHER or not new or any(a["name"].lower() == new.lower() and a["name"] != old
                                                 for a in activities):
            return False
        for a in activities:
            if a["name"] == old:
                a["was"] = a.get("was") or old  # the app's first guesses from tags still find it
                a["name"] = new
        self.save_activities(activities)
        self._save_moments([m | {"activity": new} if m["activity"] == old else m for m in self.moments()])
        with self._lock:
            self.db.execute("UPDATE items SET theme = ? WHERE theme = ?", (new, old))
            self.db.execute("UPDATE items SET theme_auto = ? WHERE theme_auto = ?", (new, old))
            self.db.commit()
        line = self.storyline()

        def key(k: str) -> str:
            activity, _, person = k.partition("|")
            return themes.block_key(new, person or None) if activity == old else k

        blocks = [b | {"key": key(b["key"]), "activity": new if b["activity"] == old else b["activity"]}
                  for b in line["blocks"]]
        split = [new if a == old else a for a in line["split"]]
        story = line.get("story") or {}
        if story.get("parts"):
            story = story | {"parts": [p | {"entries": [e | {"key": key(e["key"])} for e in p["entries"]]}
                                       for p in story["parts"]]}
        self.set_storyline(blocks=blocks, split=split, **({"story": story} if story else {}))
        return True

    # ---------- moments ----------

    def frame_vectors(self, rows: list[dict], notify: Callable[[float, str], None] | None = None
                      ) -> dict[str, tuple[list[float], np.ndarray]]:
        """For each shot: the times looked at and a CLIP vector per frame (a frame every second inside a video,
        made once and kept; a photo is its own picture)."""
        out: dict[str, tuple[list[float], np.ndarray]] = {}
        todo = []
        for r in rows:
            if r["kind"] != "video":
                if r.get("clip") is not None:
                    out[r["path"]] = ([0.0], np.frombuffer(r["clip"], dtype=np.float32)[None, :])
                continue
            saved = self.query("SELECT t, vec FROM frames WHERE path = ? ORDER BY t", (r["path"],))
            if saved:
                out[r["path"]] = ([f["t"] for f in saved],
                                  np.array([np.frombuffer(f["vec"], dtype=np.float32) for f in saved]))
            else:
                todo.append(r)
        if todo and self.frame_embedder is None:
            if notify:
                notify(0.0, "loading the picture model")
            tagger = SceneTagger(app_dir() / "models" / "clip")
            self.frame_embedder = tagger.embed_each
        for k, r in enumerate(todo):
            if notify:
                notify(k / len(todo), f"video {k + 1} of {len(todo)}")
            with tempfile.TemporaryDirectory() as tmp:
                pictures = _video_frames(r, Path(tmp))
                if not pictures:
                    continue
                vectors = np.asarray(self.frame_embedder([p for _, p in pictures]), np.float32)
            with self._lock:
                self.db.executemany("INSERT OR REPLACE INTO frames (path, t, vec) VALUES (?,?,?)",
                                    [(r["path"], t, v.tobytes()) for (t, _), v in zip(pictures, vectors)])
                self.db.commit()
            out[r["path"]] = ([t for t, _ in pictures], vectors)
        if notify and todo:
            notify(1.0, "comparing")
        return out

    def moments(self) -> list[dict]:
        """Your moments, each inside a theme: {"name", "activity", "shots": [{"path", "t"}] you ticked}."""
        rows = self.query("SELECT value FROM meta WHERE key = 'moments'")
        return json.loads(rows[0]["value"]) if rows and rows[0]["value"] else []

    def _save_moments(self, found: list[dict]) -> None:
        self.execute("INSERT OR REPLACE INTO meta (key, value) VALUES ('moments', ?)", (json.dumps(found),))
        learned = _learned_themes()  # the names are offered again as you type, in this trip and the next ones
        names = [m for m in learned.get("moments") or []
                 if not any(f["name"].lower() == m["name"].lower() and f["activity"] == m["activity"] for f in found)]
        _save_learned(learned | {"moments": (names + [{"name": f["name"], "activity": f["activity"]}
                                                      for f in found])[-200:]})

    def moment_names(self, activity: str | None = None) -> list[str]:
        """Moment names you used before (this trip's and earlier trips'), this activity's first: offered as you type."""
        seen = self.moments() + [m for m in reversed(_learned_themes().get("moments") or [])]
        first = [m["name"] for m in seen if m["activity"] == activity]
        out: list[str] = []
        for name in first + [m["name"] for m in seen]:
            if name.lower() not in {n.lower() for n in out}:
                out.append(name)
        return out

    def moment(self, activity: str, name: str) -> dict | None:
        return next((m for m in self.moments() if m["activity"] == activity and m["name"].lower() == name.lower()),
                    None)

    def add_moment(self, activity: str, name: str) -> dict:
        name = " ".join(name.split())[:40]
        found = self.moment(activity, name)
        if found:
            return found
        new = {"name": name, "activity": activity, "shots": []}
        self._save_moments(self.moments() + [new])
        return new

    def remove_moment(self, activity: str, name: str) -> None:
        self._save_moments([m for m in self.moments() if not (m["activity"] == activity and m["name"] == name)])

    # ---------- theme folders: drag a sub-theme into another theme, out to a theme of its own, or a theme in ----------

    def move_moment(self, activity: str, name: str, to: str) -> None:
        """A sub-theme moves under another theme: its ticked shots go along (confirmed in that theme)."""
        found = self.moments()
        mo = next((m for m in found if m["activity"] == activity and m["name"] == name), None)
        if mo is None or activity == to:
            return
        found.remove(mo)
        same = next((m for m in found if m["activity"] == to and m["name"].lower() == name.lower()), None)
        if same is not None:  # one of that name there already: they become one
            have = {s["path"] for s in same["shots"]}
            same["shots"] += [s for s in mo["shots"] if s["path"] not in have]
        else:
            found.append(mo | {"activity": to})
        self.checkpoint("moving a sub-theme")
        self._save_moments(found)
        self.set_theme([s["path"] for s in mo["shots"]], to)

    def moment_to_theme(self, activity: str, name: str, vector: list[float] | None = None) -> str:
        """A sub-theme becomes a theme of its own (dragged out): its ticked shots move to it. Returns its name."""
        mo = self.moment(activity, name)
        if mo is None:
            return ""
        taken = {a["name"].lower(): a["name"] for a in self.activities()}
        new = taken.get(mo["name"].lower(), mo["name"])
        self.checkpoint("making a sub-theme a theme")
        if new.lower() not in taken:
            self.add_activity(new, vector)
        self._save_moments([m for m in self.moments() if m is not None and
                            not (m["activity"] == activity and m["name"] == mo["name"])])
        self.set_theme([s["path"] for s in mo["shots"]], new)
        return new

    def theme_to_moment(self, activity: str, to: str) -> None:
        """A theme dropped into another theme: it becomes a sub-theme there. All its shots move to that theme and
        are ticked in the sub-theme (a video at the second you highlighted, else its middle)."""
        if activity in (to, themes.OTHER) or not any(a["name"] == to for a in self.activities()):
            return
        rows = [r for r in self.theme_rows() if r["theme"] == activity]
        shots = []
        for r in rows:
            if r["kind"] == "video":
                a, b = kept_part(r) or (0.0, r["duration"] or 0.0)
                marks = [p["t"] for p in r.get("postits") or [] if a <= p["t"] <= b]
                t = marks[0] if marks else (a + b) / 2
            else:
                t = 0.0
            shots.append({"path": r["path"], "t": round(float(t), 2)})
        self.checkpoint("putting a theme inside another")
        found = [m | ({"activity": to} if m["activity"] == activity else {}) for m in self.moments()]
        same = next((m for m in found if m["activity"] == to and m["name"].lower() == activity.lower()), None)
        if same is not None:
            have = {s["path"] for s in same["shots"]}
            same["shots"] += [s for s in shots if s["path"] not in have]
        else:
            found.append({"name": activity, "activity": to, "shots": shots})
        self._save_moments(found)
        with self._lock:  # its shots go first, so taking the theme away doesn't send them anywhere else
            self.db.executemany("UPDATE items SET theme = ? WHERE path = ?", [(to, r["path"]) for r in rows])
            self.db.commit()
        self.save_activities([a for a in self.activities() if a["name"] != activity])
        self.suggest_themes()
        self._remember_themes()

    def tick_moment(self, activity: str, name: str, path: str, t: float, on: bool) -> None:
        """You marked a shot (at that second) as this moment, or took it back."""
        found = self.moments()
        for m in found:
            if m["activity"] == activity and m["name"] == name:
                m["shots"] = [s for s in m["shots"] if s["path"] != path] + ([{"path": path, "t": t}] if on else [])
        self._save_moments(found)
        if on:
            self.confirm_moment_shots()

    def confirm_moment_shots(self) -> int:
        """A shot you ticked in a moment (e.g. a deer under Deer and fish) is confirmed in that theme too: no
        question mark for it there any more. Returns how many shots it confirmed now."""
        sure = [(m["activity"], s["path"]) for m in self.moments() for s in m.get("shots") or []]
        if not sure:
            return 0
        with self._lock:
            before = self.db.total_changes
            self.db.executemany("UPDATE items SET theme = ? WHERE path = ? AND theme IS NULL "
                                "AND COALESCE(theme_auto, '') = ?", [(a, p, a) for a, p in sure])
            self.db.commit()
            done = self.db.total_changes - before
        if done:
            self._remember_themes()
        return done

    def moment_matches(self, activity: str, name: str, words: np.ndarray | None,
                       notify: Callable[[float, str], None] | None = None) -> list[dict]:
        """The shots of a theme for one of its moments: the ones you ticked first (at their second), then the rest,
        most like your words and your ticks first."""
        saved = self.moment(activity, name) or {"shots": []}
        rows = [r for r in self.theme_rows() if r["theme"] == activity]  # the same shots as the theme shows
        found = self.find_moment(rows, words, saved["shots"], notify)
        ticked = {s["path"]: s["t"] for s in saved["shots"]}
        for f in found:
            f["ticked"] = f["path"] in ticked
            if f["ticked"] and abs(f["t"] - ticked[f["path"]]) > 0.01:
                f["t"] = ticked[f["path"]]
                f["thumb"] = self.moment_picture(next(r for r in rows if r["path"] == f["path"]), f["t"]) \
                    if f["kind"] == "video" else f["thumb"]
        return sorted(found, key=lambda f: (not f["ticked"], -f["score"]))

    def find_moment(self, rows: list[dict], words: np.ndarray | None, ticked: list[dict] | None = None,
                    notify: Callable[[float, str], None] | None = None) -> list[dict]:
        """These shots, best match for the moment first: {"path", "t", "score", "likely", "thumb"}.
        ticked: [{"path", "t"}] you marked as right (the app looks for more like them)."""
        shots = self.frame_vectors(rows, notify)
        examples = []
        for tick in ticked or []:
            times, vectors = shots.get(tick["path"], ([], None))
            if times:
                examples.append(vectors[int(np.argmin([abs(t - tick["t"]) for t in times]))])
        found = moments.find(words, shots, examples)
        good = moments.likely(found) | {t["path"] for t in ticked or []}
        kinds = {r["path"]: r for r in rows}
        out = []
        for path, t, score in found:
            r = kinds[path]
            out.append({"path": path, "t": t, "score": round(score, 3), "likely": path in good,
                        "thumb": self.moment_picture(r, t) if r["kind"] == "video" else r["thumb"],
                        "kind": r["kind"], "duration": r["duration"], "row": r})
        return out

    def moment_picture(self, row: dict, t: float) -> str | None:
        """A small picture of the clip at that second (kept in the trip folder)."""
        folder = self.dir / "moments"
        folder.mkdir(exist_ok=True)
        out = folder / f"{hashlib.sha1(row['path'].encode()).hexdigest()[:12]}_{t:.1f}.jpg"
        if not out.exists():
            subprocess.run([ffmpeg_paths()[0], "-loglevel", "error", "-y", "-ss", f"{t:.2f}", "-i", row["path"],
                            "-frames:v", "1", "-vf", rotate_filter(row.get("rotation") or 0) + "scale=320:-2",
                            "-q:v", "4", str(out)], capture_output=True)
        return str(out) if out.exists() else row.get("thumb")

    def remove_activity(self, name: str) -> None:
        """Take an activity away: its shots get a new suggestion (the ones you confirmed for it too)."""
        if name == themes.OTHER:
            return
        self.save_activities([a for a in self.activities() if a["name"] != name])
        self.execute("UPDATE items SET theme = NULL WHERE theme = ?", (name,))
        self.suggest_themes()

    # ---------- the storyline ----------

    def storyline(self) -> dict:
        """How the video is told: {"order": time|theme|day, "blocks": [{"key", "activity", "person", "on"}],
        "split": activities shown per person, "finale": bool}. As it happened, with a finale, until you change it."""
        rows = self.query("SELECT value FROM meta WHERE key = 'storyline'")
        saved = json.loads(rows[0]["value"]) if rows and rows[0]["value"] else {}
        line = {"order": themes.TIME, "blocks": [], "split": [], "finale": True, "by_day": False,
                "day_titles": False} | saved
        if line["order"] == themes.DAY:  # (Day by day is now a switch of As it happened)
            line |= {"order": themes.TIME, "by_day": True}
        return line

    def set_storyline(self, **changes) -> None:
        value = self.storyline() | changes
        self.execute("INSERT OR REPLACE INTO meta (key, value) VALUES ('storyline', ?)", (json.dumps(value),))

    def storyline_blocks(self) -> list[dict]:
        """The theme blocks of your selection, in your order, each with how many shots it has."""
        line = self.storyline()
        rows = [r for r in self.theme_rows() if r["included"]]
        return themes.blocks_for(rows, line["blocks"], set(line["split"]))

    @staticmethod
    def _same_moments(rows: list[dict]) -> set[str]:
        """Files that show a moment another file already shows, so the video never shows it twice: a Live Photo's
        little video next to its photo, and videos filmed at the same time (a copy, an edited version, one
        recording saved twice). The longer video (or the photo) stays; the others are left out of Auto-edit."""
        out: set[str] = set()
        by_name: dict[tuple[str, str], list[dict]] = {}
        for r in rows:
            p = Path(r["path"])
            by_name.setdefault((str(p.parent), p.stem.lower()), []).append(r)
        for same in by_name.values():  # IMG_1234.HEIC and IMG_1234.MOV: one Live Photo
            if any(r["kind"] == "photo" for r in same):
                out |= {r["path"] for r in same if r["kind"] == "video" and (r["duration"] or 0) <= 4.0}
        videos = sorted((r for r in rows if r["kind"] == "video" and r["path"] not in out),
                        key=lambda r: (-(r["duration"] or 0), r["path"]))
        seen: list[tuple[datetime, datetime, dict]] = []
        for r in videos:  # the longest first: a shorter one filmed during it, looking the same, is a repeat
            start = datetime.fromisoformat(r["taken_at"])
            end = start + timedelta(seconds=r["duration"] or 0)
            for a, b, other in seen:
                during = start < b - timedelta(seconds=0.5) and a + timedelta(seconds=0.5) < end
                alike = (r["similar_set"] is not None and r["similar_set"] == other["similar_set"]) or (
                    abs((start - a).total_seconds()) < 2 and abs((r["duration"] or 0) - (other["duration"] or 0)) < 1)
                if during and alike:  # (someone else filming the same moment from elsewhere looks different: stays)
                    out.add(r["path"])
                    break
            seen.append((start, end, r))  # a repeat of a repeat is a repeat too
        return out

    @staticmethod
    def _mark_upright(rows: list[dict]) -> None:
        """Which shots stand upright (taller than wide, as shown: your turn counts), from their preview (made with
        your turn already applied, so the picture's own shape is what the video shows)."""
        for r in rows:
            try:
                with Image.open(r["thumb"]) as img:
                    w, h = img.size
            except (OSError, TypeError, KeyError):
                r["upright"] = False
                continue
            r["upright"] = h > w * 1.1

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
        self._mark_upright(rows)
        limits = replace(limits, landscape=self.video_size() != "phone", **self.card_rooms())
        scores = self._video_scores(rows, report)
        report(0.95, "Listening to the music")
        song = songmap.analyse(song_path, app_dir() / "music-maps")
        song.taps = taps.final_taps(song_path, app_dir() / "music-taps")  # your own cuts, if you tapped this song
        mine = list(song.taps or [])
        # The calm and lively stretches (yours, if you changed them) and the song's style.
        music = musicmood.reading(song, song_path, app_dir() / "music-taps")
        musicmood.apply(song, music["stretches"])
        gap = musicmood.STYLES[music["style"]]["pin_gap"]
        if gap and song.taps:
            song.taps = musicmood.thin(song.taps, gap)  # a calm style cuts only on some of your pins
        result = autoedit.plan_to_music(rows, scores, song, limits, target, transition_seconds, soft_transition,
                                        self._chapters)
        result.outside = self._never_offered(rows)
        if mine:  # too few shots for all your pins: remember which ones this cut used (Music & audio greys the rest)
            kept = result.kept_pins or (song.taps if len(song.taps or []) < len(mine) else None)
            taps.set_extra(song_path, app_dir() / "music-taps", "used", {"pins": mine, "kept": kept} if kept else None)
        self._apply_plan(result)
        report(1.0, "Done")
        return result

    def clear_effects(self, path: str) -> None:
        """Take the punch-in, speed ramp, push-in, shake or freeze-frame off a shot (the same part of the video
        stays in use)."""
        row = self.query("SELECT effects, trim_start, trim_end, speed FROM items WHERE path = ?", (path,))
        if row and row[0]["effects"] and json.loads(row[0]["effects"]).get("collage"):
            self.split_collage(path)
            return
        if row and row[0]["effects"] and held(json.loads(row[0]["effects"])) and row[0]["trim_end"] is not None:
            # The freeze-frame's time goes back to the video playing, so the shot stays just as long.
            more = held(json.loads(row[0]["effects"])) * (row[0]["speed"] or 1.0)
            self.execute("UPDATE items SET trim_end = ? WHERE path = ?", (row[0]["trim_end"] + more, path))
        if row and row[0]["effects"] and json.loads(row[0]["effects"]).get("ramp") and row[0]["trim_end"] is not None:
            # A ramp used 40% more video for the same time on screen: keep the shot just as long.
            start, end = row[0]["trim_start"], row[0]["trim_end"]
            self.execute("UPDATE items SET trim_end = ? WHERE path = ?", (start + (end - start) / RAMP_SOURCE, path))
        self.execute("UPDATE items SET effects = NULL, effects_why = NULL WHERE path = ?", (path,))

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
        # Videos measured by older versions lack the sharpness and camera curves: measure those again.
        for n, r in enumerate(v for v in videos if not v["curves"] or '"sharp"' not in v["curves"]):
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
                n = max(len(c["action"]) + 1, int((r.get("full_duration") or r["duration"] or 0) * motion.FPS))
                score = autoedit.interest(c["action"], c["shake"], c["loud"], n, c.get("sharp"),
                                          c.get("camera"), c.get("bright"))
                if r.get("keep_offset") is not None:  # only the part you kept
                    first = int(round(r["keep_offset"] * motion.FPS))
                    score = score[first:first + max(1, int(round(r["duration"] * motion.FPS)))]
                scores[r["path"]] = score
        return scores

    def _never_offered(self, rows: list[dict]) -> list[str]:
        """Selected files Auto-edit did not get to choose from: repeats of another file, and (with chapters) shots
        in no chapter. They stay selected, but are out of this cut."""
        seen = {r["path"] for r in rows}
        if not self._chapters:  # without a storyline: only the repeats (switched-off stories keep their place)
            seen |= {r["path"] for r in self.items()} - self._repeats
        return [r["path"] for r in self.query("SELECT path FROM items WHERE chosen = 1 AND included = 1 AND removed = 0 "
                                              "AND part_of IS NULL ORDER BY taken_at") if r["path"] not in seen]

    def _apply_plan(self, result: autoedit.Plan) -> None:
        self.checkpoint("the first cut")
        offsets = {r["path"]: r["keep_start"] for r in self.query(
            "SELECT path, keep_start FROM items WHERE keep_start IS NOT NULL AND keep_end IS NOT NULL")}
        offsets |= self._moment_starts  # a moment's part (from the clip's start)
        with self._lock:
            # Parts of long clips: made afresh for each cut, as rows of their own copied from the clip.
            self.db.execute("DELETE FROM items WHERE part_of IS NOT NULL")
            columns = [r["name"] for r in self.db.execute("PRAGMA table_info(items)")]
            for path, (clip, _) in result.parts.items():
                if path == clip:
                    continue
                row = dict(self.db.execute("SELECT * FROM items WHERE path = ?", (clip,)).fetchone())
                row.update(path=path, part_of=clip, position=None, title_text=None, title_sub=None)
                self.db.execute(f"INSERT INTO items ({', '.join(columns)}) VALUES ({', '.join('?' * len(columns))})",
                                [row[c] for c in columns])
            for shot_no, shot in enumerate(result.shots):
                used = (shot.seconds - held(shot.effects)) * shot.speed * source_factor(shot.effects)  # of the video
                clip, into = result.parts.get(shot.path, (shot.path, 0.0))
                start = shot.start + into + offsets.get(clip, 0.0)  # Auto-edit only saw the part you kept
                trim = (start, start + used) if shot.kind == "video" else (None, None)
                self.db.execute(
                    "UPDATE items SET cut_out = 0, position = ?, trim_start = ?, trim_end = ?, transition = ?, "
                    "transition_why = ?, seconds = ?, speed = ?, speed_why = ?, effects = ?, effects_why = ? "
                    "WHERE path = ?",
                    (shot_no, *trim, shot.transition, shot.why, shot.seconds if shot.kind == "photo" else None,
                     shot.speed, shot.speed_why, json.dumps(shot.effects) if shot.effects else None, shot.effects_why,
                     shot.path),
                )
            # The photos inside a collage are part of this cut (they show through its first photo).
            parts = [(p,) for shot in result.shots for key in ("collage", "side") for p in (shot.effects or {}).get(key, [])]
            self.db.executemany("UPDATE items SET cut_out = 0, position = NULL WHERE path = ?", parts)
            # Left out of this cut, but still in your selection, so the next Auto-edit can use it again.
            self.db.executemany("UPDATE items SET cut_out = 1, position = NULL WHERE path = ?",
                                [(p,) for p in result.left_out + result.outside])
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
        result = autoedit.plan_to_music(rows, scores, steady, limits, None, transition_seconds, soft_transition,
                                        self._chapters)
        result.outside = self._never_offered(rows)
        self._apply_plan(result)
        report(1.0, "Done")
        return result

    def set_keep(self, path: str, start: float | None, end: float | None) -> None:
        """Cut a video down to the part you keep (start, end); (None, None) brings the whole clip back. Your file is
        not changed: the app simply never uses or shows the rest."""
        row = self.query("SELECT * FROM items WHERE path = ?", (path,))[0]
        if start is not None and end is not None:
            start, end = max(0.0, float(start)), min(float(end), row["duration"] or float(end))
            if end - start < 0.3:
                raise ValueError("Keep at least a third of a second")
            self.checkpoint("cutting a clip")
        else:
            start = end = None
            self.checkpoint("bringing a whole clip back")
        trim = (row["trim_start"], row["trim_end"])
        if start is not None and trim[0] is not None and not (start <= trim[0] < trim[1] <= end):
            trim = (None, None)  # the old trim reached into the part you cut off: it now plays the part you kept
        self.execute("UPDATE items SET keep_start = ?, keep_end = ?, trim_start = ?, trim_end = ? WHERE path = ?",
                     (start, end, *trim, path))
        row = self.query("SELECT * FROM items WHERE path = ?", (path,))[0]
        try:
            self.execute("UPDATE items SET thumb = ? WHERE path = ?", (str(self._make_thumb(row)), path))
        except Exception:
            traceback.print_exc()  # the old preview picture stays

    def set_reverse(self, path: str, on: bool) -> None:
        """Play a video backwards in the video (or forwards again)."""
        self.execute("UPDATE items SET reverse = ? WHERE path = ? OR part_of = ?", (int(on), path, path))

    def set_crop(self, path: str, zoom: float, x: float, y: float) -> None:
        """Zoom in on a photo or a video: the video shows only that part of it (zoom 1: the whole picture again)."""
        zoom = min(max(float(zoom), 1.0), 5.0)
        crop = None if zoom <= 1.001 else {"zoom": round(zoom, 3), "x": round(min(max(x, 0.0), 1.0), 4),
                                            "y": round(min(max(y, 0.0), 1.0), 4)}
        self.execute("UPDATE items SET crop = ? WHERE path = ? OR part_of = ?",
                     (json.dumps(crop) if crop else None, path, path))

    @property
    def shots_dir(self) -> Path:
        return self.dir / "shots"

    def make_shot(self, path: str, start: float, end: float) -> str:
        """Make a part of a video a shot of its own: the part is saved as a new clip (in the app's folder; your file
        is not changed) that gets its own tick, theme and place in the story, like any other file. It keeps the
        video's time (plus where the part starts), place, camera, turn and theme. Returns the new shot's path."""
        row = dict(self.query("SELECT * FROM items WHERE path = ?", (path,))[0])
        if row["kind"] != "video":
            raise ValueError("Only a video can be split into shots")
        start, end = max(0.0, float(start)), min(float(end), row["duration"] or float(end))
        if end - start < 0.5:
            raise ValueError("A shot needs at least half a second")
        self.checkpoint("making a shot from a video")
        self.shots_dir.mkdir(parents=True, exist_ok=True)
        stem = Path(path).stem
        n = 1
        while list(self.shots_dir.glob(f"{stem} shot {n} *")):
            n += 1
        out = self.shots_dir / f"{stem} shot {n} ({start:.1f}-{end:.1f}s).mp4"
        when = datetime.fromisoformat(row["taken_at"]) + timedelta(seconds=start)
        info = probe(Path(path))
        video = next((x for x in info.get("streams", []) if x.get("codec_type") == "video"), {})
        width, height = int(video.get("width") or 1920), int(video.get("height") or 1080)
        from .render import _run_encode
        audio = ["-map", "0:a:0?", "-c:a", "aac", "-b:a", "192k"] if row["has_audio"] else ["-an"]
        tmp = out.with_suffix(".part.mp4")
        _run_encode(lambda codec: ["-ss", f"{start:.3f}", "-i", path, "-t", f"{end - start:.3f}", "-map", "0:v:0",
                                   *codec, "-pix_fmt", "yuv420p", *audio, "-map_metadata", "-1",
                                   "-metadata", f"creation_time={when.astimezone().isoformat()}",
                                   "-movflags", "+faststart", str(tmp)],
                    width, height, False)
        tmp.rename(out)

        def inside(t: float) -> bool:
            return start <= t <= end

        locks = [[a - start, b - start] for a, b in json.loads(row["locks"] or "[]") if inside(a) and inside(b)]
        postits = [p | {"t": p["t"] - start} for p in json.loads(row["postits"] or "[]") if inside(p["t"])]
        orig = datetime.fromisoformat(row["taken_orig"] or row["taken_at"]) + timedelta(seconds=start)
        new = {k: row[k] for k in ("kind", "has_audio", "hdr", "lat", "lon", "camera", "place", "country",
                                   "place_guessed", "rotation", "rotation_checked", "theme", "theme_auto", "theme_sure",
                                   "story", "scene", "scene_user", "tags", "tags_user", "people_tag", "included",
                                   "clip_sound", "clip_volume", "crop") if k in row}
        new |= {"path": str(out), "shot_of": path, "taken_at": when.isoformat(), "taken_orig": orig.isoformat(),
                "duration": end - start, "mtime": out.stat().st_mtime, "chosen": 1, "best": 1,
                "locks": json.dumps(locks) if locks else None, "postits": json.dumps(postits) if postits else None}
        with self._lock:
            self.db.execute(f"INSERT INTO items ({', '.join(new)}) VALUES ({', '.join('?' * len(new))})",
                            tuple(new.values()))
            self.db.commit()
        made = dict(self.query("SELECT * FROM items WHERE path = ?", (str(out),))[0])
        try:
            self.execute("UPDATE items SET thumb = ? WHERE path = ?", (str(self._make_thumb(made)), str(out)))
        except Exception:
            traceback.print_exc()
        self._group_stories()
        return str(out)

    def crop_to_copy(self, path: str) -> str:
        """Crop a photo the way you zoomed and moved it: the cropped picture is saved as a copy (in the app's folder;
        your photo is not changed) that takes the photo's place: the copy is ticked, the photo unticked. Undo with
        drop_shot (and tick the photo again). Returns the copy's path."""
        row = dict(self.query("SELECT * FROM items WHERE path = ?", (path,))[0])
        crop = json.loads(row["crop"]) if row.get("crop") else None
        if row["kind"] != "photo" or not crop:
            raise ValueError("Zoom in on the photo first")
        from .render import crop_photo

        self.checkpoint("cropping a photo")
        self.shots_dir.mkdir(parents=True, exist_ok=True)
        stem = Path(path).stem
        n = 1
        while list(self.shots_dir.glob(f"{stem} crop {n}.jpg")):
            n += 1
        out = self.shots_dir / f"{stem} crop {n}.jpg"
        with Image.open(path) as img:
            picture = ImageOps.exif_transpose(img).convert("RGB").rotate(-(row["rotation"] or 0), expand=True)
        picture = crop_photo(picture, crop)
        exif = Image.Exif()
        when = datetime.fromisoformat(row["taken_orig"] or row["taken_at"]).strftime("%Y:%m:%d %H:%M:%S")
        exif[306] = when
        picture.save(out, quality=95, exif=exif)
        new = {k: row[k] for k in ("kind", "taken_at", "taken_orig", "lat", "lon", "camera", "place", "country",
                                   "theme", "theme_auto", "theme_sure", "story", "scene", "scene_user", "tags",
                                   "tags_user", "included", "seconds", "position") if k in row}
        new |= {"path": str(out), "shot_of": path, "mtime": out.stat().st_mtime, "chosen": 1, "best": 1,
                "rotation": 0, "rotation_checked": 1, "duration": 0.0, "has_audio": 0, "hdr": 0}
        with self._lock:
            self.db.execute(f"INSERT INTO items ({', '.join(new)}) VALUES ({', '.join('?' * len(new))})",
                            tuple(new.values()))
            self.db.execute("UPDATE items SET chosen = 0, position = NULL WHERE path = ?", (path,))
            self.db.commit()
        made = dict(self.query("SELECT * FROM items WHERE path = ?", (str(out),))[0])
        try:
            self.execute("UPDATE items SET thumb = ? WHERE path = ?", (str(self._make_thumb(made)), str(out)))
        except Exception:
            traceback.print_exc()
        self._group_stories()
        return str(out)

    def shots_from(self, path: str) -> list[dict]:
        """The shots you made from a video, in order: [{path, start, end}] (seconds in the video)."""
        rows = self.query("SELECT path, taken_at, duration FROM items WHERE shot_of = ? AND removed = 0", (path,))
        base = self.query("SELECT taken_at FROM items WHERE path = ?", (path,))
        if not base:
            return []
        t0 = datetime.fromisoformat(base[0]["taken_at"])
        out = [{"path": r["path"], "start": (datetime.fromisoformat(r["taken_at"]) - t0).total_seconds(),
                "end": (datetime.fromisoformat(r["taken_at"]) - t0).total_seconds() + (r["duration"] or 0)}
               for r in rows]
        return sorted(out, key=lambda x: x["start"])

    def drop_shot(self, path: str) -> None:
        """Take a shot you made back out (its file in the app's folder goes too)."""
        self.checkpoint("removing a shot made from a video")
        with self._lock:
            self.db.execute("DELETE FROM faces WHERE path = ?", (path,))
            self.db.execute("DELETE FROM items WHERE path = ? AND shot_of IS NOT NULL", (path,))
            self.db.commit()
        if Path(path).parent == self.shots_dir:
            Path(path).unlink(missing_ok=True)

    def filmstrip(self, path: str, frames: int = 12) -> Path | None:
        """A row of small pictures from through a video (the part you kept, if you cut it), to show under the
        handles when you cut it. Each picture is a quick jump into the file, so even a long 4K clip takes a moment."""
        row = dict(self.query("SELECT * FROM items WHERE path = ?", (path,))[0])
        rotation = row["rotation"] or 0
        start, end = kept_part(row) or (0.0, row["duration"] or 0)
        name = f"{hashlib.sha1(path.encode()).hexdigest()[:16]}_{rotation}_{start:.1f}-{end:.1f}_strip.jpg"
        out = self.dir / "thumbs" / name
        if out.exists() or end <= start:
            return out if out.exists() else None

        def grab(t: float) -> Image.Image | None:
            done = subprocess.run(
                [ffmpeg_paths()[0], "-loglevel", "error", "-ss", f"{t:.2f}", "-i", path, "-frames:v", "1",
                 "-vf", rotate_filter(rotation) + "scale=-2:60", "-f", "image2pipe", "-c:v", "mjpeg", "-"],
                capture_output=True,
            )
            try:
                return Image.open(io.BytesIO(done.stdout)).convert("RGB") if done.stdout else None
            except OSError:
                return None

        step = (end - start) / frames
        with ThreadPoolExecutor(4) as pool:
            pictures = [p for p in pool.map(grab, [start + step * (k + 0.5) for k in range(frames)]) if p]
        if not pictures:
            return None
        width = pictures[0].width
        strip = Image.new("RGB", (width * len(pictures), 60))
        for k, picture in enumerate(pictures):
            strip.paste(picture.resize((width, 60)), (k * width, 0))
        strip.save(out, quality=80)
        return out

    def set_trim(self, path: str, start: float | None, end: float | None) -> None:
        self.execute("UPDATE items SET trim_start = ?, trim_end = ? WHERE path = ?", (start, end, path))

    def set_clip_sound(self, path: str, on: bool, volume: float | None = None) -> None:
        """Use a video's own sound in the finished video (off by default, so the music plays alone)."""
        if volume is None:
            self.execute("UPDATE items SET clip_sound = ? WHERE path = ?", (int(on), path))
        else:
            self.execute("UPDATE items SET clip_sound = ?, clip_volume = ? WHERE path = ?", (int(on), volume, path))

    def set_transition(self, path: str, transition: str | None, seconds: float | None = None,
                       why: str | None = None) -> None:
        """The transition out of this shot into the next one, and how long it lasts (None: the video's length)."""
        self.execute("UPDATE items SET transition = ?, transition_len = ?, transition_why = ? WHERE path = ?",
                     (transition, seconds, why, path))

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
        p.running, p.notes, p.fraction, p.stop = True, [], 0.0, False
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
        if self.shots_dir.exists():  # good parts you made into shots of their own
            found += scan_folder(self.shots_dir)[0]
        p.notes.extend(problems)
        self._store_found(found)
        self._fill_cameras(found)
        self._apply_time_shifts()

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
        todo = self.query("SELECT path, kind, duration, rotation FROM items WHERE thumb IS NULL AND removed = 0 ORDER BY taken_at")
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
        for row in self.query("SELECT path, thumb FROM items WHERE thumb IS NOT NULL AND dhash IS NULL AND removed = 0"):
            try:
                dhash, colours, sharp = stories.fingerprint(row["thumb"])
            except OSError:
                continue
            self.execute("UPDATE items SET dhash = ?, colours = ?, sharpness = ? WHERE path = ?", (dhash, colours, sharp, row["path"]))
        if not self.query("SELECT 1 FROM meta WHERE key = 'video_looks'"):
            # Videos used to be compared on one frame: look at them again on several.
            self.execute("UPDATE items SET features = NULL WHERE kind = 'video'")
            self.execute("INSERT OR REPLACE INTO meta (key, value) VALUES ('video_looks', '1')")
        todo = self.query("SELECT path, kind, thumb, duration, rotation FROM items "
                          "WHERE thumb IS NOT NULL AND features IS NULL AND removed = 0")
        for i, row in enumerate(todo):
            if i % 50 == 0 or row["kind"] == "video":
                self.progress.message = f"Comparing pictures and videos for look-alikes: {i + 1} of {len(todo)}"
            try:
                if row["kind"] == "video":
                    blob, sharp = self._video_looks(row)
                    self.execute("UPDATE items SET features = ?, sharpness = ? WHERE path = ?", (blob, sharp, row["path"]))
                else:
                    self.execute("UPDATE items SET features = ? WHERE path = ?", (stories.features(row["thumb"]), row["path"]))
            except (OSError, cv2.error, ValueError):
                continue
        rows = [dict(r) for r in self.query(
            "SELECT path, kind, taken_at, duration, lat, lon, dhash, colours, sharpness, features, motion_shake "
            "FROM items WHERE part_of IS NULL AND removed = 0")]
        groups = stories.group(rows, self.look_alike_level(), self.story_links())
        cleaned = self.cleaned_sets()
        for r in self.query("SELECT path FROM items WHERE shot_of IS NOT NULL AND removed = 0"):
            cleaned.setdefault(r["path"], 1)  # a shot made from a video is never a double of its video
        if cleaned:  # sets you sorted out with Keep marked: each of their shots stands alone now
            free = max((sim for _, sim, _ in groups.values() if sim is not None), default=-1) + 1
            for path in sorted(cleaned):
                if path in groups:
                    groups[path] = (groups[path][0], free, bool(cleaned[path]))
                    free += 1
        with self._lock:
            self.db.executemany(
                "UPDATE items SET story = ?, similar_set = ?, best = ? WHERE path = ?",
                [(st, sim, int(best), path) for path, (st, sim, best) in groups.items()],
            )
            self.db.commit()

    def cleaned_sets(self) -> dict[str, int]:
        """Shots of look-alike sets you have sorted out (Keep marked): path -> 1 if you kept it, else 0."""
        rows = self.query("SELECT value FROM meta WHERE key = 'sets_cleaned'")
        return json.loads(rows[0]["value"]) if rows else {}

    def keep_marked(self, paths: list[str]) -> None:
        """A set of look-alikes is sorted out: the ticked shots stay as single shots, the frame and the star go.
        Remembered, also when the app looks for doubles again."""
        cleaned = self.cleaned_sets()
        chosen = {r["path"]: r["chosen"] for r in self.query(
            f"SELECT path, chosen FROM items WHERE path IN ({','.join('?' * len(paths))})", tuple(paths))}
        cleaned.update({p: int(chosen.get(p, 0)) for p in paths})
        self.execute("INSERT OR REPLACE INTO meta (key, value) VALUES ('sets_cleaned', ?)", (json.dumps(cleaned),))
        free = (self.query("SELECT MAX(similar_set) AS m FROM items")[0]["m"] or 0) + 1
        self.db_write_many("UPDATE items SET similar_set = ?, best = ? WHERE path = ?",
                           [(free + i, cleaned[p], p) for i, p in enumerate(paths)])

    def regroup_sets(self) -> None:
        """Undo every Keep marked: the look-alike frames come back."""
        self.execute("DELETE FROM meta WHERE key = 'sets_cleaned'")
        self._group_stories()

    def story_links(self) -> list[str]:
        rows = self.query("SELECT value FROM meta WHERE key = 'story_links'")
        return json.loads(rows[0]["value"]) if rows else []

    def _save_links(self, links: list[str]) -> None:
        self.execute("INSERT OR REPLACE INTO meta (key, value) VALUES ('story_links', ?)", (json.dumps(sorted(set(links))),))
        self._group_stories()

    def link_story(self, number: int, to_next: bool) -> None:
        """Join a story with the one before it (or after it, to_next) in time: they become one story."""
        firsts = {}
        for r in self.query("SELECT path, taken_at, story FROM items WHERE part_of IS NULL AND removed = 0 AND story IS NOT NULL "
                            "ORDER BY taken_at DESC, path DESC"):
            firsts[r["story"]] = f"{r['taken_at']}|{r['path']}"
        later = number + 1 if to_next else number
        if later in firsts and later - 1 in firsts:
            self._save_links(self.story_links() + [firsts[later]])

    def unlink_story(self, number: int) -> None:
        """Split a story you joined back into the stories it was made of."""
        rows = self.query("SELECT path, taken_at FROM items WHERE part_of IS NULL AND removed = 0 AND story = ?", (number,))
        if not rows:
            return
        first, last = min((r["taken_at"], r["path"]) for r in rows), max((r["taken_at"], r["path"]) for r in rows)
        keep = [link for link in self.story_links() if not (first < tuple(link.split("|", 1)) <= last)]
        self._save_links(keep)

    def joined_stories(self) -> set[int]:
        """Story numbers that you made by joining stories."""
        links = [tuple(link.split("|", 1)) for link in self.story_links()]
        spans: dict[int, list] = {}
        for r in self.query("SELECT story, MIN(taken_at || '|' || path) AS a, MAX(taken_at || '|' || path) AS b "
                            "FROM items WHERE part_of IS NULL AND removed = 0 AND story IS NOT NULL GROUP BY story"):
            spans[r["story"]] = [tuple(r["a"].split("|", 1)), tuple(r["b"].split("|", 1))]
        return {n for n, (a, b) in spans.items() if any(a < j <= b for j in links)}

    def _video_looks(self, row) -> tuple[bytes, float]:
        """A video's look from frames at its start, middle and end (falls back to its preview)."""
        duration = row["duration"] or 0
        with tempfile.TemporaryDirectory() as tmp:
            frames = []
            for k, share in enumerate(stories.VIDEO_FRAMES if duration >= 2 else (0.5,)):
                out = Path(tmp) / f"{k}.jpg"
                subprocess.run(
                    [ffmpeg_paths()[0], "-loglevel", "error", "-y", "-ss", f"{duration * share:.2f}", "-i", row["path"],
                     "-frames:v", "1", "-vf", rotate_filter(row["rotation"] or 0) + "scale=320:-2", "-q:v", "4", str(out)],
                    capture_output=True,
                )
                if out.exists():
                    frames.append(str(out))
            if not frames:
                frames = [row["thumb"]]
            return stories.video_features(frames)

    def look_alike_level(self) -> str:
        rows = self.query("SELECT value FROM meta WHERE key = 'look_alike'")
        return rows[0]["value"] if rows and rows[0]["value"] in stories.STRICTNESS else "normal"

    def set_look_alike_level(self, level: str) -> None:
        """How alike shots must be to count as doubles; the sets are worked out again (your ticks stay)."""
        self.execute("INSERT OR REPLACE INTO meta (key, value) VALUES ('look_alike', ?)", (level,))
        self._group_stories()

    def untick_doubles(self) -> int:
        """Untick every shot that has a better look-alike (the ⭐ of its set); other ticks stay as you set them."""
        with self._lock:
            n = self.db.execute("UPDATE items SET chosen = 0 WHERE best = 0 AND chosen = 1 AND removed = 0").rowcount
            self.db.commit()
        return n

    def suggest_selection(self) -> None:
        """Tick only the best shot of each set of similar shots (single shots stay ticked)."""
        with self._lock:
            self.db.execute("UPDATE items SET chosen = best WHERE removed = 0")
            self.db.execute("INSERT OR REPLACE INTO meta (key, value) VALUES ('suggested', '1')")
            self.db.commit()

    def _analyse(self, notify: Callable[[], None]) -> None:
        self._prepare(notify)
        p = self.progress
        todo = self.query(
            "SELECT * FROM items WHERE part_of IS NULL AND chosen = 1 AND (faces_done = 0 OR scene_done = 0 OR tags_done = 0) ORDER BY taken_at"
        )
        faces = scenes = tagger = None
        if todo:
            p.message = "Loading the AI models (the first time this downloads about 1 GB)…"
            notify()
            faces = self._load(lambda: FaceFinder(app_dir() / "models"), "people")
            if faces is not None:
                self.execute("DELETE FROM meta WHERE key = 'faces_error'")
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
        # Faces become people first, so a problem in one of the later steps never leaves the People tab empty.
        steps = [
            ("Grouping faces into people…", self._group_people),
            ("Reading camera moves…", lambda: self._find_camera_moves(notify)),
            ("Spotting look-alike shots…", self._group_stories),  # pictures turned upright get a fresh fingerprint
            ("Spotting look-alike shots…", self._group_look_alike),
            ("Naming the stories…", lambda: self._find_subjects(tagger, notify)),
        ]
        for message, step in steps:
            p.message = message
            notify()
            try:
                step()
            except Exception as exc:
                traceback.print_exc()
                p.notes.append(f"{message.rstrip('…')} stopped: {exc}")

    def _find_camera_moves(self, notify: Callable[[], None]) -> None:
        """How the camera moves in each video (pans, zooms, whips...), for videos analysed before this was added."""
        rows = self.query("SELECT * FROM items WHERE part_of IS NULL AND chosen = 1 AND kind = 'video' AND camera_moves IS NULL")
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
                # Your own name first, then the storyline's you chose, then the story plan's theme.
                "name": saved.get("name") or saved.get("ai_name") or saved.get("theme") or auto,
                "renamed": bool(saved.get("name")),
                "theme": saved.get("theme"), "idea": saved.get("idea"),
                "notes": json.loads(saved["plan_notes"]) if saved.get("plan_notes") else {},
                "planned": any(m["plan_role"] for m in members),
                "enabled": saved.get("enabled", 1) != 0, "position": saved.get("position"),
                "start": members[0]["taken_at"], "end": members[-1]["taken_at"],
            })
        placed = sorted((s for s in result if s["position"] is not None), key=lambda s: s["position"])
        return placed + [s for s in result if s["position"] is None]

    def _split_ai_names(self) -> None:
        """Older versions saved a storyline's story names as if you had typed them, so a later storyline could
        not rename them: keep those as the AI's names (once, when updating)."""
        rows = self.db.execute("SELECT value FROM meta WHERE key = 'storylines'").fetchall()
        names = json.loads(rows[0][0]).get("names", {}) if rows else {}
        for key, name in names.items():
            self.db.execute("UPDATE story_info SET ai_name = name, name = NULL WHERE key = ? AND name = ?", (key, name))
        self.db.commit()

    def _story_info(self, key: str) -> None:
        self.execute("INSERT OR IGNORE INTO story_info (key) VALUES (?)", (key,))

    def rename_story(self, key: str, name: str) -> None:
        self._story_info(key)
        self.execute("UPDATE story_info SET name = ? WHERE key = ?", ((name or "").strip() or None, key))

    def set_story_enabled(self, key: str, enabled: bool) -> None:
        """Switch a story on or off yourself: the AI's versions then leave it as you set it."""
        self._story_info(key)
        self.execute("UPDATE story_info SET enabled = ?, mine_enabled = ? WHERE key = ?", (int(enabled), int(enabled), key))

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

    # ---------- AI director ----------

    def describe_clips(self, ai, on_change: Callable[[], None] | None = None, again: bool = False) -> None:
        """Let the AI describe each chosen file (and say how much it stands out). Safe to re-run."""
        self._run(lambda notify: self._describe(ai, notify, again), on_change)

    def _describe(self, ai, notify: Callable[[], None], again: bool) -> None:
        """Describe each chosen file. Look-alike shots are described once: the ⭐ of the set is shown to the AI and
        its doubles get the same description, which often halves the work."""
        p = self.progress
        chosen = self.query("SELECT * FROM items WHERE part_of IS NULL AND chosen = 1 AND thumb IS NOT NULL AND removed = 0 "
                            "ORDER BY taken_at, path")
        sets: dict[object, list] = {}
        for row in chosen:
            sets.setdefault(row["similar_set"] if row["similar_set"] is not None else row["path"], []).append(row)
        todo = []  # (the shot the AI looks at, the shots that get its description)
        for members in sets.values():
            if not again and all(m["ai_caption"] for m in members):
                continue
            known = next((m for m in members if m["ai_caption"]), None)
            if not again and known:  # a double ticked later: it gets its set's description, no AI needed
                self.db_write_many("UPDATE items SET ai_caption = ?, ai_score = ? WHERE path = ?",
                                   [(known["ai_caption"], known["ai_score"], m["path"]) for m in members])
                continue
            star = next((m for m in members if m["best"]), members[0])
            todo.append((star, members))
        done = 0
        started = time.monotonic()
        for start in range(0, len(todo), ai.batch):
            if p.stop:
                p.message = (f"Stopped. The AI described {done} files; the other {len(todo) - start} to look at "
                             "have no description yet (Describe clips carries on from there).")
                p.notes.append(p.message)
                return
            batch = todo[start : start + ai.batch]
            p.fraction = start / len(todo)
            left = ""
            if start:
                minutes = (time.monotonic() - started) / start * (len(todo) - start) / 60
                left = f" · about {minutes:.0f} min left" if minutes >= 1 else " · less than a minute left"
            p.message = (f"The AI is looking at {start + 1}–{start + len(batch)} of {len(todo)} "
                         f"({len(chosen)} files; doubles share a description){left}")
            notify()
            clips = []
            for row, _ in batch:
                try:
                    clips.append(self._ai_frames(row, getattr(ai, "frames", 3), getattr(ai, "size", 512)))
                except Exception as exc:
                    clips.append([])
                    p.notes.append(f"Could not read {Path(row['path']).name} for the AI: {exc}")
            ready = [(members, frames) for (_, members), frames in zip(batch, clips) if frames]
            if not ready:
                continue
            captions = ai.describe([frames for _, frames in ready])  # a DirectorError stops the whole run
            with self._lock:
                for (members, _), caption in zip(ready, captions):
                    if caption:
                        self.db.executemany("UPDATE items SET ai_caption = ?, ai_score = ? WHERE path = ?",
                                            [(caption.text, caption.highlight, m["path"]) for m in members])
                        done += len(members)
                self.db.commit()
        p.message = f"The AI described {done} files."

    def _ai_frames(self, row, count: int = 3, size: int = 512) -> list[bytes]:
        """Small pictures for the AI: the photo's preview, or a few frames spread through a video."""
        if row["kind"] == "photo":
            if size >= 480:
                return [Path(row["thumb"]).read_bytes()]
            with Image.open(row["thumb"]) as img, tempfile.SpooledTemporaryFile() as out:
                img.thumbnail((size, size))
                img.convert("RGB").save(out, "JPEG", quality=80)
                out.seek(0)
                return [out.read()]
        offset, duration = 0.0, row["duration"] or 0
        if kept_part(dict(row)):  # only the part you kept
            offset, end = kept_part(dict(row))
            duration = end - offset
        frames = []
        shares = (0.2, 0.5, 0.8) if count >= 3 and duration >= 3 else (0.5,)
        with tempfile.TemporaryDirectory() as tmp:
            for i, share in enumerate(shares):
                out = Path(tmp) / f"{i}.jpg"
                subprocess.run(
                    [ffmpeg_paths()[0], "-loglevel", "error", "-y", "-ss", f"{offset + duration * share:.2f}", "-i", row["path"],
                     "-frames:v", "1", "-vf", rotate_filter(row["rotation"] or 0) + f"scale={size}:-2", "-q:v", "5", str(out)],
                    capture_output=True,
                )
                if out.exists():
                    frames.append(out.read_bytes())
        return frames or [Path(row["thumb"]).read_bytes()]

    # ---------- story director: a plan for each story ----------

    PLAN_MAX_SHOTS = 120  # a very long story: the AI sees its best shots only

    def plan_stories(self, ai, keys: list[str] | None = None, on_change: Callable[[], None] | None = None) -> None:
        """Let the AI plan each story (or just these): a theme, and its shots in order as opening, montage,
        highlight... Auto-edit then cuts the story that way."""
        self._run(lambda notify: self._plan_stories(ai, notify, keys), on_change)

    def _plan_stories(self, ai, notify: Callable[[], None], keys: list[str] | None) -> None:
        p = self.progress
        todo = [s for s in self._director_stories() if keys is None or s["key"] in keys]
        started = time.monotonic()
        for k, story in enumerate(todo):
            if p.stop:
                p.message = f"Stopped. The AI planned {k} of {len(todo)} stories; the others keep their old plan or none."
                p.notes.append(p.message)
                return
            p.fraction = k / max(1, len(todo))
            left = ""
            if k:
                minutes = (time.monotonic() - started) / k * (len(todo) - k) / 60
                left = f" · about {minutes:.0f} min left" if minutes >= 1 else " · less than a minute left"
            p.message = f"The AI is planning story {k + 1} of {len(todo)}: {story['name']}{left}"
            notify()
            self._plan_story(ai, story)
        p.message = f"The AI planned {len(todo)} stories."

    def _plan_story(self, ai, story: dict) -> None:
        shots = [it for it in story["items"] if it["chosen"]]
        if len(shots) > self.PLAN_MAX_SHOTS:
            keep = sorted(shots, key=lambda it: (-(it.get("ai_score") or 3), -bool(it["best"])))[: self.PLAN_MAX_SHOTS]
            shots = [it for it in shots if it in keep]
        if not shots:
            return
        places = [it["place"] or it["country"] for it in shots if it["place"] or it["country"]]
        start, end = (datetime.fromisoformat(story[k]).astimezone() for k in ("start", "end"))
        prompt = director.plan_prompt(
            {"name": story["name"], "when": f"{start:%a %d %b %H:%M}–{end:%H:%M}",
             "place": max(set(places), key=places.count) if places else ""},
            [{"caption": it.get("ai_caption"), "highlight": it.get("ai_score"), "kind": it["kind"],
              "seconds": (kept_part(it)[1] - kept_part(it)[0]) if kept_part(it) else it["duration"] or 0, "time": f"{datetime.fromisoformat(it['taken_at']).astimezone():%H:%M}"}
             for it in shots])
        plan = director.clean_plan(ai.plan(prompt), len(shots))  # a DirectorError stops the whole run
        if not plan["parts"]:
            return
        notes = {}
        rows = []
        pos = 0
        for part in plan["parts"]:
            notes[part["role"]] = part["note"]
            for n in part["shots"]:
                rows.append((part["role"], pos, shots[n - 1]["path"]))
                pos += 1
        with self._lock:
            self.db.executemany("UPDATE items SET plan_role = NULL, plan_pos = NULL WHERE path = ?",
                                [(it["path"],) for it in story["items"]])
            self.db.executemany("UPDATE items SET plan_role = ?, plan_pos = ? WHERE path = ?", rows)
            self.db.execute("INSERT OR IGNORE INTO story_info (key) VALUES (?)", (story["key"],))
            self.db.execute("UPDATE story_info SET theme = ?, idea = ?, plan_notes = ? WHERE key = ?",
                            (plan["theme"] or None, plan["idea"] or None, json.dumps(notes), story["key"]))
            # A new plan may use shots the last first cut left out: they come back.
            self.db.executemany("UPDATE items SET cut_out = 0 WHERE path = ?", [(p,) for _, _, p in rows])
            self._replace_in_storyboard([it["path"] for it in story["items"]])
            self.db.commit()

    def set_plan_role(self, path: str, role: str | None) -> None:
        """Change what one shot does in its story's plan (None: leave it out of the plan)."""
        if role is None:
            self.execute("UPDATE items SET plan_role = NULL, plan_pos = NULL WHERE path = ?", (path,))
            return
        # A shot added to the plan comes back into the storyboard, even if the last first cut left it out.
        self.execute("UPDATE items SET plan_role = ?, cut_out = 0, plan_pos = COALESCE(plan_pos, "
                     "(SELECT COALESCE(MAX(plan_pos), 0) + 1 FROM items)) WHERE path = ?", (role, path))

    def move_in_plan(self, path: str, steps: int) -> None:
        """Move a shot earlier (negative) or later in its story's plan."""
        rows = self.query("SELECT story FROM items WHERE path = ?", (path,))
        if not rows:
            return
        order = [r["path"] for r in self.query("SELECT path FROM items WHERE story = ? AND plan_role IS NOT NULL "
                                               "ORDER BY plan_pos", (rows[0]["story"],))]
        if path not in order:
            return
        i = order.index(path)
        order.insert(max(0, min(len(order) - 1, i + steps)), order.pop(i))
        self.db_write_many("UPDATE items SET plan_pos = ? WHERE path = ?", [(n, p) for n, p in enumerate(order)])
        with self._lock:
            self._replace_in_storyboard(order)
            self.db.commit()

    def _replace_in_storyboard(self, paths: list[str]) -> None:
        """After a story's plan changed: its shots take their new plan order in the storyboard, where the
        story's first placed shot was. (Call with the lock held.)"""
        if not paths:
            return
        marks = ",".join("?" * len(paths))
        first = self.db.execute(f"SELECT MIN(position) FROM items WHERE path IN ({marks})", paths).fetchone()[0]
        if first is None:
            return  # not in the storyboard's own order yet: it follows the plan anyway
        rows = self.db.execute(f"SELECT path, plan_pos, taken_at FROM items WHERE path IN ({marks}) "
                               "AND plan_role IS NOT NULL", paths).fetchall()
        planned = [r[0] for r in sorted(rows, key=lambda r: (r[1], r[2]))]
        self.db.execute(f"UPDATE items SET position = NULL WHERE path IN ({marks})", paths)
        # Spread the story's shots just after the story's old start, before whatever came next.
        nxt = self.db.execute("SELECT MIN(position) FROM items WHERE position > ?", (first,)).fetchone()[0]
        step = ((nxt if nxt is not None else first + len(planned) + 1) - first) / (len(planned) + 1)
        self.db.executemany("UPDATE items SET position = ? WHERE path = ?",
                            [(first + step * n, p) for n, p in enumerate(planned)])

    def forget_plan(self, key: str) -> None:
        story = next((s for s in self.stories() if s["key"] == key), None)
        if story is None:
            return
        self.db_write_many("UPDATE items SET plan_role = NULL, plan_pos = NULL WHERE path = ?",
                           [(it["path"],) for it in story["items"]])
        self.execute("UPDATE story_info SET theme = NULL, idea = NULL, plan_notes = NULL WHERE key = ?", (key,))

    def stop_ai(self) -> None:
        """Stop the AI: it finishes the clip or story it is on, keeps what it has done, and leaves the rest."""
        self.progress.stop = True

    def _new_turn(self) -> int:
        self.progress.stop = False
        self.progress.turn += 1
        return self.progress.turn

    def _stopped(self, turn: int) -> bool:
        """Whether you stopped this AI request (or started another since)."""
        return self.progress.stop or self.progress.turn != turn

    def _director_stories(self) -> list[dict]:
        return [s for s in self.stories() if any(it["chosen"] for it in s["items"])]

    def suggest_storylines(self, ai) -> dict:
        """Ask the AI for a few ways to tell the trip; they are kept so you can try them one after another."""
        stories_now = self._director_stories()
        if not stories_now:
            raise director.DirectorError("No stories yet. Click Analyse first.")
        by_time = sorted(stories_now, key=lambda s: s["start"])
        described = []
        for s in by_time:
            members = [it for it in s["items"] if it["chosen"]]
            places = [it["place"] or it["country"] for it in members if it["place"] or it["country"]]
            start = datetime.fromisoformat(s["start"]).astimezone()
            described.append({
                "name": s["name"], "count": len(members),
                "when": f"{start:%a %d %b %H:%M}–{datetime.fromisoformat(s['end']).astimezone():%H:%M}",
                "place": max(set(places), key=places.count) if places else "",
                "captions": [it["ai_caption"] for it in members if it.get("ai_caption")],
                "idea": s.get("idea"),
            })
        mine = self._new_turn()
        answer = director.clean_storylines(ai.storylines(director.storyline_prompt(described)), len(by_time))
        if self._stopped(mine):
            raise director.Stopped("Stopped: the storylines you had are kept.")
        if not answer["storylines"]:
            raise director.DirectorError("The AI did not come up with a usable storyline. Try again.")
        saved = {
            "storylines": [{**s, "stories": [by_time[n - 1]["key"] for n in s["stories"]]} for s in answer["storylines"]],
            "names": {by_time[n - 1]["key"]: name for n, name in answer["names"].items()},
        }
        self.execute("INSERT OR REPLACE INTO meta (key, value) VALUES ('storylines', ?)", (json.dumps(saved),))
        return saved

    def saved_storylines(self) -> dict | None:
        rows = self.query("SELECT value FROM meta WHERE key = 'storylines'")
        return json.loads(rows[0]["value"]) if rows else None

    def apply_storyline(self, index: int) -> None:
        """Put the stories in this storyline's order, switch off the ones it leaves out, and give stories you
        haven't named yourself the AI's names. The storyboard then follows it until the next first cut."""
        saved = self.saved_storylines()
        if not saved or not 0 <= index < len(saved["storylines"]):
            return
        self._use_stories(saved["storylines"][index]["stories"], saved["names"])
        self.execute("INSERT OR REPLACE INTO meta (key, value) VALUES ('storyline_used', ?)", (str(index),))

    def _use_stories(self, chosen: list[str], names: dict[str, str] | None = None) -> None:
        """Tell these stories (keys) in this order and switch the others off; the storyboard starts again from them
        (the last first cut's order and left-outs go). names: the AI's names for stories (None keeps them)."""
        keys = [s["key"] for s in self.stories()]
        chosen = [k for k in chosen if k in keys]
        rest = [k for k in keys if k not in chosen]
        with self._lock:
            for n, key in enumerate(chosen + rest):
                self.db.execute("INSERT OR IGNORE INTO story_info (key) VALUES (?)", (key,))
                # A story you switched on or off yourself stays that way.
                self.db.execute("UPDATE story_info SET position = ?, enabled = COALESCE(mine_enabled, ?) WHERE key = ?",
                                (n, int(key in chosen), key))
                if names is not None:
                    self.db.execute("UPDATE story_info SET ai_name = ? WHERE key = ?", (names.get(key), key))
            self.db.execute("UPDATE items SET position = NULL, cut_out = 0")
            self.db.commit()

    # ---------- the video you want ----------

    def _wish_stories(self) -> tuple[list[dict], list[str]]:
        """The trip for the AI: each story with its clips' captions and the people in it; and the named people."""
        stories = sorted(self._director_stories(), key=lambda s: s["start"])
        described = []
        for s in stories:
            members = [it for it in s["items"] if it["chosen"]]
            places = [it["place"] or it["country"] for it in members if it["place"] or it["country"]]
            start = datetime.fromisoformat(s["start"]).astimezone()
            described.append({
                "key": s["key"], "name": s["name"], "count": len(members),
                "when": f"{start:%a %d %b %H:%M}–{datetime.fromisoformat(s['end']).astimezone():%H:%M}",
                "place": max(set(places), key=places.count) if places else "",
                "captions": [it["ai_caption"] for it in members if it.get("ai_caption")],
                "people": sorted({p for it in members for p in it["people"] if not p.startswith("Person ")}),
            })
        return described, self.person_names()

    def make_wish(self, ai, wish: str, song_seconds: float | None = None, music: str = "") -> dict:
        """Ask the AI for 3 versions of the video you described (nothing typed: it surprises you) and use the first.
        What you set yourself (story names, stories you switched on or off, your title) is kept."""
        stories, people = self._wish_stories()
        if not stories:
            raise director.DirectorError("No stories yet. Click Analyse first.")
        mine = self._new_turn()
        answer = director.clean_wish(ai.wish(director.wish_prompt(wish, stories, people, song_seconds, music)),
                                     len(stories), people)
        if self._stopped(mine):
            raise director.Stopped("Stopped: nothing was changed.")
        self.checkpoint("the AI's answer to your wish")
        # The length is not the wish's: the video lasts as long as your track (trim the song in Music & audio), or
        # the length you set yourself on the Storyboard.
        answer["length"], answer["length_seconds"] = None, 0
        saved = {"wish": " ".join(wish.split()), **answer, "used": 0,
                 "names": {stories[n - 1]["key"]: name for n, name in answer["names"].items()},
                 "versions": [{**v, "stories": [stories[n - 1]["key"] for n in v["stories"]],
                               "story_names": [answer["names"].get(n) or stories[n - 1]["name"] for n in v["stories"]]}
                              for v in answer["versions"]]}
        if saved["names"]:
            with self._lock:
                for key, name in saved["names"].items():
                    self.db.execute("INSERT OR IGNORE INTO story_info (key) VALUES (?)", (key,))
                    self.db.execute("UPDATE story_info SET ai_name = ? WHERE key = ?", (name, key))
                self.db.commit()
        self.set_title_settings(ai_title=answer["title"] or None)
        if answer["format"]:
            self.set_video_size("phone" if answer["format"] == "phone" else "landscape")
        self._save_wish(saved)
        self.use_version(0, undo=False)
        return self.wish()

    def use_version(self, index: int, undo: bool = True) -> None:
        """Tell the video the way this version of your wish does: its stories, in its order, and its look."""
        rows = self.query("SELECT value FROM meta WHERE key = 'wish'")
        saved = json.loads(rows[0]["value"]) if rows else None
        if not saved or not 0 <= index < len(saved.get("versions", [])):
            return
        if undo:
            self.checkpoint(f"version {saved.get('used', 0) + 1} of your wish")
        version = saved["versions"][index]
        if version["stories"]:
            self._use_stories(version["stories"])
        if version["look"]:
            self.set_look(version["look"])
        saved["used"] = index
        self._save_wish(saved)

    def set_wish_length(self, length: dict | None) -> None:
        """Change the length of the video you asked for: {"seconds": 80} or {"song": 0.5}."""
        rows = self.query("SELECT value FROM meta WHERE key = 'wish'")
        if not rows:
            return
        saved = json.loads(rows[0]["value"])
        saved["length"] = length
        saved["length_seconds"] = round(length["seconds"]) if length and "seconds" in length else 0
        self._save_wish(saved)

    def _save_wish(self, saved: dict) -> None:
        self.execute("INSERT OR REPLACE INTO meta (key, value) VALUES ('wish', ?)", (json.dumps(saved),))

    def wish(self) -> dict | None:
        """The video you asked for, with the version you use now (it steers Auto-edit until you forget it)."""
        rows = self.query("SELECT value FROM meta WHERE key = 'wish'")
        if not rows:
            return None
        saved = json.loads(rows[0]["value"])
        versions = saved.get("versions") or [{}]  # a wish saved by 0.34 to 0.40 has its settings at the top
        return {**saved, **versions[min(saved.get("used", 0), len(versions) - 1)]}

    def forget_wish(self) -> None:
        self.checkpoint("forgetting your wish")
        self.execute("DELETE FROM meta WHERE key = 'wish'")

    def video_size(self) -> str:
        """'landscape' or 'phone' (upright 9:16)."""
        rows = self.query("SELECT value FROM meta WHERE key = 'video_size'")
        return rows[0]["value"] if rows and rows[0]["value"] in ("landscape", "phone") else "landscape"

    def song(self) -> str | None:
        """The song you chose last for this trip (or your trimmed version of it), so it is there after a restart."""
        rows = self.query("SELECT value FROM meta WHERE key = 'song'")
        return rows[0]["value"] if rows and rows[0]["value"] else None

    def set_song(self, path: str | None) -> None:
        self.execute("INSERT OR REPLACE INTO meta (key, value) VALUES ('song', ?)", (path or "",))

    # ---------- Camera clocks ----------

    def _fill_cameras(self, found: list[MediaItem]) -> None:
        """Files stored by an older version don't know their camera or their own time yet: the scan just read both."""
        with self._lock:
            self.db.executemany("UPDATE items SET camera = ? WHERE path = ? AND camera IS NULL",
                                [(it.camera or timeshift.OTHER, str(it.path)) for it in found])
            # Before any shift, the time stored is the file's own: keep it, so a shift can always be undone.
            self.db.execute("UPDATE items SET taken_orig = taken_at WHERE taken_orig IS NULL")
            self.db.execute("UPDATE items SET camera = (SELECT p.camera FROM items p WHERE p.path = items.part_of) "
                            "WHERE camera IS NULL AND part_of IS NOT NULL")
            self.db.commit()

    def time_shifts(self) -> dict[str, float]:
        """{camera: seconds its clock is moved} (later is positive)."""
        rows = self.query("SELECT value FROM meta WHERE key = 'time_shifts'")
        try:
            return {k: float(v) for k, v in json.loads(rows[0]["value"]).items()} if rows else {}
        except (ValueError, AttributeError):
            return {}

    def _apply_time_shifts(self) -> None:
        shifts = self.time_shifts()
        changes = []
        for r in self.query("SELECT path, camera, taken_at, taken_orig FROM items WHERE taken_orig IS NOT NULL"):
            when = datetime.fromisoformat(r["taken_orig"]) + timedelta(seconds=shifts.get(r["camera"] or timeshift.OTHER, 0))
            if when.isoformat() != r["taken_at"]:
                changes.append((when.isoformat(), r["path"]))
        if changes:
            with self._lock:
                self.db.executemany("UPDATE items SET taken_at = ? WHERE path = ?", changes)
                self.db.commit()

    def cameras(self) -> list[dict]:
        """Each camera in the trip: {camera, count, videos, photos, first, last (its own clock), shift}."""
        shifts = self.time_shifts()
        out = []
        for r in self.query("SELECT COALESCE(camera, ?) AS camera, COUNT(*) AS count, SUM(kind = 'video') AS videos, "
                            "MIN(taken_orig) AS first, MAX(taken_orig) AS last FROM items "
                            "WHERE part_of IS NULL AND removed = 0 GROUP BY COALESCE(camera, ?) ORDER BY count DESC",
                            (timeshift.OTHER, timeshift.OTHER)):
            out.append({**dict(r), "photos": r["count"] - (r["videos"] or 0), "shift": shifts.get(r["camera"], 0.0)})
        return out

    def set_time_shift(self, camera: str, seconds: float) -> int:
        """Move every file of this camera by seconds (replacing an earlier shift), then group the stories again.
        Returns how many files moved."""
        shifts = self.time_shifts()
        if abs(shifts.get(camera, 0.0) - seconds) < 0.5:
            return 0
        self.checkpoint(f"changing the {camera} clock")
        if seconds:
            shifts[camera] = float(seconds)
        else:
            shifts.pop(camera, None)
        self.execute("INSERT OR REPLACE INTO meta (key, value) VALUES ('time_shifts', ?)", (json.dumps(shifts),))
        self._apply_time_shifts()
        self._group_stories()
        return self.query("SELECT COUNT(*) AS n FROM items WHERE part_of IS NULL AND COALESCE(camera, ?) = ?",
                          (timeshift.OTHER, camera))[0]["n"]

    def suggest_time_shift(self, camera: str) -> dict | None:
        """What this camera's clock is probably off by, from its shots that look like other cameras' shots of the
        same moment (timeshift.suggest). Uses the AI's picture understanding (CLIP, after Analyse) where both shots
        have it, else their colours and layout. Adds "how": "ai" or "colours". None if nothing agrees."""
        rows = [dict(r) for r in self.query(
            "SELECT path, camera, taken_orig, clip, features FROM items WHERE part_of IS NULL AND removed = 0 "
            "AND taken_orig IS NOT NULL")]
        shifts = self.time_shifts()

        def when(r):  # other cameras at their (already corrected) time, this one on its own clock
            t = datetime.fromisoformat(r["taken_orig"]).timestamp()
            return t if (r["camera"] or timeshift.OTHER) == camera else t + shifts.get(r["camera"] or timeshift.OTHER, 0)

        mine = [r for r in rows if (r["camera"] or timeshift.OTHER) == camera]
        others = [r for r in rows if (r["camera"] or timeshift.OTHER) != camera]
        how = "ai"
        a = [(r, np.frombuffer(r["clip"], dtype=np.float32)) for r in mine if r["clip"] is not None]
        b = [(r, np.frombuffer(r["clip"], dtype=np.float32)) for r in others if r["clip"] is not None]
        if len(a) < 2 or len(b) < 2:
            how = "colours"
            a = [(r, v) for r in mine if (v := _look_vector(r["features"])) is not None]
            b = [(r, v) for r in others if (v := _look_vector(r["features"])) is not None]
        found = timeshift.suggest([(when(r), v) for r, v in a], [(when(r), v) for r, v in b],
                                  least=0.75 if how == "ai" else 0.0)
        if found:
            found["how"] = how
            found["examples"] = [(a[i][0]["path"], b[j][0]["path"]) for i, j, _ in found["pairs"]]
        return found

    def own_length(self) -> float | None:
        """The length you typed on the Storyboard ("Use a different length"), or None: as long as your track."""
        rows = self.query("SELECT value FROM meta WHERE key = 'own_length'")
        try:
            return float(rows[0]["value"]) if rows and rows[0]["value"] else None
        except ValueError:
            return None

    def set_own_length(self, seconds: float | None) -> None:
        self.execute("INSERT OR REPLACE INTO meta (key, value) VALUES ('own_length', ?)",
                     (str(round(seconds, 1)) if seconds else "",))

    def set_video_size(self, size: str) -> None:
        self.execute("INSERT OR REPLACE INTO meta (key, value) VALUES ('video_size', ?)", (size,))

    def hand_changes(self) -> list[str]:
        """What you set yourself, which the AI never changes."""
        info = self.query("SELECT name, mine_enabled FROM story_info")
        named = sum(1 for r in info if r["name"])
        on = sum(1 for r in info if r["mine_enabled"] == 1)
        off = sum(1 for r in info if r["mine_enabled"] == 0)
        rows = self.query("SELECT value FROM meta WHERE key = 'titles'")
        titled = bool(rows and json.loads(rows[0]["value"]).get("trip_title"))
        own = self.query("SELECT COUNT(*) AS n FROM items WHERE title_text IS NOT NULL AND title_text != ''")[0]["n"]
        changes = [f"{named} story name{'s' * (named != 1)}" if named else "",
                   f"{off} stor{'ies' if off != 1 else 'y'} switched off" if off else "",
                   f"{on} stor{'ies' if on != 1 else 'y'} switched on" if on else "",
                   "the opening title" if titled else "",
                   f"{own} title{'s' * (own != 1)} on shots" if own else ""]
        return [c for c in changes if c]

    def forget_my_story_switches(self) -> None:
        """Let the AI's versions decide which stories are in again."""
        self.execute("UPDATE story_info SET mine_enabled = NULL")

    # ---------- undo ----------

    UNDO_STEPS = 5

    def _undo_list(self) -> list[dict]:
        try:
            return json.loads((self.dir / "undo" / "steps.json").read_text())
        except (OSError, ValueError):
            return []

    def _save_undo_list(self, steps: list[dict]) -> None:
        (self.dir / "undo").mkdir(exist_ok=True)
        (self.dir / "undo" / "steps.json").write_text(json.dumps(steps))

    def checkpoint(self, label: str) -> None:
        """Remember the trip as it is now, so Undo can bring it back (the last few steps are kept)."""
        folder = self.dir / "undo"
        folder.mkdir(exist_ok=True)
        name = f"{time.time_ns()}.db"
        with self._lock:
            self.db.commit()
            copy = sqlite3.connect(folder / name)
            self.db.backup(copy)
            copy.close()
        steps = self._undo_list() + [{"label": label, "file": name}]
        for old in steps[: -self.UNDO_STEPS]:
            (folder / old["file"]).unlink(missing_ok=True)
        self._save_undo_list(steps[-self.UNDO_STEPS:])

    def undo_label(self) -> str | None:
        """What Undo would take back, e.g. "the first cut"."""
        steps = self._undo_list()
        return steps[-1]["label"] if steps else None

    def undo(self) -> str | None:
        """Bring the trip back to how it was before the last AI answer or first cut. Returns what was undone."""
        steps = self._undo_list()
        if not steps:
            return None
        last = steps.pop()
        path = self.dir / "undo" / last["file"]
        if path.exists():
            with self._lock:
                saved = sqlite3.connect(path)
                saved.backup(self.db)
                saved.close()
            path.unlink()
        self._save_undo_list(steps)
        return last["label"]

    @staticmethod
    def wish_match(row: dict, wish: dict | None) -> float:
        """How well a shot fits the people and things you asked for: 0 when you asked for none,
        else -0.5 for a shot without any of them, up to +2 for one with both."""
        if not wish or not (wish.get("people") or wish.get("focus")):
            return 0.0
        score = 0.0
        if set(wish.get("people") or []) & set(row.get("people") or []):
            score += 1.0
        words = " ".join([row.get("ai_caption") or "", row.get("scene_final") or "", row.get("place") or "",
                          " ".join(row.get("tags_final") or [])]).lower()
        forms = {f for w in wish.get("focus") or [] for f in (w, w[:-1] if w.endswith("s") else w,
                                                                 w[:-2] if w.endswith("es") else w) if len(f) > 2}
        if any(f in words for f in forms):  # "beaches" finds "beach", "kids" finds "kid"
            score += 1.0
        return score or -0.5

    def storyline_used(self) -> int | None:
        rows = self.query("SELECT value FROM meta WHERE key = 'storyline_used'")
        return int(rows[0]["value"]) if rows else None

    def _thumb_path(self, path: str, rotation: int = 0, kept: tuple[float, float] | None = None) -> Path:
        # The rotation (and the part of a video you kept) is part of the name, so the browser shows the new
        # picture straight away.
        cut = f"_{kept[0]:.1f}-{kept[1]:.1f}" if kept else ""
        return self.dir / "thumbs" / f"{hashlib.sha1(path.encode()).hexdigest()[:16]}_{rotation}{cut}.jpg"

    def _make_thumb(self, row) -> Path:
        rotation = row["rotation"] or 0
        kept = kept_part(dict(row)) if "keep_start" in row.keys() else None
        out = self._thumb_path(row["path"], rotation, kept)
        if row["kind"] == "photo":
            with Image.open(row["path"]) as img:
                img.draft("RGB", (960, 960))  # fast for JPEGs
                img = ImageOps.exif_transpose(img).convert("RGB").rotate(-rotation, expand=True)
                img.thumbnail((480, 480))
                img.save(out, quality=85)
        else:
            subprocess.run(
                [ffmpeg_paths()[0], "-loglevel", "error", "-y", "-ss",
                 f"{sum(kept or (0, row['duration'] or 0)) / 2:.2f}",
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
            if what == "people":
                self.execute("INSERT OR REPLACE INTO meta (key, value) VALUES ('faces_error', ?)", (str(exc)[:300],))
            return None

    def face_status(self) -> dict:
        """For the People tab: how far finding faces got among the chosen files."""
        row = self.query(
            "SELECT COUNT(*) AS chosen, SUM(faces_done) AS done, "
            "(SELECT COUNT(*) FROM faces f JOIN items i ON i.path = f.path WHERE i.chosen = 1) AS faces, "
            "(SELECT COUNT(*) FROM faces f JOIN items i ON i.path = f.path WHERE i.chosen = 1 AND f.person_id IS NULL) "
            "AS ungrouped FROM items WHERE part_of IS NULL AND chosen = 1")[0]
        error = self.query("SELECT value FROM meta WHERE key = 'faces_error'")
        return {"chosen": row["chosen"] or 0, "done": row["done"] or 0, "faces": row["faces"] or 0,
                "ungrouped": row["ungrouped"] or 0, "error": error[0]["value"] if error else None}

    def look_for_faces_again(self) -> None:
        """Forget the faces found in the chosen files, so the next Analyse looks again (names you gave stay)."""
        with self._lock:
            self.db.execute("DELETE FROM faces WHERE path IN (SELECT path FROM items WHERE part_of IS NULL AND chosen = 1)")
            self.db.execute("UPDATE items SET faces_done = 0 WHERE chosen = 1")
            self.db.execute("DELETE FROM meta WHERE key = 'faces_error'")
            self.db.commit()

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
                    "INSERT INTO items (path, kind, taken_at, taken_orig, duration, has_audio, hdr, mtime, lat, lon, camera) "
                    "VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                    (path, item.kind, item.taken_at.isoformat(), item.taken_at.isoformat(), item.duration, int(item.has_audio),
                     int(item.hdr), mtime, item.lat, item.lon, item.camera or timeshift.OTHER),
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
        rows = self.query("SELECT path, clip FROM items WHERE part_of IS NULL AND chosen = 1 AND clip IS NOT NULL ORDER BY taken_at")
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


def LEARNED_FILE() -> Path:
    return app_dir() / "themes-learned.json"


def _learned_themes() -> dict:
    """What earlier trips taught about your themes: {"activities": [...], "trips": {trip: {activity: [vectors]}}}."""
    try:
        return json.loads(LEARNED_FILE().read_text())
    except (OSError, ValueError):
        return {}


def _save_learned(learned: dict) -> None:
    try:
        LEARNED_FILE().write_text(json.dumps(learned))
    except OSError:
        pass


def _look_vector(blob: bytes | None) -> np.ndarray | None:
    """A shot's colours and layout as one vector (for a video: its middle frame), for timeshift.suggest
    when the AI has not looked at the shots yet."""
    if blob is None:
        return None
    try:
        look = stories._unpack(blob)
    except (OSError, ValueError, KeyError):
        return None
    if "frames" in look:
        if not look["frames"]:
            return None
        look = look["frames"][len(look["frames"]) // 2]
    colours, layout = np.sqrt(np.maximum(look["colours"], 0)), look["layout"]
    return np.concatenate([colours / (np.linalg.norm(colours) + 1e-9), layout / (np.linalg.norm(layout) + 1e-9)])


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
    camera = [round(float(np.hypot(x, y)), 2) for x, y in zip(moved.move_x, moved.move_y)]
    return json.dumps({"action": moved.action_steps, "shake": moved.shake_steps, "loud": loud,
                       "sharp": moved.sharp_steps, "bright": moved.bright_steps, "camera": camera})


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


def kept_part(row: dict) -> tuple[float, float] | None:
    """The part of a video you kept when you cut it (Choose files or Library), or None for the whole clip."""
    start, end = row.get("keep_start"), row.get("keep_end")
    return (float(start), float(end)) if start is not None and end is not None and end > start else None


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


def _video_frames(row: dict, out_dir: Path) -> list[tuple[float, Path]]:
    """Small frames about every second inside a video, with their times. Its key frames first (one quick pass, no
    decoding of the rest); if they are too far apart, frames read at each second."""
    turn = rotate_filter(row.get("rotation") or 0)
    result = subprocess.run(
        [ffmpeg_paths()[0], "-hide_banner", "-skip_frame", "nokey", "-i", row["path"], "-an",
         "-vf", turn + "scale=224:-2,showinfo", "-vsync", "vfr", "-q:v", "5", str(out_dir / "k%04d.jpg")],
        capture_output=True, text=True)
    times = [float(m) for m in re.findall(r"pts_time:([0-9.]+)", result.stderr)]
    files = sorted(out_dir.glob("k*.jpg"))
    if len(files) == len(times):
        chosen = moments.pick_frames(times, row["duration"] or 0)
        if chosen:
            return [(round(times[k], 2), files[k]) for k in chosen]
    pictures = []
    for n, t in enumerate(moments.frame_times(row["duration"] or 0)[::2]):  # every 2 s: each one is a seek
        shot = out_dir / f"f{n}.jpg"
        subprocess.run([ffmpeg_paths()[0], "-loglevel", "error", "-y", "-ss", f"{t:.2f}", "-i", row["path"],
                        "-frames:v", "1", "-vf", turn + "scale=224:-2", "-q:v", "5", str(shot)], capture_output=True)
        if shot.exists():
            pictures.append((t, shot))
    return pictures


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
