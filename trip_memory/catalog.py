"""The trip catalogue: everything the app has learned about a trip, saved so it is only worked out once.

Each trip folder gets its own small database (SQLite) plus thumbnail pictures, stored in
~/Library/Application Support/TripMemory, never inside your photo folders.
"""

from __future__ import annotations

import hashlib
import io
import json
import sqlite3
import subprocess
import sys
import tempfile
import time
import threading
import traceback
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

from .faces import FaceFinder, group_faces
from .media import MediaItem, scan_folder
from . import autoedit, director, motion, musicmood, stories, themes
from .scenes import SceneTagger, mean_vector
from .tags import TAG_GROUP, Tagger, group_name, look_alike_groups, people_tag, story_subject
from .tools import ffmpeg_paths, probe
from .looks import LOOKS
from .render import RAMP_SOURCE, held, source_factor


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
            "theme TEXT", "theme_auto TEXT", "theme_sure REAL",
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
        inside = {p for r in rows if r["effects"] for p in json.loads(r["effects"]).get("collage", [])}
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
                    seconds=r["seconds"],
                    sound=bool(r["clip_sound"]),  # videos are silent unless you switch their sound on
                    volume=1.0 if r["clip_volume"] is None else r["clip_volume"],
                    speed=r["speed"] or 1.0,
                    effects=self._with_parts(r["effects"], turned),
                    overlay=shown.get(r["path"]),
                    focus=where.get(r["path"]),
                )
            )
        return result

    @staticmethod
    def _with_parts(effects: str | None, turned: dict[str, int]) -> dict | None:
        """A shot's effects; a collage also gets how each of its photos is turned."""
        if not effects:
            return None
        fx = json.loads(effects)
        if fx.get("collage"):
            fx["turns"] = [turned.get(p, 0) for p in fx["collage"]]
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
                "end_sub": self.folder.name.replace("_", " "), "style": "bold", "end_black": 3.0, **saved}

    def end_card(self) -> tuple[float, dict | None]:
        """(seconds of black at the end, the words on it) for the video's ending."""
        t = self.title_settings()
        words = ({"text": t["end_text"], "sub": t["end_sub"], "style": "elegant", "where": "center"}
                 if t["ending"] and (t["end_text"] or t["end_sub"]) else None)
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
        chapters_on = self.storyline()["order"] != themes.TIME
        result: dict[str, dict] = {}
        previous = object()
        for k, r in enumerate(board):
            if r["story"] != previous and t["stories"] and k > 0 and not chapters_on:
                places = [x["place"] or x["country"] for x in board if x["story"] == r["story"] and (x["place"] or x["country"])]
                when = datetime.fromisoformat(r["taken_at"]).astimezone()
                sub = (max(set(places), key=places.count) + " · " if places else "") + f"{when:%a %-d %b}"
                result[r["path"]] = {"text": names.get(r["story"], ""), "sub": sub, "style": t["story_style"],
                                     "where": "", "auto": "story"}
            previous = r["story"]
        if t["opening"] and (t["trip_title"] or t["trip_sub"]):
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
                "UPDATE items SET rotation = ?, rotation_checked = 1, faces_done = 0, scene_done = 0, tags_done = 0, dhash = NULL, features = NULL "
                "WHERE path = ?",
                (rotation, path),
            )
            self.db.commit()
        row = self.query("SELECT * FROM items WHERE path = ?", (path,))[0]
        self.execute("UPDATE items SET thumb = ? WHERE path = ?", (str(self._make_thumb(row)), path))

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
        return pool

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
        chapters, where = themes.chapters_for(storyline, rows)
        self._chapters = chapters
        pool = []
        for r, chapter in zip(rows, where):
            if chapter is None:
                continue
            if kept_part(r):  # a clip you cut: Auto-edit only sees the part you kept
                start, end = kept_part(r)
                r = r | {"duration": end - start, "keep_offset": start, "full_duration": r["duration"]}
            pool.append(r | {"chapter": chapter, "pace_want": chapters[chapter].get("pace")})
        return pool

    # ---------- themes ----------

    def activities(self) -> list[dict]:
        """The activities a shot can have (the app's own and the ones you added), with their CLIP vectors."""
        rows = self.query("SELECT value FROM meta WHERE key = 'activities'")
        return json.loads(rows[0]["value"]) if rows else themes.default_activities()

    def save_activities(self, activities: list[dict]) -> None:
        self.execute("INSERT OR REPLACE INTO meta (key, value) VALUES ('activities', ?)", (json.dumps(activities),))

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
        seen = [r for r in rows if r["clip"] is not None]
        vectors = np.array([np.frombuffer(r["clip"], dtype=np.float32) for r in seen]) if seen else np.zeros((0, 512))
        confirmed = [(i, r["theme"]) for i, r in enumerate(seen) if r["theme"] in names]
        guesses = {r["path"]: g for r, g in zip(seen, themes.suggest(vectors, activities, confirmed))}
        updates = []
        for r in rows:
            if r["path"] in guesses:
                name, sure = guesses[r["path"]]
            else:  # no picture fingerprint yet: a first guess from its tags
                tags = json.loads(r["tags_user"]) if r["tags_user"] is not None else json.loads(r["tags"] or "[]")
                name, sure = themes.by_tags(tags, r["scene_user"] or r["scene"]), 0.3
            updates.append((name, round(sure, 3), r["path"]))
        with self._lock:
            self.db.executemany("UPDATE items SET theme_auto = ?, theme_sure = ? WHERE path = ?", updates)
            self.db.commit()

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

    def unconfirm_theme(self, paths: list[str]) -> None:
        with self._lock:
            self.db.executemany("UPDATE items SET theme = NULL WHERE path = ?", [(p,) for p in paths])
            self.db.commit()
        self.suggest_themes()

    def add_activity(self, name: str, vector: list[float] | None) -> None:
        activities = self.activities()
        if any(a["name"].lower() == name.lower() for a in activities):
            return
        new = themes.new_activity(name) | {"vector": vector}
        at = next((k for k, a in enumerate(activities) if a["name"] == themes.OTHER), len(activities))
        activities.insert(at, new)
        self.save_activities(activities)
        self.suggest_themes()

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
        return {"order": themes.TIME, "blocks": [], "split": [], "finale": True} | saved

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
            parts = [(p,) for shot in result.shots for p in (shot.effects or {}).get("collage", [])]
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
        with self._lock:
            self.db.executemany(
                "UPDATE items SET story = ?, similar_set = ?, best = ? WHERE path = ?",
                [(st, sim, int(best), path) for path, (st, sim, best) in groups.items()],
            )
            self.db.commit()

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
