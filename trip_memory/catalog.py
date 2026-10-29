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
from .scenes import SceneTagger
from .tools import ffmpeg_paths


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
    faces_done INTEGER DEFAULT 0,
    scene_done INTEGER DEFAULT 0
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
        for column in ("chosen INTEGER DEFAULT 1", "position REAL", "trim_start REAL", "trim_end REAL", "transition TEXT"):
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
        for row in rows:
            ids = sorted(by_path.get(row["path"], set()))
            row["person_ids"] = ids
            row["people"] = [people[i]["label"] for i in ids]
            row["scene_final"] = row["scene_user"] or row["scene"]
            row["day"] = row["taken_at"][:10]
            row["place_guessed"] = False
            row.pop("clip", None)
        _guess_places(rows)
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

        Files you moved keep their place; anything not placed yet follows in date order.
        """
        rows = [it for it in self.items() if it["chosen"] and it["included"]]
        placed = sorted((r for r in rows if r["position"] is not None), key=lambda r: r["position"])
        return placed + [r for r in rows if r["position"] is None]

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
                    trim_start=r["trim_start"],
                    trim_end=r["trim_end"],
                    transition=r["transition"],
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
            self.db.executemany("UPDATE items SET included = ? WHERE path = ?", [(int(included), p) for p in paths])
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

    def reset_order(self) -> None:
        self.execute("UPDATE items SET position = NULL")

    def set_trim(self, path: str, start: float | None, end: float | None) -> None:
        self.execute("UPDATE items SET trim_start = ?, trim_end = ? WHERE path = ?", (start, end, path))

    def set_transition(self, path: str, transition: str | None) -> None:
        self.execute("UPDATE items SET transition = ? WHERE path = ?", (transition, path))

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

        todo = self.query("SELECT path, kind, duration FROM items WHERE thumb IS NULL ORDER BY taken_at")
        for i, row in enumerate(todo):
            p.fraction = i / len(todo)
            p.message = f"Making previews: {i + 1} of {len(todo)}"
            if i % 10 == 0:
                notify()
            try:
                self.execute("UPDATE items SET thumb = ? WHERE path = ?", (str(self._make_thumb(row)), row["path"]))
            except Exception as exc:
                p.notes.append(f"No preview for {Path(row['path']).name}: {exc}")

    def _analyse(self, notify: Callable[[], None]) -> None:
        self._prepare(notify)
        p = self.progress
        todo = self.query("SELECT * FROM items WHERE chosen = 1 AND (faces_done = 0 OR scene_done = 0) ORDER BY taken_at")
        faces = scenes = None
        if todo:
            p.message = "Loading the AI models (the first time this downloads about 1 GB)…"
            notify()
            faces = self._load(lambda: FaceFinder(app_dir() / "models"), "people")
            scenes = self._load(lambda: SceneTagger(app_dir() / "models" / "clip"), "scenes")

        for i, row in enumerate(todo):
            p.fraction = i / len(todo)
            p.message = f"Analysing {i + 1} of {len(todo)}: {Path(row['path']).name}"
            notify()
            try:
                self._analyse_item(row, faces, scenes)
            except Exception as exc:
                traceback.print_exc()
                p.notes.append(f"Could not analyse {Path(row['path']).name}: {exc}")

        p.message = "Grouping faces into people…"
        notify()
        self._group_people()

    def _thumb_path(self, path: str) -> Path:
        return self.dir / "thumbs" / f"{hashlib.sha1(path.encode()).hexdigest()[:16]}.jpg"

    def _make_thumb(self, row) -> Path:
        out = self._thumb_path(row["path"])
        if row["kind"] == "photo":
            with Image.open(row["path"]) as img:
                img.draft("RGB", (960, 960))  # fast for JPEGs
                img = ImageOps.exif_transpose(img).convert("RGB")
                img.thumbnail((480, 480))
                img.save(out, quality=85)
        else:
            subprocess.run(
                [ffmpeg_paths()[0], "-loglevel", "error", "-y", "-ss", f"{(row['duration'] or 0) / 2:.2f}",
                 "-i", row["path"], "-frames:v", "1", "-vf", "scale=480:-2", "-q:v", "4", str(out)],
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

    def _analyse_item(self, row: sqlite3.Row, faces, scenes) -> None:
        path = Path(row["path"])
        key = hashlib.sha1(str(path).encode()).hexdigest()[:16]
        with tempfile.TemporaryDirectory() as tmp:
            frames = _frames(path, row["kind"], row["duration"], Path(tmp))
            if not frames:
                raise RuntimeError("no picture could be read")
            updates: dict = {}

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

            if scenes is not None and not row["scene_done"]:
                vector = scenes.embed(frames)
                top = scenes.classify(vector)
                updates.update(scene=top[0][0], scene_top=json.dumps(top), clip=vector.tobytes(), scene_done=1)

        if not updates:
            return
        sets = ", ".join(f"{k} = ?" for k in updates)
        self.execute(f"UPDATE items SET {sets} WHERE path = ?", (*updates.values(), str(path)))

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


def _frames(path: Path, kind: str, duration: float, out_dir: Path) -> list[Path]:
    """Pictures to look at: the photo itself, or a few frames spread through a video."""
    if kind == "photo":
        out = out_dir / "photo.jpg"
        with Image.open(path) as img:
            img = ImageOps.exif_transpose(img).convert("RGB")
            img.thumbnail((1600, 1600))
            img.save(out, quality=90)
        return [out]
    count = 1 if duration < 4 else min(5, int(duration // 3))
    times = [duration / 2] if count == 1 else [duration * (0.1 + 0.8 * i / (count - 1)) for i in range(count)]
    frames = []
    for i, t in enumerate(times):
        out = out_dir / f"frame_{i}.jpg"
        subprocess.run(
            [ffmpeg_paths()[0], "-loglevel", "error", "-y", "-ss", f"{t:.2f}", "-i", str(path),
             "-frames:v", "1", "-vf", "scale='min(1600,iw)':-2", "-q:v", "3", str(out)],
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
