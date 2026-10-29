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
    faces_done INTEGER DEFAULT 0,
    scene_done INTEGER DEFAULT 0
);
CREATE TABLE IF NOT EXISTS people (
    id INTEGER PRIMARY KEY,
    name TEXT,
    hidden INTEGER DEFAULT 0
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
        """All items in date order, with their people's names."""
        rows = [dict(r) for r in self.query("SELECT * FROM items ORDER BY taken_at, path")]
        names = self.person_names()
        by_path: dict[str, set[int]] = {}
        for f in self.query("SELECT path, person_id FROM faces WHERE person_id IS NOT NULL"):
            by_path.setdefault(f["path"], set()).add(f["person_id"])
        hidden = {r["id"] for r in self.query("SELECT id FROM people WHERE hidden = 1")}
        for row in rows:
            ids = sorted(by_path.get(row["path"], set()) - hidden)
            row["person_ids"] = ids
            row["people"] = [names[i] for i in ids]
            row["scene_final"] = row["scene_user"] or row["scene"]
            row["day"] = row["taken_at"][:10]
            row.pop("clip", None)
        return rows

    def person_names(self) -> dict[int, str]:
        return {r["id"]: r["name"] or f"Person {r['id']}" for r in self.query("SELECT id, name FROM people")}

    def people(self) -> list[dict]:
        """People with how often they appear and a few example face pictures."""
        result = []
        for p in self.query("SELECT * FROM people WHERE hidden = 0"):
            faces = self.query(
                "SELECT crop, path FROM faces WHERE person_id = ? ORDER BY score DESC", (p["id"],)
            )
            if not faces:
                continue
            result.append(
                {
                    "id": p["id"],
                    "name": p["name"] or "",
                    "label": p["name"] or f"Person {p['id']}",
                    "faces": [f["crop"] for f in faces[:6]],
                    "count": len({f["path"] for f in faces}),
                }
            )
        result.sort(key=lambda p: (not p["name"], -p["count"]))
        return result

    def media_items(self, only_included: bool = True) -> list[MediaItem]:
        sql = "SELECT * FROM items" + (" WHERE included = 1" if only_included else "") + " ORDER BY taken_at, path"
        return [
            MediaItem(
                path=Path(r["path"]),
                kind=r["kind"],
                taken_at=datetime.fromisoformat(r["taken_at"]),
                duration=r["duration"],
                has_audio=bool(r["has_audio"]),
                hdr=bool(r["hdr"]),
            )
            for r in self.query(sql)
        ]

    # ---------- changes made by you ----------

    def set_included(self, paths: list[str], included: bool) -> None:
        with self._lock:
            self.db.executemany("UPDATE items SET included = ? WHERE path = ?", [(int(included), p) for p in paths])
            self.db.commit()

    def set_scene(self, path: str, scene: str | None) -> None:
        self.execute("UPDATE items SET scene_user = ? WHERE path = ?", (scene, path))

    def name_person(self, person_id: int, name: str) -> None:
        """Name a person. Using a name that already exists merges the two."""
        name = name.strip()
        if not name:
            self.execute("UPDATE people SET name = NULL WHERE id = ?", (person_id,))
            return
        same = self.query("SELECT id FROM people WHERE name = ? AND id != ?", (name, person_id))
        with self._lock:
            if same:
                self.db.execute("UPDATE faces SET person_id = ? WHERE person_id = ?", (same[0]["id"], person_id))
                self.db.execute("DELETE FROM people WHERE id = ?", (person_id,))
            else:
                self.db.execute("UPDATE people SET name = ? WHERE id = ?", (name, person_id))
            self.db.commit()

    def hide_person(self, person_id: int) -> None:
        self.execute("UPDATE people SET hidden = 1 WHERE id = ?", (person_id,))

    # ---------- analysis ----------

    def analyse(self, on_change: Callable[[], None] | None = None) -> None:
        """Scan the folder and work out places, scenes and people for anything new. Safe to re-run."""
        p = self.progress
        p.running, p.notes, p.fraction = True, [], 0.0
        notify = on_change or (lambda: None)
        try:
            p.message = "Looking through the folder…"
            notify()
            found, problems = scan_folder(self.folder)
            p.notes.extend(problems)
            self._store_found(found)

            p.message = "Finding place names…"
            notify()
            self._name_places()

            todo = self.query("SELECT * FROM items WHERE faces_done = 0 OR scene_done = 0 OR thumb IS NULL ORDER BY taken_at")
            faces = scenes = None
            if todo:
                p.message = "Loading the AI models (the first time this downloads about 650 MB)…"
                notify()
                faces = self._load(lambda: FaceFinder(app_dir() / "models"), "people")
                scenes = self._load(lambda: SceneTagger(app_dir() / "models" / "clip"), "scenes")

            for i, row in enumerate(todo):
                p.fraction = i / len(todo)
                p.message = f"Looking at {i + 1} of {len(todo)}: {Path(row['path']).name}"
                notify()
                try:
                    self._analyse_item(row, faces, scenes)
                except Exception as exc:
                    traceback.print_exc()
                    p.notes.append(f"Could not analyse {Path(row['path']).name}: {exc}")

            p.message = "Grouping faces into people…"
            notify()
            self._group_people()
            p.fraction, p.message = 1.0, "Done"
        finally:
            p.running = False
            notify()

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
            thumb = self.dir / "thumbs" / f"{key}.jpg"
            with Image.open(frames[len(frames) // 2]) as img:
                img.thumbnail((480, 480))
                img.save(thumb, quality=85)
            updates = {"thumb": str(thumb)}

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
