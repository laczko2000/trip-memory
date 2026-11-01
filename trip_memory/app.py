"""The Trip Memory screen. Runs locally and opens in your web browser."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
import hashlib
import json
import re
import subprocess
import sys
import time
import traceback
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from urllib.parse import quote

import numpy as np
from nicegui import app, background_tasks, run, ui

from .catalog import Trip, app_dir, kept_part
from .media import MediaItem
from . import transitions
from . import themefinder
from . import director
from . import looks
from . import titles
from . import stories
from . import music as free_music
from . import songmap
from . import musicmood
from . import pinfill
from . import themes
from . import moments
from . import highlights
from . import timeshift
from . import taps as song_taps
from .beats import song_length
from .autoedit import CUT_ON, shot_value
from .autoedit import Limits as AutoLimits
from .render import END_FADE, OVERLAP_SHARE, TRANSITIONS, RenderSettings, clip_range, held, render, shot_times, source_factor, style_between
from .scenes import SCENES, text_vectors
from .tags import ALL_TAGS, EDITABLE_TAGS, TAG_GROUP

OUTPUT_DIR = Path.home() / "Movies" / "Trip Memory"
MUSIC_SITES = [("YouTube Audio Library", "https://studio.youtube.com/channel/UC/music"),
               ("Pixabay Music", "https://pixabay.com/music/"),
               ("iTunes Store", "itms://itunes.apple.com/")]  # (opens the Mac's Music app on its store)
# Where a song you buy in the iTunes Store lands on a Mac (the Music app's own folder, by artist and album).
BOUGHT_SONGS = ["Music/Music/Media.localized/Music", "Music/Music/Media/Music", "Music/iTunes/iTunes Media/Music"]


def new_song_since(since: float) -> Path | None:
    """A song that arrived after `since`: in Downloads, or bought in the iTunes Store (the Music app's folder)."""
    found = free_music.find_download("", since)
    if found is not None:
        return found
    fresh = []
    for folder in BOUGHT_SONGS:
        root = Path.home() / folder
        try:
            for f in root.glob("*/*/*"):  # artist / album / song
                if f.suffix.lower() in free_music.AUDIO_SUFFIXES and f.stat().st_mtime >= since - 2:
                    fresh.append(f)
        except OSError:
            continue
    return max(fresh, key=lambda f: f.stat().st_mtime) if fresh else None

RESOLUTIONS = {"Full HD (1080p)": (1920, 1080), "HD (720p, faster)": (1280, 720),
               "Phone, upright 9:16 (Instagram, TikTok)": (1080, 1920)}
PHONE = "Phone, upright 9:16 (Instagram, TikTok)"
PAGE_SIZE = 48
TILES_PER_PAGE = 60  # Choose files draws this many at a time as you scroll
STORY_ROWS = 60
STORY_PAGE_SIZE = 25
DEFAULT_TRANSITION = "Default (from Make video)"
ALL = "All"
SETTINGS = app_dir() / "settings.json"


def load_settings() -> dict:
    try:
        return json.loads(SETTINGS.read_text())
    except (OSError, ValueError):
        return {}


def clock(seconds: float) -> str:
    """0:12.4 - a time in the video to a tenth of a second."""
    return f"{int(seconds // 60)}:{seconds % 60:04.1f}"


def save_settings(**values) -> None:
    SETTINGS.write_text(json.dumps({**load_settings(), **values}))
    SETTINGS.chmod(0o600)  # it can hold your Claude API key: only you may read it


def choose_with_finder(kind: str) -> str | None:
    """Open the Mac's own folder/file picker. Returns None elsewhere or if cancelled."""
    if sys.platform != "darwin":
        return None
    prompt = {
        "folder": 'choose folder with prompt "Choose your trip folder"',
        "music": 'choose file with prompt "Choose a music file" of type {"public.audio"}',
    }[kind]
    result = subprocess.run(
        ["osascript", "-e", "activate", "-e", f"POSIX path of ({prompt})"],
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() or None


def pil_data_url(picture) -> str:
    """A Pillow picture as an address the page can show directly."""
    import base64
    import io

    out = io.BytesIO()
    picture.save(out, "JPEG", quality=85)
    return "data:image/jpeg;base64," + base64.b64encode(out.getvalue()).decode()


def autoedit_pack() -> dict:
    from .autoedit import PACK
    return PACK


def data_url(path: str | None) -> str:
    """Web address for a thumbnail stored in the app's data folder."""
    if not path:
        return ""
    return "/data/" + str(Path(path).relative_to(app_dir()))


def fmt_time(iso: str) -> str:
    return datetime.fromisoformat(iso).astimezone().strftime("%a %d %b, %H:%M")


def clip_id(path: str) -> str:
    """A short id for a file, used in the ids of its pictures on the page."""
    return hashlib.sha1(path.encode()).hexdigest()[:12]


def fmt_length(seconds: float) -> str:
    m, s = divmod(round(seconds), 60)
    return f"{m}:{s:02d}"


def rotate_buttons(on_turn) -> None:
    """Two small buttons that turn a picture a quarter turn either way."""
    ui.button(icon="rotate_left", on_click=lambda: on_turn(-90)).props("flat dense round size=sm").tooltip(
        "Turn 90° anti-clockwise"
    )
    ui.button(icon="rotate_right", on_click=lambda: on_turn(90)).props("flat dense round size=sm").tooltip(
        "Turn 90° clockwise"
    )



def app_version() -> str:
    """The version you are running (it says which download you started)."""
    try:
        from importlib.metadata import version

        return version("trip-memory")
    except Exception:
        return "?"

class Page:
    """Everything on the screen for one browser tab."""

    def __init__(self) -> None:
        self.trip: Trip | None = None
        self.all_items: list[dict] = []  # every file in the folder
        self.items: list[dict] = []  # the files you chose for this trip
        self.choose_filters = {"day": ALL, "kind": ALL, "show": ALL}
        self.stale = False  # quick ticks in Choose files: the other tabs catch up when you open them
        self.board_limit = STORY_ROWS  # Storyboard rows shown (more on request: a long list is slow to draw)
        self.story_page = 1
        self.choose_rest: list = []  # Choose files: parts of the long list not drawn yet (drawn as you scroll)
        self.choose_left = 0
        self.choose_list = None
        self.choose_more = None
        self.choose_tiles: dict[str, tuple] = {}
        self.choose_counts: list[tuple] = []
        self.choose_focus: int | None = None  # a story to scroll to after the list is drawn again
        self.choose_focus_card = None
        self.choose_focus_path: str | None = None  # ...or the file to scroll to
        self.choose_whole: dict[int, list[dict]] = {}
        self.joined: set[int] = set()
        self.last_story = 0
        self.open_plans: set[str] = set()  # stories whose plan is shown in the Stories tab
        self.media_route: str | None = None
        self.auto_limits = AutoLimits(**load_settings().get("limits", {}))
        self.library_columns = load_settings().get("library_columns", 3)  # pictures in a row in the Library
        self.look_previews: dict[tuple[str, str], str] = {}  # (picture, look) -> a small preview
        self.song_maps: dict = {}
        self.track_lengths: dict = {}
        self.own_length_open = False  # "Use a different length" switched on, length not typed yet
        self.dirty = False
        self.last_refresh = 0.0
        self.filters = {"day": ALL, "theme": ALL, "place": ALL, "scene": ALL, "tag": ALL, "look": ALL, "person": ALL,
                        "show": ALL}
        self.draft_pick: int | None = None  # the clip chosen on the strip under the draft
        self.editor_full = False  # the editor open over the whole screen
        self.editor_zoom = 40  # pixels per second on its timeline (0: the whole video fits)
        self.draft_title: str | None = None  # the timeline title being changed
        self.auto_update = load_settings().get("auto_update", True)  # the draft updates by itself after a pause
        self.update_due: float | None = None
        self.draft_resume = 0.0
        self.draft_busy = False
        self.lock_views: dict[str, list[tuple]] = {}  # clip -> the places on screen that show its locked moments
        self.library_focus: dict | None = None  # one shot opened by double-click: {path, tab, name, scroll}
        self.pending_click = None  # a single click waiting to see whether it becomes a double-click
        self.page = 1
        self.rendering = {"busy": False, "progress": 0.0, "message": "", "result": None}
        self.ollama_found: tuple[list[str], list[str]] | None = None  # (models that see pictures, all models)
        self.ollama_checked = False
        self.ai_busy = False  # the AI director is working (its progress shows in its own card)
        self.recording = False  # the browser is listening for "speak your idea"
        self.voice_base = ""  # what was in the box before you started speaking
        self.talk_button = None
        self.wish_box = None  # where "speak your idea" writes
        self.story_stop = None
        self.theme_shown = ""  # the activity open in the Themes tab
        self.theme_moment = None  # the moment being marked there (a name), or None
        self.moment_from_story = None  # the My story part you came from to mark a moment, or None
        self.moment_found: dict = {}  # (activity, moment) -> the shots in the order found
        self.theme_selected: set[str] = set()
        self.theme_limit = 60
        self.nav_order: dict[str, list[str]] = {}  # Previous / Next in the big window: the tiles where you opened it
        self.vzoom_moved = None  # dragging the zoom frame on a video in the big window
        self.theme_ticks: dict = {}  # path -> (tick box, its mark, sure): ticking changes only that box
        self.theme_rows_shown: dict = {}
        self.theme_summary = None
        self.theme_badge = None
        self.moment_count = None
        self.theme_counts = [0, 0]
        self.theme_tiles: dict = {}  # path -> its tile, to show a selection without building the page again
        self.theme_count = self.theme_move = None

    # ---------- data ----------

    def reload(self) -> None:
        self.all_items = self.trip.items() if self.trip else []
        self.items = [it for it in self.all_items if it["chosen"]]
        self.dirty = False
        self.last_refresh = time.time()

    def shown(self) -> list[dict]:
        if self.library_focus:  # only the shot you double-clicked
            return [it for it in self.all_items if it["path"] == self.library_focus["path"]]
        f = self.filters
        out = []
        for it in self.items:
            if f["day"] != ALL and it["day"] != f["day"]:
                continue
            if f["theme"] != ALL and self.theme_of(it) != f["theme"]:
                continue
            if f["place"] != ALL and (it["place"] or "Unknown place") != f["place"]:
                continue
            if f["scene"] != ALL and (it["scene_final"] or "Not sorted yet") != f["scene"]:
                continue
            if f["tag"] != ALL and f["tag"] not in it["tags_final"] and f["tag"] != it["people_tag"]:
                continue
            if f["look"] != ALL and it["look_name"] != f["look"]:
                continue
            if f["person"] != ALL and f["person"] not in it["people"]:
                continue
            if f["show"] == "In the video" and not it["included"]:
                continue
            if f["show"] == "Left out" and it["included"]:
                continue
            out.append(it)
        return out

    @staticmethod
    def theme_of(it: dict) -> str:
        """A shot's theme: yours if you confirmed one, else the app's suggestion."""
        return it.get("theme") or it.get("theme_auto") or themes.OTHER

    def options(self) -> dict[str, dict]:
        days = sorted({it["day"] for it in self.items})
        day_opts = {ALL: "All days"}
        for n, d in enumerate(days, 1):
            day_opts[d] = f"Day {n} · {datetime.fromisoformat(d):%a %d %b}"
        places = sorted({it["place"] or "Unknown place" for it in self.items})
        scenes = sorted({it["scene_final"] or "Not sorted yet" for it in self.items})
        people = sorted({p for it in self.items for p in it["people"]})
        present = {t for it in self.items for t in [*it["tags_final"], it["people_tag"]] if t}
        tags = {t: f"{TAG_GROUP[t]}: {t}" for t in ALL_TAGS if t in present}
        looks = sorted({it["look_name"] for it in self.items if it["look_name"]}, key=lambda n: int(n.rsplit(" ", 1)[1]))
        theme_names = sorted({self.theme_of(it) for it in self.items})
        return {
            "day": day_opts,
            "theme": {ALL: "All themes", **{t: t for t in theme_names}},
            "place": {ALL: "All places", **{p: p for p in places}},
            "scene": {ALL: "All scenes", **{s: s for s in scenes}},
            "tag": {ALL: "All tags", **tags},
            "look": {ALL: "All", **{n: n for n in looks}},
            "person": {ALL: "Everyone", **{p: p for p in people}},
            "show": {ALL: "In and out of the video", "In the video": "Only in the video", "Left out": "Only left out"},
        }

    # ---------- building the screen ----------

    def build(self) -> None:
        ui.colors(primary="#2563eb")
        ui.on("tm_dictation", self.dictation)
        ui.on("tm_story_drop", self.story_drop)
        ui.on("tm_theme_drop", self.theme_drop)
        ui.on("tm_board_drop", self.board_drop)
        ui.on("tm_draft_drop", self.draft_drop)
        ui.on("tm_title_move", self.title_move)
        ui.on("tm_title_click", self.title_click)
        ui.on("tm_title_menu", self.title_menu)
        ui.on("tm_postit", self.postit_key)
        ui.timer(1.0, self.auto_update_tick)  # (here, outside every redrawn part, so it keeps running)
        self.window_box = ui.element("div")  # the shot window lives here: redrawing a tab never closes it
        self.window_timers = ui.element("div")
        ui.add_css(".thumb { height: 150px; object-fit: cover; }")
        # The big window fills the screen without scrolling: the video takes whatever room the rest leaves.
        ui.add_css(".tm-fill { display: flex; flex-direction: column; } .tm-fill > * { flex: 1 1 0; min-height: 0; }"
                   ".tm-fill > * > .tm-hover { height: 100% !important; }"
                   ".tm-bigwin { display: flex; flex-direction: column; flex-wrap: nowrap; overflow: hidden; }"
                   ".tm-bigwin > * { flex: none; }")
        with ui.column().classes("w-full max-w-6xl mx-auto p-4 gap-3"):
            with ui.row().classes("items-baseline gap-2"):
                ui.label("Trip Memory").classes("text-3xl font-bold")
                ui.label(f"version {app_version()}").classes("text-sm text-gray-500")
            self.build_folder_row()
            self.guided = bool(load_settings().get("guided", True))
            self.guide_step = 0
            self.guide_detour = False
            self.guide_bar()
            with ui.tabs().classes("w-full") as tabs:
                # The path to a video, left to right; then the extra tabs for looking things up.
                self.choose_tab = ui.tab("Choose files", icon="checklist")
                self.panels = None
                themes_tab = self.themes_tab = ui.tab("Themes", icon="category")
                music_tab = ui.tab("Music & audio", icon="graphic_eq")
                story_tab = self.storyboard_tab = ui.tab("Storyboard", icon="view_timeline")
                video_tab = self.video_tab = ui.tab("Make video", icon="movie")
                ui.separator().props("vertical").classes("mx-2")
                library_tab = ui.tab("Library", icon="photo_library").classes("opacity-70")
                people_tab = ui.tab("People", icon="people").classes("opacity-70")
                stories_tab = ui.tab("Stories", icon="auto_stories").classes("opacity-70")
            self.library_tab = library_tab
            self.music_tab = music_tab
            self.people_tab = people_tab
            self.stories_tab = stories_tab
            self.tabs_bar = tabs
            tabs.set_visibility(not self.guided)
            with ui.tab_panels(tabs, value=self.choose_tab, on_change=self.tab_opened).classes("w-full") as self.panels:
                with ui.tab_panel(self.choose_tab):
                    self.choose_view()
                with ui.tab_panel(themes_tab):
                    self.themes_view()
                with ui.tab_panel(music_tab):
                    self.music_tab_view()
                with ui.tab_panel(story_tab):
                    self.storyboard_view()
                with ui.tab_panel(video_tab):
                    self.build_video()
                with ui.tab_panel(library_tab):
                    self.build_library()
                with ui.tab_panel(people_tab):
                    self.people_view()
                with ui.tab_panel(stories_tab):
                    self.director_card()
                    self.stories_view()
        ui.timer(1.0, self.tick)
        last = load_settings().get("folder")
        if last and Path(last).is_dir():
            self.folder_input.value = last
            ui.timer(0.1, self.open_trip, once=True)

    def build_folder_row(self) -> None:
        with ui.card().classes("w-full"):
            with ui.row().classes("w-full items-center"):
                self.folder_input = ui.input("Trip folder", placeholder="/Users/you/Pictures/Lisbon 2025").classes("flex-grow")
                if sys.platform == "darwin":
                    ui.button("Browse…", on_click=self.browse_folder).props("outline")
                self.open_button = ui.button("Open folder", icon="folder_open", on_click=self.open_trip)
            self.progress_bar = ui.linear_progress(value=0, show_value=False).classes("w-full")
            self.progress_label = ui.label(
                "Choose a trip folder, then pick the files you want. The app finds places, scenes and people in them."
            ).classes("text-gray-600")
            self.notes_label = ui.label("").classes("text-orange-600 text-sm whitespace-pre-line")
            self.progress_bar.set_visibility(False)
            # Any AI step can be stopped; what it finished so far is kept.
            self.ai_stop = ui.button("Stop the AI (keep what it has done)", icon="stop_circle", on_click=self.stop_ai) \
                .props("outline color=negative dense no-caps")
            self.ai_stop.set_visibility(False)

    async def browse_folder(self) -> None:
        path = await run.io_bound(choose_with_finder, "folder")
        if path:
            self.folder_input.value = path
            await self.open_trip()

    async def open_trip(self) -> None:
        """Open a folder and take the quick first look (dates, places, previews)."""
        folder = (self.folder_input.value or "").strip().strip("'\"")
        if not folder or not Path(folder).expanduser().is_dir():
            ui.notify("Please choose a folder that exists.", type="warning")
            return
        if self.trip and self.trip.progress.running:
            ui.notify("Still working, please wait.", type="warning")
            return
        self.trip = Trip(folder)
        self.media_route = serve_trip_folder(self.trip)
        save_settings(folder=str(self.trip.folder))
        self.filters = {k: ALL for k in self.filters}
        self.choose_filters = {k: ALL for k in self.choose_filters}
        song = self.trip.song()
        if song and Path(song).is_file() and hasattr(self, "music_input") and self.music_input.value != song:
            self.music_input.value = song  # the song you chose last time
        self.refresh_all()
        await self.run_step(self.trip.prepare)
        analysed = any(it["faces_done"] or it["scene_done"] for it in self.items)
        self.panels.value = self.library_tab if analysed else self.choose_tab

    async def run_step(self, step) -> None:
        self.open_button.disable()
        self.progress_bar.set_visibility(True)
        try:
            await run.io_bound(step, self.mark_dirty)
        except director.DirectorError as exc:
            ui.notify(str(exc), type="warning", multi_line=True, timeout=10000)
        except Exception as exc:
            traceback.print_exc()
            ui.notify(f"Stopped: {exc}", type="negative")
        finally:
            self.open_button.enable()
        self.refresh_all()

    async def analyse_chosen(self) -> None:
        if not self.trip or self.trip.progress.running:
            return
        if not self.items:
            ui.notify("Choose at least one file first.", type="warning")
            return
        await self.run_step(self.trip.analyse)
        self.panels.value = self.library_tab

    def mark_dirty(self) -> None:
        self.dirty = True

    def tick(self) -> None:
        if self.trip:
            p = self.trip.progress
            if p.running:
                self.progress_bar.set_visibility(True)
                self.progress_bar.value = p.fraction
                self.progress_label.text = p.message
                # (while it works through a big trip, redraw less often: every redraw of a page with thousands
                # of files competes with the work for the computer, and the page can lose its connection)
                if self.dirty and time.time() - self.last_refresh > max(8.0, len(self.all_items) / 40):
                    self.refresh_all()
                if getattr(self, "ai_bar", None) is not None and self.ai_busy:
                    self.ai_bar.set_visibility(True)
                    self.ai_bar.value = p.fraction
                    self.ai_label.text = p.message
            else:
                self.progress_bar.set_visibility(False)
                self.progress_label.text = (
                    f"{self.trip.folder.name}: {len(self.all_items)} files in the folder, {len(self.items)} chosen."
                )
            self.notes_label.text = "\n".join(p.notes[:8])
        if getattr(self, "ai_stop", None) is not None:
            self.ai_stop.set_visibility(self.ai_busy)
            if getattr(self, "ai_stop_here", None) is not None:
                self.ai_stop_here.set_visibility(self.ai_busy)
        r = self.rendering
        if r["busy"]:
            self.render_progress.value = r["progress"]
            self.render_status.text = r["message"]

    def refresh_all(self) -> None:
        slow = []
        for name in ("reload", "sync_video_size", "choose_view", "filters_view", "gallery", "people_view",
                     "video_summary", "editor_card", "titles_card", "looks_card", "storyboard_view", "themes_view",
                     "stories_view", "director_card"):
            started = time.time()
            part = getattr(self, name)
            part.refresh() if hasattr(part, "refresh") else part()
            if time.time() - started > 0.3:
                slow.append(f"{name} {time.time() - started:.1f} s")
        if slow:  # (in the Terminal: what makes the page slow with a big trip)
            print("Slow to draw:", ", ".join(slow), flush=True)

    # ---------- Choose files ----------

    # ---------- the guided way: one step after another, Next and Back ----------

    GUIDE = [
        ("Files", "Choose files", None,
         "Open your trip folder. The app looks at everything once (faces, places, look-alikes). Untick what you "
         "don't want in the video."),
        ("Doubles", "Choose files", "tm-step-doubles",
         "Look-alike shots are grouped: keep the best of each (⭐ is the app's pick), leave the rest out."),
        ("Themes & best bits", "Themes", None,
         "Confirm what each shot is: tick the right ones, ⇄ moves a wrong one. Run the Second check. Mark the best "
         "bits in your videos with post-its."),
        ("Music", "Music & audio", None, "Choose the song, trim it, and put pins where you want the cuts."),
        ("Story", "Storyboard", None,
         "Choose how the story runs (As it happened, By theme or My story) and press Make the first cut."),
        ("Watch & finish", "Make video", None,
         "Open the editor: watch it, change clips, titles, cards and transitions, then Finalize."),
    ]

    def guide_tab(self, name: str):
        return {"Choose files": self.choose_tab, "Themes": self.themes_tab, "Music & audio": self.music_tab,
                "Storyboard": self.storyboard_tab, "Make video": self.video_tab}[name]

    @ui.refreshable
    def guide_bar(self) -> None:
        if not self.guided:
            with ui.row().classes("w-full justify-end -mb-2"):
                ui.button("Guided steps", icon="assistant_navigation", on_click=lambda: self.set_guided(True)) \
                    .props("flat dense no-caps size=sm").tooltip("One step after another, with Next and Back")
            return
        k = self.guide_step
        title, _, _, words = self.GUIDE[k]
        with ui.card().classes("w-full gap-2 bg-slate-50"):
            with ui.row().classes("w-full items-center gap-1 no-wrap overflow-x-auto"):
                for n, (name, *_rest) in enumerate(self.GUIDE):
                    done, now = n < k, n == k
                    ui.button(f"{'✓' if done else n + 1} {name}", on_click=lambda n=n: self.go_step(n)) \
                        .props("dense no-caps rounded " + ("unelevated color=primary" if now else
                                                          "flat color=positive" if done else "flat color=grey-8"))
                    if n < len(self.GUIDE) - 1:
                        ui.icon("chevron_right").classes("text-gray-400")
                ui.space()
                with ui.button("More", icon="more_horiz").props("flat dense no-caps"):
                    with ui.menu():
                        ui.menu_item("Library: all shots", on_click=lambda: self.guide_open(self.library_tab))
                        ui.menu_item("People", on_click=lambda: self.guide_open(self.people_tab))
                        ui.menu_item("Stories", on_click=lambda: self.guide_open(self.stories_tab))
                        ui.separator()
                        ui.menu_item("Show all the tabs instead", on_click=lambda: self.set_guided(False))
            with ui.row().classes("w-full items-center gap-3 no-wrap"):
                ui.label(f"Step {k + 1} of {len(self.GUIDE)}: {title}").classes("text-lg font-semibold flex-none")
                ui.label(words).classes("text-sm text-gray-700 flex-1")
                if k == 2:
                    ui.button("📌 Mark the best bits", on_click=self.guide_best_bits).props("outline dense no-caps")
                if k == 5:
                    ui.button("Open the editor", icon="movie_edit", on_click=lambda: self.open_editor(True)) \
                        .props("unelevated dense no-caps color=positive")
                if k:
                    ui.button("Back", icon="chevron_left", on_click=lambda: self.go_step(k - 1)) \
                        .props("outline dense no-caps")
                if k < len(self.GUIDE) - 1:
                    ui.button(f"Next: {self.GUIDE[k + 1][0]}",
                              on_click=lambda: self.go_step(k + 1)).props("unelevated dense no-caps")
        if k < len(self.GUIDE) - 1:  # (also at the bottom of the screen, wherever you scrolled)
            ui.button(f"Next: {self.GUIDE[k + 1][0]} →", on_click=lambda: self.go_step(k + 1)) \
                .props("unelevated no-caps rounded color=primary").classes("shadow-lg") \
                .style("position:fixed;right:24px;bottom:24px;z-index:1500")

    def set_guided(self, on: bool) -> None:
        self.guided = on
        save_settings(guided=on)
        self.tabs_bar.set_visibility(not on)
        if on:
            self.guide_step = self._step_for_panel(default=self.guide_step)
        self.guide_bar.refresh()
        self.storyboard_view.refresh()

    def _step_for_panel(self, default: int = 0) -> int:
        here = [n for n, g in enumerate(self.GUIDE) if self.showing(g[1])]
        return default if not here or default in here else here[0]

    def go_step(self, k: int) -> None:
        self.guide_step = max(0, min(len(self.GUIDE) - 1, k))
        self.guide_detour = False
        _, tab, anchor, _ = self.GUIDE[self.guide_step]
        self.panels.set_value(self.guide_tab(tab))
        self.guide_bar.refresh()
        if anchor:
            ui.run_javascript(f"setTimeout(() => {{ const e = document.getElementById('{anchor}'); "
                              f"if (e) e.scrollIntoView({{behavior: 'smooth', block: 'start'}}); }}, 400)")
        else:
            ui.run_javascript("setTimeout(() => window.scrollTo({top: 0, behavior: 'smooth'}), 200)")

    def guide_open(self, tab) -> None:
        self.guide_detour = True
        self.panels.set_value(tab)

    def guide_best_bits(self) -> None:
        self.guide_detour = True
        self.panels.set_value(self.choose_tab)
        ui.run_javascript("setTimeout(() => { const e = document.getElementById('tm-step-bestbits'); "
                          "if (e) e.scrollIntoView({behavior: 'smooth', block: 'start'}); }, 500)")

    def tab_opened(self) -> None:
        # Tabs are only built while open (a big trip would otherwise make the page very slow), so build this one now.
        if getattr(self, "guided", False) and not self.guide_detour:
            step = self._step_for_panel(default=self.guide_step)
            if step != self.guide_step:  # (a button took you to another step's screen)
                self.guide_step = step
                self.guide_bar.refresh()
        self.stale = False
        if self.library_focus and not self.showing("Library"):
            self.library_focus = None  # you left the Library by its tab: it shows everything again next time
        self.refresh_all()

    def showing(self, name: str) -> bool:
        """Whether this tab is the one open (only the open tab is built)."""
        if self.panels is None:
            return name == "Choose files"
        value = self.panels.value
        return (value if isinstance(value, str) else getattr(value, "_props", {}).get("name")) == name

    def choose_shown(self) -> list[dict]:
        f = self.choose_filters
        if f["show"] == "removed":
            return self.trip.items(removed=True) if self.trip else []
        sets: dict[int, int] = {}
        for it in self.all_items:
            sets[it["similar_set"] or -1] = sets.get(it["similar_set"] or -1, 0) + 1
        return [
            it
            for it in self.all_items
            if (f["day"] == ALL or it["day"] == f["day"]) and (f["kind"] == ALL or it["kind"] == f["kind"])
            and (f["show"] != "sets" or sets.get(it["similar_set"] or -1, 0) > 1)
            and (f["show"] != "ticked" or it["chosen"]) and (f["show"] != "unticked" or not it["chosen"])
        ]

    def sort_out_card(self) -> None:
        """The first pass over a big folder: throw out doubles and anything you don't want, before the slow steps."""
        total = len(self.all_items)
        chosen = sum(1 for it in self.all_items if it["chosen"])
        removed = self.trip.removed_count()
        size = sum(_file_size(it["path"]) for it in self.all_items)
        doubles = total - len({it["similar_set"] for it in self.all_items})
        with ui.card().classes("w-full bg-amber-50").props('id="tm-step-doubles"'):
            ui.label("Sort out doubles" if getattr(self, "guided", False) else "Step 1: sort out").classes("text-lg font-semibold")
            self.sort_label = ui.label(self.sort_text(total, chosen, size, removed)).classes("text-sm")
            ui.label(
                f"Go through quickly and untick what you don't want. {doubles} files are doubles or triples of another "
                "shot; Show → Only doubles and triples lists just those. Then remove the unticked files from the trip: "
                "they disappear from every tab and are never analysed, so everything after this is faster. "
                "Your original files on the Mac are not touched."
            ).classes("text-sm text-gray-700")
            with ui.row().classes("items-center gap-2"):
                level = self.trip.look_alike_level()
                ui.label("Doubles:").classes("text-sm font-medium")
                for key, name in stories.STRICTNESS_NAMES.items():
                    ui.button(name, on_click=lambda k=key: self.set_look_alike(k)).props(
                        "dense no-caps size=sm " + ("unelevated color=amber-9" if key == level else "outline color=amber-9"))\
                        .tooltip("Look for doubles at this level. Start with level 1, accept, then go on to 2 and 3: "
                                 "the shots you kept can pair up again at the next level. Never more than 10 min apart.")
            with ui.row().classes("items-center gap-2"):
                ui.button("✓ Accept all markings", icon="done_all", on_click=lambda: self.accept_sets(None)) \
                    .props("unelevated no-caps color=primary") \
                    .tooltip("Every look-alike set: the ticks as they are (a set you didn't touch keeps only its ⭐), "
                             "and the set is done")
                ui.button("Keep only the ⭐ of every set", icon="star", on_click=self.suggest).props("outline") \
                    .tooltip("Untick every double or triple except the best one (sharpest, best exposed)")
                self.remove_button = ui.button(f"Remove {total - chosen} unticked from the trip", icon="delete_sweep",
                                               on_click=self.confirm_remove).props("color=negative")
                self.remove_button.set_visibility(total > chosen)
                cleaned = len(self.trip.cleaned_sets())
                if cleaned:
                    ui.button("Show the similar-shot frames again", icon="undo", on_click=self.regroup_sets) \
                        .props("flat dense").tooltip(f"{cleaned} shots were sorted out with Keep marked")
                if removed:
                    ui.button(f"See removed ({removed})", icon="restore_from_trash",
                              on_click=lambda: self.set_choose_filter("show", "removed")).props("flat")

        # Always in view while you scroll and untick: how many are unticked, and the button to remove them.
        with ui.element("div").style(
            "position:fixed;top:12px;left:50%;transform:translateX(-50%);z-index:3000"
        ) as self.remove_bar:
            with ui.card().classes("bg-red-50 border border-red-200 shadow-lg py-1 px-3"):
                with ui.row().classes("items-center gap-3 no-wrap"):
                    self.remove_bar_label = ui.label("").classes("text-sm font-medium")
                    ui.button("Remove them from the trip", icon="delete_sweep", on_click=self.confirm_remove) \
                        .props("color=negative dense")
        self.update_sort_counts()

    def camera_clocks_card(self) -> None:
        """Each camera in the trip, and its clock: a GoPro set to the wrong time puts its shots in the wrong place,
        so its whole clock can be moved at once (or the app finds by how much from shots of the same moment)."""
        cams = self.trip.cameras()
        if len(cams) < 2 and not any(c["shift"] for c in cams):
            return
        moved = [c for c in cams if c["shift"]]
        caption = ", ".join(f"{c['camera']} {timeshift.fmt_shift(c['shift'])}" for c in moved) if moved \
            else "Is a camera's time wrong (a GoPro often is)? Put it right here"
        with ui.expansion(f"Camera clocks · {len(cams)} cameras", caption=caption, icon="schedule") \
                .classes("w-full mt-2 bg-sky-50 rounded"):
            ui.label("Each camera's shots go into the trip by the time its clock gave them. If a camera's clock was "
                     "wrong, its shots land in the wrong story: move its clock and all its files move together. "
                     "Your files are not changed, only where they sit in the trip.").classes("text-sm text-gray-700")
            for c in cams:
                self.camera_row(c)

    # ---------- post-its: mark the best bits of each video ----------

    def postit_card(self) -> None:
        videos = [it for it in self.all_items if it["kind"] == "video" and it["chosen"]]
        if not videos:
            return
        marked = sum(1 for it in videos if it.get("postits"))
        with ui.card().classes("w-full mt-2 bg-yellow-50").props('id="tm-step-bestbits"'):
            with ui.row().classes("w-full items-center gap-3"):
                ui.label("📌 Mark the best bits" if getattr(self, "guided", False) else "Step 1b: mark the best bits") \
                    .classes("text-lg font-semibold")
                ui.label(f"{marked} of {len(videos)} videos marked").classes("text-sm text-gray-600")
                ui.space()
                ui.button("Mark the best bits", icon="sticky_note_2",
                          on_click=lambda: self.postit_dialog(0)).props("unelevated color=amber-8 no-caps")
            ui.label("Watch each video and tap the Shift key at every exciting bit (a jump, a splash, a laugh) to stick "
                     "a post-it there. Your first post-it becomes the video's picture everywhere, which makes sorting "
                     "themes much quicker, and the first cut uses the seconds around it. Optional: skip any video.") \
                .classes("text-sm text-gray-700")
            self.highlight_ideas_box()

    def highlight_ideas_box(self) -> None:
        """The app learns your highlights (all trips) and suggests more in orange: keep or drop each one."""
        learned, dropped = self.trip.highlights_learned()
        ideas = [i for i in self.trip.highlight_ideas() if any(x["path"] == i["path"] for x in self.all_items)]
        ready = learned >= highlights.START_FROM
        with ui.row().classes("w-full items-center gap-3 mt-1"):
            ui.label(f"⭐ The app has {learned} of your highlights to learn from (all trips)"
                     + (f", and {dropped} it got wrong" if dropped else "") + "."
                     + ("" if ready else f" From {highlights.START_FROM} it suggests more like them.")) \
                .classes("text-sm text-amber-900")
            ui.space()
            ui.button("Suggest highlights like mine", icon="auto_awesome", on_click=self.suggest_highlights) \
                .props("outline no-caps color=orange-8" + ("" if ready else " disable")) \
                .tooltip("Looks at a frame every second of the chosen videos without a ⭐ and finds the seconds most "
                         "like your highlights (the first time takes a while).")
        if not ideas:
            return
        with ui.row().classes("w-full items-center gap-2"):
            ui.label(f"Suggested ({len(ideas)}): keep the right ones, drop the wrong ones. Both teach the app.") \
                .classes("text-sm font-medium text-orange-800")
            ui.space()
            ui.button("Review one by one", icon="slideshow", on_click=lambda: self.open_idea(ideas, 0)) \
                .props("flat dense no-caps color=orange-9") \
                .tooltip("Each suggestion big at its second: Keep or Not this, then the next one")
            ui.button("Keep all", icon="done_all", on_click=lambda: self.idea_answer(ideas, True)) \
                .props("flat dense no-caps color=orange-9")
            ui.button("Drop all", icon="clear_all", on_click=lambda: self.idea_answer(ideas, False)) \
                .props("flat dense no-caps color=grey-8")
        rows = {x["path"]: x for x in self.all_items}
        with ui.element("div").classes("w-full grid gap-2").style("grid-template-columns:repeat(auto-fill,minmax(11rem,1fr))"):
            for n, idea in enumerate(ideas):
                row = rows[idea["path"]]
                with ui.column().classes("gap-1 p-1 rounded border-2 border-orange-400 border-dashed bg-white"):
                    with ui.element("div").classes("w-full").tooltip("Double-click: open it big at that very second") as wrap:
                        thumb = self.trip.moment_picture(row, idea["t"])
                        a, b = moments.window(idea["t"], row["duration"] or 0)
                        self.preview(row | {"thumb": thumb}, "width:100%;aspect-ratio:16/10", a, b, view="moment")
                    self.click_or_double(wrap, None, lambda i=idea: self.shot_window(i["path"], "Choose files",
                                                                                     at=i["t"]))
                    with ui.row().classes("w-full items-center gap-1 no-wrap"):
                        ui.label(f"{Path(idea['path']).name[:16]} · {clock(idea['t'])}").classes("text-xs truncate")
                        ui.space()
                        ui.button(icon="star", on_click=lambda i=idea: self.idea_answer([i], True)) \
                            .props("flat dense round size=sm color=amber-8").tooltip("Keep: it becomes a ⭐ highlight")
                        ui.button(icon="close", on_click=lambda i=idea: self.idea_answer([i], False)) \
                            .props("flat dense round size=sm color=grey-7").tooltip("Drop: not a highlight")

    def suggestion_window(self, items: list[dict], k: int, title: str, keep, drop, done,
                          highlights: bool = False) -> None:
        """One suggestion big (a highlight like yours, or a moment the app found): it opens at the very second the
        app picked, plays the seconds around it again and again (green on the strip, the second itself in orange),
        and you answer: Keep (at that second, or where you put the red line) or Not this. Then the next one comes,
        without closing; ← → go back and forth. items: [{"path", "t", "row"}]; keep(item, t) and drop(item) save
        the answer; done() redraws the page once the window closes."""
        if not items:
            done()
            return
        k = max(0, min(k, len(items) - 1))
        f = items[k]
        row = f["row"]
        full = kept_part(row) or (0.0, float(row["duration"] or 0))
        a, b = moments.window(f["t"], row["duration"] or 0, *full)
        state = {"nav": False}
        box = f"tm-part-{clip_id(row['path'])}"

        def go(to: int, old=None) -> None:
            """The next suggestion. old: this window, closed only once the next one is open (no flicker), unless
            both show the same video (then this one goes first)."""
            state["nav"] = True
            same = old is not None and items[to]["row"]["path"] == row["path"]
            if same:
                old.close()

            def swap() -> None:
                self.suggestion_window(items, to, title, keep, drop, done, highlights)
                if old is not None and not same:
                    old.close()

            with self.window_timers:
                ui.timer(0.01 if old is not None else 0.05, swap, once=True)

        async def red_or_suggested() -> float:
            try:  # the red line, if you stopped the video somewhere; else the second the app suggested
                at = await ui.run_javascript(
                    f"(() => {{ const b = document.getElementById('{box}'), v = b && b.querySelector('video'); "
                    f"return v && v.src && v.paused && b.dataset.red ? parseFloat(b.dataset.red) : null; }})()",
                    timeout=3)
            except Exception:
                at = None
            return round(float(at), 2) if at is not None else f["t"]

        async def kept(lo: float, hi: float) -> bool:
            t = await red_or_suggested()
            result = keep(f, t)
            if asyncio.iscoroutine(result):
                await result
            items.pop(k)
            ui.notify(f"⭐ Kept at {fmt_length(t)}.", timeout=1200, type="positive")
            if items:
                go(min(k, len(items) - 1))
            return True

        async def dropped(lo: float, hi: float) -> bool:
            result = drop(f)
            if asyncio.iscoroutine(result):
                await result
            items.pop(k)
            ui.notify("Not this one: the app learns from it.", timeout=1200)
            if items:
                go(min(k, len(items) - 1))
            return True

        def extra(dialog) -> None:
            with ui.row().classes("w-full items-center gap-3 px-2 py-1 rounded bg-orange-50"):
                picture = self.trip.moment_picture(row, f["t"])
                if picture:
                    ui.image(data_url(picture)).classes("w-40 rounded border-2 border-orange-500") \
                        .style("aspect-ratio:16/10").props("fit=contain no-spinner")
                with ui.column().classes("gap-0 flex-1"):
                    ui.label(f"The app's pick: this frame, at {clock(f['t'])} (the orange mark on the strip; the "
                             "video stands still on it, the red line on it).").classes("text-sm font-medium "
                                                                                        "text-orange-900")
                    ui.label("Right frame? Press ⭐ Keep. A bit off? Move the mouse along the strip to the right "
                             "frame (or Space to play, Space to stop), then ⭐ Keep: the ⭐ goes where the red line "
                             "is.").classes("text-xs text-orange-800")
                ui.button("Back to the app's pick", icon="my_location", on_click=lambda: ui.run_javascript(
                    f"tmShowAt('{box}', {a:.2f}, {b:.2f}, {f['t']:.2f})")).props("flat dense no-caps color=orange-9")
            if highlights:
                self.highlight_cards(row, box, mine_box)
            for side, step, icon in (("left", -1, "chevron_left"), ("right", 1, "chevron_right")):
                if 0 <= k + step < len(items):
                    ui.button(icon=icon, on_click=lambda s=step: go(k + s, dialog)) \
                        .props(f"round unelevated color=grey-9 size=xl data-nav={step}") \
                        .style(f"position:fixed;{side}:10px;top:{'28%' if highlights else '50%'};"
                               "transform:translateY(-50%);z-index:20;opacity:.85") \
                        .tooltip(("Previous" if step < 0 else "Next") + " suggestion (← →)")

        def hidden() -> None:
            if not state["nav"]:
                done()

        mine_box: dict = {}  # the row of your own highlights in this video (redrawn when you add or take one off)

        def add_highlight(t: float) -> None:  # Shift: one more ⭐ of your own, at the red line
            self.trip.add_postit(row["path"], round(t, 2))
            ui.run_javascript(f"tmHighlightMark({t:.2f})")
            ui.notify(f"⭐ Your highlight at {clock(t)} (the picture is below).", timeout=1500, type="positive")
            if mine_box.get("draw"):
                mine_box["draw"].refresh()

        self.part_dialog(row | {"ideas": [{"t": f["t"]}]}, a, b,
                         f"{title}: suggestion {k + 1} of {len(items)} · {Path(row['path']).name} at {clock(f['t'])}",
                         [("Not this", "close", dropped, "outline no-caps color=grey-8"),
                          ("Keep at the red line", "star", kept, "unelevated no-caps color=orange-8")],
                         extra=extra, on_hide=hidden, full=True, focus=f["t"],
                         on_highlight=add_highlight if highlights else None,
                         )

    def highlight_cards(self, row: dict, box: str, holder: dict) -> None:
        """Your own ⭐ highlights in this video, each as its picture with its time: click to go to it, ✕ to take it
        off. Shift adds one at the red line."""
        path = row["path"]

        @ui.refreshable
        def draw() -> None:
            marks = self.trip.postits(path)
            with ui.row().classes("w-full items-center gap-2 px-2"):
                ui.label(f"⭐ Your highlights in this video ({len(marks)}):" if marks else
                         "⭐ Another great moment in this video? Move the mouse along the strip to it and tap Shift: it becomes one "
                         "of your highlights too.").classes("text-sm font-medium text-amber-900")
                for n, m in enumerate(marks):
                    with ui.column().classes("gap-0 p-1 rounded border-2 border-amber-400 bg-amber-50 items-center"):
                        if m.get("thumb"):
                            ui.image(data_url(m["thumb"])).classes("w-28 rounded cursor-pointer") \
                                .style("aspect-ratio:16/10").props("fit=contain no-spinner") \
                                .on("click", lambda t=m["t"]: ui.run_javascript(
                                    f"(() => {{ const b = document.getElementById('{box}'); if (b) tmShowAt(b.id, "
                                    f"parseFloat(b.dataset.start), parseFloat(b.dataset.end), {t:.2f}); }})()")) \
                                .tooltip("Go to this frame")
                        with ui.row().classes("items-center gap-0 no-wrap"):
                            ui.label(f"⭐ {clock(m['t'])}").classes("text-xs font-semibold")
                            ui.button(icon="close", on_click=lambda n=n: take_off(n)) \
                                .props("flat dense round size=xs").tooltip("Take this highlight off")

        def take_off(n: int) -> None:
            self.trip.change_postit(path, n, remove=True)
            ui.run_javascript(f"tmHighlightsSet({json.dumps([m['t'] for m in self.trip.postits(path)])})")
            ui.notify("Highlight taken off.", timeout=1200)
            draw.refresh()

        holder["draw"] = draw
        draw()

    def open_idea(self, ideas: list[dict], k: int) -> None:
        rows = {x["path"]: x for x in self.all_items}
        items = [{"path": i["path"], "t": i["t"], "row": rows[i["path"]]} for i in ideas if i["path"] in rows]

        def done() -> None:
            self.choose_keep_scroll = True
            self.refresh_all()
            self.choose_keep_scroll = False

        self.suggestion_window(items, k, "⭐ Highlight like yours",
                               lambda f, t: run.io_bound(self.trip.keep_idea, f["path"], t, f["t"]),
                               lambda f: run.io_bound(self.trip.drop_idea, f["path"], f["t"]), done, highlights=True)

    def learn_quietly(self) -> None:
        try:
            self.trip.learn_highlights()
        except Exception:
            traceback.print_exc()

    async def suggest_highlights(self) -> None:
        if self.ai_busy:
            ui.notify("Still busy with the last request.", type="warning")
            return
        self.ai_busy = True
        done = {"share": 0.0, "step": "starting"}
        note = ui.notification("", spinner=True, timeout=None, multi_line=True)
        started = time.time()

        def tick() -> None:
            gone = int(time.time() - started)
            note.message = (f"⭐ Looking for highlights like yours: {done['step']}, {int(100 * done['share'])}% "
                            f"({gone // 60}:{gone % 60:02d}). The first time takes longest: the app looks at a frame "
                            "about every second of each video and remembers it.")

        tick()
        ticker = ui.timer(1.0, tick)
        try:
            ideas = await run.io_bound(self.trip.suggest_highlights,
                                       lambda share, step: done.update(share=share, step=step))
            ui.notify(f"⭐ {len(ideas)} suggested highlights, in orange under Step 1b." if ideas else
                      "No seconds clearly like your highlights in the videos without a ⭐.", multi_line=True)
        except Exception as exc:
            traceback.print_exc()
            ui.notify(f"Couldn't look for highlights: {exc}", type="negative", multi_line=True)
        finally:
            self.ai_busy = False
            ticker.cancel()
            note.dismiss()
        self.choose_keep_scroll = True
        self.refresh_all()
        self.choose_keep_scroll = False

    async def idea_answer(self, ideas: list[dict], keep: bool) -> None:
        for idea in list(ideas):
            await run.io_bound(self.trip.keep_idea if keep else self.trip.drop_idea, idea["path"], idea["t"])
        if keep:
            self.stale = True
        ui.notify(("⭐ Kept: " if keep else "Dropped: ") + f"{len(ideas)} · the app learns from it.", timeout=1500)
        self.choose_keep_scroll = True
        self.refresh_all()
        self.choose_keep_scroll = False

    BEST_BITS = "Mark the best bits"

    def postit_dialog(self, k: int) -> None:
        """Mark the best bits: the chosen videos one after the other in the same big window as everywhere (full
        screen, the strip to look through with the mouse, Shift for a ⭐, big ‹ › on the sides and ← →)."""
        videos = [it["path"] for it in self.all_items if it["kind"] == "video" and it["chosen"]]
        if not videos:
            return
        self.nav_order[self.BEST_BITS] = videos
        self.shot_window(videos[max(0, min(k, len(videos) - 1))], self.BEST_BITS)

    def postit_strip_html(self, it: dict, found: list[dict], start: float, end: float, picture) -> str:
        span = max(0.1, end - start)
        flags = "".join(f'<div title="{fmt_length(p["t"])}" style="position:absolute;top:0;bottom:0;left:'
                        f'{100 * (p["t"] - start) / span:.2f}%;width:3px;background:#f59e0b;box-shadow:0 0 3px #000">'
                        f'<span style="position:absolute;top:-2px;left:-6px;font-size:14px">📌</span></div>'
                        for p in found if start <= p["t"] <= end)
        back = f"url('{data_url(str(picture))}') center/100% 100%" if picture else "#e2e8f0"
        return (f'<div style="position:relative;height:60px;border-radius:4px;background:{back};cursor:col-resize" '
                f'onmousemove="tmPostitScrub(event, this, {start:.2f}, {end:.2f})">{flags}</div>')

    def postit_key(self, e) -> None:
        """Shift in the post-it window: a post-it at the video's current second."""
        if not getattr(self, "postit_dialog_open", None) or not getattr(self, "postit_video", None):
            return
        try:
            t = float((e.args or {}).get("t"))
        except (TypeError, ValueError, AttributeError):
            return
        it = self.postit_video
        self.trip.add_postit(it["path"], t)
        ui.notify(f"📌 Post-it at {fmt_length(t)}", timeout=1200)
        self.postit_body.refresh()

    def lock_postit(self, it: dict, p: dict) -> None:
        start, end = kept_part(it) or (0.0, float(it["duration"] or 0))
        a, b = max(start, p["t"] - 1.5), min(end, p["t"] + 1.5)
        self.trip.add_lock(it["path"], a, b)
        self._locks_changed(it["path"])
        ui.notify(f"🔒 Locked {fmt_length(a)}–{fmt_length(b)}: every cut uses exactly these seconds.", type="positive")

    def camera_row(self, c: dict) -> None:
        first, last = (datetime.fromisoformat(c[k]).astimezone() if c[k] else None for k in ("first", "last"))
        what = ", ".join(x for x in (f"{c['videos']} videos" if c["videos"] else "",
                                     f"{c['photos']} photos" if c["photos"] else "") if x)
        span = ""
        if first and last:
            until = f"{last:%H:%M}" if first.date() == last.date() else f"{last:%a %d %b %H:%M}"
            span = f" · its clock says {first:%a %d %b %H:%M} – {until}"
        with ui.row().classes("w-full items-center gap-2 mt-1"):
            ui.label(f"📷 {c['camera']}").classes("font-semibold min-w-40")
            ui.label(what + span).classes("text-sm text-gray-600")
            ui.space()
            box = ui.input("Move its clock by", value=timeshift.fmt_shift(c["shift"]) if c["shift"] else "",
                           placeholder="e.g. +2:00 or -30 min").props("dense clearable").classes("w-44") \
                .tooltip("+2:00 = its shots happened 2 hours later than it says; -0:30 = half an hour earlier")
            ui.button("Apply", icon="check", on_click=lambda b=box, cam=c["camera"]: self.apply_shift(cam, b.value)) \
                .props("dense unelevated")
            ui.button("Find it for me", icon="travel_explore",
                      on_click=lambda cam=c["camera"]: self.suggest_shift(cam)).props("dense outline") \
                .tooltip("Look for this camera's shots that show the same moment as another camera's shots, "
                         "and work out how far apart their clocks are")
            if c["shift"]:
                ui.label(timeshift.shift_words(c["shift"])).classes("text-xs text-sky-800")
                ui.button("Undo shift", on_click=lambda cam=c["camera"]: self.apply_shift(cam, "0")).props("flat dense size=sm")

    async def apply_shift(self, camera: str, text: str | float | None) -> None:
        seconds = float(text) if isinstance(text, (int, float)) else timeshift.parse_shift(text or "0")
        if seconds is None:
            ui.notify("Type how much to move the clock, like +2:00, -1:30 or 45 min.", type="warning")
            return
        note = ui.notification(f"Moving the {camera} shots…", spinner=True, timeout=None)
        try:
            n = await run.io_bound(self.trip.set_time_shift, camera, seconds)
        finally:
            note.dismiss()
        if n:
            ui.notify(f"{n} {camera} files now sit {timeshift.shift_words(seconds)} than its clock said."
                      if seconds else f"{camera} files are back at the time its clock gave them.", type="positive")
        self.refresh_all()

    async def suggest_shift(self, camera: str) -> None:
        note = ui.notification(f"Looking for {camera} shots of the same moments as the other cameras…",
                               spinner=True, timeout=None)
        try:
            found = await run.io_bound(self.trip.suggest_time_shift, camera)
        finally:
            note.dismiss()
        if not found:
            ui.notify(f"No {camera} shots clearly match another camera's shots, so the app can't tell. Type the "
                      "shift yourself (for example from a shot where you know the time). After Analyse, matching "
                      "works better.", type="warning", multi_line=True, timeout=10000)
            return
        current = next((c["shift"] for c in self.trip.cameras() if c["camera"] == camera), 0.0)
        thumbs = {it["path"]: it["thumb"] for it in self.all_items}
        with ui.dialog() as dialog, ui.card().classes("w-[44rem] max-w-full"):
            if abs(found["seconds"] - current) < 60:
                ui.label(f"The {camera} clock {'is already right' if current else 'looks right'}") \
                    .classes("text-lg font-semibold")
            else:
                ui.label(f"Move the {camera} clock {timeshift.fmt_shift(found['seconds'])}?").classes("text-lg font-semibold")
            ui.label(f"{found['shots']} of its {found['of']} shots look like another camera's shots taken "
                     f"{timeshift.shift_words(found['seconds'])} by that camera's clock"
                     + (" (matched by the AI's picture understanding)." if found["how"] == "ai"
                        else " (matched by colours and layout; after Analyse the app can match better).")) \
                .classes("text-sm")
            ui.label("Same moment? Left: this camera · right: the other camera").classes("text-xs text-gray-500 mt-1")
            for mine, other in found["examples"][:4]:
                with ui.row().classes("gap-2 items-center"):
                    for path in (mine, other):
                        if thumbs.get(path):
                            ui.image(data_url(thumbs[path])).props("fit=contain") \
                                .style("width:200px;height:120px").classes("rounded bg-gray-800")
                        else:
                            ui.label(Path(path).name).classes("text-xs w-48")
            with ui.row().classes("justify-end w-full"):
                ui.button("Not right", on_click=dialog.close).props("flat")

                async def use() -> None:
                    dialog.close()
                    await self.apply_shift(camera, found["seconds"])

                ui.button(f"Use {timeshift.fmt_shift(found['seconds'])}", icon="check", on_click=use).props("unelevated")
        dialog.open()

    def update_sort_counts(self) -> None:
        """The counts in Step 1 and the floating Remove bar, after ticking or unticking."""
        total = len(self.all_items)
        ticked = sum(1 for it in self.all_items if it["chosen"])
        unticked = total - ticked
        if getattr(self, "sort_label", None) is not None:
            self.sort_label.text = self.sort_text(total, ticked, sum(_file_size(x["path"]) for x in self.all_items),
                                                  self.trip.removed_count())
            self.remove_button.text = f"Remove {unticked} unticked from the trip"
            self.remove_button.set_visibility(unticked > 0)
            self.remove_bar_label.text = f"{unticked} unticked"
            self.remove_bar.set_visibility(unticked > 0 and self.choose_filters["show"] != "removed")

    def sort_text(self, total: int, chosen: int, size: int, removed: int) -> str:
        amount = f"{size / 1e9:.1f} GB" if size >= 1e9 else f"{size / 1e6:.0f} MB"
        return (f"{total} files in the trip ({amount}): {chosen} ticked, {total - chosen} unticked"
                + (f" · {removed} removed earlier" if removed else ""))

    async def set_look_alike(self, level: str) -> None:
        """A level of doubles: found at that level, and only they are shown (no scrolling past everything else)."""
        if self.trip.look_alike_level() != level:
            note = ui.notification("Finding doubles again…", spinner=True, timeout=None)
            try:
                await run.io_bound(self.trip.set_look_alike_level, level)
            finally:
                note.dismiss()
            self.reload()
        counts: dict = {}
        for it in self.all_items:
            counts[it["similar_set"]] = counts.get(it["similar_set"], 0) + 1
        sets = sum(1 for k, n in counts.items() if k is not None and n > 1)
        self.choose_filters["show"] = "sets"  # only the doubles of this level, in front
        self.story_page = 1
        name = stories.STRICTNESS_NAMES[level].split(" · ")[0]
        ui.notify(f"{name}: {sets} sets of doubles, shown on their own. Check them, then ✓ Accept all markings." if sets
                  else f"{name}: no doubles left. Go on to the next level, or Show → Everything.",
                  multi_line=True, timeout=8000)
        self.refresh_all()
        ui.run_javascript("setTimeout(() => { const e = document.getElementById('tm-step-doubles'); "
                          "if (e) e.scrollIntoView({behavior: 'smooth', block: 'start'}); }, 300)")

    async def confirm_remove(self) -> None:
        """Remove the unticked files at once (no question: See removed brings them back)."""
        await self.remove_unchosen()

    async def remove_unchosen(self) -> None:
        # The first ticked file in view stays exactly where it is on the screen: nothing jumps.
        try:
            anchor = await ui.run_javascript(
                "(() => { const el = [...document.querySelectorAll('[data-tile]')].find(e => "
                "e.getBoundingClientRect().bottom > 80 && !e.classList.contains('opacity-40')); "
                "return el ? [el.dataset.tile, el.getBoundingClientRect().top] : null; })()", timeout=3)
        except Exception:
            anchor = None
        n = self.trip.remove_unchosen()
        ui.notify(f"Removed {n} files from the trip (See removed brings them back).", type="positive")
        it = next((x for x in self.all_items if anchor and clip_id(x["path"]) == anchor[0]), None)
        if it is None:
            self.refresh_all()
            return
        self.choose_focus, self.choose_focus_path = it["story"], it["path"]  # draw down to it
        self.choose_keep_scroll = True
        self.refresh_all()
        self.choose_keep_scroll = False
        ui.run_javascript(f"for (const ms of [100, 350, 800]) setTimeout(() => {{ const el = document.querySelector("
                          f"'[data-tile=\"{anchor[0]}\"]'); if (el) window.scrollBy(0, el.getBoundingClientRect().top "
                          f"- {float(anchor[1])}); }}, ms)")

    def restore(self, paths: list[str]) -> None:
        self.trip.restore(paths)
        ui.notify(f"Brought back {len(paths)} files (unticked: tick the ones you want).", type="positive")
        if not self.trip.removed_count():
            self.choose_filters["show"] = ALL
        self.refresh_all()

    @ui.refreshable
    def choose_view(self) -> None:
        if not self.showing("Choose files"):
            return
        if not self.all_items:
            ui.label("Open a trip folder to see its files here.").classes("text-gray-500")
            return
        days = sorted({it["day"] for it in self.all_items})
        day_opts = {ALL: "All days", **{d: f"Day {n} · {datetime.fromisoformat(d):%a %d %b}" for n, d in enumerate(days, 1)}}
        if self.choose_filters["day"] not in day_opts:
            self.choose_filters["day"] = ALL
        shown = self.choose_shown()
        chosen = len(self.items)
        self.sort_out_card()
        self.camera_clocks_card()
        self.postit_card()
        with ui.row().classes("w-full items-center gap-3"):
            ui.label("Analyse:" if getattr(self, "guided", False) else "Step 2:").classes("text-lg font-semibold")
            ui.button(
                f"Analyse the {chosen} chosen files", icon="auto_awesome", on_click=self.analyse_chosen
            ).props("" if chosen else "disable")
        ui.label(
            "Files are grouped into stories (same time and place). Similar shots sit together in a box, and the ⭐ one "
            "is the app's suggestion. Hover over a video to play it."
        ).classes("text-sm text-gray-600")
        with ui.row().classes("w-full items-end gap-3"):
            ui.select(day_opts, value=self.choose_filters["day"], label="Day",
                      on_change=lambda e: self.set_choose_filter("day", e.value)).classes("min-w-48")
            ui.select({ALL: "Photos and videos", "photo": "Only photos", "video": "Only videos"},
                      value=self.choose_filters["kind"], label="Type",
                      on_change=lambda e: self.set_choose_filter("kind", e.value)).classes("min-w-40")
            ui.select({ALL: "Everything", "sets": "Only doubles and triples", "ticked": "Only ticked",
                       "unticked": "Only unticked", "removed": "Removed from the trip"},
                      value=self.choose_filters["show"], label="Show",
                      on_change=lambda e: self.set_choose_filter("show", e.value)).classes("min-w-52")
            ui.button("Tick all shown", icon="check_box", on_click=lambda: self.choose_shown_files(True)).props("flat dense")
            ui.button("Untick all shown", icon="check_box_outline_blank", on_click=lambda: self.choose_shown_files(False)).props("flat dense")

        if self.choose_filters["show"] == "removed":
            self.removed_view(shown)
            return
        if self.choose_filters["show"] == "sets" and not shown:  # every double of this level sorted out
            levels = list(stories.STRICTNESS_NAMES)
            level = self.trip.look_alike_level()
            later = levels[levels.index(level) + 1:] if level in levels else []
            with ui.card().classes("w-full items-center p-6 bg-green-50"):
                ui.label(f"✓ No doubles left at {stories.STRICTNESS_NAMES[level].split(' · ')[0]}.") \
                    .classes("text-lg font-medium text-green-900")
                with ui.row().classes("gap-2"):
                    if later:
                        ui.button(f"Go on: {stories.STRICTNESS_NAMES[later[0]]}", icon="arrow_forward",
                                  on_click=lambda k=later[0]: self.set_look_alike(k)).props("unelevated no-caps color=amber-9")
                    ui.button("Show everything", on_click=lambda: self.set_choose_filter("show", ALL)) \
                        .props("outline no-caps")
            return
        # One long list. It is drawn TILES_PER_PAGE files at a time: the next part is added by itself as you scroll
        # near the bottom, so even thousands of files stay quick (drawing them all at once makes the page reload).
        self.choose_whole = {}  # story number: all its files shown (a long story is drawn in parts)
        for it in shown:
            self.choose_whole.setdefault(it["story"] or 0, []).append(it)
        parts: list[list[tuple[int, list[dict]]]] = [[]]
        in_part = 0
        for number in sorted({it["story"] or 0 for it in shown}):
            members = [it for it in shown if (it["story"] or 0) == number]
            sets: dict[int, list[dict]] = {}
            for it in members:
                sets.setdefault(it["similar_set"] or 0, []).append(it)
            chunk: list[dict] = []
            for group in sets.values():  # a set of look-alikes is never split
                if in_part + len(group) > TILES_PER_PAGE and in_part:
                    if chunk:
                        parts[-1].append((number, chunk))
                    parts.append([])
                    chunk, in_part = [], 0
                chunk += group
                in_part += len(group)
            if chunk:
                parts[-1].append((number, chunk))
        self.joined = self.trip.joined_stories()
        self.last_story = max((it["story"] or 0 for it in self.all_items), default=0)
        self.choose_focus_card = None
        drawn = 1
        while self.choose_focus is not None and drawn < len(parts) \
                and not any(n == self.choose_focus and (self.choose_focus_path is None
                                                        or any(it["path"] == self.choose_focus_path for it in m))
                            for part in parts[:drawn] for n, m in part):
            drawn += 1  # after a link: draw down to that story, then scroll to it
        parts = [[x for part in parts[:drawn] for x in part]] + parts[drawn:]
        self.choose_rest = parts[1:]
        self.choose_tiles: dict[str, tuple] = {}  # path: (file, checkbox, tile), for ticking without a redraw
        self.choose_counts: list[tuple] = []  # (story's "x of y chosen" label, its files)
        self.choose_left = len(shown) - sum(len(m) for _, m in parts[0])
        with ui.column().classes("w-full gap-0") as self.choose_list:
            with ui.column().classes("w-full gap-0"):  # each part in its own box: adding one leaves the others alone
                for number, members in parts[0]:
                    self.story_block(number, members)
        self.choose_more = None
        self.more_files_button()
        if self.choose_focus_card is not None and not getattr(self, "choose_keep_scroll", False):
            ui.run_javascript(f"setTimeout(() => getHtmlElement({self.choose_focus_card.id})"
                              f".scrollIntoView({{block: 'center'}}), 300)")
        self.choose_focus = None
        self.choose_focus_path = None

    def more_files_button(self) -> None:
        """At the bottom of the list: shows the next files. The page presses it by itself when you scroll near it."""
        if self.choose_more is not None:
            self.choose_more.delete()
            self.choose_more = None
        if not self.choose_rest:
            return
        with self.choose_list:
            self.choose_more = ui.button(f"Loading more… ({self.choose_left} files below)", icon="expand_more",
                                         on_click=self.more_files).props("outline").classes("w-full mt-2 tm-more")

    def more_files(self) -> None:
        if not self.choose_rest:
            return
        part = self.choose_rest.pop(0)
        with self.choose_list, ui.column().classes("w-full gap-0"):
            for number, members in part:
                self.story_block(number, members)
        self.choose_left -= sum(len(m) for _, m in part)
        self.more_files_button()

    def removed_view(self, removed: list[dict]) -> None:
        if not removed:
            ui.label("Nothing removed.").classes("text-gray-500")
            return
        with ui.row().classes("items-center gap-2"):
            ui.label(f"{len(removed)} files removed from the trip").classes("font-semibold")
            ui.button("Bring them all back", icon="restore",
                      on_click=lambda: self.restore([it["path"] for it in removed])).props("outline dense")
        with ui.row().classes("w-full gap-2"):
            for it in removed[:120]:
                with ui.column().classes("w-36 gap-0"):
                    if it["thumb"]:
                        ui.image(data_url(it["thumb"])).props("fit=contain").style("width:144px;height:96px") \
                            .classes("rounded bg-gray-800")
                    else:
                        ui.label(Path(it["path"]).name).classes("text-xs w-36 h-24 bg-gray-100 p-1 break-all")
                    ui.button("Bring back", on_click=lambda p=it["path"]: self.restore([p])).props("flat dense size=sm")
        if len(removed) > 120:
            ui.label(f"…and {len(removed) - 120} more (Bring them all back covers them too).").classes("text-sm")

    def story_block(self, number: int, chunk: list[dict]) -> None:
        """A story (or, for a long one, the next part of it) in Choose files."""
        items = self.choose_whole.get(number, chunk)
        continued = chunk[0] is not items[0]
        first, last = datetime.fromisoformat(items[0]["taken_at"]).astimezone(), datetime.fromisoformat(items[-1]["taken_at"]).astimezone()
        places = [it["place"] or it["country"] for it in items if it["place"] or it["country"]]
        place = max(set(places), key=places.count) if places else "Unknown place"
        picked = sum(it["chosen"] for it in items)
        # content-visibility: the browser skips stories that are off screen, so a long list stays quick
        until = f"{last:%H:%M}" if first.date() == last.date() else f"{last:%a %d %b, %H:%M}"
        with ui.card().classes("w-full p-3 mt-2").style("content-visibility:auto;contain-intrinsic-size:auto 400px") as card:
            if number == self.choose_focus and not continued and self.choose_focus_path is None:
                self.choose_focus_card = card
            if continued:
                ui.label(f"Story {number + 1}, continued").classes("text-sm text-gray-500")
            with ui.row().classes("w-full items-center gap-2") as head:
                head.set_visibility(not continued)
                ui.label(f"Story {number + 1} · {first:%a %d %b, %H:%M}–{until} · {place}").classes("font-semibold")
                count = ui.label(f"{picked} of {len(items)} chosen").classes("text-sm text-gray-500")
                self.choose_counts.append((count, items))
                paths = [it["path"] for it in items]
                ui.button("Tick story", on_click=lambda p=paths: self.choose_paths(p, True)).props("flat dense size=sm")
                ui.button("Untick story", on_click=lambda p=paths: self.choose_paths(p, False)).props("flat dense size=sm")
                if len({it["similar_set"] for it in items}) < len(items):  # (some look-alike set here)
                    ui.button("✓ Accept all in this story", on_click=lambda n=number: self.accept_sets(n)) \
                        .props("unelevated dense no-caps size=sm color=primary") \
                        .tooltip("Every look-alike set of this story: the ticks as they are (a set you didn't touch "
                                 "keeps only its ⭐), and the set is done")
                ui.space()
                if number > 0:
                    ui.button("Link to previous", icon="north", on_click=lambda n=number: self.link_story(n, False)) \
                        .props("flat dense size=sm").tooltip(f"Join story {number + 1} onto story {number}: they become one story")
                if number < self.last_story:
                    ui.button("Link to next", icon="south", on_click=lambda n=number: self.link_story(n, True)) \
                        .props("flat dense size=sm").tooltip(f"Join story {number + 2} onto this one: they become one story")
                if number in self.joined:
                    ui.button("Split again", icon="call_split", on_click=lambda n=number: self.unlink_story(n)) \
                        .props("flat dense size=sm").tooltip("Undo the links: back to the stories it was made of")
            sets: dict[int, list[dict]] = {}
            for it in chunk:
                sets.setdefault(it["similar_set"] or 0, []).append(it)
            with ui.row().classes("w-full gap-2 items-start"):
                for members in sets.values():
                    if len(members) == 1:
                        self.choose_tile(members[0], in_set=False)
                        continue
                    with ui.card().tight().classes("p-1 bg-blue-50 border border-blue-200"):
                        with ui.row().classes("items-center gap-1 px-1"):
                            ui.label(f"{len(members)} similar shots").classes("text-xs text-blue-800")
                            ui.button("keep ⭐ only", on_click=lambda m=members: self.keep_best(m)).props("flat dense size=xs")
                            ui.button("✓ Keep marked", on_click=lambda m=members, n=number: self.keep_marked(m, n)) \
                                .props("unelevated dense size=xs color=primary") \
                                .tooltip("Done with this set: the ticked shots stay as single shots, the frame goes")
                        with ui.row().classes("gap-1"):
                            for it in members:
                                self.choose_tile(it, in_set=True)

    def choose_tile(self, it: dict, in_set: bool) -> None:
        with ui.column().classes("w-36 gap-0" + ("" if it["chosen"] else " opacity-40")) \
                .props(f'data-tile="{clip_id(it["path"])}"') as tile:
            if it["path"] == self.choose_focus_path:
                self.choose_focus_card = tile
            with ui.element("div").tooltip("Double-click: open it big (cut, split into shots, lock, theme)") as wrap:
                self.preview(it, "width:144px;height:96px", start=self.hover_start(it))
            self.click_or_double(wrap, None, lambda p=it["path"]: self.shot_window(p, "Choose files"))
            with ui.row().classes("items-center gap-0 no-wrap"):
                box = ui.checkbox(
                    value=bool(it["chosen"]),
                    on_change=lambda e, it=it, tile=tile: self.quick_tick(it, e.value, tile),
                ).props("dense size=xs")
                self.choose_tiles[it["path"]] = (it, box, tile)
                rotate_buttons(lambda degrees, path=it["path"]: self.rotate(path, degrees))
                cut = {"tile": tile, "wide": True}
                self.cut_button(it, cut)
                star = "⭐ " if in_set and it["best"] else ""
                ui.label(f"{star}{datetime.fromisoformat(it['taken_at']).astimezone():%H:%M:%S}").classes("text-xs text-gray-600")
            cut["area"] = ui.column().classes("w-full gap-1")  # the cutting handles, when you press ✂

    def preview(self, it: dict, size: str, start: float = 0, end: float | None = None, view: str = "",
                sound: bool = False) -> str:
        """A preview picture; for a video, hovering plays it (only the part from start to end).

        Returns the element id so the storyboard can change the part being played."""
        box = f"tm-{view}-{clip_id(it['path'])}"
        thumb = data_url(it["thumb"])
        if it["kind"] != "video" or not self.media_route:
            if thumb:
                ui.image(thumb).props(f'fit=contain data-shot="{clip_id(it["path"])}"').style(size) \
                    .classes("rounded bg-gray-800")
            return box
        file = Path(it.get("part_of") or it["path"])  # a part of a long clip plays from the clip itself
        src = self.video_src(file)
        given_end = end is not None
        end = (it["duration"] or 0) if end is None else end
        kept = kept_part(it)
        if kept:  # a clip you cut: only the part you kept plays
            end = min(end, kept[1])
            start = start if kept[0] <= start < end else kept[0]
        # Skimming: the mouse from left to right goes through the clip (the part you kept, or the part shown).
        skim0 = start if given_end else (kept[0] if kept else 0.0)
        skim = "" if view in ("part", "cut") else f'data-skim0="{skim0:.2f}" data-skim1="{end:.2f}" '
        length = f"✂ {fmt_length(kept[1] - kept[0])}" if kept else fmt_length(it["duration"])
        ui.html(
            f'<div id="{box}" class="tm-hover" data-shot="{clip_id(it["path"])}" {skim}data-sound="{int(sound)}" data-start="{start:.2f}" data-end="{end:.2f}" data-rot="{it.get("rotation") or 0}" '
            f'style="position:relative;overflow:hidden;border-radius:4px;background:#222;{size}">'
            f'<img src="{thumb}" style="width:100%;height:100%;object-fit:contain">'
            f'<video data-src="{src}" {"" if sound else "muted "}playsinline preload="none" '
            f'style="display:none;position:absolute;inset:0;width:100%;height:100%;object-fit:contain;background:#222"></video>'
            f'<span class="tm-len" style="position:absolute;left:4px;bottom:2px;color:white;font-size:11px;'
            f'text-shadow:0 0 3px black">▶ {length}</span></div>',
            sanitize=False,
        ).style("width:100%" if "100%" in size else "")  # else a tall picture squeezes a full-width frame
        return box

    def video_src(self, file: Path) -> str:
        """Where the page streams a video from: the trip folder, or the app's folder for shots you made."""
        try:
            return f"{self.media_route}/{quote(str(file.relative_to(self.trip.folder)))}"
        except ValueError:
            return f"{self.media_route}-shots/{quote(str(file.relative_to(self.trip.shots_dir)))}"

    @staticmethod
    def hover_start(it: dict) -> float:
        start, end = kept_part(it) or (0.0, it["duration"] or 0)
        if it.get("postits"):  # from just before your first post-it
            return min(max(start, it["postits"][0]["t"] - 1.0), max(start, end - 0.5))
        return start + (end - start) * 0.3

    def cut_button(self, it: dict, cut: dict) -> None:
        """✂ opens the cutting handles right on the tile (cut["area"], filled in once the tile is built)."""
        if it["kind"] == "video":
            ui.button(icon="content_cut", on_click=lambda: self.toggle_cut(it, cut)) \
                .props("flat dense round size=sm" + (" color=green" if kept_part(it) else "")) \
                .tooltip("Cut this clip: keep only the part you want" + (" (cut: ✂ " + fmt_length(
                    kept_part(it)[1] - kept_part(it)[0]) + " kept)" if kept_part(it) else ""))

    def toggle_cut(self, it: dict, cut: dict) -> None:
        area = cut["area"]
        if cut.get("open"):
            self.close_cut(cut)
            return
        cut["open"] = True
        if cut.get("wide"):  # a small tile in Choose files grows while you cut
            cut["tile"].classes(remove="w-36", add="w-96")
        with area:
            self.cut_editor(it, cut)

    def close_cut(self, cut: dict) -> None:
        cut["open"] = False
        if cut.get("box"):  # the Library picture plays by itself again when you hover over it
            ui.run_javascript(f"const b = document.getElementById('{cut['box']}'); if (b) {{ delete b.dataset.scrub; "
                              f"const v = b.querySelector('video'); if (v) v.pause(); }}")
        cut["area"].clear()
        if cut.get("wide"):
            cut["tile"].classes(remove="w-96", add="w-36")

    def cut_editor(self, it: dict, cut: dict) -> None:
        """Drag the two handles in to the part you want, then Cut. The clip is then that shorter clip everywhere
        (cut it again to make it shorter still). Your file is not changed: Undo cut brings the whole clip back."""
        start, end = kept_part(it) or (0.0, it["duration"] or 0)
        start, end = round(start, 1), round(end, 1)
        length = end - start
        part = {"min": start, "max": end}
        name = Path(it["path"]).name
        box = cut.get("box") or self.preview(it, "width:100%;height:210px", start, end, view="cut")
        # The clip as a row of pictures (made while you look); the parts you cut off are greyed out.
        shade = "position:absolute;top:0;bottom:0;background:rgba(15,23,42,.65)"
        # Like the editor: press and drag on it and the video follows the red line; a click puts the line there.
        ui.html(f'<div data-scrub-for="{box}" data-press="1" style="position:relative;height:40px;margin:4px 10px 0;'
                f'border-radius:4px;overflow:hidden;background:#e2e8f0;cursor:ew-resize;touch-action:none">'
                f'<img id="{box}-strip" style="width:100%;height:100%;object-fit:fill;display:none;pointer-events:none">'
                f'<div id="{box}-l" style="{shade};left:0;width:0;pointer-events:none"></div>'
                f'<div id="{box}-r" style="{shade};right:0;width:0;pointer-events:none"></div>'
                f'{RED_LINE}</div>', sanitize=False).classes("w-full")
        ui.timer(0.1, lambda: ui.run_javascript(
            f"const b = document.getElementById('{box}'); if (b) {{ b.dataset.full0 = {start}; "
            f"b.dataset.full1 = {end}; }}"), once=True)
        ui.range(min=start, max=end, step=0.1, value=dict(part),
                 on_change=lambda e: moved(e.value["min"], e.value["max"])).props("color=green dense drag-range").classes("w-full px-2")
        label = ui.label().classes("text-xs font-medium text-green-800 px-2")

        def show() -> None:  # times from the start of the clip as it is now
            label.text = (f"Keeping {fmt_length(part['min'] - start)}–{fmt_length(part['max'] - start)}: "
                          f"{part['max'] - part['min']:.1f} s of {length:.1f} s")
            left, right = 100 * (part["min"] - start) / (length or 1), 100 * (end - part["max"]) / (length or 1)
            ui.run_javascript(f"for (const [s, w] of [['l', {left:.2f}], ['r', {right:.2f}]]) {{"
                              f"const e = document.getElementById('{box}-' + s); if (e) e.style.width = w + '%'; }}")

        def moved(lo: float, hi: float) -> None:
            # The still frame where the handle you drag is (the start, or the last frame of the part)
            at = lo if lo != part["min"] or hi == part["max"] else max(lo, hi - 0.05)
            part.update(min=lo, max=hi)
            show()
            ui.run_javascript(f"tmShowAt('{box}', {lo}, {hi}, {at})")

        async def apply(whole: bool = False) -> None:
            if not whole and part["min"] <= start + 0.05 and part["max"] >= end - 0.05:
                ui.notify("Drag the green handles in first: nothing is cut off yet.", type="warning")
                return
            try:
                await run.io_bound(self.trip.set_keep, it["path"], None if whole else part["min"],
                                   None if whole else part["max"])
            except Exception as exc:
                traceback.print_exc()
                ui.notify(f"Could not cut {name}: {exc}", type="negative", multi_line=True)
                return
            self.show_cut(it["path"])
            ui.notify(f"{name}: the whole clip again." if whole else
                      f"{name} is now {fmt_length(part['max'] - part['min'])} long. Auto-edit and the video only use "
                      "this part.", type="positive")
            self.close_cut(cut)  # last: this removes the button you pressed

        async def strip() -> None:
            picture = await run.io_bound(self.trip.filmstrip, it["path"])
            if picture:
                ui.run_javascript(f"const s = document.getElementById('{box}-strip'); "
                                  f"if (s) {{ s.src = '{data_url(str(picture))}'; s.style.display = 'block'; }}")

        show()
        ui.timer(0.05, strip, once=True)
        with ui.row().classes("w-full items-center gap-1 px-1 pb-1"):
            ui.button("Play", icon="play_arrow",
                      on_click=lambda: ui.run_javascript(f"tmPlayPart('{box}', {part['min']}, {part['max']}, {part['min']})")) \
                .props("outline dense no-caps")
            ui.button("Cut", icon="content_cut", on_click=lambda: apply()).props("color=green dense no-caps") \
                .tooltip("Keep only this part")
            if kept_part(it):
                ui.button("Undo cut", icon="undo", on_click=lambda: apply(whole=True)).props("flat dense no-caps") \
                    .tooltip("The whole clip again")
            ui.button("Cancel", on_click=lambda: self.close_cut(cut)).props("flat dense no-caps")

    # ---------- locked moments (one per clip, the same in Library, Themes and Storyboard) ----------

    @staticmethod
    def clip_of(it: dict) -> str:
        return it.get("part_of") or it["path"]

    def lock_area(self, it: dict, mode: str) -> None:
        """Where a clip's locked moments show: "full" (Library), "compact" (a Themes tile) or "story" (a shot in
        the Storyboard). Change them in any of these places and the others follow."""
        box = ui.element("div").classes("w-full" if mode != "compact" else "")
        self.lock_views.setdefault(self.clip_of(it), []).append((box, it, mode))
        with box:
            self._draw_locks(it, mode)

    def _locks_changed(self, clip: str) -> None:
        alive = []
        for box, it, mode in self.lock_views.get(clip, []):
            if box.is_deleted:
                continue
            box.clear()
            with box:
                self._draw_locks(it, mode)
            alive.append((box, it, mode))
        self.lock_views[clip] = alive

    def _draw_locks(self, it: dict, mode: str) -> None:
        clip = self.clip_of(it)
        locks = self.trip.locks(clip)
        words = [f"{fmt_length(a)}–{fmt_length(b)}" for a, b in locks]
        if mode == "compact":
            button = ui.button(str(len(locks)) if locks else "", icon="lock" if locks else "lock_open") \
                .props("round dense size=sm " + ("color=amber-8" if locks else "color=white text-color=grey-7")) \
                .classes("shadow").tooltip("Locked moments: " + ", ".join(words) if locks else "Lock a moment")
            with button, ui.menu():
                for k, w in enumerate(words):
                    ui.menu_item(f"🔒 {w}", on_click=lambda k=k: self.edit_lock(it, k))
                ui.menu_item("🔒 Lock a moment…", on_click=lambda: self.new_lock(it))
            return
        if mode == "story":
            lo, hi = self.kept_part(it)
            match = next((k for k, (a, b) in enumerate(locks) if abs(a - lo) < 0.15 and abs(b - hi) < 0.15), None)
            with ui.row().classes("items-center gap-1"):
                ui.button("Choose part", icon="open_in_full", on_click=lambda: self.choose_shot_part(it)) \
                    .props("flat dense no-caps size=sm").tooltip("Watch the whole clip big and pick the part to use")
                if match is not None:
                    ui.label("🔒 This part is locked").classes("text-xs font-medium text-amber-800")
                    ui.button("Unlock", icon="lock_open", on_click=lambda: self.unlock(clip, match)) \
                        .props("flat dense no-caps size=sm")
                else:
                    ui.button("Lock this part", icon="lock", on_click=lambda: self.lock(clip, lo, hi)) \
                        .props("flat dense no-caps size=sm color=amber-9") \
                        .tooltip("The next cut keeps exactly this part (it may move it); the rest of the clip stays free")
                others = [w for k, w in enumerate(words) if k != match]
                if others:
                    ui.label("· also locked: " + ", ".join(others)).classes("text-xs text-amber-800")
            return
        with ui.row().classes("items-center gap-1"):
            for k, w in enumerate(words):
                ui.button(f"🔒 {w}", on_click=lambda k=k: self.edit_lock(it, k)) \
                    .props("flat dense no-caps size=sm color=amber-9").tooltip("Watch it, move it or unlock it")
            ui.button("Lock a moment", icon="lock", on_click=lambda: self.new_lock(it)) \
                .props("flat dense no-caps size=sm").tooltip("Pick the best part of this clip: every cut keeps it")

    def lock(self, clip: str, start: float, end: float) -> None:
        self.trip.add_lock(clip, start, end)
        ui.notify(f"🔒 Locked {fmt_length(start)}–{fmt_length(end)}. The next cut keeps exactly this part.",
                  type="positive")
        self._locks_changed(clip)

    def unlock(self, clip: str, k: int) -> None:
        self.trip.change_lock(clip, k)
        ui.notify("Unlocked: the next cut may use any part of the clip again.")
        self._locks_changed(clip)

    def _clip_row(self, it: dict) -> dict:
        """The clip itself (for a shot that is a part of a long clip)."""
        clip = self.clip_of(it)
        return next((r for r in self.all_items if r["path"] == clip), it)

    def new_lock(self, it: dict) -> None:
        row = self._clip_row(it)
        a, b = moments.window(self.hover_start(row), row["duration"] or 0, *(kept_part(row) or
                                                                           (0.0, row["duration"] or 0)))

        def save(lo: float, hi: float) -> bool:
            self.lock(row["path"], lo, hi)
            return True

        self.part_dialog(row, a, b, "Lock a moment of this clip",
                         [("Lock this moment", "lock", save, "no-caps color=amber-9")])

    def edit_lock(self, it: dict, k: int) -> None:
        row = self._clip_row(it)
        locks = self.trip.locks(row["path"])
        if not 0 <= k < len(locks):
            return

        def save(lo: float, hi: float) -> bool:
            self.trip.change_lock(row["path"], k, lo, hi)
            ui.notify(f"🔒 Locked moment moved to {fmt_length(lo)}–{fmt_length(hi)}.", type="positive")
            self._locks_changed(row["path"])
            return True

        def free(_lo: float, _hi: float) -> bool:
            self.unlock(row["path"], k)
            return True

        self.part_dialog(row, *locks[k], "Locked moment",
                         [("Unlock", "lock_open", free, "flat no-caps color=negative"),
                          ("Save", "lock", save, "no-caps color=amber-9")])

    def choose_shot_part(self, it: dict) -> None:
        """A shot in the Storyboard: watch its whole clip big, use another part, and lock it if you like."""
        lo, hi = self.kept_part(it)

        def use(a: float, b: float, locked: bool = False) -> bool:
            self.trip.set_trim(it["path"], a, b)
            if locked:
                self.trip.add_lock(self.clip_of(it), a, b)
            ui.notify(f"This shot now uses {fmt_length(a)}–{fmt_length(b)}" + (", locked 🔒." if locked else "."),
                      type="positive")
            self.reload()
            self.storyboard_view.refresh()
            if locked:
                self._locks_changed(self.clip_of(it))
            return True

        self.part_dialog(it, lo, hi, "Choose the part of this shot",
                         [("Use this part", "check", use, "outline no-caps"),
                          ("Use and lock", "lock", lambda a, b: use(a, b, True), "no-caps color=amber-9")],
                         taken=self.trip.used_parts(self.clip_of(it), but=it["path"]))

    def part_dialog(self, it: dict, a: float, b: float, title: str, actions: list[tuple],
                    taken: list[tuple[float, float, int]] | None = None,
                    made: list[tuple[float, float, str]] | None = None, extra=None, on_hide=None,
                    full: bool = False, on_highlight=None, focus: float | None = None,
                    video_height: str | None = None) -> None:
        """focus: open paused on this very frame, the red line on it (a suggested second you check and move).
        A big window to choose a part of a video: it plays that part (silently, again and again), the strip below
        shows the whole clip (move the mouse along it to look), and the green handles set the part.
        actions: (label, icon, callback(start, end), props) buttons; a callback returning True closes the window."""
        full0, full1 = kept_part(it) or (0.0, float(it["duration"] or 0))
        part = {"min": round(max(full0, a), 1), "max": round(min(full1, b), 1)}
        with ui.dialog().props("maximized transition-show=none transition-hide=none" if full else "") as dialog, \
                ui.card().classes("w-full h-full gap-1 p-2 tm-bigwin" if full else "w-[60rem] max-w-full gap-2"):
            ui.label(title).classes("text-base font-semibold truncate" if full else "text-lg font-semibold")
            height = video_height or ("100%" if full else "min(60vh,520px)")
            with ui.element("div").classes("w-full" + (" tm-fill" if full else "")) \
                    .style("flex:1 1 0;min-height:160px" if full else "") \
                    .on("dblclick", lambda: dialog.close()).tooltip("Double-click: close and go back") as holder:
                holder.on("tmvzoom", lambda e: self.vzoom_moved and self.vzoom_moved(e), ["detail"])
                if on_highlight is not None:  # Space plays and pauses, Shift puts a highlight here
                    holder.props("data-shotwin")
                    holder.on("tmhighlight", lambda e: on_highlight(float(e.args["detail"]["t"])), ["detail"])
                box = self.preview(it, f"width:100%;height:{height}", part["min"], part["max"], view="part")
            shade = "position:absolute;top:0;bottom:0;background:rgba(15,23,42,.65)"
            ui.html(f'<div data-scrub-for="{box}" data-press="1" style="position:relative;height:48px;'
                    f'margin:4px 10px 0;border-radius:4px;overflow:hidden;background:#e2e8f0;cursor:ew-resize;'
                    f'touch-action:none">'
                    f'<img id="{box}-strip" style="width:100%;height:100%;object-fit:fill;display:none;'
                    f'pointer-events:none">'
                    f'<div id="{box}-l" style="{shade};left:0;width:0;pointer-events:none"></div>'
                    f'<div id="{box}-r" style="{shade};right:0;width:0;pointer-events:none"></div>'
                    + RED_LINE
                    + "".join(f'<div title="Used by shot {n}" style="position:absolute;top:0;bottom:0;'
                              f'left:{100 * (x - full0) / ((full1 - full0) or 1):.2f}%;'
                              f'width:{100 * (y - x) / ((full1 - full0) or 1):.2f}%;background:rgba(220,38,38,.55);'
                              f'border-left:2px solid #dc2626;border-right:2px solid #dc2626;pointer-events:none">'
                              f'<span style="color:white;font:600 11px sans-serif;padding:2px">shot {n}</span></div>'
                              for x, y, n in taken or [])
                    + "".join(f'<div title="{label}" style="position:absolute;top:0;bottom:0;'
                              f'left:{100 * (x - full0) / ((full1 - full0) or 1):.2f}%;'
                              f'width:{100 * (y - x) / ((full1 - full0) or 1):.2f}%;background:rgba(22,163,74,.45);'
                              f'border-left:2px solid #16a34a;border-right:2px solid #16a34a;pointer-events:none">'
                              f'<span style="color:white;font:600 11px sans-serif;padding:2px;text-shadow:0 0 3px #000">'
                              f'{label}</span></div>' for x, y, label in made or [])
                    + "".join(f'<div class="tm-hl" style="position:absolute;top:0;bottom:0;width:3px;'
                              f'left:{100 * (t - full0) / ((full1 - full0) or 1):.2f}%;background:#facc15;'
                              f'box-shadow:0 0 3px #000;pointer-events:none"></div>'
                              for t in [p["t"] for p in it.get("postits") or [] if full0 <= p["t"] <= full1])
                    + "".join(f'<div class="tm-idea" data-t="{t:.2f}" style="position:absolute;top:0;bottom:0;'
                              f'width:4px;margin-left:-2px;left:{100 * (t - full0) / ((full1 - full0) or 1):.2f}%;'
                              f'background:repeating-linear-gradient(#f97316 0 6px,#fff7ed 6px 9px);'
                              f'box-shadow:0 0 3px #000;'
                              f'pointer-events:none"></div>'
                              for t in [i["t"] for i in it.get("ideas") or [] if full0 <= i["t"] <= full1])
                    + '</div>', sanitize=False).classes("w-full")
            if taken:
                ui.label("Red: parts other shots in your video already use (never shown twice).") \
                    .classes("text-xs text-red-700 px-2")
            ui.range(min=full0, max=full1, step=0.1, value=dict(part),
                     on_change=lambda e: moved(e.value["min"], e.value["max"])).props("color=green dense drag-range") \
                .classes("w-full px-2")
            with ui.row().classes("w-full items-center gap-3 px-2 no-wrap"):
                label = ui.label().classes("text-sm font-medium text-green-800 whitespace-nowrap")
                ui.label("Move the mouse along the strip to look through it · Space plays and stops · the green bar is "
                         "the part used (drag its middle or ends) · double-click the picture to close") \
                    .classes("text-xs text-gray-500 truncate")

            def show() -> None:
                label.text = (f"{fmt_length(part['min'])}–{fmt_length(part['max'])} · "
                              f"{part['max'] - part['min']:.1f} s")
                length = full1 - full0 or 1
                left, right = 100 * (part["min"] - full0) / length, 100 * (full1 - part["max"]) / length
                ui.run_javascript(f"for (const [s, w] of [['l', {left:.2f}], ['r', {right:.2f}]]) {{"
                                  f"const e = document.getElementById('{box}-' + s); if (e) e.style.width = w + '%'; }}")

            def moved(lo: float, hi: float) -> None:
                # The still frame where the handle you drag is (the start, or the last frame of the part)
                at = lo if lo != part["min"] or hi == part["max"] else max(lo, hi - 0.05)
                part.update(min=lo, max=hi)
                show()
                ui.run_javascript(f"tmShowAt('{box}', {lo}, {hi}, {at})")

            async def start() -> None:
                ui.run_javascript(f"const b = document.getElementById('{box}'); if (b) {{ b.dataset.full0 = {full0}; "
                                  f"b.dataset.full1 = {full1}; }} " +
                                  (f"tmShowAt('{box}', {part['min']}, {part['max']}, {focus:.2f})" if focus is not None
                                   else f"tmPlayPart('{box}', {part['min']}, {part['max']}, {part['min']})"))
                show()
                picture = await run.io_bound(self.trip.filmstrip, self.clip_of(it))
                if picture:
                    ui.run_javascript(f"const s = document.getElementById('{box}-strip'); "
                                      f"if (s) {{ s.src = '{data_url(str(picture))}'; s.style.display = 'block'; }}")

            async def act(callback) -> None:
                clash = next((n for x, y, n in taken or [] if part["min"] < y - 0.05 and x + 0.05 < part["max"]), None)
                if clash is not None:
                    ui.notify(f"Shot {clash} already uses part of these seconds (red on the strip). Choose seconds "
                              "outside the red, so nothing shows twice.", type="warning", multi_line=True)
                    return
                ui.run_javascript(f"const b = document.getElementById('{box}'); if (b) tmStop(b);")
                done = callback(part["min"], part["max"])
                if asyncio.iscoroutine(done):
                    done = await done
                if dialog.is_deleted:  # (what it changed redrew the place the window was opened from)
                    return
                if done:
                    dialog.close()
                else:
                    ui.run_javascript(f"tmPlayPart('{box}', {part['min']}, {part['max']}, {part['min']})")

            with ui.row().classes("w-full items-center justify-end gap-2"):
                ui.button("Play", icon="play_arrow", on_click=lambda: ui.run_javascript(
                    f"tmPlayRed('{box}', {part['min']}, {part['max']})")).props("outline dense no-caps") \
                    .tooltip("From the red line (Space)")
                ui.space()
                ui.button("Close", on_click=dialog.close).props("flat no-caps")
                for text, icon, callback, props in actions:
                    ui.button(text, icon=icon, on_click=lambda c=callback: act(c)).props(props)
            if extra is not None:
                extra(dialog)
            ui.timer(0.3, start, once=True)

        def hidden() -> None:
            ui.run_javascript(f"const b = document.getElementById('{box}'); if (b) tmStop(b);")
            if on_hide is not None:
                on_hide()

        dialog.on("hide", hidden)
        dialog.open()

    def show_cut(self, path: str) -> None:
        """After a cut: change that one clip where it is on screen (its picture, length, what hovering plays),
        rather than redrawing every tab."""
        row = next((r for r in self.trip.items() if r["path"] == path), None)
        if row is None:
            return
        for items in (self.all_items, self.items):
            for it in items:
                if it["path"] == path:
                    it.update(keep_start=row["keep_start"], keep_end=row["keep_end"], thumb=row["thumb"],
                              trim_start=row["trim_start"], trim_end=row["trim_end"])
        start, end = kept_part(row) or (0.0, row["duration"] or 0)
        words = f"✂ {fmt_length(end - start)}" if kept_part(row) else fmt_length(row["duration"])
        ui.run_javascript(f"tmShowCut({json.dumps(clip_id(path))}, {start + (end - start) * 0.3:.2f}, {end:.2f}, "
                          f"{json.dumps(data_url(row['thumb']))}, {json.dumps(words)})")
        self.storyboard_view.refresh()  # only drawn when you are on the Storyboard

    def suggest(self) -> None:
        n = self.trip.untick_doubles()
        ui.notify(f"Unticked {n} doubles: only the ⭐ of each set stays ticked. Everything else you chose is unchanged.")
        self.refresh_all()

    async def link_story(self, number: int, to_next: bool) -> None:
        later = [it for it in self.all_items if it["story"] == number + (1 if to_next else 0)]
        await run.io_bound(self.trip.link_story, number, to_next)
        self.choose_focus = number - (0 if to_next else 1)
        if later:  # stay where the two stories now meet
            self.choose_focus_path = min(later, key=lambda r: (r["taken_at"], r["path"]))["path"]
        self.stale = True
        self.refresh_choose()

    async def unlink_story(self, number: int) -> None:
        await run.io_bound(self.trip.unlink_story, number)
        self.choose_focus = number
        self.stale = True
        self.refresh_choose()

    async def keep_marked(self, members: list[dict], number: int) -> None:
        paths = [m["path"] for m in members]
        await run.io_bound(self.trip.keep_marked, paths)
        kept = sum(1 for m in members if m["chosen"])
        ui.notify(f"Sorted out: kept {kept} of {len(members)}." + (" The unticked ones go when you remove the unticked "
                  "files." if kept < len(members) else ""), type="positive")
        self.choose_focus, self.choose_focus_path = number, paths[0]  # stay where you were
        self.stale = True
        self.refresh_choose()

    async def accept_sets(self, story: int | None) -> None:
        sets, dropped = await run.io_bound(self.trip.accept_sets, story)
        if not sets:
            ui.notify("No look-alike sets left here.")
            return
        ui.notify(f"Accepted {sets} sets" + (f" ({dropped} doubles unticked)" if dropped else "")
                  + (". Next: the next level of doubles, or remove the unticked files." if story is None else "."),
                  type="positive", multi_line=True)
        if story is not None:
            self.choose_focus, self.choose_focus_path = story, None  # stay where you were
        self.stale = True
        self.refresh_choose()

    async def regroup_sets(self) -> None:
        await run.io_bound(self.trip.regroup_sets)
        ui.notify("The similar-shot frames are back.")
        self.stale = True
        self.refresh_choose()

    def keep_best(self, members: list[dict]) -> None:
        self.tick_in_place([m["path"] for m in members if not m["best"]], False)
        self.tick_in_place([m["path"] for m in members if m["best"]], True)

    async def rotate(self, path: str, degrees: int) -> None:
        try:
            await run.io_bound(self.trip.rotate, path, degrees)
        except Exception as exc:
            traceback.print_exc()
            ui.notify(f"Could not turn {Path(path).name}: {exc}", type="negative", multi_line=True)
            return
        way = "clockwise" if degrees > 0 else "anti-clockwise"
        ui.notify(f"Turned {Path(path).name} {way}.", timeout=1500)
        # Stay on the picture you turned: where it is on the screen now, it is after the page is drawn again.
        cid = clip_id(path)
        try:
            top = await ui.run_javascript(
                f"(() => {{ const e = document.querySelector('[data-tile=\"{cid}\"]'); "
                "return e ? e.getBoundingClientRect().top : null; })()", timeout=2.0)
        except Exception:
            top = None
        it = next((x for x in self.all_items if x["path"] == path), None)
        if it is not None:
            self.choose_focus, self.choose_focus_path = it["story"], path  # (drawn down to it)
        self.choose_keep_scroll = True
        self.refresh_all()  # last: this rebuilds the tile the button sits on
        self.choose_keep_scroll = False
        if top is not None:
            ui.run_javascript(
                f"setTimeout(() => {{ const e = document.querySelector('[data-tile=\"{cid}\"]'); "
                f"if (e) window.scrollBy(0, e.getBoundingClientRect().top - ({float(top):.1f})); }}, 150)")

    def choose_paths(self, paths: list[str], chosen: bool) -> None:
        self.tick_in_place(paths, chosen)

    def tick_in_place(self, paths: list[str], chosen: bool) -> None:
        """Tick or untick files without redrawing the list, so you stay where you are in it."""
        self.trip.set_chosen(paths, chosen)
        for path in paths:
            if path not in self.choose_tiles:
                continue
            it, box, tile = self.choose_tiles[path]
            it["chosen"] = int(chosen)
            box.set_value(chosen)  # quick_tick sees nothing changed
            tile.classes(remove="opacity-40") if chosen else tile.classes(add="opacity-40")
        for label, items in self.choose_counts:
            label.set_text(f"{sum(i['chosen'] for i in items)} of {len(items)} chosen")
        self.items = [x for x in self.all_items if x["chosen"]]
        self.stale = True
        self.update_sort_counts()

    def quick_tick(self, it: dict, chosen: bool, tile) -> None:
        """Tick or untick one file at once, without rebuilding the page (fast with thousands of files)."""
        if bool(it["chosen"]) == chosen:
            return
        self.trip.set_chosen([it["path"]], chosen)
        it["chosen"] = int(chosen)
        tile.classes(remove="opacity-40") if chosen else tile.classes(add="opacity-40")
        for label, items in self.choose_counts:
            if any(i is it for i in items):
                label.set_text(f"{sum(i['chosen'] for i in items)} of {len(items)} chosen")
        self.items = [x for x in self.all_items if x["chosen"]]
        self.stale = True
        self.update_sort_counts()

    def refresh_choose(self) -> None:
        """After ticking in Choose files: redraw just this tab; the others catch up when you open them."""
        self.reload()
        self.stale = True
        self.choose_view.refresh()

    def set_choose_filter(self, key: str, value: str) -> None:
        self.choose_filters[key] = value
        self.story_page = 1
        self.choose_view.refresh()

    def choose_order(self) -> list[str]:
        """The files of Choose files in the order they are on the page (story by story, look-alikes together)."""
        shown = self.choose_shown()
        out = []
        for number in sorted({it["story"] or 0 for it in shown}):
            sets: dict[int, list[str]] = {}
            for it in shown:
                if (it["story"] or 0) == number:
                    sets.setdefault(it["similar_set"] or 0, []).append(it["path"])
            out += [p for group in sets.values() for p in group]
        return out

    def choose_shown_files(self, chosen: bool) -> None:
        self.trip.set_chosen([it["path"] for it in self.choose_shown()], chosen)
        self.refresh_choose()

    # ---------- Library ----------

    def build_library(self) -> None:
        self.filters_view()
        self.gallery()

    @ui.refreshable
    def filters_view(self) -> None:
        if not self.items or not self.showing("Library"):
            return
        if self.library_focus:
            with ui.row().classes("w-full items-center gap-3 p-2 rounded bg-green-50 border border-green-200"):
                ui.button(f"Back to {self.library_focus['name']}", icon="arrow_back", on_click=self.library_back) \
                    .props("color=green unelevated no-caps")
                ui.label("This is the shot you double-clicked: lock a moment, cut it, change its theme or turn it here. "
                         "Double-click the picture to go back.").classes("text-sm text-green-900 flex-1")
                ui.button("Show the whole Library", on_click=self.library_all).props("flat dense no-caps")
            return
        opts = self.options()
        with ui.row().classes("w-full items-end gap-3"):
            for key, label in (("day", "Day"), ("theme", "Theme"), ("place", "Place"), ("scene", "Scene"), ("tag", "Tag"),
                               ("look", "Look-alike group"), ("person", "Person"), ("show", "Show")):
                if self.filters[key] not in opts[key]:
                    self.filters[key] = ALL
                ui.select(opts[key], value=self.filters[key], label=label, on_change=lambda e, k=key: self.set_filter(k, e.value)).classes("min-w-36")
        with ui.row().classes("items-center gap-2"):
            ui.button("Put all shown in the video", icon="check_box", on_click=lambda: self.include_shown(True)).props("flat dense")
            ui.button("Leave all shown out", icon="check_box_outline_blank", on_click=lambda: self.include_shown(False)).props("flat dense")

    # ---------- double-click: a shot in the Library, and back ----------

    def click_or_double(self, element, single, double) -> None:
        """A click waits a moment: a second click turns it into a double-click (and the single one is dropped)."""
        async def clicked() -> None:
            if self.pending_click is not None:
                return  # the second click of a double-click
            async def later() -> None:
                await asyncio.sleep(0.3)
                self.pending_click = None
                with self.window_timers:  # (this runs on its own: the page's slot, for notes and JavaScript)
                    result = single()
                    if asyncio.iscoroutine(result):
                        await result
            self.pending_click = asyncio.create_task(later())

        async def doubled() -> None:
            if self.pending_click is not None:
                self.pending_click.cancel()
                self.pending_click = None
            try:
                result = double()
                if asyncio.iscoroutine(result):
                    await result
            except Exception:
                traceback.print_exc()  # (so a failure shows in the Terminal)
                raise

        if single is not None:
            element.on("click", clicked)
        element.on("dblclick", doubled)

    def shot_window(self, path: str, name: str = "", changed: dict | None = None, at: float | None = None,
                    moment: dict | None = None, whole: bool = False) -> None:
        """Double-click on a shot anywhere: it opens big, right where you are. Watch it, cut it, make good parts of
        a video shots of their own (and drop the rest), lock a part, tick it, give it a theme.
        at: open it at this second (a suggestion: the seconds around it). moment: {"activity", "name", "t"} when
        opened from a moment's page: This is the moment / Not this, right there."""
        it = next((x for x in self.trip.items(parts=True) if x["path"] == path), None)
        if it is None:
            return
        if not whole and name in ("Make video", "the Storyboard") and self.trip.together(path)[0]:
            self.collage_window(path)  # (a collage or videos side by side: its own window)
            return
        if it.get("part_of"):  # a part Auto-edit made: its video
            it = next((x for x in self.trip.items() if x["path"] == it["part_of"]), it)
        path = it["path"]
        changed = changed if changed is not None else {"any": False}
        changed["again"] = False
        left = {"yes": False}  # this window was swapped for the next one (no closing work for it then)
        history = changed.setdefault("undo", [])  # (what, how to take it back): the Undo button in the window
        # Previous / Next go through the files in the order of the place you opened it from.
        order = self.choose_order() if name == "Choose files" else self.nav_order.get(name) or []
        if path not in order:
            order = [x["path"] for x in self.all_items]
        story = it.get("story")
        for old in list(self.window_box):  # windows closed before
            if isinstance(old, ui.dialog) and not old.value:
                old.delete()

        def after() -> None:  # when the window closes: the page behind shows what changed, at the same spot
            if left["yes"]:  # (it went on to the next shot, or drew itself again: that window carries on)
                return
            if not changed["again"]:  # the app learns your new highlights quietly, for this trip and the next
                background_tasks.create(run.io_bound(self.learn_quietly))
            if name == "Make video" and not changed["again"]:
                back_to_clip()
                if not changed["any"]:
                    self.editor_card.refresh()
            if changed["any"] and not changed["again"]:
                self.stale = True
                self.choose_focus, self.choose_focus_path = story, path  # Choose files draws down to the shot
                self.choose_keep_scroll = True
                self.refresh_all()
                self.choose_keep_scroll = False
                if name != "Make video":
                    ui.run_javascript("for (const ms of [150, 450, 900]) setTimeout(tmBackToShot, ms)")
            elif changed.get("moved") and not changed["again"] and name != "Make video":
                ui.run_javascript("for (const ms of [50, 300]) setTimeout(tmBackToShot, ms)")

        def back_to_clip() -> None:
            """Back in the editor: on the clip you opened, the red line at its start, the timeline showing it."""
            board = self.trip.storyboard()
            k = next((i for i, r in enumerate(board) if (in_video and r["path"] == in_video["path"])), None)
            if k is None:
                k = next((i for i, r in enumerate(board) if (r.get("part_of") or r["path"]) == path), None)
            if k is None:
                return
            self.draft_pick = k
            at = self.editor_times(board)[k]["start"] + 0.05
            self.draft_resume = at
            ui.run_javascript(f"window.tmDraftT = {at:.2f}; window.tmBackClip = {k}; "
                              "for (const ms of [300, 700, 1200]) setTimeout(() => { "
                              "const el = document.getElementById('tm-ed-' + window.tmBackClip), "
                              "sc = document.getElementById('tm-tl-scroll'); "
                              "if (el && sc) { sc.scrollLeft = Math.max(0, el.offsetLeft - sc.clientWidth * 0.3); "
                              "window.tmTlScroll = sc.scrollLeft; } "
                              "const v = document.getElementById('tm-draft'); "
                              "if (v && Math.abs(v.currentTime - window.tmDraftT) > 0.2) v.currentTime = window.tmDraftT; }, ms)")

        def mark() -> None:
            changed["any"] = True

        async def moment_answer(on: bool) -> None:
            t = moment["t"]
            if on:
                try:  # where you stopped the red line, else the second the app found
                    red = await ui.run_javascript(
                        f"(() => {{ const b = document.getElementById('tm-part-{clip_id(path)}'), "
                        "v = b && b.querySelector('video'); return v && v.src && v.paused && b.dataset.red ? "
                        "parseFloat(b.dataset.red) : null; })()", timeout=3)
                    t = round(float(red), 2) if red is not None else t
                except Exception:
                    pass
            await run.io_bound(self.trip.tick_moment, moment["activity"], moment["name"], path, t, on)
            mark()
            ui.notify(f"🔍 {moment['name']}: " + (f"this is it, at {fmt_length(t)}." if on else "not this one."),
                      type="positive" if on else "info", timeout=1800)

        def moment_row() -> None:
            if not moment:
                return
            with ui.row().classes("w-full items-center gap-2 px-2 py-1 rounded bg-purple-50"):
                ui.label(f"🔍 {moment['name']}: found at {fmt_length(moment['t'])} (orange on the strip).") \
                    .classes("text-sm text-purple-900")
                ui.button("This is the moment", icon="check", on_click=lambda: moment_answer(True)) \
                    .props("dense unelevated no-caps color=purple").tooltip("At the red line if you stopped it, "
                                                                            "else at the second found")
                ui.button("Not this", icon="close", on_click=lambda: moment_answer(False)) \
                    .props("dense outline no-caps color=grey-8")

        def controls(dialog) -> None:
            moment_row()
            shots = self.trip.shots_from(path) if it["kind"] == "video" else []
            if path in order:  # big Previous / Next on the sides: the next file without closing
                k = order.index(path)
                for side, step, icon in (("left", -1, "chevron_left"), ("right", 1, "chevron_right")):
                    target = order[k + step] if 0 <= k + step < len(order) else None
                    if target is None:
                        continue
                    ui.button(icon=icon, on_click=lambda t=target: go(t, dialog)) \
                        .props(f"round unelevated color=grey-9 size=xl data-nav={step}") \
                        .style(f"position:fixed;{side}:10px;top:50%;transform:translateY(-50%);z-index:20;opacity:.85") \
                        .tooltip(("Previous" if step < 0 else "Next") + " (← →)")
            with ui.row().classes("w-full items-center gap-3 px-2"):
                ui.checkbox("In the video", value=bool(it["chosen"]),
                            on_change=lambda e: (self.trip.set_chosen([path], e.value), mark())).props("dense")
                names = [a["name"] for a in self.trip.activities()]
                now = self.theme_of(it)

                async def theme(value: str) -> None:
                    await run.io_bound(self.trip.set_theme, [path], value)
                    mark()
                    ui.notify(f"Theme: {value}", timeout=1500)

                ui.select(sorted(set(names) | {now}), value=now, label="Theme",
                          on_change=lambda e: theme(e.value)).props("dense options-dense").classes("w-56")
                if it["kind"] == "video":
                    def backwards(on: bool) -> None:
                        self.trip.set_reverse(path, on)
                        history.append(("play backwards" if on else "play forwards",
                                        lambda: self.trip.set_reverse(path, not on)))
                        mark()
                        ui.notify("⇆ This shot plays backwards in the video (see it in the draft)." if on else
                                  "This shot plays forwards again.", timeout=2500)

                    ui.switch("⇆ Play backwards", value=bool(it.get("reverse")),
                              on_change=lambda e: backwards(e.value)).props("dense") \
                        .tooltip("The shot plays in reverse in the video. The preview here plays forwards; "
                                 "the draft shows it backwards.")
                ui.label(f"🕒 {fmt_time(it['taken_at'])} · 📍 {it['place'] or it['country'] or 'Unknown place'}") \
                    .classes("text-xs text-gray-600")
                ui.space()
                async def to_library() -> None:
                    dialog.close()
                    await self.open_in_library(path, name or "here")

                ui.button("Open in the Library", icon="photo_library", on_click=to_library) \
                    .props("flat dense no-caps size=sm")
            if in_video:
                with ui.row().classes("w-full items-center gap-2 px-2"):
                    def alternative(r=in_video) -> None:
                        dialog.close()
                        self.draft_swap_dialog(r, self.theme_of(r))

                    ui.button("Find an alternative", icon="auto_awesome", on_click=alternative) \
                        .props("unelevated no-caps color=purple") \
                        .tooltip("Don't like this one? The app suggests shots that could take its place")
            if it["kind"] == "video":
                video_tools(dialog)
            if history:
                with ui.row().classes("w-full items-center gap-2 px-2"):
                    ui.button(f"Undo: {history[-1][0]}", icon="undo", on_click=lambda: undo(dialog)) \
                        .props("outline dense no-caps")
            if in_video:
                in_video_box()
            if it["kind"] == "video":
                highlights_row()
            else:
                star_row()
            if it.get("shot_of"):
                ui.label(("✂ A shot you made from " if it["kind"] == "video" else "✂ A cropped copy of ")
                         + Path(it["shot_of"]).name).classes("text-xs text-green-800 px-2")
            if shots:
                with ui.column().classes("w-full gap-1 px-2"):
                    ui.label(f"Shots you made from this video ({len(shots)}): each is its own shot now, with its own "
                             "tick and theme.").classes("text-sm font-medium text-green-800")
                    with ui.row().classes("gap-2 items-center"):
                        for k, shot in enumerate(shots, 1):
                            with ui.row().classes("items-center gap-0 bg-green-50 border border-green-300 rounded px-2"):
                                ui.label(f"✂ Shot {k}: {fmt_length(shot['start'])}–{fmt_length(shot['end'])}") \
                                    .classes("text-xs")
                                ui.button(icon="close", on_click=lambda p=shot["path"]: take_back(p, dialog)) \
                                    .props("flat dense round size=xs").tooltip("Take this shot out again")
                        if it["chosen"]:
                            ui.button("Drop the rest of the video", icon="delete_sweep",
                                      on_click=lambda: drop_rest(dialog)).props("outline dense no-caps color=negative") \
                                .tooltip("Untick the whole video: only your shots go in. It leaves the trip when you "
                                         "remove the unticked files.")
                        else:
                            ui.label("The rest of the video is dropped (unticked).").classes("text-xs text-gray-500")

        def video_tools(dialog) -> None:
            """Turn the video, and zoom in on part of it (drag the yellow frame on the picture to move it)."""
            crop = json.loads(it["crop"]) if it.get("crop") else {}
            z = {"zoom": float(crop.get("zoom", 1.0)), "x": float(crop.get("x", 0.5)), "y": float(crop.get("y", 0.5))}
            box = f"tm-part-{clip_id(path)}"

            def show_zoom(save: bool = True) -> None:
                ui.run_javascript(f"tmVZoom('{box}', {z['zoom']}, {z['x']}, {z['y']})")
                zoom_words.text = ("Whole picture" if z["zoom"] <= 1.001 else
                                   f"Zoomed in {z['zoom']:.1f}×: the video keeps what's inside the yellow frame "
                                   "(drag it to move)")
                if save:
                    self.trip.set_crop(path, z["zoom"], z["x"], z["y"])
                    mark()

            def zoom_to(value: float) -> None:
                z["zoom"] = round(min(max(value, 1.0), 3.0), 2)
                if slider.value != z["zoom"]:
                    slider.value = z["zoom"]
                show_zoom()

            def dragged(e) -> None:
                z["x"], z["y"] = float(e.args["detail"]["x"]), float(e.args["detail"]["y"])
                show_zoom()

            with ui.row().classes("w-full items-center gap-2 px-2"):
                ui.button("Turn left", icon="rotate_left", on_click=lambda: rotate(-90, dialog)) \
                    .props("outline dense no-caps").tooltip("Turn the video 90° anti-clockwise")
                ui.button("Turn right", icon="rotate_right", on_click=lambda: rotate(90, dialog)) \
                    .props("outline dense no-caps").tooltip("Turn the video 90° clockwise")
                ui.button("Upside down", icon="sync", on_click=lambda: rotate(180, dialog)) \
                    .props("outline dense no-caps").tooltip("Turn it round (180°)")
                ui.separator().props("vertical")
                ui.button(icon="zoom_out", on_click=lambda: zoom_to(z["zoom"] / 1.2)).props("flat round dense") \
                    .tooltip("Zoom out")
                slider = ui.slider(min=1.0, max=3.0, step=0.05, value=z["zoom"]) \
                    .props("dense").classes("w-48 tm-zoom").on("change", lambda e: zoom_to(float(e.args)))
                ui.button(icon="zoom_in", on_click=lambda: zoom_to(z["zoom"] * 1.2)).props("flat round dense") \
                    .tooltip("Zoom in")
                ui.button("Whole picture", icon="fit_screen",
                          on_click=lambda: (z.update(x=0.5, y=0.5), zoom_to(1.0))).props("flat dense no-caps")
                zoom_words = ui.label().classes("text-xs text-gray-600")
                ui.label("· Trim: drag the green bar, then Cut: keep only this part").classes("text-xs text-gray-500")
            ui.timer(0.6, lambda: show_zoom(save=False), once=True)
            self.vzoom_moved = dragged  # (the big window passes on where you dragged the frame)

        @ui.refreshable
        def star_row() -> None:  # a photo's ⭐: like a video's highlight, it is always in the video
            on = self.trip.starred(path)
            with ui.row().classes("w-full items-center gap-2 px-2"):
                ui.button("⭐ Highlight: always in the video" if on else "☆ Make it a highlight",
                          on_click=lambda: star(not on)) \
                    .props(("unelevated color=amber-8" if on else "outline color=amber-9") + " dense no-caps data-photostar") \
                    .tooltip("Press Shift for the same. A highlight photo is always used in the video.")
                ui.label("Shift: highlight on / off").classes("text-xs text-amber-800")

        def star(on: bool) -> None:
            self.trip.set_star(path, on)
            history.append(("highlight" if on else "take off highlight", lambda: self.trip.set_star(path, not on)))
            mark()
            ui.notify("⭐ Highlight: this photo is always in the video." if on else "Highlight taken off.",
                      timeout=1500)
            star_row.refresh()

        @ui.refreshable
        def highlights_row() -> None:
            marks = self.trip.postits(path)
            with ui.row().classes("w-full items-center gap-2 px-2"):
                ui.label("⭐ Highlights:" if marks else "Space plays and pauses · tap Shift at a great moment to put "
                         "a ⭐ highlight there: it is always in the video.").classes("text-xs text-amber-800")
                for k, m in enumerate(marks):
                    with ui.row().classes("items-center gap-1 bg-amber-50 border border-amber-300 rounded px-1"):
                        if m.get("thumb"):
                            ui.image(data_url(m["thumb"])).classes("w-20 rounded cursor-pointer") \
                                .style("aspect-ratio:16/10").props("fit=contain no-spinner") \
                                .on("click", lambda t=m["t"]: ui.run_javascript(
                                    f"(() => {{ const b = document.getElementById('tm-part-{clip_id(path)}'); if (b) "
                                    f"tmShowAt(b.id, parseFloat(b.dataset.start), parseFloat(b.dataset.end), "
                                    f"{t:.2f}); }})()")).tooltip("Go to this frame")
                        ui.label(f"⭐ {clock(m['t'])}" + (f" {m['name']}" if m.get("name") else "")).classes("text-xs")
                        ui.button(icon="close", on_click=lambda k=k: unhighlight(k)) \
                            .props("flat dense round size=xs").tooltip("Take this highlight off")
                for idea in self.trip.highlight_ideas(path):
                    with ui.row().classes("items-center gap-0 bg-orange-50 border border-orange-400 border-dashed "
                                          "rounded px-2").tooltip("The app suggests this frame: click to see it. Keep "
                                                                  "puts the ⭐ at the red line (drag it to move)"):
                        ui.label(f"Suggested {clock(idea['t'])}").classes("text-xs text-orange-800 cursor-pointer "
                                                                          "underline") \
                            .on("click", lambda t=idea["t"]: ui.run_javascript(
                                f"(() => {{ const b = document.getElementById('tm-part-{clip_id(path)}'); if (b) "
                                f"tmShowAt(b.id, parseFloat(b.dataset.start), parseFloat(b.dataset.end), {t:.2f}); }})()"))
                        ui.button(icon="star", on_click=lambda t=idea["t"]: answer(t, True)) \
                            .props("flat dense round size=xs color=amber-8").tooltip("Keep: a ⭐ highlight")
                        ui.button(icon="close", on_click=lambda t=idea["t"]: answer(t, False)) \
                            .props("flat dense round size=xs").tooltip("Drop: not a highlight")

        async def answer(t: float, keep: bool) -> None:
            idea_t = t
            if keep:  # at the red line if you moved it near there (within 5 s), else at the suggested frame
                at = t
                try:
                    red = await ui.run_javascript(
                        f"(() => {{ const b = document.getElementById('tm-part-{clip_id(path)}'), "
                        "v = b && b.querySelector('video'); return v && v.src && v.paused && b.dataset.red ? "
                        "parseFloat(b.dataset.red) : null; })()", timeout=3)
                    if red is not None and abs(float(red) - t) <= 5:
                        at = round(float(red), 2)
                except Exception:
                    pass
                await run.io_bound(self.trip.keep_idea, path, at, t)
                t = at
            else:
                await run.io_bound(self.trip.drop_idea, path, t)
            if keep:
                history.append((f"highlight {clock(t)}", lambda t=t: drop_highlight_at(t)))
                ui.run_javascript(f"tmHighlightMark({t:.2f})")
            mark()
            ui.run_javascript(f"document.querySelectorAll('.q-dialog .tm-idea[data-t=\"{idea_t:.2f}\"]')"
                              ".forEach(el => el.remove())")
            ui.notify(("⭐ Kept" if keep else "Dropped") + ": the app learns from it.", timeout=1500)
            highlights_row.refresh()

        def highlight(t: float) -> None:
            self.trip.add_postit(path, round(t, 2))
            history.append((f"highlight {clock(t)}", lambda t=round(t, 2): drop_highlight_at(t)))
            mark()
            ui.notify(f"⭐ Highlight at {clock(t)}: it is always in the video.", timeout=1500)
            ui.run_javascript(f"tmHighlightMark({t:.2f})")
            highlights_row.refresh()

        def drop_highlight_at(t: float) -> None:
            marks = self.trip.postits(path)
            k = min(range(len(marks)), key=lambda k: abs(marks[k]["t"] - t), default=None)
            if k is not None and abs(marks[k]["t"] - t) < 0.4:
                self.trip.change_postit(path, k, remove=True)

        def unhighlight(k: int) -> None:
            marks = self.trip.postits(path)
            if not 0 <= k < len(marks):
                return
            t = marks[k]["t"]
            self.trip.change_postit(path, k, remove=True)
            history.append((f"take off highlight {clock(t)}", lambda t=t: self.trip.add_postit(path, t)))
            mark()
            ui.notify(f"Highlight at {clock(t)} taken off.", timeout=1500)
            ui.run_javascript(f"tmHighlightsSet({json.dumps([m['t'] for m in self.trip.postits(path)])})")
            highlights_row.refresh()

        def go(target: str, dialog) -> None:  # the next (or previous) file, in the same window
            changed["again"] = True
            changed["moved"] = True  # (closing it brings you to this file, not the one you opened first)
            ui.run_javascript(f"if (window.tmShotAnchor) window.tmShotAnchor.sel = "
                              f"'[data-tile=\"{clip_id(target)}\"], [data-shot=\"{clip_id(target)}\"]'")
            left["yes"] = True

            def swap() -> None:  # the next one opens first, on top; then this one goes: no flicker in between
                self.shot_window(target, name, changed)
                dialog.close()

            with self.window_timers:
                ui.timer(0.01, swap, once=True)

        def again() -> None:  # the same window, drawn again with what changed
            changed["again"] = True
            left["yes"] = True
            with self.window_timers:
                ui.timer(0.2, lambda: self.shot_window(path, name, changed), once=True)

        async def undo(dialog) -> None:
            what, back = history.pop()
            done = back()
            if asyncio.iscoroutine(done):
                await done
            mark()
            ui.notify(f"Undone: {what}", timeout=2000)
            if self.trip.query("SELECT 1 FROM items WHERE path = ?", (path,)):
                again()
            else:  # (undoing a crop: back to the photo)
                go(changed.get("back_to") or path, dialog)
                return
            dialog.close()

        async def rotate(degrees: int, dialog) -> None:
            await run.io_bound(self.trip.rotate, path, degrees)
            history.append(("turn", lambda: run.io_bound(self.trip.rotate, path, -degrees)))
            mark()
            again()
            dialog.close()

        async def crop(dialog) -> None:
            try:
                copy = await run.io_bound(self.trip.crop_to_copy, path)
            except Exception as exc:
                ui.notify(f"Could not crop: {exc}", type="warning")
                return

            async def back(copy=copy) -> None:
                self.trip.execute("UPDATE items SET position = (SELECT position FROM items WHERE path = ?) "
                                  "WHERE path = ?", (copy, path))
                self.trip.set_chosen([path], True)
                await run.io_bound(self.trip.drop_shot, copy)

            history.append(("crop", back))
            changed["back_to"] = path
            mark()
            ui.notify("Cropped: the cropped photo is used from now on (your own photo is not changed).",
                      type="positive")
            go(copy, dialog)

        async def take_back(shot: str, dialog) -> None:
            await run.io_bound(self.trip.drop_shot, shot)
            mark()
            again()
            dialog.close()

        def drop_rest(dialog) -> None:
            history.append(("drop the rest", lambda: self.trip.set_chosen([path], True)))
            self.trip.set_chosen([path], False)
            mark()
            ui.notify("Only the shots you made go in now.", type="positive")
            again()
            dialog.close()

        # Opened from the video editor: the part this clip plays in the video, and a button to use a new one.
        in_video = None
        if name == "Make video":
            board = self.trip.storyboard()
            mine = [r for r in board if (r.get("part_of") or r["path"]) == path]
            pick = self.draft_pick
            in_video = board[pick] if pick is not None and pick < len(board) and board[pick] in mine else \
                (mine[0] if mine else None)
        slot = Trip.screen_length(in_video) if in_video else 0.0


        async def make(a: float, b: float) -> bool:
            n = ui.notification("Making the shot…", spinner=True, timeout=None)
            try:
                made = await run.io_bound(self.trip.make_shot, path, a, b)
                history.append((f"shot {fmt_length(a)}–{fmt_length(b)}",
                                lambda made=made: run.io_bound(self.trip.drop_shot, made)))
            except Exception as exc:
                traceback.print_exc()
                n.dismiss()
                ui.notify(f"Could not make the shot: {exc}", type="negative", multi_line=True)
                return False
            n.dismiss()
            mark()
            ui.notify(f"New shot: {fmt_length(a)}–{fmt_length(b)}. Mark another good part, or close.",
                      type="positive")
            return True

        async def make_and_reopen(a: float, b: float) -> bool:
            if await make(a, b):
                again()
                return True
            return False

        async def keep(a: float, b: float) -> bool:
            was = (it.get("keep_start"), it.get("keep_end"))
            await run.io_bound(self.trip.set_keep, path, a, b)
            history.append(("cut", lambda: run.io_bound(self.trip.set_keep, path, *was)))
            mark()
            ui.notify(f"Cut: only {fmt_length(a)}–{fmt_length(b)} is used.", type="positive")
            again()  # the window stays open, now showing the part you kept
            return True

        def lock(a: float, b: float) -> bool:
            was = self.trip.locks(path)
            self.trip.add_lock(path, a, b)
            history.append(("lock", lambda: self.trip.set_locks(path, was, checkpoint=False)))
            mark()
            ui.notify(f"Locked {fmt_length(a)}–{fmt_length(b)}: the video always uses it whole.", type="positive")
            again()
            return True

        shots = self.trip.shots_from(path)
        length = it["duration"] or 0
        a, b = (shots[-1]["end"], min(length, shots[-1]["end"] + 4)) if shots and shots[-1]["end"] < length - 1 \
            else (length * 0.3, min(length, length * 0.3 + 4))
        if in_video and in_video.get("trim_start") is not None and in_video.get("trim_end") is not None:
            a, b = float(in_video["trim_start"]), float(in_video["trim_end"])
        if at is not None and it["kind"] == "video":  # a suggested second: the seconds around it
            a, b = moments.window(at, length, *(kept_part(it) or (0.0, length)))

        def in_video_box() -> None:
            """Opened from the editor: this clip's speed in the video, and the draft made again right here."""
            with ui.column().classes("w-full gap-1 px-2 py-1 rounded bg-indigo-50"):
                if it["kind"] == "video":
                    speed_row()
                    sound_row(in_video)
                with ui.row().classes("w-full items-center gap-2"):
                    ui.button("Show me in the video", icon="play_circle", on_click=update_and_show) \
                        .props("unelevated no-caps color=indigo") \
                        .tooltip("Plays this clip in the video straight away, with a second before and after it")
                    ui.label("Clips play at normal speed unless you pick another here. Use this part in the video: "
                             + ("the part you set ends on the nearest pin and the clips after it move along."
                                if self.trip.edit_stage() == "fitted" else
                                "the clip plays that part (rough cut: Fit to the music puts it on a pin later).")
                             if it["kind"] == "video" else
                             "Turn or crop the photo above, then see it in the video here.") \
                        .classes("text-xs text-gray-600 flex-1")
                preview_box()

        def sound_row(r: dict) -> None:
            with ui.row().classes("w-full items-center gap-2"):
                ui.label("Its own sound in the video:").classes("text-sm font-medium text-indigo-900 w-48")
                self.sound_control(r, on_change=lambda: (mark(), self.sound_changed()))

        @ui.refreshable
        def speed_row() -> None:
            row = next((r for r in self.trip.storyboard() if r["path"] == in_video["path"]), in_video)
            speed = row.get("speed") or 1.0
            with ui.row().classes("w-full items-center gap-1 no-wrap"):
                ui.label(f"Speed in the video: {speed:g}×").classes("text-sm font-medium text-indigo-900 w-48")
                for v in self.SPEEDS:
                    ui.button(f"{v:g}×", on_click=lambda v=v: set_speed(v)) \
                        .props("dense no-caps size=sm " + ("unelevated color=indigo" if abs(v - speed) < 0.01
                                                           else "outline color=indigo"))
                ui.slider(min=0.25, max=3.0, step=0.05, value=speed).props("dense label color=indigo") \
                    .classes("flex-1 min-w-32").on("change", lambda e: set_speed(float(e.args)))

        def set_speed(value: float) -> None:
            row = next((r for r in self.trip.storyboard() if r["path"] == in_video["path"]), in_video)
            was = (row["trim_start"], row["trim_end"], row.get("speed") or 1.0)
            value = round(max(0.25, min(3.0, value)), 2)
            self.trip.set_speed(in_video["path"], value)
            if self.trip.edit_stage() == "fitted":  # its place, at the speed you chose (rough cut: the same part)
                self.trip.fit_length(in_video["path"], slot)
            self.reflow_board()
            history.append((f"speed {value:g}×", lambda: (self.trip.set_trim(in_video["path"], was[0], was[1]),
                                                          self.trip.set_speed(in_video["path"], was[2]))))
            mark()
            ui.notify(f"Plays at {value:g}×. Show me in the video to see it.", timeout=2500)
            speed_row.refresh()

        @ui.refreshable
        def preview_box() -> None:
            shown = changed.get("preview")
            if not shown:
                return
            ui.html(self.instant_html("tm-mini", "width:100%;height:34vh;border-radius:6px"),
                    sanitize=False).classes("w-full")
            ui.run_javascript(f"setTimeout(() => {{ const m = tmInstant('tm-mini', {json.dumps(shown['data'])}); "
                              "if (m) m.play(); }, 50)")
            ui.label(f"Your clip in the video ({shown['when']}), with a second before and after, again and again.") \
                .classes("text-xs text-indigo-900")

        async def update_and_show() -> None:
            board = self.trip.storyboard()
            k = next((i for i, r in enumerate(board) if r["path"] == in_video["path"]), None)
            if k is None:
                ui.notify("This clip is not in the video now.", type="warning")
                return
            times = self.editor_times(board)
            start, length = times[k]["start"], times[k]["length"]
            loop = (max(0.0, start - 1.0), start + length + 1.0)
            changed["preview"] = {"data": self.instant_data(board, times, loop),
                                  "when": f"{fmt_length(start)}–{fmt_length(start + length)}"}
            preview_box.refresh()
            ui.run_javascript("setTimeout(() => { const m = document.getElementById('tm-mini'); "
                              "if (m) m.scrollIntoView({behavior: 'smooth', block: 'center'}); }, 200)")

        def use_in_video(a: float, b: float) -> bool:
            """The part you set plays in the clip's slot: the cuts stay on your pins, the speed fits it in."""
            self.trip.checkpoint("using a new part of a clip")
            self.trip.set_trim(in_video["path"], round(a, 3), round(b, 3))
            gone = self.reflow_board()  # at its own speed: it ends on the nearest pin, the rest move along
            history.append(("use in the video", lambda: self.trip.undo()))
            mark()
            fitted = self.trip.edit_stage() == "fitted"
            ui.notify(f"The video now uses {fmt_length(a)}–{fmt_length(b)} at its own speed"
                      + (", ending on the nearest pin; the clips after it moved along" if fitted else
                         "; the clips after it moved along (Fit to the music puts the cuts on your pins)")
                      + (f" ({gone} at the end left the video)" if gone else "") + ".", type="positive",
                      multi_line=True)
            return True

        if it["kind"] != "video":
            with self.window_box, ui.dialog().props("maximized transition-show=none transition-hide=none") as dialog, \
                    ui.card().classes("w-full h-full gap-2 overflow-auto"):
                ui.label(Path(path).name).classes("text-lg font-semibold")
                self.photo_editor(it, dialog, mark, lambda d, dialog=dialog: rotate(d, dialog),
                                  lambda dialog=dialog: crop(dialog))
                controls(dialog)
                with ui.row().classes("w-full justify-end"):
                    ui.button("Close", on_click=dialog.close).props("flat no-caps")
            dialog.on("hide", after)
            dialog.open()
            return

        actions = [("Lock this part", "lock", lock, "outline no-caps"),
                   ("Cut: keep only this part", "content_cut", keep, "outline no-caps"),
                   ("Make this a shot", "add_box", make_and_reopen, "outline no-caps" if in_video else "color=green no-caps")]
        if in_video:
            actions.append(("Use this part in the video", "published_with_changes", use_in_video,
                            "unelevated no-caps color=deep-orange"))
        it = it | {"ideas": self.trip.highlight_ideas(path)  # the app's suggested highlights, in orange
                   + ([{"t": at}] if at is not None and it["kind"] == "video" else [])}
        with self.window_box:
            self.part_dialog(
            it, a, b, (f"📌 Video {order.index(path) + 1} of {len(order)} · {Path(path).name} · press and drag "
                       "along the strip to look through it, tap Shift at a great moment" if name == self.BEST_BITS
                       else f"{Path(path).name} · " + (f"clip in the video ({slot:.1f} s): set its part" if in_video
                                                       else "mark a good part")),
            actions,
            made=[(x["start"], x["end"], f"shot {k}") for k, x in enumerate(shots, 1)],
            extra=controls, on_hide=after, full=True, on_highlight=highlight,
            focus=at if at is not None and it["kind"] == "video" else None)

    def photo_editor(self, it: dict, dialog, mark, on_rotate=None, on_crop=None) -> None:
        """Zoom in on a photo and drag it to choose what shows: the video uses exactly that part."""
        path = it["path"]
        from PIL import Image

        crop = json.loads(it["crop"]) if it.get("crop") else {}
        state = {"zoom": float(crop.get("zoom", 1.0)), "x": float(crop.get("x", 0.5)), "y": float(crop.get("y", 0.5))}
        try:
            with Image.open(it["thumb"]) as img:
                ratio = img.width / img.height
        except (OSError, TypeError, AttributeError, ValueError):
            ratio = 1.5
        key = f"tm-crop-{clip_id(path)}"
        # Sharp when zoomed in: the photo itself (if the browser can show it and it isn't turned), else its preview.
        sharp = data_url(it["thumb"])
        if not it.get("rotation") and self.media_route and Path(path).suffix.lower() in (".jpg", ".jpeg", ".png", ".heic"):
            try:
                sharp = self.video_src(Path(path))
            except ValueError:
                pass
        # The photo in a frame of its own shape; zooming and dragging move it inside the frame.
        with ui.element("div").classes("w-full flex justify-center"):
            frame = ui.html(
                f'<div id="{key}" onpointerdown="tmCropDown(event, this)" style="position:relative;overflow:hidden;'
                f'border-radius:6px;background:#1f2937;cursor:grab;touch-action:none;aspect-ratio:{ratio:.4f};'
                f'height:min(calc(100vh - 300px), calc((100vw - 80px) / {ratio:.4f}));max-width:100%">'
                f'<img src="{sharp}" onerror="this.onerror=null;this.src=\'{data_url(it["thumb"])}\'" draggable="false" style="position:absolute;inset:0;width:100%;'
                f'height:100%;object-fit:fill;transform-origin:center;pointer-events:none"></div>', sanitize=False)
        frame.on("dblclick", lambda: dialog.close())
        info = ui.label().classes("text-xs text-gray-600 px-2")

        def show(save: bool = True) -> None:
            z = state["zoom"]
            state["x"] = min(max(state["x"], 0.5 / z), 1 - 0.5 / z)
            state["y"] = min(max(state["y"], 0.5 / z), 1 - 0.5 / z)
            ui.run_javascript(f"tmCropShow('{key}', {z}, {state['x']}, {state['y']})")
            info.text = ("The whole photo is shown. Zoom in to show only a part of it." if z <= 1.001 else
                         f"Zoomed in {z:.1f}×: the video shows just this part. Drag the photo to move it.")
            if save:
                self.trip.set_crop(path, z, state["x"], state["y"])
                mark()

        def dragged(e) -> None:
            state["x"], state["y"] = float(e.args["detail"]["x"]), float(e.args["detail"]["y"])
            show()

        frame.on("tmcrop", dragged, ["detail"])

        def zoom_to(value: float) -> None:
            state["zoom"] = min(max(value, 1.0), 5.0)
            slider.value = state["zoom"]
            show()

        with ui.row().classes("w-full items-center gap-2 px-2"):
            ui.button(icon="zoom_out", on_click=lambda: zoom_to(state["zoom"] / 1.25)).props("flat round dense") \
                .tooltip("Zoom out")
            slider = ui.slider(min=1.0, max=5.0, step=0.05, value=state["zoom"],
                               on_change=lambda e: (state.update(zoom=float(e.value)), show())) \
                .classes("w-64 tm-zoom").props("dense")
            ui.button(icon="zoom_in", on_click=lambda: zoom_to(state["zoom"] * 1.25)).props("flat round dense") \
                .tooltip("Zoom in")
            ui.button("Whole photo", icon="fit_screen",
                      on_click=lambda: (state.update(x=0.5, y=0.5), zoom_to(1.0))).props("flat dense no-caps")
            if on_crop is not None:
                ui.button("Crop", icon="crop", on_click=lambda: on_crop() if state["zoom"] > 1.001 else
                          ui.notify("Zoom in and move the photo first: Crop keeps what you see.", type="info")) \
                    .props("unelevated dense no-caps color=primary") \
                    .tooltip("Keep only what you see: saved as a cropped copy, your photo is not changed")
            if on_rotate is not None:
                ui.separator().props("vertical")
                ui.button("Turn left", icon="rotate_left", on_click=lambda: on_rotate(-90)) \
                    .props("outline dense no-caps").tooltip("Turn 90° anti-clockwise")
                ui.button("Turn right", icon="rotate_right", on_click=lambda: on_rotate(90)) \
                    .props("outline dense no-caps").tooltip("Turn 90° clockwise")
                ui.button("Upside down", icon="sync", on_click=lambda: on_rotate(180)) \
                    .props("outline dense no-caps").tooltip("Turn it round (180°)")
            ui.label("← → next photo (hold to go through) · double-click the photo to close.") \
                .classes("text-xs text-gray-500")
        ui.timer(0.2, lambda: show(save=False), once=True)

    async def open_in_library(self, path: str, name: str) -> None:
        """Double-click on a shot anywhere: show just that shot in the Library, with a button back here."""
        try:
            scroll = float(await ui.run_javascript("window.scrollY", timeout=2) or 0)
        except Exception:
            scroll = 0.0
        clip = next((it.get("part_of") or it["path"] for it in self.all_items if it["path"] == path), path)
        self.library_focus = {"path": clip, "tab": self.panels.value, "name": name, "scroll": scroll}
        self.panels.set_value(self.library_tab)
        ui.run_javascript("setTimeout(() => window.scrollTo(0, 0), 300)")

    def library_back(self) -> None:
        focus, self.library_focus = self.library_focus, None
        if not focus:
            return
        if focus["name"] == "Choose files":  # a long list is drawn in parts: draw down to the shot, then show it
            it = next((x for x in self.all_items if x["path"] == focus["path"]), None)
            if it:
                self.choose_focus, self.choose_focus_path = it["story"], it["path"]
        self.panels.set_value(focus["tab"])
        if focus["name"] != "Choose files":
            ui.run_javascript(f"setTimeout(() => window.scrollTo(0, {focus['scroll']}), 700)")

    def library_all(self) -> None:
        self.library_focus = None
        self.filters_view.refresh()
        self.gallery.refresh()

    def set_filter(self, key: str, value: str) -> None:
        self.filters[key] = value
        self.page = 1
        self.gallery.refresh()

    def include_shown(self, included: bool) -> None:
        self.trip.set_included([it["path"] for it in self.shown()], included)
        self.reload()
        self.gallery.refresh()
        self.video_summary.refresh()
        self.storyboard_view.refresh()
        self.stories_view.refresh()

    @ui.refreshable
    def gallery(self) -> None:
        if not self.showing("Library"):
            return
        if not self.items:
            if self.trip:
                ui.label("Nothing to show yet.").classes("text-gray-500")
            return
        shown = self.shown()
        in_video = sum(it["included"] for it in self.items)
        with ui.row().classes("w-full items-center"):
            ui.label(f"Showing {len(shown)} of {len(self.items)} · {in_video} in the video").classes("text-sm text-gray-500")
            ui.space()
            ui.label("In a row:").classes("text-sm text-gray-600")
            ui.toggle({2: "2", 3: "3", 4: "4"}, value=self.library_columns,
                      on_change=lambda e: self.set_library_columns(e.value)).props("dense no-caps")
        pages = max(1, -(-len(shown) // PAGE_SIZE))
        self.page = min(self.page, pages)
        columns = 2 if self.library_focus else self.library_columns
        with ui.grid().classes("w-full gap-3").style(f"grid-template-columns:repeat({columns},minmax(0,1fr))"):
            for it in shown[(self.page - 1) * PAGE_SIZE : self.page * PAGE_SIZE]:
                self.card(it)
        if pages > 1:
            ui.pagination(1, pages, direction_links=True, value=self.page, on_change=lambda e: self.go_to_page(e.value))

    def set_library_columns(self, columns: int) -> None:
        self.library_columns = columns
        save_settings(library_columns=columns)
        self.gallery.refresh()

    def go_to_page(self, page: int) -> None:
        self.page = page
        self.gallery.refresh()

    def card(self, it: dict) -> None:
        with ui.card().tight().classes("w-full" + ("" if it["included"] else " opacity-50")) as card:
            if it["thumb"]:
                height = {2: 320, 3: 220}.get(self.library_columns, 150)  # fewer in a row: bigger pictures
                if self.library_focus:
                    height = 420
                    card.classes("ring-4 ring-green-500")
                with ui.element("div").classes("w-full").on("dblclick", self.library_back if self.library_focus else
                                                            lambda it=it: self.shot_window(it["path"], "the Library")):
                    cut = {"box": self.preview(it, f"width:100%;height:{height}px", start=self.hover_start(it))}
                cut["area"] = ui.column().classes("w-full gap-1")  # the cutting handles, when you press ✂
            else:
                ui.label("Analysing…").classes("w-full thumb flex items-center justify-center bg-gray-100 text-gray-400")
            with ui.column().classes("p-2 gap-1 w-full"):
                kept = kept_part(it)
                icon = ("🎬 " + (f"✂ {fmt_length(kept[1] - kept[0])}" if kept else fmt_length(it["duration"]))
                        if it["kind"] == "video" else "📷")
                with ui.row().classes("gap-1 items-center"):
                    if it["kind"] == "video":
                        ui.label("🎬").classes("text-xs")
                        ui.label(icon[2:]).classes(f"text-xs text-gray-600 tm-len-{clip_id(it['path'])}")
                    else:
                        ui.label("📷").classes("text-xs")
                    ui.label(f"· {fmt_time(it['taken_at'])}").classes("text-xs text-gray-600")
                if it.get("ai_caption"):
                    ui.label(f"📝 {it['ai_caption']}" + " ★" * (it["ai_score"] or 0)).classes("text-xs italic") \
                        .tooltip("The AI director's description, and how much it stands out (1 to 5 stars)")
                cut_note = ui.label("✂ Selected, but not in the current Auto-edit cut").classes("text-xs text-orange-700")
                cut_note.visible = bool(it["included"] and it["cut_out"])
                ui.label(f"🏷 {self.theme_of(it)}" + ("" if it.get("theme") else " (suggested)")) \
                    .classes("text-xs text-indigo-800")
                if it["kind"] == "video" and it["thumb"]:
                    self.lock_area(it, "full")
                place = ", ".join(x for x in (it["place"], it["country"]) if x) or "Unknown place"
                if it["place_guessed"]:
                    ui.label(f"📍 {place} (guessed from photos taken around then)").classes("text-xs text-gray-500")
                else:
                    ui.label(f"📍 {place}").classes("text-xs")
                ui.select(
                    [*SCENES],
                    value=it["scene_final"],
                    label="Scene",
                    on_change=lambda e, path=it["path"]: self.trip.set_scene(path, e.value),
                ).props("dense options-dense").classes("w-full text-xs")
                ui.select(
                    EDITABLE_TAGS,
                    value=[t for t in it["tags_final"] if t in EDITABLE_TAGS],
                    multiple=True,
                    label="Tags",
                    on_change=lambda e, path=it["path"]: self.set_tags(path, e.value),
                ).props("dense options-dense use-chips").classes("w-full text-xs")
                with ui.row().classes("items-center gap-1"):
                    if it["people_tag"]:
                        ui.label(f"👥 {it['people_tag']}").classes("text-xs text-gray-600")
                    if it["look_name"]:
                        ui.button(
                            f"🔗 {it['look_name'].split(' · group')[0]}",
                            on_click=lambda name=it["look_name"]: self.show_look_alike(name),
                        ).props("flat dense no-caps size=sm").tooltip("Show everything that looks like this")
                if it["people"]:
                    with ui.row().classes("gap-1"):
                        for name in it["people"]:
                            ui.badge(name).props("outline")
                with ui.row().classes("items-center justify-between w-full no-wrap"):
                    ui.checkbox(
                        "In the video",
                        value=bool(it["included"]),
                        on_change=lambda e, path=it["path"], card=card, note=cut_note: self.toggle(path, e.value, card, note),
                    ).props("dense")
                    with ui.row().classes("gap-0 no-wrap items-center"):
                        if it["rotation"]:
                            ui.label(f"↻ {it['rotation']}°").classes("text-xs text-orange-700") \
                                .tooltip("Turned by the app or by you; use the arrows to change it")
                        rotate_buttons(lambda degrees, path=it["path"]: self.rotate(path, degrees))
                        if it["thumb"]:
                            self.cut_button(it, cut)

    def set_tags(self, path: str, tags: list[str]) -> None:
        self.trip.set_tags(path, tags)
        for it in self.items:
            if it["path"] == path:
                it["tags_final"] = tags

    def show_look_alike(self, name: str) -> None:
        self.filters["look"] = name
        self.page = 1
        self.filters_view.refresh()
        self.gallery.refresh()

    def toggle(self, path: str, included: bool, card: ui.card | None = None, note: ui.label | None = None) -> None:
        """Put a file in the video or take it out; it shows at once (and counts in the video's length)."""
        self.trip.set_included([path], included)
        for it in self.items:
            if it["path"] == path:
                it["included"], it["cut_out"] = int(included), 0
        if card is not None:
            card.classes(remove="opacity-50") if included else card.classes(add="opacity-50")
        if note is not None:
            note.visible = False
        if included:
            ui.notify("Added to the video. It's in the Storyboard, in time order.", type="positive", timeout=2000)
        self.video_summary.refresh()
        self.storyboard_view.refresh()
        self.stories_view.refresh()

    # ---------- People ----------

    @ui.refreshable
    def people_view(self) -> None:
        if not self.showing("People"):
            return
        if not self.trip:
            ui.label("Open a trip first.").classes("text-gray-500")
            return
        people = self.trip.people()
        if not people:
            self.no_people_yet()
            return
        names = self.trip.person_names()
        named = [p for p in people if p["status"] == "named"]
        new = [p for p in people if p["status"] == "new"]
        rare = [p for p in people if p["status"] == "rare"]
        unknown = [p for p in people if p["status"] == "unknown"]

        ui.label(
            "Pick a name from the list or type a new one. Picking a name you already used joins the two groups. "
            "Click Unknown for strangers; they disappear from the Library labels and filters."
        ).classes("text-sm text-gray-600")

        if new:
            ui.label(f"Who is this? ({len(new)})").classes("text-lg font-semibold mt-2")
            self.person_grid(new, names)
        if named:
            ui.label(f"Named ({len(named)})").classes("text-lg font-semibold mt-2")
            self.person_grid(named, names)
        if rare:
            with ui.row().classes("w-full items-center mt-2"):
                ui.label(f"Seen only once or twice ({len(rare)})").classes("text-lg font-semibold")
                ui.button(
                    "Mark all of these as Unknown",
                    icon="visibility_off",
                    on_click=lambda ids=[p["id"] for p in rare]: self.mark_unknown(ids),
                ).props("outline dense")
            ui.label(
                "Often people in the background. Check for anyone you know (including yourself) "
                "before marking the rest as Unknown."
            ).classes("text-sm text-gray-500")
            self.person_grid(rare, names)
        if unknown:
            with ui.expansion(f"Unknown and strangers ({len(unknown)})", icon="visibility_off").classes("w-full mt-2"):
                ui.label("They are left out of the Library labels and filters. Restore anyone you know.").classes("text-sm text-gray-500")
                with ui.grid().classes("w-full gap-2 grid-cols-2 sm:grid-cols-4 md:grid-cols-6"):
                    for person in unknown:
                        with ui.card().tight().classes("w-full p-2 items-center"):
                            if person["faces"]:
                                ui.image(data_url(person["faces"][0])).classes("w-16 h-16 rounded")
                            ui.label(f"In {person['count']}").classes("text-xs text-gray-500")
                            ui.button("Restore", on_click=lambda pid=person["id"]: self.restore_person(pid)).props("flat dense size=sm")

    def no_people_yet(self) -> None:
        """Why the People tab is empty, and what to do about it."""
        st = self.trip.face_status()
        if st["error"]:
            ui.label(f"The face finder could not start: {st['error']}").classes("text-orange-700")
            ui.label("Check your internet connection (it downloads once, about 290 MB), then click Analyse again.") \
                .classes("text-sm text-gray-600")
        elif st["done"] < st["chosen"]:
            ui.label(f"{st['chosen'] - st['done']} of your {st['chosen']} chosen files haven't been checked for faces yet.") \
                .classes("text-gray-700")
            ui.button("Analyse the chosen files", icon="auto_awesome", on_click=self.analyse_chosen)
        elif st["ungrouped"]:
            ui.label(f"{st['faces']} faces were found but not sorted into people yet.").classes("text-gray-700")
            ui.button("Sort them into people", icon="people", on_click=self.analyse_chosen)
        else:
            ui.label(f"No faces were found in your {st['chosen']} chosen files.").classes("text-gray-700")
        if st["done"]:
            ui.button("Look for faces again", icon="refresh", on_click=self.look_for_faces_again).props("outline")

    async def look_for_faces_again(self) -> None:
        if self.trip.progress.running:
            ui.notify("Still working, please wait.", type="warning")
            return
        self.trip.look_for_faces_again()
        await self.analyse_chosen()
        self.panels.value = self.people_tab

    def person_grid(self, people: list[dict], names: list[str]) -> None:
        with ui.grid().classes("w-full gap-3 grid-cols-1 sm:grid-cols-2 md:grid-cols-3"):
            for person in people:
                with ui.card().classes("w-full"):
                    with ui.row().classes("gap-1"):
                        for crop in person["faces"]:
                            ui.image(data_url(crop)).classes("w-14 h-14 rounded")
                    ui.label(f"In {person['count']} photos and videos").classes("text-xs text-gray-500")
                    with ui.row().classes("w-full items-center no-wrap"):
                        ui.select(
                            names,
                            value=person["name"] or None,
                            label="Who is this?",
                            with_input=True,
                            new_value_mode="add-unique",
                            on_change=lambda e, pid=person["id"], old=person["name"]: self.rename(pid, e.value, old),
                        ).props("input-debounce=0").classes("flex-grow")
                        ui.button("Unknown", icon="visibility_off", on_click=lambda pid=person["id"]: self.mark_unknown([pid])).props("flat dense")

    def rename(self, person_id: int, name: str | None, old: str) -> None:
        if (name or "").strip() == (old or ""):
            return
        self.trip.name_person(person_id, name or "")
        ui.notify(f"Saved: {name}" if (name or "").strip() else "Name removed")
        self.refresh_all()

    def mark_unknown(self, person_ids: list[int]) -> None:
        self.trip.mark_unknown(person_ids)
        self.refresh_all()

    def restore_person(self, person_id: int) -> None:
        self.trip.restore_person(person_id)
        self.refresh_all()

    # ---------- Storyboard ----------

    def settings_now(self) -> RenderSettings:
        width, height = RESOLUTIONS[self.resolution.value]
        return RenderSettings(
            width=width,
            height=height,
            photo_seconds=self.photo_len.value,
            max_clip_seconds=self.clip_len.value,
            transition=self.transition.value,
            transition_seconds=self.t_len.value,
            music_volume=self.music_vol.value,
            look=self.trip.look() if self.trip else "natural",
        )

    # ---------- Music: tap your own cuts ----------

    @ui.refreshable
    def music_tab_view(self) -> None:
        music = self.music_path()
        with ui.row().classes("w-full items-center gap-3"):
            ui.label("Song:").classes("font-medium")
            ui.label(Path(music).name if music else "none chosen yet").classes("text-gray-700")
            if sys.platform == "darwin":
                ui.button("Choose song…", icon="music_note", on_click=self.browse_music).props("outline dense")
            else:
                ui.input("Song file", on_change=lambda e: setattr(self.music_input, "value", e.value)).classes("w-96")
        with ui.row().classes("w-full items-center gap-2"):
            ui.label("Get a song:").classes("text-sm text-gray-600")
            for name, url in MUSIC_SITES:
                ui.button(name, icon="open_in_new", on_click=lambda n=name, u=url: self.music_site(n, u)) \
                    .props("outline dense no-caps size=sm color=purple") \
                    .tooltip("A song you buy there is yours to use in your own videos, but not on YouTube"
                             if url.startswith("itms") else
                             "Songs you may use on YouTube. Download one there: the app picks it up from Downloads")
        if not music:
            ui.label("Choose a song first. Then play it and tap the right Shift key wherever you want a cut.") \
                .classes("text-gray-500")
            return
        can_trim = self.trim_card(music)
        saved = song_taps.load(music, app_dir() / "music-taps")
        ui.label("Trim, fade and pin the song").classes("font-semibold mt-2")
        if can_trim:
            ui.label(
                "Trim: drag the two green handles at the ends of the line to cut off what you don't want (a long intro, "
                "the ending); the grey parts are cut off. Fade: drag the small purple dot at the top of each handle "
                "inwards; the purple slope shows the music getting louder at the start and quieter at the end. "
                "When you let go of a handle or a dot you hear it straight away: the start plays from there, the end "
                "plays the last seconds up to it. Play skips the grey parts and plays the fades, so you hear exactly "
                "the track you will get."
            ).classes("text-sm text-gray-600")
        ui.label(
            "Pins: press Play and tap the RIGHT Shift key wherever you want the video to cut. Each tap puts a pin on the "
            "line. Every pin makes a click as it passes. Scroll on the line to zoom in (the spikes are the hits), drag a "
            "pin to move it, or click it and nudge it with the arrow keys; Delete removes it. Pins in the grey parts "
            "are dropped when you trim. Your pins are saved as you go, and Auto-edit always cuts on them."
        ).classes("text-sm text-gray-600")
        ui.html(f'<audio id="tm-song" preload="auto" src="{serve_song(music)}" onloadedmetadata="tmTapDraw()" '
                f'onplay="tmTapOnPlay()" onpause="tmTapUnschedule(); tmTapDraw()"></audio>', sanitize=False)
        with ui.row().classes("w-full items-center gap-2"):
            ui.button("Play / pause", icon="play_arrow", on_click=lambda: ui.run_javascript("tmTapPlay()"))
            ui.button("5 s back", icon="replay_5", on_click=lambda: ui.run_javascript("tmTapSeek(-5)")).props("outline")
            ui.button("From the start", icon="skip_previous", on_click=lambda: ui.run_javascript("tmTapStart()")).props("outline")
            with ui.button_group().props("outline"):
                for span, label in ((0, "Whole song"), (30, "30 s"), (10, "10 s"), (4, "4 s")):
                    ui.button(label, on_click=lambda span=span: ui.run_javascript(f"tmTapZoom({span})")) \
                        .props("outline no-caps").tooltip("How much of the song the line shows (or scroll on it)")
            ui.button("Delete selected pin", icon="delete", on_click=lambda: ui.run_javascript("tmTapDelete()")).props("flat")
            ui.button("Clear all", icon="clear_all", on_click=self.clear_taps).props("flat")
            ui.button("Finish my pins", icon="auto_fix_high", on_click=self.finish_pins).props("outline no-caps") \
                .tooltip("Pin the start of the song by hand; the app suggests the rest the same way")
            ui.button("Half the pins", icon="exposure_neg_1", on_click=self.halve_pins).props("outline no-caps") \
                .tooltip("Take away every second pin, for longer shots. The first and last pins stay, and so do "
                         "pins where the music changes. Press again to halve again")
            ui.button("Double the pins", icon="exposure_plus_1", on_click=self.double_pins).props("outline no-caps") \
                .tooltip("A new pin between every two pins, on the nearest beat, for shorter shots. Press again to "
                         "double again")
        ui.html('<div id="tm-tl" style="position:relative;width:100%;height:150px;background:#f1f5f9;border-radius:8px;'
                'touch-action:none;user-select:none;cursor:pointer;overflow:hidden"></div>', sanitize=False).classes("w-full")
        ui.html('<div id="tm-trim-info" style="font-size:14px;font-weight:600;color:#15803d"></div>', sanitize=False)
        ui.html('<div id="tm-tap-info" style="font-size:13px;color:#475569"></div>', sanitize=False)
        self.suggesting = False  # the line starts afresh with only your own pins
        self.finish_row()
        self.pins_before_half = None
        self.half_row()
        if can_trim:
            with ui.row().classes("items-center gap-2"):
                ui.button("Hear the start", icon="play_arrow",
                          on_click=lambda: ui.run_javascript("tmTrimHear(0)")).props("outline dense no-caps")
                ui.button("Hear the end", icon="play_arrow",
                          on_click=lambda: ui.run_javascript("tmTrimHear(1)")).props("outline dense no-caps")
                with ui.button_group().props("outline"):
                    for secs, label in ((0, "No fades"), (3, "Fades 3 s"), (6, "6 s"), (9, "9 s")):
                        ui.button(label, on_click=lambda secs=secs: ui.run_javascript(f"tmFadeSet({secs}, {secs})")) \
                            .props("outline no-caps dense").tooltip("Set both fades at once (or drag the purple dots)")
                # Your own fades, any number of seconds, the start and the end each their own.
                fade_in = ui.number("Fade in (s)", min=0, max=60, step=0.5, format="%.1f").props("dense outlined").classes("w-28")
                fade_out = ui.number("Fade out (s)", min=0, max=60, step=0.5, format="%.1f").props("dense outlined").classes("w-28")
                ui.button("Set fades", on_click=lambda: ui.run_javascript(
                    f"tmFadeSet({float(fade_in.value or 0)}, {float(fade_out.value or 0)})")).props("outline dense no-caps") \
                    .tooltip("Longer than half the kept part is shortened to fit")
                ui.switch("Play skips the cut-off parts", value=True,
                          on_change=lambda e: ui.run_javascript(f"tm.tap.skip = {json.dumps(bool(e.value))}; tmTapDraw()"))
                ui.button("Use this part as the track", icon="content_cut", on_click=lambda: self.trim_song(music))
        ui.button("Snap to the music", icon="center_focus_strong", on_click=self.snap_taps).props("outline")
        self.saved_pins = saved["taps"]
        self.pin_summary(music, saved["taps"])
        self.video_length_line(music)
        self.music_reading(music)
        ui.timer(0.2, lambda: ui.run_javascript(f"tmTapInit({json.dumps(saved['taps'])}, 0, {json.dumps(can_trim)}); "
                                                f"tm.tap.song = {json.dumps(music)}"), once=True)
        ui.timer(2.0, self.autosave_taps)  # pins are kept as you place them: no save button to forget

    @ui.refreshable
    def music_reading(self, music: str) -> None:
        """The song's calm, medium and lively stretches (lined up with your pins), what fits in each, and its style.
        Change any of it; Auto-edit and the AI follow it."""
        ui.label("Read the music").classes("font-semibold mt-3")
        song = self.song_maps.get(music)
        if song is None:
            with ui.row().classes("items-center"):
                ui.spinner()
                ui.label("Listening to the song…").classes("text-sm text-gray-600")

            async def study() -> None:
                try:
                    self.song_maps[music] = await run.io_bound(songmap.analyse, music, app_dir() / "music-maps")
                except Exception as exc:
                    traceback.print_exc()
                    ui.notify(f"Could not study the song: {exc}", type="negative")
                    return
                self.music_reading.refresh(music)

            ui.timer(0.1, study, once=True)
            return
        folder = app_dir() / "music-taps"
        reading = musicmood.reading(song, music, folder)
        stretches = reading["stretches"]
        ui.timer(0.3, lambda: ui.run_javascript(
            "window.tmMoodsSet && tmMoodsSet(" + json.dumps([[b["start"], b["end"], musicmood.COLOURS[b["mood"]]]
                                                             for b in stretches]) + ")"), once=True)
        ui.label("The coloured strip on the line shows how the music feels: light blue calm, blue medium, orange lively, "
                 "each change moved onto your nearest pin. Auto-edit puts the fitting shots in each stretch. Change any "
                 "stretch you hear differently.").classes("text-sm text-gray-600")
        with ui.row().classes("w-full gap-2"):
            for n, b in enumerate(stretches):
                with ui.card().classes("p-2 gap-1").style(f"border-left: 6px solid {musicmood.COLOURS[b['mood']]}"):
                    ui.label(f"{fmt_length(b['start'])}–{fmt_length(b['end'])}").classes("text-xs text-gray-500")
                    ui.select(list(musicmood.MOODS), value=b["mood"],
                              on_change=lambda e, n=n: self.set_mood(music, n, e.value)).props("dense").classes("w-28")
                    ui.label(musicmood.FITS[b["mood"]]).classes("text-xs text-gray-600 w-28")
        if reading["mine"]:
            ui.button("Back to what the app heard", icon="restart_alt",
                      on_click=lambda: self.reset_moods(music)).props("flat dense no-caps")
        styles = {k: v["label"] + (" (suggested)" if k == reading["suggested"] else "") for k, v in musicmood.STYLES.items()}
        with ui.row().classes("items-center gap-3 mt-1"):
            ui.select(styles, value=reading["style"], label="The song's style",
                      on_change=lambda e: self.set_song_style(music, e.value)).classes("w-56")
            ui.label(musicmood.STYLES[reading["style"]]["what"] + ".").classes("text-sm text-gray-600")

    def set_mood(self, music: str, n: int, mood: str) -> None:
        song = self.song_maps.get(music)
        if not song:
            return
        folder = app_dir() / "music-taps"
        stretches = [dict(b) for b in musicmood.reading(song, music, folder)["stretches"]]
        stretches[n]["mood"] = mood
        stretches[n]["energy"] = musicmood.MOODS[mood]
        song_taps.set_extra(music, folder, "moods", stretches)
        self.music_reading.refresh(music)

    def reset_moods(self, music: str) -> None:
        song_taps.set_extra(music, app_dir() / "music-taps", "moods", None)
        self.music_reading.refresh(music)

    def set_song_style(self, music: str, style: str) -> None:
        song_taps.set_extra(music, app_dir() / "music-taps", "style", style)
        ui.notify(f"Style: {musicmood.STYLES[style]['label']}. Auto-edit follows it at the next cut.", type="positive")
        self.music_reading.refresh(music)

    def music_words(self) -> str:
        """The song's style and stretches, for the AI's versions."""
        music = self.music_path()
        song = self.song_maps.get(music) if music else None
        if not song:
            return ""
        reading = musicmood.reading(song, music, app_dir() / "music-taps")
        return (f"The song's style: {musicmood.STYLES[reading['style']]['label']}. "
                f"How the music feels: {musicmood.words(reading['stretches'])}.")

    def trim_card(self, music: str) -> bool:
        """Whether the song can still be trimmed (on the pin line); for a track already cut out of a song, says so."""
        source = song_taps.trimmed_from(music, app_dir() / "music-taps")
        if not source:
            return True
        whole, start, end = source
        with ui.row().classes("items-center gap-3"):
            fades = song_taps.fades_of(music, app_dir() / "music-taps")
            faded = (f"fading in over {fades[0]:.1f} s and out over {fades[1]:.1f} s" if fades
                     else "with a short fade in and out")
            ui.label(f"✓ The track is {fmt_length(start)} to {fmt_length(end)} of {Path(whole).name}, {faded}.") \
                .classes("text-sm text-green-700")
            ui.button("Trim again from the whole song", icon="content_cut",
                      on_click=lambda: self.music_input.set_value(whole)).props("flat dense no-caps")
        return False

    async def trim_song(self, music: str) -> None:
        try:
            part = await ui.run_javascript("tmTrimGet()", timeout=5)
            whole = song_length(music)
        except Exception:
            ui.notify("The song is still loading: try again in a moment.", type="warning")
            return
        if not part:
            return
        start, end, fade_in, fade_out = (float(v) for v in part)
        if start < 0.05 and end > whole - 0.05 and fade_in < 0.05 and fade_out < 0.05:
            ui.notify("Drag a green handle or a purple fade dot first: the whole song is still chosen, unchanged.",
                      type="warning")
            return
        await self.autosave_taps()  # the pins you just placed come along into the track
        try:
            track = await run.io_bound(song_taps.trim, music, start, end, app_dir() / "music-trimmed",
                                       app_dir() / "music-taps", fade_in, fade_out)
        except Exception as exc:
            traceback.print_exc()
            ui.notify(f"Could not trim the song: {exc}", type="negative")
            return
        self.music_input.set_value(track)
        ui.notify(f"The track is now {fmt_length(start)} to {fmt_length(end)} of the song, fading in over "
                  f"{fade_in:.1f} s and out over {fade_out:.1f} s. Pin it below.", type="positive")

    @ui.refreshable
    def pin_summary(self, music: str, taps: list[float]) -> None:
        used = song_taps.extra(music, app_dir() / "music-taps", "used") if taps else None
        mine = sorted(round(p, 3) for p in taps or [])
        if used and sorted(round(p, 3) for p in used["pins"]) != mine:
            used = None  # you changed your pins since that cut
        kept = {round(p, 3) for p in used["kept"]} if used else set()
        skipped = [p for p in mine if p not in kept] if used else []
        ui.run_javascript(f"tm.tap.unused = new Set({json.dumps([f'{p:.3f}' for p in skipped])}); tmTapDraw()")
        if taps:
            ui.label(f"✓ {len(taps)} pins saved: Auto-edit (in the Storyboard) cuts exactly there.") \
                .classes("text-sm text-green-700")
            if skipped:
                with ui.row().classes("items-center gap-2 p-2 rounded bg-slate-100"):
                    ui.label(f"The last cut used {len(mine) - len(skipped)} of your {len(mine)} pins: there weren't "
                             "enough shots for all of them, so it skipped the grey ones and kept your pattern. Add more "
                             "clips or photos and the next cut uses more.").classes("text-sm text-slate-700 flex-1")
                    ui.button("Make these my pins", icon="check",
                              on_click=lambda: self.use_kept_pins(music, sorted(kept))).props("outline dense no-caps") \
                        .tooltip("Delete the grey pins for good")
            self.tap_lengths(music, taps)
        else:
            ui.label("No pins yet: Auto-edit finds the beats by itself.").classes("text-sm text-gray-500")

    def use_kept_pins(self, music: str, kept: list[float]) -> None:
        song_taps.save(music, app_dir() / "music-taps", kept, True)
        song_taps.set_extra(music, app_dir() / "music-taps", "used", None)
        self.saved_pins = kept
        ui.run_javascript(f"tmTapSet({json.dumps(kept)})")
        ui.notify(f"Your pins are now the {len(kept)} the last cut used.", type="positive")
        self.pin_summary.refresh(music, kept)  # last: this removes the button you pressed
        self.music_reading.refresh(music)

    async def autosave_taps(self) -> None:
        music = self.music_path()
        if not music or not self.showing("Music & audio"):
            return
        try:
            # Only the pins of this song (just after you change songs, the line may still show the last one's).
            pins = await ui.run_javascript(f"(window.tm && tm.tap && tm.tap.song === {json.dumps(music)}) "
                                           "? tmTapGet() : null", timeout=3)
        except Exception:
            return  # the page is busy or reloading: next time
        if pins is None:
            return
        pins = sorted(round(float(p), 3) for p in pins or [])
        if pins == [round(p, 3) for p in self.saved_pins]:
            return
        song_taps.save(music, app_dir() / "music-taps", pins, True)
        self.saved_pins = pins
        self.pin_summary.refresh(music, pins)
        self.music_reading.refresh(music)  # the stretches line up with the pins

    def tap_lengths(self, music: str, taps: list[float]) -> None:
        """The shots your pins make, one box each."""
        song = self.song_maps.get(music)
        try:
            duration = song.duration if song else song_length(music)
        except Exception:
            duration = max(taps) + 1
        lengths = song_taps.shot_lengths(taps, max(taps))  # up to your last pin; after it the app cuts by itself
        with ui.row().classes("w-full gap-1"):
            for n, length in enumerate(lengths, 1):
                ui.chip(f"{n}: {length:.1f} s", color="blue-grey", text_color="white").props("dense square") \
                    .tooltip(f"Shot {n}: from pin {n - 1 if n > 1 else 'start'} to pin {n}")
        rest = duration - max(taps)
        notes = ["Each grey box is one shot, in order: the time from one pin to the next. Your pins are the only cuts, "
                 "however long or short.",
                 f"After your last pin ({fmt_length(max(taps))}) the app cuts the remaining {fmt_length(rest)} by itself."
                 if rest > 1 else ""]
        for note in filter(None, notes):
            ui.label(note).classes("text-xs text-gray-600")

    async def snap_taps(self) -> None:
        music = self.music_path()
        pins = await ui.run_javascript("tmTapGet()", timeout=5)
        if not music or not pins:
            ui.notify("Place some pins first: play the song and tap the right Shift key.")
            return
        snapped, delay = await run.io_bound(song_taps.snap_song, music, [float(p) for p in pins])
        ui.run_javascript(f"tmTapSet({json.dumps(snapped)})")
        habit = (f" You usually tap about {1000 * abs(delay):.0f} ms {'late' if delay > 0 else 'early'}; "
                 "that is taken off every pin." if abs(delay) >= 0.02 else "")
        ui.notify(f"Snapped {len(snapped)} pins onto the beats and hits.{habit} Play it back to check.",
                  type="positive", multi_line=True, timeout=8000)

    async def clear_taps(self) -> None:
        ui.run_javascript("tmTapClear()")
        self.suggesting = False
        self.finish_row.refresh()

    async def halve_pins(self) -> None:
        music = self.music_path()
        if not music:
            return
        pins = [float(p) for p in await ui.run_javascript("tmTapGet()", timeout=5) or []]
        if len(pins) < 3:
            ui.notify("There are too few pins to halve.", type="info")
            return
        song = self.song_maps.get(music)
        if song is None:
            song = self.song_maps[music] = await run.io_bound(songmap.analyse, music, app_dir() / "music-maps")
        halved = song_taps.halve(pins, [s.start for s in song.sections[1:]])
        if len(halved) == len(pins):
            ui.notify("Every pin sits on a change in the music, so none were taken away.", type="info")
            return
        if self.pins_before_half is None:  # Undo goes back to before the first halving
            self.pins_before_half = pins
        self.pins_halved = len(halved)
        ui.run_javascript(f"tmTapSet({json.dumps(halved)})")  # saved with your pins within 2 seconds
        self.half_row.refresh()

    async def double_pins(self) -> None:
        music = self.music_path()
        if not music:
            return
        pins = [float(p) for p in await ui.run_javascript("tmTapGet()", timeout=5) or []]
        if len(pins) < 2:
            ui.notify("Pin at least two cuts first.", type="info")
            return
        song = self.song_maps.get(music)
        if song is None:
            song = self.song_maps[music] = await run.io_bound(songmap.analyse, music, app_dir() / "music-maps")
        doubled = song_taps.double(pins, song.beat_times() if song.clear_beat else [])
        if len(doubled) == len(pins):
            ui.notify("The pins are too close together to add more between them.", type="info")
            return
        if self.pins_before_half is None:  # Undo goes back to before the first change
            self.pins_before_half = pins
        self.pins_halved = len(doubled)
        ui.run_javascript(f"tmTapSet({json.dumps(doubled)})")  # saved with your pins within 2 seconds
        self.half_row.refresh()

    @ui.refreshable
    def half_row(self) -> None:
        before = getattr(self, "pins_before_half", None)
        if not before:
            return
        with ui.row().classes("items-center gap-2 p-2 rounded bg-sky-50"):
            more = self.pins_halved > len(before)
            ui.label(f"{len(before)} pins → {self.pins_halved}, so shots are "
                     f"{'shorter' if more else 'longer'}. Undo brings back your {len(before)} pins.") \
                .classes("text-sm text-sky-900")
            ui.button("Undo", icon="undo", on_click=self.unhalve_pins).props("flat dense no-caps")

    def unhalve_pins(self) -> None:
        ui.run_javascript(f"tmTapSet({json.dumps(self.pins_before_half)})")
        ui.notify(f"Your {len(self.pins_before_half)} pins are back.", type="positive")
        self.pins_before_half = None
        self.half_row.refresh()  # last: this removes the button you pressed

    async def finish_pins(self) -> None:
        """Learn how you pinned the start of the song, and suggest pins for the rest in the same way."""
        music = self.music_path()
        if not music:
            return
        got = await ui.run_javascript("[tmTapGet(), tm.tap.trim ? tmTrimPart()[1] : null]", timeout=5)
        pins, until = [float(p) for p in got[0] or []], got[1]
        if len(pins) < pinfill.LEAST:
            ui.notify(f"Pin at least {pinfill.LEAST} cuts by hand first (a minute or so of the song works best).",
                      type="warning")
            return
        song = self.song_maps.get(music)
        if song is None:
            song = self.song_maps[music] = await run.io_bound(songmap.analyse, music, app_dir() / "music-maps")
        stretches = musicmood.reading(song, music, app_dir() / "music-taps")["stretches"]
        suggested = await run.io_bound(pinfill.suggest, song, pins, stretches, until)
        if not suggested:
            ui.notify("Your pins already reach the end of the song.", type="info")
            return
        ui.run_javascript(f"tmTapSuggest({json.dumps(suggested)})")
        self.suggesting = True
        self.finish_row.refresh()
        ui.notify(f"{len(suggested)} pins suggested in orange, the way you pinned the start. Play it to listen; "
                  "drag or delete any you don't like, then Keep these pins.", type="positive", multi_line=True, timeout=9000)

    @ui.refreshable
    def finish_row(self) -> None:
        if not getattr(self, "suggesting", False):
            return
        with ui.row().classes("items-center gap-2 p-2 rounded bg-amber-50"):
            ui.label("🟠 Suggested pins (not saved yet):").classes("text-sm text-amber-900")
            ui.button("Keep these pins", icon="check", on_click=self.keep_suggested).props("color=green dense no-caps")
            ui.button("Take them away", icon="close", on_click=self.drop_suggested).props("flat dense no-caps")

    async def keep_suggested(self) -> None:
        ui.run_javascript("tmTapKeepAi()")  # they are yours now: saved with your pins within 2 seconds
        ui.notify("Kept. They are now ordinary pins: drag or delete any of them as before.", type="positive")
        self.suggesting = False
        self.finish_row.refresh()  # last: this removes the button you pressed

    def drop_suggested(self) -> None:
        ui.run_javascript("tmTapDropAi()")
        self.suggesting = False
        self.finish_row.refresh()  # last: this removes the button you pressed

    # ---------- Stories ----------

    @ui.refreshable
    def director_card(self) -> None:
        if not self.trip or not self.showing("Stories"):
            self.ai_bar = self.ai_stop_here = None
            return
        ai = load_settings().get("ai", {})
        provider = ai.get("provider", "ollama")
        chosen = [it for it in self.items if it["thumb"]]
        described = sum(1 for it in chosen if it.get("ai_caption"))
        with ui.card().classes("w-full bg-indigo-50"):
            with ui.row().classes("w-full items-center gap-3"):
                ui.label("🎬 AI director").classes("text-lg font-semibold")
                ui.label(f"{described} of {len(chosen)} chosen files described").classes("text-sm text-gray-600")
            ui.label(
                "The AI looks at your clips, writes what each one shows and how much it stands out (Auto-edit then "
                "prefers the best moments), and suggests a few ways to tell the trip."
            ).classes("text-sm text-gray-600")
            with ui.expansion("Which AI", icon="settings", value=not self.director_ready(ai)).classes("w-full"):
                ui.select(director.PROVIDERS, value=provider, label="Run it",
                          on_change=lambda e: self.save_ai(provider=e.value)).classes("w-72")
                if provider == "ollama":
                    self.ollama_settings(ai)
                elif provider == "claude_code":
                    self.claude_code_settings(ai)
                else:
                    self.claude_settings(ai)
            # The AI's progress, right here (the main progress bar is at the top of the page, often out of sight).
            self.ai_bar = ui.linear_progress(value=0, show_value=False).classes("w-full")
            self.ai_label = ui.label("").classes("text-sm text-indigo-800")
            self.ai_bar.set_visibility(self.ai_busy)
            self.ai_stop_here = ui.button("Stop (keep what the AI has done)", icon="stop_circle", on_click=self.stop_ai) \
                .props("outline color=negative dense no-caps")
            self.ai_stop_here.set_visibility(self.ai_busy)
            with ui.row().classes("items-center gap-2"):
                left = len(chosen) - described
                looks = len({it["similar_set"] if it["similar_set"] is not None else it["path"]
                             for it in chosen if not it["ai_caption"]})  # doubles share their set's description
                ui.button(f"Describe clips ({looks} to look at)" if left else "Describe clips", icon="visibility",
                          on_click=lambda: self.describe_clips(again=False)).props("" if left else "outline")
                if described:
                    ui.button("Describe all again", on_click=lambda: self.describe_clips(again=True)).props("flat dense no-caps")
                ui.button("Plan the stories", icon="movie_filter", on_click=lambda: self.plan_stories(None)) \
                    .tooltip("For every story: a theme, and its shots in order as opening, montage, highlight, "
                             "slow moment and ending. Auto-edit then cuts each story that way.")
            ui.label("Which stories go in the video, and in what order, is chosen in the Storyboard: tell the app the "
                     "video you want there.").classes("text-xs text-gray-500")

    def director_ready(self, ai: dict) -> bool:
        if ai.get("provider") == "claude":
            return bool(ai.get("api_key"))
        if ai.get("provider") == "claude_code":
            return bool(director.claude_code_path())
        return bool(ai.get("ollama_model"))

    def ollama_settings(self, ai: dict) -> None:
        if not self.ollama_checked:
            self.ollama_checked = True
            ui.timer(0.1, self.check_ollama, once=True)
        found = self.ollama_found
        if found is None:
            ui.markdown(
                "Free, and your pictures stay on this Mac. **One-time setup:** download Ollama from "
                "[ollama.com](https://ollama.com/download) and open it. Then in Terminal run "
                f"`ollama pull {director.OLLAMA_SUGGESTED}` (about 3 GB), and click **Check again**."
            ).classes("text-sm")
        elif not found[0]:
            ui.markdown(
                "Ollama is running, but none of its models can look at pictures. In Terminal run "
                f"`ollama pull {director.OLLAMA_SUGGESTED}`, then click **Check again**."
            ).classes("text-sm")
        else:
            model = ai.get("ollama_model") if ai.get("ollama_model") in found[0] else found[0][0]
            if model != ai.get("ollama_model"):
                self.save_ai(refresh=False, ollama_model=model)
            ui.select(found[0], value=model, label="Model",
                      on_change=lambda e: self.save_ai(ollama_model=e.value)).classes("w-72")
            ui.label("Ollama is ready. Bigger models describe better but are slower.").classes("text-xs text-green-700")
            if director.OLLAMA_FASTEST not in " ".join(found[0]):
                ui.markdown(f"Too slow? A smaller model is about 3× faster (simpler descriptions): in Terminal run "
                            f"`ollama pull {director.OLLAMA_FASTEST}`, click **Check again** and pick it above.") \
                    .classes("text-xs")
        ui.button("Check again", icon="refresh", on_click=self.check_ollama).props("flat dense no-caps")

    def claude_code_settings(self, ai: dict) -> None:
        found = director.claude_code_path()
        ui.markdown(
            "Uses your own Claude account (Pro) through the Claude Code app on this Mac: the strongest model writes "
            "your stories, with no extra cost. Only the words go to Claude (theme names, times and your idea), "
            "not your pictures."
        ).classes("text-sm")
        if found:
            ui.label(f"Claude Code found ({found}).").classes("text-xs text-green-700")
        else:
            ui.markdown(
                "**One-time setup.** In Terminal run:\n\n"
                f"`{director.CLAUDE_CODE_INSTALL}`\n\n"
                "then type `claude`, press Enter and sign in with your Claude account (then close it with Ctrl+C). "
                "Click **Check again** when done."
            ).classes("text-sm")
        with ui.row().classes("items-center gap-2"):
            ui.button("Check again", icon="refresh", on_click=self.director_card.refresh).props("flat dense no-caps")
            if found:
                ui.button("Test it", icon="play_arrow", on_click=self.test_claude_code).props("flat dense no-caps")
        ui.label("Describing pictures stays free on this Mac (Ollama):").classes("text-xs text-gray-600 mt-2")
        self.ollama_settings(ai)

    async def test_claude_code(self) -> None:
        note = ui.notification("Asking Claude…", spinner=True, timeout=None)
        try:
            ok = await run.io_bound(director.test_claude_code)
            note.message, note.type, note.spinner = f"Claude answered: {ok}", "positive", False
        except director.DirectorError as exc:
            note.message, note.type, note.spinner, note.multi_line = str(exc), "warning", False, True
        note.timeout = 8

    async def check_ollama(self) -> None:
        self.ollama_found = await run.io_bound(director.ollama_models)
        self.director_card.refresh()

    def claude_settings(self, ai: dict) -> None:
        ui.markdown(
            "Very good descriptions and storylines, and much faster than on this Mac. Paid per use to Anthropic: "
            "describing a trip costs a few cents with the fast model; thinking about the stories (always the best "
            "model) a few cents more. **One-time setup:** at [console.anthropic.com](https://console.anthropic.com) add some "
            "credit, create an API key, and paste it here. It is kept only on this Mac."
        ).classes("text-sm")
        key = ai.get("api_key", "")
        box = ui.input("API key", password=True, password_toggle_button=True,
                       placeholder="sk-ant-…" if not key else "saved: …" + key[-4:]).classes("w-96")
        with ui.row().classes("items-center gap-2"):
            ui.button("Save key", on_click=lambda: (self.save_ai(api_key=box.value.strip()) if box.value.strip() else None)) \
                .props("dense")
            if key:
                ui.button("Forget key", on_click=lambda: self.save_ai(api_key="")).props("flat dense no-caps")
        current = ai.get("claude_model") or director.CLAUDE_MODEL
        ui.select({**director.CLAUDE_MODELS, **({} if current in director.CLAUDE_MODELS else {current: current})},
                  value=current, label="Describing pictures with",
                  on_change=lambda e: self.save_ai(refresh=False, claude_model=e.value)).classes("w-96")

    def save_ai(self, refresh: bool = True, **values) -> None:
        save_settings(ai={**load_settings().get("ai", {}), **values})
        if "api_key" in values:
            ui.notify("Key saved" if values["api_key"] else "Key removed")
        if refresh:
            self.director_card.refresh()

    def make_director(self):
        try:
            return director.make(load_settings().get("ai", {}))
        except director.DirectorError as exc:
            ui.notify(str(exc), type="warning", multi_line=True, timeout=10000)
            return None

    def busy(self) -> bool:
        if self.trip and (self.trip.progress.running or self.ai_busy):
            ui.notify("Still working on the last step; its progress shows above. Please wait for it to finish.",
                      type="warning")
            return True
        return not self.trip

    async def describe_clips(self, again: bool) -> None:
        if self.busy():
            return
        ai = self.make_director()
        if not ai:
            return
        before = sum(1 for it in self.items if it.get("ai_caption"))
        self.ai_busy = True
        self.ai_bar.set_visibility(True)
        self.ai_label.text = "Starting the AI (the first time it loads the model, which can take a minute)…"
        try:
            await self.run_step(lambda notify: self.trip.describe_clips(ai, notify, again))
        finally:
            self.ai_busy = False
        now = sum(1 for it in self.items if it.get("ai_caption"))
        if self.trip.progress.stop:
            return  # Stop says what was kept
        if now > before or again:
            ui.notify(f"The AI described {now if again else now - before} files.", type="positive")

    async def plan_stories(self, keys: list[str] | None) -> None:
        if self.busy():
            return
        ai = self.make_director()
        if not ai:
            return
        self.ai_busy = True
        self.ai_bar.set_visibility(True)
        self.ai_label.text = "The AI is reading the stories…"
        try:
            await self.run_step(lambda notify: self.trip.plan_stories(ai, keys, notify))
        finally:
            self.ai_busy = False
        if keys:
            self.open_plans.update(keys)
        if self.trip.progress.stop:
            return
        ui.notify("Plans ready: open a story's plan below to see or change it. Make a first cut in the Storyboard "
                  "to use them.", type="positive", multi_line=True)

    async def stoppable(self, fn, *args):
        """Run one AI request; Stop gives up waiting for it straight away (its answer is then thrown away).
        None when stopped."""
        self.ai_stopped = asyncio.Event()
        job = asyncio.ensure_future(run.io_bound(fn, *args))
        halt = asyncio.ensure_future(self.ai_stopped.wait())
        await asyncio.wait({job, halt}, return_when=asyncio.FIRST_COMPLETED)
        halt.cancel()
        if not job.done():
            job.add_done_callback(lambda j: j.exception())  # its late answer is ignored (catalog checks Stop)
            return None
        return job.result()

    def stop_ai(self) -> None:
        if not self.trip or not self.ai_busy:
            return
        self.trip.stop_ai()
        if getattr(self, "ai_stopped", None) is not None:
            self.ai_stopped.set()
        ui.notify("Stopping the AI. Everything it finished is kept; the rest is left as it was.", type="info")

    @ui.refreshable
    def stories_view(self) -> None:
        if not self.showing("Stories"):
            return
        if not self.trip:
            ui.label("Open a trip first.").classes("text-gray-500")
            return
        stories = [s for s in self.trip.stories() if any(it["chosen"] for it in s["items"])]
        if not stories:
            ui.label("No stories yet. Click Analyse to group your files into stories.").classes("text-gray-500")
            return
        with ui.row().classes("w-full items-center gap-4"):
            ui.label(f"{len(stories)} stories").classes("text-lg font-semibold")
            ui.button("Back to date order", icon="restart_alt", on_click=self.reset_story_order).props("flat dense")
        ui.label(
            "Each story is a group of shots of the same thing, at the same time and place. Give them names, put them in "
            "the order you want the video to tell them, and switch off the ones you don't want. Auto-edit and the "
            "Storyboard follow this order."
        ).classes("text-sm text-gray-600")
        if not any(it.get("subjects") for s in stories for it in s["items"]):
            ui.label("Tip: click Analyse again so the app can name stories by what they show (fish, deer, running…).") \
                .classes("text-sm text-amber-700")
        for n, story in enumerate(stories, 1):
            self.story_card(n, story, last=n == len(stories))

    def story_card(self, n: int, story: dict, last: bool) -> None:
        members = [it for it in story["items"] if it["chosen"]]
        included = sum(1 for it in members if it["included"])
        with ui.card().classes("w-full p-2" + ("" if story["enabled"] else " opacity-50")):
            with ui.row().classes("w-full items-center no-wrap gap-3"):
                with ui.column().classes("gap-0 items-center"):
                    ui.button(icon="arrow_upward", on_click=lambda k=story["key"]: self.move_story(k, -1)) \
                        .props("flat dense round size=sm" + (" disable" if n == 1 else ""))
                    ui.label(str(n)).classes("text-xs text-gray-500")
                    ui.button(icon="arrow_downward", on_click=lambda k=story["key"]: self.move_story(k, 1)) \
                        .props("flat dense round size=sm" + (" disable" if last else ""))
                with ui.row().classes("gap-1 no-wrap"):
                    for it in members[:4]:
                        if it["thumb"]:
                            self.preview(it, "width:90px;height:64px;flex:none", view=f"tale{n}")
                with ui.column().classes("flex-grow gap-1"):
                    name = ui.input("Story name", value=story["name"]).props("dense").classes("w-full")
                    name.on("blur", lambda e, k=story["key"], box=name: self.rename_story(k, box.value))
                    name.on("keydown.enter", lambda e, k=story["key"], box=name: self.rename_story(k, box.value))
                    places = [it["place"] or it["country"] for it in members if it["place"] or it["country"]]
                    place = max(set(places), key=places.count) if places else ""
                    begin, finish = (datetime.fromisoformat(story[k]).astimezone() for k in ("start", "end"))
                    start = fmt_time(story["start"])
                    end = finish.strftime("%H:%M" if finish.date() == begin.date() else "%a %d %b, %H:%M")
                    ui.label(f"{start}–{end}" + (f" · {place}" if place else "") +
                             f" · {len(members)} files, {included} in the video").classes("text-xs text-gray-600")
                    if story["renamed"]:
                        ui.label(f"The app's name: {story['auto_name']}").classes("text-xs text-gray-400")
                ui.switch("In the video", value=story["enabled"],
                          on_change=lambda e, k=story["key"]: self.set_story_enabled(k, e.value))
            self.story_plan(story, members)

    def story_plan(self, story: dict, members: list[dict]) -> None:
        """The story director's plan for a story: what each shot does, in the order Auto-edit will use."""
        key = story["key"]
        with ui.row().classes("w-full items-center gap-2 pl-10"):
            if not story["planned"]:
                ui.label("No plan yet.").classes("text-xs text-gray-500")
                ui.button("Plan this story", icon="movie_filter", on_click=lambda: self.plan_stories([key])) \
                    .props("flat dense size=sm")
                return
            is_open = key in self.open_plans
            ui.button(icon="expand_less" if is_open else "expand_more", on_click=lambda: self.toggle_plan(key)) \
                .props("flat dense round size=sm")
            ui.label(f"🎬 {story['theme'] or 'Plan'}").classes("text-sm font-semibold text-indigo-800")
            if story["idea"]:
                ui.label(story["idea"]).classes("text-xs text-gray-600")
            ui.space()
            ui.button("Plan again", on_click=lambda: self.plan_stories([key])).props("flat dense size=sm")
            ui.button("Forget plan", on_click=lambda: self.forget_plan(key)).props("flat dense size=sm")
        if key not in self.open_plans:
            return
        planned = sorted((it for it in members if it["plan_role"]), key=lambda it: it["plan_pos"])
        parts: list[tuple[str, list[dict]]] = []
        for it in planned:
            if parts and parts[-1][0] == it["plan_role"]:
                parts[-1][1].append(it)
            else:
                parts.append((it["plan_role"], [it]))
        with ui.row().classes("w-full gap-2 pl-10 items-start"):
            for role, shots in parts:
                with ui.card().tight().classes("p-1 bg-indigo-50 border border-indigo-200"):
                    ui.label(director.ROLES.get(role, role)).classes("text-xs font-semibold text-indigo-800 px-1")
                    if story["notes"].get(role):
                        ui.label(story["notes"][role]).classes("text-xs text-gray-600 px-1 max-w-80")
                    with ui.row().classes("gap-1 no-wrap"):
                        for it in shots:
                            self.plan_tile(it)
        left_out = [it for it in members if not it["plan_role"]]
        if left_out:
            with ui.row().classes("w-full items-center gap-2 pl-10"):
                ui.label(f"Left out of the plan ({len(left_out)}; Auto-edit uses them only if the song needs more):") \
                    .classes("text-xs text-gray-500")
            with ui.row().classes("w-full gap-1 pl-10"):
                for it in left_out[:24]:
                    self.plan_tile(it)

    def plan_tile(self, it: dict) -> None:
        with ui.column().classes("gap-0 items-center" + ("" if it["plan_role"] else " opacity-60")):
            self.preview(it, "width:96px;height:64px;flex:none", view="plan")
            with ui.row().classes("gap-0 no-wrap items-center"):
                if it["plan_role"]:
                    ui.button(icon="chevron_left", on_click=lambda p=it["path"]: self.move_in_plan(p, -1)) \
                        .props("flat dense round size=xs").tooltip("Earlier")
                with ui.button(icon="edit").props("flat dense round size=xs").tooltip("What this shot does"):
                    with ui.menu():
                        for role, label in director.ROLES.items():
                            ui.menu_item(label, on_click=lambda p=it["path"], r=role: self.set_plan_role(p, r))
                        if it["plan_role"]:
                            ui.menu_item("Leave out of the plan", on_click=lambda p=it["path"]: self.set_plan_role(p, None))
                if it["plan_role"]:
                    ui.button(icon="chevron_right", on_click=lambda p=it["path"]: self.move_in_plan(p, 1)) \
                        .props("flat dense round size=xs").tooltip("Later")
            if it.get("ai_caption"):
                ui.label(it["ai_caption"]).classes("text-[10px] leading-tight text-gray-500 w-24 line-clamp-2")

    def toggle_plan(self, key: str) -> None:
        self.open_plans ^= {key}
        self.stories_view.refresh()

    def set_plan_role(self, path: str, role: str | None) -> None:
        self.trip.set_plan_role(path, role)
        self.reload()
        self.stories_view.refresh()

    def move_in_plan(self, path: str, steps: int) -> None:
        self.trip.move_in_plan(path, steps)
        self.stories_view.refresh()

    def forget_plan(self, key: str) -> None:
        self.trip.forget_plan(key)
        self.reload()
        self.stories_view.refresh()

    def rename_story(self, key: str, name: str) -> None:
        current = next((s for s in self.trip.stories() if s["key"] == key), None)
        if current and (name or "").strip() != current["name"]:
            self.trip.rename_story(key, name)
            ui.notify("Story renamed")
            self.stories_view.refresh()

    def move_story(self, key: str, steps: int) -> None:
        self.trip.move_story(key, steps)
        self.stories_view.refresh()
        self.storyboard_view.refresh()

    def set_story_enabled(self, key: str, enabled: bool) -> None:
        self.trip.set_story_enabled(key, enabled)
        self.stories_view.refresh()
        self.storyboard_view.refresh()
        self.video_summary.refresh()

    def reset_story_order(self) -> None:
        self.trip.reset_story_order()
        self.stories_view.refresh()
        self.storyboard_view.refresh()

    @ui.refreshable
    def storyboard_view(self) -> None:
        if not self.showing("Storyboard"):
            return
        if not self.trip:
            ui.label("Open a trip first.").classes("text-gray-500")
            return
        self.storyline_card()
        board = self.trip.storyboard()
        if not board:
            ui.label("Nothing is in the video yet. Tick files in the Library, or let Auto-edit choose.").classes("text-gray-500")
            return
        if getattr(self, "guided", False) and not getattr(self, "board_list_open", False):
            with ui.row().classes("w-full items-center gap-3"):
                ui.label(f"{len(board)} clips in the first cut. You change them in the editor (next step: Watch & "
                         "finish).").classes("text-sm text-gray-700")
                ui.button("Show the clip list", on_click=lambda: (setattr(self, "board_list_open", True),
                                                                   self.storyboard_view.refresh())) \
                    .props("flat dense no-caps size=sm")
            return
        with ui.row().classes("w-full items-center gap-4"):
            ui.label(f"{len(board)} clips in this order").classes("text-lg font-semibold")
            ui.button("Back to time order", icon="restart_alt", on_click=self.reset_order).props("flat dense") \
                .tooltip("Forget the order you set here: the shots go back to the order they were taken")
        ui.label(
            "Drag a clip by its ⠿ handle up or down to change the order. For videos, drag the middle of the bar to use "
            "the same length from another moment, or its two ends to make it longer or shorter. "
            "The transition setting is for the cut after that clip."
        ).classes("text-sm text-gray-600")
        self.shown_titles = self.trip.shot_titles(board)
        self.board_order = [it["path"] for it in board]  # (a dropped clip is found by its number)
        for n, it in enumerate(board[: self.board_limit], 1):  # one long list you scroll through
            self.story_row(n, it, last=n == len(board))
        if len(board) > self.board_limit:
            ui.button(f"Show {min(STORY_ROWS, len(board) - self.board_limit)} more of {len(board) - self.board_limit} left",
                      icon="expand_more", on_click=self.more_board).props("outline").classes("w-full")

    def more_board(self) -> None:
        self.board_limit += STORY_ROWS
        self.storyboard_view.refresh()

    PACES = {  # what "calm", "medium" and "lively" mean for Auto-edit
        "calm": {"cut_on": "changes", "max_video": 6.0, "max_photo": 4.0, "min_shot": 1.5},
        "medium": {"cut_on": "strong", "max_video": 4.0, "max_photo": 2.5, "min_shot": 1.0},
        "lively": {"cut_on": "strong", "max_video": 2.5, "max_photo": 1.5, "min_shot": 0.5},
    }
    ORDER_WORDS = {
        themes.TIME: "The whole trip in the order it happened, day after day.",
        themes.THEME: "Your themes in the order you set; inside each one, the shots run in time order.",
        themes.DAY: "Day after day; inside each day, its themes in the order you set.",
        themes.STORY: "Describe (or say) the story you want: the AI arranges your themes into parts, each with its own "
                      "pace. Free with Ollama; without it, the app follows the order you name the themes in.",
    }
    PACE_NAMES = {"calm": "🌙 Calm", "medium": "🚶 Medium", "lively": "⚡ Lively"}

    def storyline_card(self) -> None:
        """The one card at the top of the Storyboard: how the video is told, and the button that cuts it."""
        line = self.trip.storyline()
        music = self.music_path()
        pins = song_taps.final_taps(music, app_dir() / "music-taps") if music else None
        with ui.card().classes("w-full bg-amber-50"):
            ui.label("Storyline").classes("text-lg font-semibold")
            with ui.row().classes("w-full items-center gap-2 text-sm text-gray-700"):
                ui.label(f"🎵 {Path(music).name}" if music else "🎵 No song yet").classes("font-medium")
                ui.label(f"· {self.length_words()}")
                ui.button("change", on_click=lambda: self.panels.set_value(self.music_tab)) \
                    .props("flat dense no-caps size=sm")
                ui.switch("Use a different length", value=bool(self.trip.own_length()),
                          on_change=lambda e: self.own_length_switched(e.value)).props("dense")
            if self.trip.own_length() is not None or self.own_length_open:
                with ui.row().classes("items-center gap-2"):
                    field = ui.input("Length (m:ss)", value=fmt_length(self.trip.own_length())
                                     if self.trip.own_length() else "", placeholder="e.g. 1:05") \
                        .props("dense outlined").classes("w-36")
                    field.on("keydown.enter", lambda: self.set_own_length(field.value))
                    ui.button("Set", on_click=lambda: self.set_own_length(field.value)).props("outline dense no-caps")
            ui.toggle(themes.ORDERS, value=line["order"], on_change=lambda e: self.set_order(e.value)) \
                .props("no-caps unelevated toggle-color=primary color=white text-color=primary") \
                .classes("border border-blue-600 rounded")
            ui.label(self.ORDER_WORDS[line["order"]]).classes("text-sm text-gray-600")
            if line["order"] == themes.STORY:
                self.my_story_view(line)
            else:
                if line["order"] == themes.TIME:
                    with ui.row().classes("items-center gap-4"):
                        ui.switch("A title for each day", value=line["day_titles"],
                                  on_change=lambda e: self.set_line(day_titles=e.value)).props("dense") \
                            .tooltip("\"Day 2 · Lisbon · Tue 15 Jul\" over the first shot of each day")
                        ui.switch("Keep each day's themes together", value=line["by_day"],
                                  on_change=lambda e: self.set_line(by_day=e.value)).props("dense") \
                            .tooltip("Inside each day: all its kitesurfing, then all its food... in the theme order "
                                     "below, instead of mixed as you filmed them")
                if line["order"] == themes.THEME or line["by_day"]:
                    if line["by_day"] and line["order"] == themes.TIME:
                        ui.label("The order of the themes inside each day:").classes("text-sm text-gray-600")
                    self.storyline_blocks_view(line)
            if line["order"] != themes.STORY:  # my story ends with its last part, nothing after it
                with ui.row().classes("items-center gap-3"):
                    ui.switch("Finale: end with the most exciting shots", value=line["finale"],
                              on_change=lambda e: self.set_finale(e.value))
            with ui.row().classes("w-full items-center gap-3"):
                ui.label("Moments per cut:").classes("text-sm")
                ui.toggle({"normal": "Normal", "more": "More", "max": "As many as fit"},
                          value=self.auto_limits.pack if self.auto_limits.pack in autoedit_pack() else "more",
                          on_change=lambda e: self.set_limit("pack", e.value)).props("dense no-caps") \
                    .tooltip("More: photos you have no room for become collages of 2–4 on one cut (a photo alone is "
                             "rare), and upright videos play side by side in a landscape video. As many as fit: up "
                             "to 4 photos and 3 videos on every cut. Normal: one shot a cut.")
            with ui.row().classes("w-full items-center gap-3"):
                ui.button("Make the first cut", icon="auto_fix_high", on_click=self.run_auto_edit) \
                    .tooltip("Replaces the storyboard's order, trims and transitions; Undo brings them back")
                ui.button("Update the video", icon="published_with_changes", on_click=self.update_video) \
                    .props("outline no-caps") \
                    .tooltip("Make the video again with your changes here, keeping this cut: only the clips you "
                             "changed are made again, the rest is reused")
                ui.label("The music decides how many shots there are and where the cuts go." if music else
                         "Without a song the cuts follow a steady pace.").classes("text-sm text-gray-600 flex-1")
                undo = self.trip.undo_label()
                if undo:
                    ui.button(f"Undo {undo}", icon="undo", on_click=self.undo).props("flat dense no-caps size=sm")
            if music and pins:
                ui.label(f"Cuts on your {len(pins)} pins from Music & audio.").classes("text-xs text-green-700")
            locked = self.trip.locked_count()
            if locked:
                ui.label(f"🔒 {locked} locked moment{'s' if locked > 1 else ''} stay in, each played whole; the cut "
                         "fills the rest around them. Unlock one in the Library, Themes or the clips below.") \
                    .classes("text-xs text-amber-800")
            with ui.expansion("More settings: shot lengths, effects", icon="tune").classes("w-full").props("dense"):
                self.auto_edit_settings(self.auto_limits, pins)

    def storyline_blocks_view(self, line: dict) -> None:
        blocks = self.trip.storyline_blocks()
        if not blocks:
            ui.label("No themes yet: open the Themes tab first.").classes("text-sm text-orange-700")
            return
        emoji = {a["name"]: a["emoji"] for a in self.trip.activities()}
        split = set(line["split"])
        rows = self.trip.theme_rows()
        with ui.column().classes("w-full gap-1"):
            for n, b in enumerate(blocks):
                with ui.row().classes("w-full items-center no-wrap gap-2 p-1 rounded bg-white border" +
                                      ("" if b["on"] else " opacity-50")):
                    with ui.column().classes("gap-0"):
                        ui.button(icon="arrow_upward", on_click=lambda n=n: self.move_block(n, -1)) \
                            .props("flat dense round size=xs").set_enabled(n > 0)
                        ui.button(icon="arrow_downward", on_click=lambda n=n: self.move_block(n, 1)) \
                            .props("flat dense round size=xs").set_enabled(n < len(blocks) - 1)
                    name = f"{emoji.get(b['activity'], '⭐')} {b['activity']}" + (f" · {b['person']}" if b["person"] else "")
                    ui.label(name).classes("font-medium min-w-48")
                    ui.label(f"{b['count']} shot{'s' * (b['count'] != 1)}").classes("text-xs text-gray-500 w-16")
                    with ui.row().classes("gap-1 no-wrap overflow-hidden flex-grow"):
                        for it in [r for r in rows if themes.block_of(r, split) == b["key"]][:6]:
                            if it.get("thumb"):
                                ui.image(data_url(it["thumb"])).classes("w-12 h-8 rounded").props("fit=cover no-spinner")
                    if b["person"] is None and b["count"] >= 4:
                        ui.button("By person", on_click=lambda a=b["activity"]: self.split_block(a, True)) \
                            .props("flat dense no-caps size=sm").tooltip("One block per person in these shots")
                    elif b["person"] is not None:
                        ui.button("Join", on_click=lambda a=b["activity"]: self.split_block(a, False)) \
                            .props("flat dense no-caps size=sm").tooltip("One block for all the people again")
                    ui.switch(value=b["on"], on_change=lambda e, k=b["key"]: self.block_on(k, e.value)).props("dense") \
                        .tooltip("Off: these shots stay out of the video")

    def my_story_view(self, line: dict) -> None:
        blocks = [b for b in self.trip.storyline_blocks() if b["on"]]
        if not blocks:
            ui.label("No themes yet: open the Themes tab first.").classes("text-sm text-orange-700")
            return
        story = line.get("story") or {}
        emoji = {a["name"]: a["emoji"] for a in self.trip.activities()}
        names = {b["key"]: f"{emoji.get(b['activity'], '⭐')} {themes.block_name(b)}" for b in blocks}
        self.wish_box = ui.textarea(
            placeholder="e.g. Start calm with the boat trip, then the beach and food, build up to Laszlo's best "
                        "kitesurfing jumps at the end", value=story.get("idea", "")) \
            .props("outlined autogrow").classes("w-full")
        with ui.row().classes("items-center gap-2"):
            self.talk_button = ui.button("Speak your idea", icon="mic", on_click=lambda: self.toggle_voice(self.talk_button)) \
                .props("outline no-caps")
            ui.button("Make my story", icon="auto_awesome", on_click=lambda: self.make_story(False)).props("no-caps")
            if story.get("parts"):
                ui.button("Try another idea", icon="refresh", on_click=lambda: self.make_story(True)) \
                    .props("flat no-caps")
            self.story_stop = ui.button("Stop", icon="stop_circle", on_click=self.stop_ai) \
                .props("outline color=negative dense no-caps")
            self.story_stop.set_visibility(self.ai_busy)
            ui.select(director.PROVIDERS, value=load_settings().get("ai", {}).get("provider", "ollama"), label="AI",
                      on_change=lambda e: self.story_ai(e.value)).props("dense").classes("w-72 ml-auto")
        if not story.get("parts"):
            ui.label("No story yet: until you make one, the video runs as it happened.").classes("text-sm text-gray-500")
            return
        parts = story["parts"]
        split = set(line["split"])
        shots_of: dict[str, list[dict]] = {}
        for r in self.trip.theme_rows():
            if r["included"]:
                shots_of.setdefault(themes.block_of(r, split), []).append(r)
        used = {it["part_of"] or it["path"] for it in self.trip.storyboard()}
        total = self.video_seconds()
        room = total  # the parts share the whole video: the last part is the end
        times = themes.fit_times(parts, room)
        counts = {b["key"]: b["count"] for b in blocks}
        with ui.row().classes("items-center gap-2"):
            ui.label(f"✨ {story.get('title') or 'Your story'}").classes("text-base font-semibold")
            ui.label(f"· {fmt_length(total)} in all").classes("text-sm text-gray-600")
            if story.get("by") == "words":
                ui.label("(made from your words; start Ollama for a smarter story)").classes("text-xs text-gray-500")
        # The whole song, shared between the parts (and the finale).
        colours = {"calm": "#5b8def", "medium": "#9b8afb", "lively": "#f08a24"}
        with ui.element("div").classes("w-full flex rounded overflow-hidden text-white text-xs font-semibold") \
                .style("height:26px"):
            for n, (part, t) in enumerate(zip(parts, times)):
                with ui.element("div").classes("flex items-center px-1 overflow-hidden whitespace-nowrap") \
                        .style(f"flex:{t};background:{colours.get(part['pace'], '#9b8afb')};"
                               f"border-right:2px solid white"):
                    ui.label(f"{n + 1}. {part['name']} {fmt_length(t)}").tooltip(f"{part['name']}: {fmt_length(t)}")
        drag = ("this.style.opacity=0.5;event.dataTransfer.setData('text/plain','{src}');event.stopPropagation()")
        over = "event.preventDefault();this.style.outline='2px dashed #2563eb'"
        leave = "this.style.outline=''"
        drop = ("event.preventDefault();event.stopPropagation();this.style.outline='';"
                "emitEvent('tm_story_drop',{{src:event.dataTransfer.getData('text/plain'),dst:'{dst}'}})")
        for n, part in enumerate(parts):
            card = ui.element("div").classes("w-full p-2 rounded border bg-white flex flex-col gap-1")
            card._props.update({"id": f"tm-part-{n}", "draggable": "true", "ondragstart": drag.format(src=f"p:{n}"),
                                "ondragend": "this.style.opacity=1", "ondragover": over, "ondragleave": leave,
                                "ondrop": drop.format(dst=f"p:{n}")})
            with card:
                with ui.row().classes("w-full items-center no-wrap gap-2"):
                    ui.icon("drag_indicator").classes("text-gray-400 cursor-grab").tooltip("Drag to move this part")
                    ui.label(f"{n + 1}.").classes("font-semibold")
                    title = ui.input(value=part["name"]).props("dense borderless").classes("font-medium flex-grow")
                    title.on("blur", lambda e, n=n, f=title: self.edit_part(n, name=f.value))
                    ui.select(self.PACE_NAMES, value=part["pace"],
                              on_change=lambda e, n=n: self.edit_part(n, pace=e.value)).props("dense outlined") \
                        .classes("w-32").tooltip("How lively this part feels")
                    ui.button(icon="remove", on_click=lambda n=n: self.part_time(n, -5)) \
                        .props("outline dense round size=sm").tooltip("5 seconds shorter (the other parts get them)")
                    ui.label(fmt_length(times[n])).classes("font-semibold w-12 text-center")
                    ui.button(icon="add", on_click=lambda n=n: self.part_time(n, 5)) \
                        .props("outline dense round size=sm").tooltip("5 seconds longer (taken from the other parts)")
                    ui.button(icon="delete_outline", on_click=lambda n=n: self.remove_part(n)) \
                        .props("flat dense round size=sm").tooltip("Leave this part out")
                with ui.row().classes("w-full items-center gap-2"):
                    for m, entry in enumerate(part["entries"]):
                        if entry["key"] not in names:
                            continue
                        added = entry.get("added")
                        chip = ui.element("div").classes("flex items-center no-wrap gap-1 px-2 rounded-full border "
                                                         "cursor-grab " + ("bg-amber-50 border-dashed border-amber-500"
                                                                           if added else "bg-blue-50"))
                        if added:
                            chip.tooltip("Added by the app: the AI left this theme out, so the app put it where it "
                                         "was filmed nearest in time. Drag it elsewhere or remove it.")
                        chip._props.update({"draggable": "true", "ondragstart": drag.format(src=f"e:{n}:{m}"),
                                            "ondragend": "this.style.opacity=1", "ondragover": over,
                                            "ondragleave": leave, "ondrop": drop.format(dst=f"e:{n}:{m}")})
                        with chip:
                            ui.icon("drag_indicator").classes("text-gray-400 text-sm")
                            ui.label(names[entry["key"]]).classes("text-sm")
                            if added:
                                ui.label("added by the app").classes("text-xs text-amber-700 italic")
                            ui.select(themes.PICKS, value=entry["pick"],
                                      on_change=lambda e, n=n, m=m: self.edit_entry(n, m, pick=e.value)) \
                                .props("dense borderless options-dense").classes("text-xs w-36")
                            mine = shots_of.get(entry["key"], [])
                            dropped = set(entry.get("drop") or [])
                            n_in = sum(1 for r in mine if (r["path"] in used or r["path"] in (entry.get("keep") or []))
                                       and r["path"] not in dropped)
                            ui.button(f"{n_in} of {len(mine)} used", icon="photo_library",
                                      on_click=lambda n=n, m=m, e=entry, rows=mine, t=times[n]:
                                      self.shots_dialog(n, m, e, rows, used, t)) \
                                .props("flat dense no-caps size=sm").classes("text-xs") \
                                .tooltip("See this theme's shots: tick to put one in, untick to keep it out")
                            if entry["pick"] == "moment":
                                activity = entry["key"].partition("|")[0]
                                look = ui.input(value=entry.get("moment", ""), placeholder="e.g. kite jump",
                                                autocomplete=self.trip.moment_names(activity)) \
                                    .props("dense borderless").classes("text-xs w-28") \
                                    .tooltip("Type a few letters: names you used before are offered (Tab takes one)")
                                look.on("blur", lambda e, n=n, m=m, f=look: self.edit_entry(n, m, moment=f.value))
                                saved = self.trip.moment(activity, entry.get("moment") or "") \
                                    if entry.get("moment") else None
                                found = len(saved["shots"]) if saved else 0
                                ui.button(f"{found} marked" if found else "Mark in Themes", icon="search",
                                          on_click=lambda n=n, m=m, a=activity, f=look: self.mark_moment(n, m, a, f.value)) \
                                    .props(("outline" if found else "unelevated") + " dense no-caps size=sm") \
                                    .tooltip("Opens Themes: tick a few shots of this moment, the app finds the rest")
                            ui.button(icon="close", on_click=lambda n=n, m=m: self.remove_entry(n, m)) \
                                .props("flat dense round size=xs")
                    ui.button("Add theme", icon="add", on_click=lambda n=n: self.add_theme_dialog(n, blocks, names)) \
                        .props("flat dense no-caps size=sm")
                if part.get("why"):
                    ui.label(f"AI: {part['why']}").classes("text-xs text-gray-500 italic")
                warning = self._part_warning(part, times[n], counts)
                if warning:
                    ui.label(f"⚠️ {warning}").classes("text-xs text-orange-700")
        with ui.row().classes("w-full items-center gap-2"):
            ui.button("New part", icon="add", on_click=self.new_part).props("flat dense no-caps size=sm")
            ui.label("Drag parts and themes (⠿) to reorder them or move a theme to another part. − / + takes the "
                     "time from the other parts, so the total stays the song's length.").classes("text-xs text-gray-500")
        missing = themes.left_out(story, blocks)
        if missing:
            with ui.row().classes("w-full items-center gap-2"):
                ui.label("Not in the story:").classes("text-sm text-gray-600")
                for b in missing:
                    ui.button(f"+ {names[b['key']]} ({b['count']})", on_click=lambda k=b["key"]: self.add_entry(k)) \
                        .props("flat dense no-caps size=sm").tooltip("Add to the last part")

    def shots_dialog(self, n: int, m: int, entry: dict, rows: list[dict], used: set[str], seconds: float) -> None:
        """A theme's shots in My story: the ones in the cut ticked first, then the rest, best first. Tick a shot to
        put it in, untick it to keep it out; the next cut follows."""
        keep, drop = set(entry.get("keep") or []), set(entry.get("drop") or [])
        state = {r["path"]: (r["path"] in keep or r["path"] in used) and r["path"] not in drop for r in rows}

        def value(r: dict) -> float:
            try:
                return shot_value(r)
            except (KeyError, TypeError):
                return 0.0

        ordered = sorted(rows, key=lambda r: (not state[r["path"]], r["taken_at"] if state[r["path"]] else -value(r)))
        limits = self.limits_now()

        def need() -> float:  # about how long the ticked shots take on screen
            return sum((limits.max_video * 0.7 if r["kind"] == "video" else limits.max_photo)
                       for r in rows if state[r["path"]])

        with ui.dialog() as dialog, ui.card().classes("w-[64rem] max-w-full"):
            with ui.row().classes("w-full items-center gap-3"):
                ui.label(themes.block_name({"key": entry["key"], "activity": entry["key"].partition("|")[0],
                                            "person": entry["key"].partition("|")[2] or None})) \
                    .classes("text-lg font-semibold")
                hint = ui.label("").classes("text-sm text-gray-600")
            ui.label("Ticked shots are in the video (the ones the last cut used, and any you tick). Untick a shot "
                     "to keep it out. Make the first cut again to see the change.").classes("text-xs text-gray-500")

            def show_hint() -> None:
                ticked = sum(state.values())
                hint.set_text(f"{ticked} ticked of {len(rows)} · they need about {fmt_length(need())}; this part "
                              f"has {fmt_length(seconds)}")
                hint.classes(replace="text-sm " + ("text-orange-700" if need() > seconds * 1.3 else "text-gray-600"))

            with ui.element("div").classes("w-full grid grid-cols-6 gap-2 max-h-[65vh] overflow-auto"):
                for r in ordered[:300]:
                    path = r["path"]
                    with ui.element("div").classes("relative rounded p-1 border-2 cursor-pointer") as tile:
                        if r.get("thumb"):
                            ui.image(data_url(r["thumb"])).classes("w-full rounded").style("aspect-ratio:16/10") \
                                .props("fit=cover no-spinner")
                        mark = ui.label("").classes("absolute top-2 left-2 w-6 h-6 rounded border-2 text-center "
                                                    "font-bold shadow")
                        ui.label(("🎬 " if r["kind"] == "video" else "📷 ") + r["taken_at"][5:16].replace("T", " ")) \
                            .classes("text-xs text-gray-600")

                    def show(tile=tile, mark=mark, path=path) -> None:
                        on = state[path]
                        mark.set_text("✓" if on else "")
                        mark.classes(replace="absolute top-2 left-2 w-6 h-6 rounded border-2 text-center font-bold "
                                             "shadow " + ("bg-green-600 border-green-600 text-white" if on else
                                                          "bg-white border-gray-500"))
                        tile.classes(replace="relative rounded p-1 border-2 cursor-pointer " +
                                             ("border-green-600" if on else "border-transparent opacity-60"))

                    def toggle(path=path, show=show) -> None:
                        state[path] = not state[path]
                        show()
                        show_hint()

                    show()

                    async def to_library(path=path) -> None:
                        done()  # your ticks so far are kept as you leave
                        await self.open_in_library(path, "My story")

                    self.click_or_double(tile, toggle, to_library)
            show_hint()

            def done() -> None:
                was_in = {p for p in state if (p in used or p in keep) and p not in drop}
                new_keep = sorted(p for p, on in state.items() if on and (p not in used or p in drop or p in keep))
                new_drop = sorted(p for p, on in state.items() if not on and (p in was_in or p in drop))
                dialog.close()
                self.edit_entry(n, m, keep=new_keep, drop=new_drop)
                ui.notify("Saved. Make the first cut again to use your choice.", type="positive")

            with ui.row().classes("w-full justify-end gap-2"):
                ui.button("Cancel", on_click=dialog.close).props("flat no-caps")
                ui.button("Use these shots", icon="check", on_click=done).props("no-caps")
        dialog.open()

    def video_seconds(self) -> float:
        target = self.length_now()
        if target == "song":
            return self.track_seconds(self.music_path()) or 180.0
        return float(target)

    def _part_warning(self, part: dict, seconds: float, counts: dict) -> str:
        """Too little time for a part, or too few shots for its time."""
        shots = sum(counts.get(e["key"], 0) * (1 if e["pick"] == "all" else 0.5) for e in part["entries"])
        if not part["entries"]:
            return "No themes yet: drag one here or press Add theme."
        limits = self.limits_now()
        if seconds < 6:
            return f"Only {seconds:.0f} s: about {max(1, round(seconds / 2))} shots. Make it longer or remove it?"
        longest = max(limits.max_video, limits.max_photo)
        if shots * longest < seconds * 0.9:
            return (f"Not enough shots for {fmt_length(seconds)}: {shots:.0f} shots fill about "
                    f"{fmt_length(shots * longest)}. Make it shorter or add a theme.")
        return ""

    def _block_times(self, blocks: list[dict]) -> dict[str, float]:
        """Each theme block's middle time (when most of it was filmed), to place a theme the AI left out."""
        split = set(self.trip.storyline()["split"])
        times: dict[str, list[float]] = {}
        for r in self.trip.theme_rows():
            times.setdefault(themes.block_of(r, split), []).append(datetime.fromisoformat(r["taken_at"]).timestamp())
        return {b["key"]: sorted(times[b["key"]])[len(times[b["key"]]) // 2] for b in blocks if times.get(b["key"])}

    def _story_blocks_info(self, blocks: list[dict]) -> list[dict]:
        """What the AI hears about each theme: its name, how many shots, which days and where."""
        line = self.trip.storyline()
        rows = self.trip.theme_rows()
        out = []
        for b in blocks:
            mine = [r for r in rows if themes.block_of(r, set(line["split"])) == b["key"]]
            days = sorted({r["taken_at"][:10] for r in mine})
            places = sorted({r["place"] for r in mine if r.get("place")})
            out.append({"name": themes.block_name(b), "count": b["count"],
                        "days": ", ".join(datetime.fromisoformat(d).strftime("%-d %b") for d in days[:4]),
                        "places": ", ".join(places[:3])})
        return out

    def story_ai(self, provider: str) -> None:
        self.save_ai(refresh=False, provider=provider)
        if provider == "claude_code" and not director.claude_code_path():
            ui.notify("Claude Code isn't on this Mac yet. In Terminal run: " + director.CLAUDE_CODE_INSTALL +
                      "  then type claude, press Enter and sign in. More in Stories > AI director > Which AI.",
                      type="warning", multi_line=True, timeout=20000)

    async def make_story(self, again: bool) -> None:
        """Make my story: first the AI reads your idea back to you (fixing dictation slips, a few questions), then
        it writes the story. Try another idea skips the reading and uses your last answers."""
        if not self.trip:
            return
        if self.ai_busy:  # (analysing files in the background doesn't stop you)
            ui.notify("The AI is still busy with the last request. Wait for it, or press Stop.", type="warning")
            return
        idea = (self.wish_box.value or "").strip() if self.wish_box is not None else ""
        blocks = [b for b in self.trip.storyline_blocks() if b["on"]]
        if not blocks:
            return
        try:
            ai, failed = director.make(load_settings().get("ai", {})), ""
        except director.DirectorError as exc:
            ai, failed = None, str(exc)
        old = self.trip.storyline().get("story") or {}
        if ai is None or not idea or again:
            await self.write_story(again, idea, old.get("answers") or [] if again else [], ai, failed)
            return
        moments = [f"{m['name']} (in {m['activity']})" for m in self.trip.moments()]
        prompt = director.understand_prompt(idea, [themes.block_name(b) for b in blocks], moments)
        try:
            answer = await self._think(ai, ai.understand, prompt, "reading your idea")
        except Exception as exc:  # no reading: straight on to the story
            print(f"My story: the AI couldn't read the idea back ({exc}); making the story.", flush=True)
            await self.write_story(False, idea, [], ai, "")
            return
        if answer is None:
            ui.notify("Stopped: your story stays as it was. Press Make my story to try again.")
            return
        self.understood_dialog(director.clean_understood(answer, idea), idea, ai)

    def understood_dialog(self, understood: dict, said: str, ai) -> None:
        """What the AI understood: your idea, cleaned up (change anything), and its few questions to click."""
        picked: list[dict] = []
        with ui.dialog() as dialog, ui.card().classes("w-[44rem] max-w-full gap-3"):
            ui.label("🤖 What I understood").classes("text-lg font-semibold")
            box = ui.textarea("Your idea (change anything)", value=understood["idea"]) \
                .props("outlined autogrow").classes("w-full")
            if understood["idea"] != said:
                ui.label(f"You said: {said}").classes("text-xs text-gray-500 italic")
            for q in understood["questions"]:
                ui.label(q["question"]).classes("text-sm font-medium mt-2")
                choice = ui.toggle(q["answers"], value=None, clearable=True) \
                    .props("no-caps unelevated toggle-color=primary color=white text-color=primary") \
                    .classes("border border-blue-600 rounded")
                own = ui.input(placeholder="or type your own answer").props("dense").classes("w-full")
                picked.append({"question": q["question"], "choice": choice, "own": own})

            async def go() -> None:
                answers = [[p["question"], (p["own"].value or "").strip() or (p["choice"].value or "")] for p in picked]
                answers = [a for a in answers if a[1]]
                idea = (box.value or "").strip() or said
                if self.wish_box is not None:
                    self.wish_box.value = idea
                dialog.close()
                await self.write_story(False, idea, answers, ai, "")

            with ui.row().classes("w-full justify-end gap-2 mt-2"):
                ui.button("Cancel", on_click=dialog.close).props("flat no-caps")
                ui.button("Make my story", icon="auto_awesome", on_click=go).props("no-caps")
        dialog.open()

    async def _think(self, ai, ask, prompt: str, what: str):
        """Ask the AI with a running clock and Stop; None if you stopped it (errors are raised)."""
        self.ai_busy = True
        note = ui.notification("", spinner=True, timeout=None, multi_line=True, close_button="Hide")
        started = time.time()

        def thinking() -> None:
            gone = int(time.time() - started)
            usual = ("Claude usually takes under a minute." if isinstance(ai, director.ClaudeCode) else
                     "On a MacBook Air it usually takes 1 to 3 minutes (longer the first time, while the model "
                     "loads).")
            note.message = (f"🤖 The AI is {what}: {gone // 60}:{gone % 60:02d} so far. {usual} "
                            "Stop is next to the story buttons.")

        thinking()
        ticker = ui.timer(1.0, thinking)
        if self.story_stop is not None:
            self.story_stop.set_visibility(True)
        print(f"My story: the AI ({type(ai).__name__}) is {what}…", flush=True)
        try:
            answer = await self.stoppable(ask, prompt)
            print(f"My story: the AI answered after {time.time() - started:.0f} s." if answer is not None else
                  "My story: stopped.", flush=True)
            return answer
        finally:
            self.ai_busy = False
            ticker.cancel()
            note.dismiss()
            if self.story_stop is not None:
                self.story_stop.set_visibility(False)

    async def write_story(self, again: bool, idea: str, answers: list, ai, failed: str) -> None:
        blocks = [b for b in self.trip.storyline_blocks() if b["on"]]
        old = self.trip.storyline().get("story") or {}
        before = (old.get("before") or []) + ([old["title"]] if again and old.get("title") else []) if again else []
        story = None
        if ai is not None:
            try:
                prompt = director.story_prompt(director.with_answers(idea, [tuple(a) for a in answers]),
                                               self._story_blocks_info(blocks), before[-4:] or None,
                                               self.video_seconds(), packed=self.auto_limits.pack in ("more", "max"))
                answer = await self._think(ai, ai.story, prompt, "working on your story")
                if answer is None:
                    ui.notify("Stopped: your story stays as it was. Press Make my story to try again.")
                    return
                story = themes.story_keys(director.clean_story(answer, len(blocks)), blocks)
                story["by"] = "ai"
            except Exception as exc:  # Ollama not running, a bad answer...: your words still work
                print(f"My story: the AI couldn't make a story ({exc}); following your words instead.", flush=True)
                failed = str(exc)
        if not story or not story.get("parts"):
            if not idea:
                ui.notify("Write or say your idea first (without the AI, the app follows the themes you name).",
                          type="warning", multi_line=True)
                return
            words = {a["name"]: a["describe"] for a in self.trip.activities()}
            story = themes.story_from_words(idea, [b | {"describe": words.get(b["activity"])} for b in blocks]) \
                | {"by": "words"}
            if not story["parts"]:
                ui.notify("None of your themes were named. Use their names, e.g. \"boat first, then food\".",
                          type="warning", multi_line=True)
                return
            ui.notify(f"The AI couldn't make the story ({failed or 'no answer'}), so the app followed the order "
                      "you named your themes in.", type="warning", multi_line=True, timeout=15000)
        else:
            ui.notify(f"Your story is ready: {story['title'] or 'have a look'}. Change anything below, then Make the "
                      "first cut.", type="positive", multi_line=True)
        missing = len(themes.left_out(story, blocks))
        story = themes.add_missing(story, blocks, self._block_times(blocks))
        if missing:
            ui.notify(f"{missing} theme{'s' if missing > 1 else ''} weren't in the story, so the app added "
                      f"{'them' if missing > 1 else 'it'} where filmed nearest in time (orange, \"added by the app\"). "
                      "Drag or remove as you like.", multi_line=True, timeout=12000)
        counts = {b["key"]: b["count"] for b in blocks}
        for p in story["parts"]:  # without the AI's times: by how many shots each part has
            if not p.get("seconds"):
                p["seconds"] = sum(counts.get(e["key"], 0) for e in p["entries"]) ** 0.5 or 1
        for p, t in zip(story["parts"], themes.fit_times(story["parts"], self.video_seconds())):
            p["seconds"] = t
        self.trip.checkpoint("story")
        self.trip.set_storyline(story=story | {"idea": idea, "before": before, "answers": answers})
        self.storyboard_view.refresh()

    def _edit_story(self, change) -> None:
        line = self.trip.storyline()
        story = dict(line.get("story") or {})
        parts = [dict(p, entries=[dict(e) for e in p["entries"]]) for p in story.get("parts") or []]
        change(parts)
        # Every part keeps its time; the total always stays the song's length (less the finale).
        for p, t in zip(parts, themes.fit_times(parts, self.video_seconds())):
            p["seconds"] = t
        story["parts"] = parts
        if self.wish_box is not None:
            story["idea"] = (self.wish_box.value or "").strip()
        self.trip.set_storyline(story=story)
        self.storyboard_view.refresh()

    def part_time(self, n: int, step: float) -> None:
        def change(parts):
            times = themes.fit_times(parts, self.video_seconds())
            for p, t in zip(parts, themes.change_time(times, n, times[n] + step)):
                p["seconds"] = t
        self._edit_story(change)

    def new_part(self) -> None:
        self._edit_story(lambda parts: parts.append({"name": f"Part {len(parts) + 1}", "pace": "medium",
                                                     "seconds": None, "entries": []}))

    async def theme_drop(self, e) -> None:
        """A theme folder dropped in the Themes list: ["t", theme], ["m", theme, sub-theme], or ["new"] (out)."""
        args = e.args if isinstance(e.args, dict) else {}
        try:
            src, dst = json.loads(args.get("src") or "null"), json.loads(args.get("dst") or "null")
        except ValueError:
            return
        if not self.trip or not isinstance(src, list) or not isinstance(dst, list) or src == dst:
            return
        if src[0] == "m" and dst[0] == "t":
            if src[1] == dst[1]:
                return
            await run.io_bound(self.trip.move_moment, src[1], src[2], dst[1])
            ui.notify(f"🔍 {src[2]} is now inside {dst[1]}; its ticked shots went along.", type="positive")
            self.theme_shown, self.theme_moment = dst[1], None
        elif src[0] == "m" and dst[0] == "new":
            vector = None
            if self.trip.has_fingerprints():
                try:
                    v = await run.io_bound(text_vectors, [f"a holiday photo of {src[2].lower()}"],
                                           app_dir() / "models" / "clip")
                    vector = [round(float(x), 5) for x in v[0]]
                except Exception:
                    traceback.print_exc()
            made = await run.io_bound(self.trip.moment_to_theme, src[1], src[2], vector)
            ui.notify(f"{made} is a theme of its own now, with its ticked shots.", type="positive")
            self.theme_shown, self.theme_moment = made, None
        elif src[0] == "t" and dst[0] == "t":
            await run.io_bound(self.trip.theme_to_moment, src[1], dst[1])
            ui.notify(f"{src[1]} is now a sub-theme inside {dst[1]}: all its shots moved there.", type="positive",
                      multi_line=True)
            self.theme_shown, self.theme_moment = dst[1], None
        else:
            return
        self.theme_selected = set()
        self.stale = True
        self.themes_view.refresh()

    def story_drop(self, e) -> None:
        """A part or a theme dropped somewhere in My story: p:N is a part, e:N:M a theme (M in part N)."""
        args = e.args if isinstance(e.args, dict) else {}
        src, dst = str(args.get("src", "")).split(":"), str(args.get("dst", "")).split(":")
        if len(src) < 2 or len(dst) < 2 or not self.trip:
            return
        try:
            src_n, dst_n = int(src[1]), int(dst[1])
            src_m = int(src[2]) if src[0] == "e" else None
            dst_m = int(dst[2]) if dst[0] == "e" else None
        except (ValueError, IndexError):
            return

        def change(parts):
            if not (0 <= src_n < len(parts) and 0 <= dst_n < len(parts)):
                return
            if src_m is None:  # a part: it goes where it was dropped
                if src_n != dst_n:
                    parts.insert(dst_n, parts.pop(src_n))
                return
            if not 0 <= src_m < len(parts[src_n]["entries"]):
                return
            entry = parts[src_n]["entries"].pop(src_m)
            at = len(parts[dst_n]["entries"]) if dst_m is None else dst_m
            if src_n == dst_n and dst_m is not None and src_m < dst_m:
                at -= 1
            parts[dst_n]["entries"].insert(max(0, at), entry)
        self._edit_story(change)

    def add_theme_dialog(self, n: int, blocks: list[dict], names: dict) -> None:
        with ui.dialog() as dialog, ui.card().classes("w-[720px] max-w-[95vw] gap-3"):
            ui.label("Which theme goes in this part?").classes("text-lg font-semibold")
            with ui.element("div").classes("w-full grid gap-2").style("grid-template-columns:repeat(3,minmax(0,1fr))"):
                for b in blocks:
                    ui.button(f"{names[b['key']]} ({b['count']})",
                              on_click=lambda k=b["key"]: (dialog.close(), self.add_entry(k, n))) \
                        .props("outline no-caps").classes("h-12 text-base w-full")
            ui.button("Cancel", on_click=dialog.close).props("flat no-caps")
        dialog.open()

    def edit_part(self, n: int, **values) -> None:
        story = self.trip.storyline().get("story") or {}
        if all(story["parts"][n].get(k) == v for k, v in values.items()):
            return
        self._edit_story(lambda parts: parts[n].update(values))

    def move_part(self, n: int, steps: int) -> None:
        def change(parts):
            if 0 <= n + steps < len(parts):
                parts.insert(n + steps, parts.pop(n))
        self._edit_story(change)

    def remove_part(self, n: int) -> None:
        self._edit_story(lambda parts: parts.pop(n))

    def edit_entry(self, n: int, m: int, **values) -> None:
        self._edit_story(lambda parts: parts[n]["entries"][m].update(values))

    def remove_entry(self, n: int, m: int) -> None:
        self._edit_story(lambda parts: parts[n]["entries"].pop(m))

    def add_entry(self, key: str, n: int = -1) -> None:
        def change(parts):
            if not parts:
                parts.append({"name": "Part 1", "pace": "medium", "entries": []})
            parts[n]["entries"].append({"key": key, "pick": "all"})
        self._edit_story(change)

    def _saved_blocks(self) -> list[dict]:
        return [{k: b[k] for k in ("key", "activity", "person", "on")} for b in self.trip.storyline_blocks()]

    def set_order(self, order: str) -> None:
        self.trip.set_storyline(order=order, blocks=self._saved_blocks())
        self.storyboard_view.refresh()

    def set_line(self, **changes) -> None:
        self.trip.set_storyline(blocks=self._saved_blocks(), **changes)
        self.storyboard_view.refresh()
        if "day_titles" in changes:
            self.refresh_all()  # (the titles show on the storyboard and in Make video)

    def set_finale(self, on: bool) -> None:
        self.trip.set_storyline(finale=on)
        if self.trip.storyline()["order"] == themes.STORY:
            self.storyboard_view.refresh()  # the finale takes its time from the parts

    def move_block(self, n: int, steps: int) -> None:
        blocks = self._saved_blocks()
        if 0 <= n + steps < len(blocks):
            blocks.insert(n + steps, blocks.pop(n))
            self.trip.set_storyline(blocks=blocks)
            self.storyboard_view.refresh()

    def block_on(self, key: str, on: bool) -> None:
        self.trip.set_storyline(blocks=[b | {"on": on} if b["key"] == key else b for b in self._saved_blocks()])
        self.storyboard_view.refresh()

    def split_block(self, activity: str, split: bool) -> None:
        line = self.trip.storyline()
        keep = [b for b in self._saved_blocks() if b["activity"] != activity]
        at = next((k for k, b in enumerate(self._saved_blocks()) if b["activity"] == activity), len(keep))
        names = set(line["split"]) | {activity} if split else set(line["split"]) - {activity}
        self.trip.set_storyline(split=sorted(names), blocks=keep)
        # The new blocks take the old one's place.
        fresh = [b for b in self._saved_blocks() if b["activity"] == activity]
        rest = [b for b in self._saved_blocks() if b["activity"] != activity]
        self.trip.set_storyline(blocks=rest[:at] + fresh + rest[at:])
        self.storyboard_view.refresh()

    # ---------- Themes ----------

    THEME_PAGE = 60

    @ui.refreshable
    def themes_view(self) -> None:
        if not self.showing("Themes"):
            return
        if not self.trip:
            ui.label("Open a trip first.").classes("text-gray-500")
            return
        self.trip.confirm_moment_shots()  # ticked in a sub-theme: confirmed in its theme too
        rows = self.trip.theme_rows()
        if not rows:
            ui.label("Nothing selected yet: tick files in Choose files first.").classes("text-gray-500")
            return
        if all(r["theme_auto"] is None for r in rows):
            ui.timer(0.1, self.suggest_themes, once=True)
            ui.label("Looking at your shots…").classes("text-gray-500")
            return
        proposals = self.trip.theme_proposals()
        if proposals:  # the trip's own themes, waiting for your answer
            self.theme_review(proposals)
            return
        self.new_themes_bar()
        self.theme_check_card()
        activities = self.trip.activities()
        names = [a["name"] for a in activities]
        if self.theme_shown not in names:
            self.theme_shown = next((a for a in names if any(r["theme"] == a for r in rows)), names[0])
        done = sum(r["theme_ok"] for r in rows)
        with ui.row().classes("w-full items-center gap-3"):
            ui.label("What is each shot?").classes("text-lg font-semibold")
            self.theme_badge = ui.badge(f"{done} of {len(rows)} confirmed",
                                        color="positive" if done == len(rows) else "grey-7")
            self.theme_counts = [done, len(rows)]
            ui.label("The app suggests the activity; you confirm or change it. Who, where and when are filled in "
                     "for you.").classes("text-sm text-gray-600 flex-1")
        earlier = self.trip.earlier_trips_learned()
        if earlier:
            ui.label(f"🧠 The app remembers {earlier} earlier trip{'s' if earlier > 1 else ''}: your own themes and the "
                     "shots you confirmed there help it suggest here too.").classes("text-sm text-indigo-700")
        if not self.trip.has_fingerprints():
            ui.label("Tip: run Analyse in Choose files first. The app then judges each picture itself; until then it "
                     "only guesses from tags.").classes("text-sm text-orange-700")
        saved_moments = self.trip.moments()
        # Like folders: drag a sub-theme onto another theme, or out (to the box below) to make it a theme of its
        # own; drag a theme onto another to put it inside as a sub-theme.
        drag = "event.dataTransfer.setData('text/plain', this.dataset.k); this.style.opacity=0.5; event.stopPropagation()"
        over = "event.preventDefault(); this.style.outline='2px dashed #2563eb'"
        leave = "this.style.outline=''"
        drop = ("event.preventDefault(); event.stopPropagation(); this.style.outline=''; "
                "emitEvent('tm_theme_drop', {src: event.dataTransfer.getData('text/plain'), dst: this.dataset.k})")

        def folder(el, key: list, target: bool, movable: bool) -> None:
            el._props["data-k"] = json.dumps(key)
            if movable:
                el._props.update({"draggable": "true", "ondragstart": drag, "ondragend": "this.style.opacity=1"})
            if target:
                el._props.update({"ondragover": over, "ondragleave": leave, "ondrop": drop})

        with ui.row().classes("w-full no-wrap items-start gap-4"):
            with ui.column().classes("w-56 gap-1 flex-none"):
                for a in activities:
                    count = sum(r["theme"] == a["name"] for r in rows)
                    todo = sum(r["theme"] == a["name"] and not r["theme_ok"] for r in rows)
                    on = a["name"] == self.theme_shown
                    with ui.row().classes("w-full items-center no-wrap gap-1 px-2 py-1 rounded cursor-pointer" +
                                          (" bg-blue-100 text-blue-800 font-semibold" if on else " hover:bg-gray-100")) \
                            .on("click", lambda n=a["name"]: self.show_theme(n)) as line:
                        folder(line, ["t", a["name"]], True, a["name"] != themes.OTHER)
                        ui.label(f"{a['emoji']} {a['name']}").classes("flex-grow text-sm")
                        ui.label(f"{count}" + (f" · {todo}?" if todo else "")).classes("text-xs text-gray-500")
                    for mo in [m for m in saved_moments if m["activity"] == a["name"]]:
                        here = on and self.theme_moment == mo["name"]
                        with ui.row().classes("w-full items-center no-wrap gap-1 pl-6 pr-2 py-0.5 rounded cursor-pointer" +
                                              (" bg-purple-100 text-purple-800 font-semibold" if here
                                               else " hover:bg-gray-100")) \
                                .on("click", lambda a=a["name"], m=mo["name"]: self.open_moment(a, m)) as sub:
                            folder(sub, ["m", a["name"], mo["name"]], False, True)
                            ui.label(f"↳ 🔍 {mo['name']}").classes("flex-grow text-xs")
                            ui.label(f"{len(mo['shots'])} ✓").classes("text-xs text-gray-500")
                if saved_moments or len(activities) > 2:
                    out = ui.label("Drag a theme onto another to put it inside · drag a sub-theme here to make it a "
                                   "theme of its own").classes("w-full text-[11px] text-gray-500 p-2 mt-1 rounded "
                                                               "border border-dashed border-gray-300")
                    folder(out, ["new"], True, False)
                with ui.row().classes("items-center gap-1 mt-2 no-wrap"):
                    new = ui.input(placeholder="Add an activity").props("dense outlined").classes("w-36")
                    new.on("keydown.enter", lambda: self.add_activity(new.value))
                    ui.button(icon="add", on_click=lambda: self.add_activity(new.value)).props("flat dense round")
            with ui.column().classes("flex-grow gap-2 min-w-0"):
                self.theme_grid(rows, activities)

    def theme_check_card(self) -> None:
        """Second check: the app sorts every shot again, learning from the ones you confirmed, and shows only the
        shots where it now thinks another theme fits better."""
        check = self.trip.last_theme_check()
        with ui.column().classes("w-full gap-2 p-2 rounded bg-teal-50"):
            with ui.row().classes("w-full items-center gap-3"):
                ui.button("Second check", icon="fact_check", on_click=self.run_theme_check) \
                    .props("unelevated no-caps color=teal") \
                    .tooltip("The app looks at every shot again, comparing it with the shots you confirmed, and "
                             "lists the ones that look like they belong to another theme. Nothing changes until you "
                             "answer.")
                if check is None:
                    ui.label("Done confirming? Run a second check to catch shots in the wrong theme.") \
                        .classes("text-sm text-teal-900 flex-1")
                elif check.get("agree") is None:
                    ui.label("The second check needs the pictures analysed first (Analyse in Choose files).") \
                        .classes("text-sm text-orange-700 flex-1")
                else:
                    left = len(check["shots"])
                    ui.label(f"{round(100 * check['agree'])}% agree"
                             + (f" · {left} shot{'s' if left > 1 else ''} to look at" if left else
                                " · nothing left to look at. Run it again after more changes.")) \
                        .classes("text-sm font-medium text-teal-900 flex-1")
            if check and check.get("shots"):
                items = {it["path"]: it for it in self.items}
                with ui.element("div").classes("w-full grid gap-2").style(
                        "grid-template-columns:repeat(auto-fill,minmax(190px,1fr))"):
                    for shot in check["shots"][:60]:
                        it = items.get(shot["path"])
                        if not it:
                            continue
                        with ui.column().classes("gap-1 p-1 rounded bg-white"):
                            box = ui.element("div").classes("w-full")
                            with box:
                                self.preview(it, "width:100%;aspect-ratio:16/10", start=self.hover_start(it), view="theme")
                            box.on("dblclick", lambda p=shot["path"]: self.shot_window(p, "Themes"))
                            ui.label(f"{shot['now']} → {shot['better']}").classes("text-sm font-semibold")
                            ui.label(("You set " + shot["now"] + "; it " if shot["yours"] else "It ") + shot["why"] + ".") \
                                .classes("text-xs text-gray-600")
                            with ui.row().classes("gap-1"):
                                ui.button(f"Move it", icon="east",
                                          on_click=lambda p=shot["path"]: self.answer_theme_check(p, True)) \
                                    .props("dense unelevated no-caps size=sm color=teal")
                                ui.button("Keep mine", on_click=lambda p=shot["path"]: self.answer_theme_check(p, False)) \
                                    .props("dense outline no-caps size=sm color=grey-8")
                if len(check["shots"]) > 60:
                    ui.label(f"…and {len(check['shots']) - 60} more after these.").classes("text-xs text-gray-600")

    async def run_theme_check(self) -> None:
        note = ui.notification("Second check: looking at every shot again…", spinner=True, timeout=None)
        try:
            check = await run.io_bound(self.trip.theme_check)
        finally:
            note.dismiss()
        left = len(check["shots"])
        ui.notify(f"{round(100 * check['agree'])}% agree with your themes." + (
            f" {left} shot{'s' if left > 1 else ''} to look at." if left else " Nothing to change.")
                  if check.get("agree") is not None else "Analyse the pictures first.", type="positive")
        self.themes_view.refresh()

    def answer_theme_check(self, path: str, move: bool) -> None:
        self.trip.answer_theme_check(path, move)
        self.reload()
        self.themes_view.refresh()

    def new_themes_bar(self) -> None:
        """A new trip: the app proposes its own themes from the shots (not the last trip's)."""
        claude = load_settings().get("ai", {}).get("provider") == "claude_code"
        if self.trip._meta_json("own_themes", False):  # done for this trip: only a small button to do it again
            with ui.row().classes("w-full justify-end"):
                ui.button("Suggest this trip's themes again", icon="auto_awesome", on_click=self.propose_trip_themes) \
                    .props("flat dense no-caps color=indigo")
            return
        with ui.row().classes("w-full items-center gap-3 p-2 rounded bg-indigo-50"):
            ui.label("✨ A new trip? Let the app look at your shots and suggest this trip's own themes and "
                     "sub-themes (instead of the last trip's)." + (" Claude (your Pro) gives them names."
                                                                    if claude else "")) \
                .classes("text-sm text-indigo-900 flex-1")
            ui.button("Suggest themes for this trip", icon="auto_awesome", on_click=self.propose_trip_themes) \
                .props("unelevated no-caps color=indigo")

    async def propose_trip_themes(self) -> None:
        if self.busy():
            return
        missing = self.trip.missing_fingerprints()
        if missing:
            ui.notify(f"First the app looks at the {missing} chosen shot{'s' if missing > 1 else ''} it hasn't "
                      "analysed yet (Analyse); the themes come right after.", multi_line=True, timeout=6000)
            await self.run_step(self.trip.analyse)
            self.panels.value = self.themes_tab
        if not self.trip.has_fingerprints():
            ui.notify("The picture model isn't ready, so the app can't see the shots yet. Try Analyse in Choose "
                      "files first.", type="warning", multi_line=True)
            return
        ai = None
        if load_settings().get("ai", {}).get("provider") == "claude_code":
            try:
                ai = director.make(load_settings().get("ai", {}))
            except director.DirectorError:
                ai = None
        state = {"share": 0.0, "step": "starting"}
        note = ui.notification("", spinner=True, timeout=None, multi_line=True)
        timer = ui.timer(0.5, lambda: setattr(note, "message", f"✨ Finding this trip's themes: {state['step']}…"))
        self.ai_busy = True
        try:
            sentences = [f"a holiday photo of {x}" for _, _, x in themefinder.phrases()]
            vectors = await run.io_bound(text_vectors, sentences, app_dir() / "models" / "clip")
            found = await run.io_bound(self.trip.propose_themes, vectors, ai,
                                       lambda f, m: state.update(share=f, step=m))
        except Exception as exc:
            traceback.print_exc()
            ui.notify(f"Could not suggest themes: {exc}", type="negative", multi_line=True)
            return
        finally:
            self.ai_busy = False
            timer.cancel()
            note.dismiss()
        self.theme_edits = {}
        ui.notify(f"{len(found)} themes suggested. Keep, rename or merge them, then Use these themes.",
                  type="positive")
        self.themes_view.refresh()

    def theme_review(self, proposals: list[dict]) -> None:
        """The proposed themes, each with its pictures: keep it or not, rename it, merge it into another, drop a
        sub-theme. Use these themes makes them this trip's themes."""
        edits = getattr(self, "theme_edits", None) or {}
        self.theme_edits = edits
        if not hasattr(self, "theme_review_all"):
            self.theme_review_all = set()
        for n, p in enumerate(proposals):
            edits.setdefault(n, {"keep": True, "name": p["name"], "emoji": p["emoji"], "into": None,
                                 "subs": [True] * len(p["subs"])})
        rows = {r["path"]: r for r in self.all_items}
        with ui.row().classes("w-full items-center gap-3"):
            ui.label("✨ This trip's themes").classes("text-lg font-semibold")
            ui.label("The app put the shots that look alike together and named each group. Untick what you don't "
                     "want, click a name to change it, or merge a group into another. Sub-themes become moments "
                     "inside a theme.").classes("text-sm text-gray-600 flex-1")
        for n, p in enumerate(proposals):
            e = edits[n]
            with ui.card().classes("w-full p-2" + ("" if e["keep"] else " opacity-50")):
                with ui.row().classes("w-full items-center gap-2 no-wrap"):
                    ui.checkbox(value=e["keep"], on_change=lambda ev, e=e: (e.update(keep=ev.value),
                                                                             self.themes_view.refresh()))
                    ui.input(value=e["emoji"], on_change=lambda ev, e=e: e.update(emoji=ev.value)) \
                        .props("dense borderless").classes("w-10 text-xl")
                    ui.input(value=e["name"], on_change=lambda ev, e=e: e.update(name=ev.value)) \
                        .props("dense outlined").classes("w-64 font-semibold")
                    ui.label(f"{len(p['paths'])} shots").classes("text-sm text-gray-600")
                    if p.get("from") == "earlier":
                        ui.badge("like an earlier trip's theme", color="indigo-3")
                    ui.space()
                    others = {m: edits[m]["name"] for m in range(len(proposals)) if m != n}
                    ui.select({None: "—", **others}, value=e["into"], label="Merge into",
                              on_change=lambda ev, e=e: (e.update(into=ev.value), self.themes_view.refresh())) \
                        .props("dense outlined").classes("w-56")
                every = n in self.theme_review_all
                shown = list(dict.fromkeys([*p["sample"], *p["paths"]])) if every else p["sample"][:8]
                with ui.row().classes("w-full gap-1 " + ("" if every else "no-wrap overflow-hidden")):
                    for path in shown:
                        it = rows.get(path)
                        if it:
                            with ui.element("div").classes("flex-none relative").style("width:120px") as tile:
                                self.preview(it, "width:120px;height:76px", start=self.hover_start(it), view="prop")
                                with ui.element("div").classes("absolute top-1 right-1").on("click.stop", lambda: None):
                                    with ui.button(icon="swap_horiz").props(
                                            "round dense size=xs color=white text-color=primary").classes("shadow") \
                                            .tooltip("Move to another theme"):
                                        with ui.menu():
                                            for m, q in enumerate(proposals):
                                                if m != n:
                                                    ui.menu_item(f"{edits[m]['emoji']} {edits[m]['name']}",
                                                                 on_click=lambda path=path, m=m: self.move_proposed(path, m))
                                if path in (p.get("moved_in") or []):
                                    ui.label("moved here").classes("absolute bottom-0 left-0 text-[10px] px-1 "
                                                                   "bg-green-600 text-white rounded")
                            tile.on("dblclick", lambda path=path: self.shot_window(path, "Themes"))
                if len(p["paths"]) > 8:
                    ui.button(f"Show all {len(p['paths'])} shots" if not every else "Show fewer",
                              on_click=lambda n=n: (self.theme_review_all.symmetric_difference_update({n}),
                                                    self.themes_view.refresh())) \
                        .props("flat dense no-caps size=sm")
                if p["subs"]:
                    with ui.row().classes("w-full items-center gap-1"):
                        ui.label("Sub-themes:").classes("text-xs text-gray-600")
                        for k, sub in enumerate(p["subs"]):
                            on = e["subs"][k]
                            ui.chip(f"↳ {sub['name']} · {len(sub['paths'])}", removable=on, selectable=False,
                                    color="purple-1" if on else "grey-3",
                                    on_click=(None if on else lambda e=e, k=k: (e["subs"].__setitem__(k, True),
                                                                                self.themes_view.refresh()))) \
                                .props("dense" + ("" if on else " outline")) \
                                .on("remove", lambda e=e, k=k: (e["subs"].__setitem__(k, False),
                                                                self.themes_view.refresh()))
        kept = sum(1 for e in edits.values() if e["keep"] and e["into"] is None)
        with ui.row().classes("w-full items-center gap-2 mt-2"):
            ui.button(f"Use these {kept} themes", icon="check", on_click=self.use_trip_themes) \
                .props("unelevated no-caps color=positive")
            ui.button("Cancel (keep the old themes)", on_click=self.cancel_trip_themes).props("flat no-caps")
            ui.label("Shots of themes you untick get a suggestion from the kept ones (or Other).") \
                .classes("text-xs text-gray-500")

    def move_proposed(self, path: str, to: int) -> None:
        self.trip.move_in_proposals(path, to)
        ui.notify(f"Moved to {self.theme_edits.get(to, {}).get('name', 'that theme')}.", type="positive")
        self.themes_view.refresh()

    def cancel_trip_themes(self) -> None:
        self.trip.drop_theme_proposals()
        self.theme_edits = {}
        self.themes_view.refresh()

    async def use_trip_themes(self) -> None:
        proposals = self.trip.theme_proposals()
        edits = getattr(self, "theme_edits", {}) or {}
        chosen: dict[int, dict] = {}
        for n, p in enumerate(proposals):
            e = edits.get(n) or {"keep": True, "name": p["name"], "emoji": p["emoji"], "into": None,
                                 "subs": [True] * len(p["subs"])}
            target = n
            seen = set()
            while edits.get(target, {}).get("into") is not None and target not in seen:  # (merged into a merged one)
                seen.add(target)
                target = edits[target]["into"]
            if not (edits.get(target) or e)["keep"]:
                continue
            te = edits.get(target) or e
            name = " ".join((te["name"] or "").split())[:30] or proposals[target]["name"]
            into = chosen.setdefault(target, {"name": name, "emoji": (te["emoji"] or "⭐").strip()[:4],
                                              "describe": list(proposals[target]["describe"]), "paths": [],
                                              "subs": [], "moved_in": []})
            into["paths"] += p["paths"]
            into["moved_in"] += p.get("moved_in") or []
            into["subs"] += [s for k, s in enumerate(p["subs"]) if k < len(e["subs"]) and e["subs"][k]]
        if not chosen:
            ui.notify("Keep at least one theme.", type="warning")
            return
        names = [c["name"].lower() for c in chosen.values()]
        if len(set(names)) < len(names):
            ui.notify("Two themes have the same name: rename one, or merge them.", type="warning")
            return
        note = ui.notification("Setting up this trip's themes…", spinner=True, timeout=None)
        for c in chosen.values():  # what the name looks like to the picture model
            c["vector"] = None
            if not self.trip.has_fingerprints():
                continue
            try:
                v = await run.io_bound(text_vectors, [f"a holiday photo of {d}" for d in
                                                      [c["name"].lower()] + c["describe"]], app_dir() / "models" / "clip")
                mean = v.mean(axis=0)
                c["vector"] = [round(float(x), 5) for x in mean / np.linalg.norm(mean)]
            except Exception:
                traceback.print_exc()
        try:
            await run.io_bound(self.trip.use_theme_proposals, list(chosen.values()))
        finally:
            note.dismiss()
        self.theme_edits = {}
        self.theme_shown = ""
        self.reload()
        ui.notify(f"{len(chosen)} themes for this trip. The clearest shots of each are confirmed; check the rest "
                  "(? marks a guess). Sub-themes are moments inside their theme.", type="positive", multi_line=True,
                  timeout=8000)
        self.themes_view.refresh()

    def theme_grid(self, rows: list[dict], activities: list[dict]) -> None:
        name = self.theme_shown
        if self.theme_moment and self.trip.moment(name, self.theme_moment):
            self.moment_grid(name, self.theme_moment, activities)
            return
        emoji = {a["name"]: a["emoji"] for a in activities}
        shown = [r for r in rows if r["theme"] == name]
        todo = [r for r in shown if not r["theme_ok"]]
        self.theme_tiles = {}
        self.theme_ticks = {}
        self.theme_rows_shown = {r["path"]: r for r in shown}
        self.theme_selected &= {r["path"] for r in shown}
        with ui.row().classes("w-full items-center gap-2"):
            ui.label(f"{emoji.get(name, '⭐')} {name}").classes("text-lg font-semibold")
            if name != themes.OTHER:
                ui.button("Rename", icon="edit", on_click=lambda: self.rename_dialog(name)) \
                    .props("flat dense no-caps size=sm")
                ui.button("Moment", icon="add", on_click=lambda: self.new_moment_dialog(name)) \
                    .props("flat dense no-caps size=sm") \
                    .tooltip("Mark a moment inside these shots, e.g. kite jump: tick a few, the app finds the rest")
            self.theme_summary = ui.label(f"{len(shown)} shots · {len(shown) - len(todo)} confirmed · "
                                          f"{len(todo)} to check").classes("text-sm text-gray-600")
        with ui.row().classes("w-full items-center gap-2"):
            ui.button("All correct", icon="done_all", on_click=lambda: self.confirm_themes([r["path"] for r in todo])) \
                .set_enabled(bool(todo))
            move = ui.select({a["name"]: f"{a['emoji']} {a['name']}" for a in activities if a["name"] != name},
                             label="Move selected to…", on_change=lambda e: e.value and self.move_selected(e.value)) \
                .props("dense outlined").classes("w-56")
            move.set_enabled(bool(self.theme_selected))
            self.theme_move = move
            self.theme_count = ui.label(self.selected_words()).classes("text-sm text-gray-600")
            if name != themes.OTHER and name not in themes.DEFAULTS:
                ui.button("Remove this activity", icon="delete_outline", on_click=lambda: self.remove_activity(name)) \
                    .props("flat dense no-caps size=sm")
        if not shown:
            ui.label("No shots here yet. Move shots here from another activity.").classes("text-gray-500")
            return
        # Unsure ones first (they need you), then in the order they were taken.
        shown.sort(key=lambda r: (r["theme_ok"], (r["theme_sure"] or 0) >= themes.SURE, r["taken_at"]))
        self.nav_order["Themes"] = [r["path"] for r in shown[: self.theme_limit]]
        with ui.element("div").classes("w-full grid gap-2").style("grid-template-columns:repeat(auto-fill,minmax(150px,1fr))"):
            for r in shown[: self.theme_limit]:
                self.theme_tile(r)
        if len(shown) > self.theme_limit:
            ui.button(f"Show more ({len(shown) - self.theme_limit} left)", icon="expand_more",
                      on_click=self.more_themes).props("outline").classes("w-full")

    def theme_tile(self, r: dict) -> None:
        picked = r["path"] in self.theme_selected
        sure = r["theme_ok"] or (r["theme_sure"] or 0) >= themes.SURE
        who = ", ".join(p for p in r["people"] if not p.startswith("Person ")) or \
            ("Nobody" if r["people_tag"] == "No people" else "Someone" if r["people_tag"] else "")
        where = r["place"] or r["country"] or ""
        when = f"{datetime.fromisoformat(r['taken_at']).astimezone():%a %-d %b}, {r['day_part']}"
        with ui.element("div").classes("relative rounded p-1 cursor-pointer border-2 " +
                                       ("border-blue-600 bg-blue-50" if picked else "border-transparent bg-gray-50")) \
                as tile:
            self.click_or_double(tile, lambda p=r["path"]: self.pick_theme_shot(p),
                                 lambda p=r["path"]: self.shot_window(p, "Themes"))
            self.theme_tiles[r["path"]] = tile
            if r["kind"] == "video":  # hover plays it (silently) so you can see what happens in it
                self.preview(r, "width:100%;aspect-ratio:16/10", start=self.hover_start(r), view="theme")
            elif r["thumb"]:
                ui.image(data_url(r["thumb"])).classes("w-full rounded").style("aspect-ratio:16/10") \
                    .props("fit=cover no-spinner")
            # Top left: tick = "right theme" (saved at once). Top right: pick another theme.
            tick = ui.element("div").classes("absolute top-2 left-2 w-7 h-7 rounded border-2 flex items-center "
                                             "justify-center text-base font-bold shadow") \
                .on("click.stop", lambda p=r["path"]: self.tick_theme(p)).tooltip("Right theme: tick to confirm")
            with tick:
                mark = ui.label("")
            self.theme_ticks[r["path"]] = (tick, mark, sure)
            self._show_tick(r["path"], r["theme_ok"])
            with ui.element("div").classes("absolute top-2 right-2").on("click.stop", lambda: None):
                ui.button(icon="swap_horiz", on_click=lambda r=r: self.theme_menu(r)) \
                    .props("round dense size=sm color=white text-color=primary").classes("shadow") \
                    .tooltip("Move to another theme")
            with ui.element("div").classes("absolute top-2 right-11 z-10").on("click.stop", lambda: None):
                ui.button(icon="delete", on_click=lambda p=r["path"]: self.remove_theme_shot(p)) \
                    .props("round dense size=sm color=white text-color=negative").classes("shadow") \
                    .tooltip("Remove from the trip (as in the clean-up; the file stays on your computer)")
            if r["kind"] == "video":
                with ui.element("div").classes("absolute top-11 right-2").on("click.stop", lambda: None):
                    self.lock_area(r, "compact")
            elif r.get("postits"):
                ui.label("⭐").classes("absolute top-11 right-3 text-lg").tooltip("Highlight: always in the video")
            with ui.column().classes("gap-0 text-xs leading-tight mt-1"):
                ui.label(("🎬 " if r["kind"] == "video" else "") + (who or "—")) \
                    .classes("font-semibold text-gray-800 truncate")
                ui.label(where).classes("text-gray-500 truncate")
                ui.label(when).classes("text-gray-500")

    async def remove_theme_shot(self, path: str) -> None:
        """🗑 on a Themes tile: the shot leaves the trip (as in the clean-up) and its tile goes at once."""
        r = self.theme_rows_shown.pop(path, None)
        tile = self.theme_tiles.pop(path, None)
        if tile is not None:
            tile.set_visibility(False)
        self.theme_selected.discard(path)
        if r is not None:
            self.theme_counts[1] -= 1
            self._theme_summary(-1 if r["theme_ok"] else 0)
        ui.notify(f"Removed {Path(path).name} from the trip. To bring it back: Choose files, show 'Removed from the trip'.",
                  type="info", timeout=5)
        await run.io_bound(self.trip.remove, [path])
        self.all_items = [it for it in self.all_items if it.get("path") != path]

    def _show_tick(self, path: str, ok: bool) -> None:
        tick, mark, sure = self.theme_ticks[path]
        looks = {True: ("✓", "bg-green-600 border-green-600 text-white"),
                 False: ("" if sure else "?", "bg-white border-gray-500 text-orange-600" if not sure else
                         "bg-white border-gray-500 text-gray-600")}
        mark.set_text(looks[ok][0])
        tick.classes(replace="absolute top-2 left-2 w-7 h-7 rounded border-2 flex items-center justify-center "
                             "text-base font-bold shadow cursor-pointer " + looks[ok][1])

    async def tick_theme(self, path: str) -> None:
        """Tick: this shot is in the right theme (saved straight away). Tick again to take it back."""
        r = self.theme_rows_shown.get(path)
        if r is None:
            return
        r["theme_ok"] = not r["theme_ok"]
        self._show_tick(path, r["theme_ok"])
        self._theme_summary(1 if r["theme_ok"] else -1)
        if r["theme_ok"]:
            await run.io_bound(self.trip.set_theme, [path], r["theme"])
        else:
            await run.io_bound(self.trip.unconfirm_theme, [path])

    def _theme_summary(self, confirmed: int = 0) -> None:
        self.theme_counts[0] += confirmed
        self.theme_badge.set_text(f"{self.theme_counts[0]} of {self.theme_counts[1]} confirmed")
        rows = list(self.theme_rows_shown.values())
        done = sum(r["theme_ok"] for r in rows)
        self.theme_summary.set_text(f"{len(rows)} shots · {done} confirmed · {len(rows) - done} to check")

    def theme_menu(self, r: dict, on_move=None) -> None:
        """A big window: the shot, and every theme as a button. One click moves it there (and confirms it).
        on_move(dialog, name, new): instead of the Themes grid's own move (e.g. on a moment's page)."""
        move = on_move or (lambda dialog, n, new=False: self.move_one(dialog, r, n, new))
        activities = self.trip.activities()
        with ui.dialog() as dialog, ui.card().classes("w-[760px] max-w-[95vw] gap-3"):
            with ui.row().classes("w-full no-wrap items-start gap-3"):
                if r.get("thumb"):
                    ui.image(data_url(r["thumb"])).classes("w-56 rounded flex-none").style("aspect-ratio:16/10") \
                        .props("fit=cover no-spinner")
                with ui.column().classes("gap-1"):
                    ui.label("Which theme is this?").classes("text-lg font-semibold")
                    ui.label(f"Now: {r['theme']}" + ("" if r["theme_ok"] else " (the app's guess)")) \
                        .classes("text-sm text-gray-600")
                    ui.label("Click the right one: the shot moves there, and the app looks again at the shots you "
                             "haven't confirmed.").classes("text-xs text-gray-500")
            with ui.element("div").classes("w-full grid gap-2").style("grid-template-columns:repeat(3,minmax(0,1fr))"):
                for a in activities:
                    now = a["name"] == r["theme"]
                    ui.button(f"{a['emoji']} {a['name']}", on_click=lambda n=a["name"]: move(dialog, n)) \
                        .props(("unelevated" if now else "outline") + " no-caps").classes("h-12 text-base w-full")
            with ui.row().classes("w-full items-center gap-2"):
                new = ui.input(placeholder="Or a new theme, e.g. Kitesurfing").props("dense outlined").classes("flex-grow")
                ui.button("Add and move here", icon="add",
                          on_click=lambda: move(dialog, " ".join((new.value or "").split())[:30], True)) \
                    .props("outline no-caps")
                ui.button("Cancel", on_click=dialog.close).props("flat no-caps")
        dialog.open()

    async def move_one(self, dialog, r: dict, name: str, new: bool = False) -> None:
        if not name:
            ui.notify("Type the new theme's name first.", type="warning")
            return
        dialog.close()
        if new:
            await self.add_activity(name, quiet=True)
        await run.io_bound(self.trip.set_theme, [r["path"]], name)
        ui.notify(f"Moved to {name}." if name != r["theme"] else f"Confirmed in {name}.", type="positive")
        was_ok = r["theme_ok"]
        if name == r["theme"]:
            r["theme_ok"] = True
            self._show_tick(r["path"], True)
        else:
            self.theme_rows_shown.pop(r["path"], None)
            self.theme_selected.discard(r["path"])
        self._theme_summary(0 if was_ok else 1)
        if name != r["theme"]:
            tile = self.theme_tiles.pop(r["path"], None)
            if tile is not None:
                tile.delete()  # last: the window lived inside it
        if new:
            self.themes_view.refresh()

    def rename_dialog(self, name: str) -> None:
        with ui.dialog() as dialog, ui.card().classes("w-96"):
            ui.label(f"Rename {name}").classes("text-lg font-semibold")
            field = ui.input("New name", value=name).props("outlined autofocus").classes("w-full")
            field.on("keydown.enter", lambda: self.rename_activity(dialog, name, field.value))
            with ui.row().classes("w-full justify-end"):
                ui.button("Cancel", on_click=dialog.close).props("flat no-caps")
                ui.button("Rename", on_click=lambda: self.rename_activity(dialog, name, field.value)).props("no-caps")
        dialog.open()

    async def rename_activity(self, dialog, old: str, new: str) -> None:
        new = " ".join((new or "").split())[:30]
        if not new or new == old:
            dialog.close()
            return
        if not await run.io_bound(self.trip.rename_activity, old, new):
            ui.notify(f"There is already a theme called {new}.", type="warning")
            return
        dialog.close()
        self.theme_shown = new
        ui.notify(f"{old} is now {new}.", type="positive")
        self.themes_view.refresh()

    # ---------- moments (in Themes) ----------

    def mark_moment(self, n: int, m: int, activity: str, name: str) -> None:
        """From My story: open Themes on this moment, ready to tick its shots."""
        name = " ".join((name or "").split())
        if not name:
            ui.notify("Write the moment first, e.g. kite jump.", type="warning")
            return
        self.edit_entry(n, m, moment=name)
        self.trip.add_moment(activity, name)
        self.panels.set_value(self.themes_tab)
        self.open_moment(activity, name)
        self.moment_from_story = n  # the part to go back to

    def back_to_story(self) -> None:
        """Done marking a moment: back to My story, at the part you came from."""
        part, self.moment_from_story = self.moment_from_story, None
        self.panels.set_value(self.storyboard_tab)
        self.storyboard_view.refresh()
        if part is not None:
            ui.timer(0.6, lambda: ui.run_javascript(
                f"const e = document.getElementById('tm-part-{part}'); "
                "if (e) e.scrollIntoView({behavior: 'smooth', block: 'center'});"), once=True)

    def new_moment_dialog(self, activity: str) -> None:
        with ui.dialog() as dialog, ui.card().classes("w-96"):
            ui.label(f"A moment inside {activity}").classes("text-lg font-semibold")
            ui.label("What happens in it? e.g. kite jump, falling in the water, sunset").classes("text-sm text-gray-600")
            names = self.trip.moment_names(activity)
            field = ui.input("Moment", autocomplete=names).props("outlined autofocus").classes("w-full")
            if names:
                ui.label("Used before (click one, or type and press Tab):").classes("text-xs text-gray-500")
                with ui.row().classes("gap-1"):
                    for name in names[:8]:
                        ui.chip(name, on_click=lambda name=name: field.set_value(name)).props("dense clickable outline")

            def go() -> None:
                name = " ".join((field.value or "").split())
                if name:
                    dialog.close()
                    self.trip.add_moment(activity, name)
                    self.open_moment(activity, name)

            field.on("keydown.enter", go)
            with ui.row().classes("w-full justify-end"):
                ui.button("Cancel", on_click=dialog.close).props("flat no-caps")
                ui.button("Mark it", on_click=go).props("no-caps")
        dialog.open()

    def open_moment(self, activity: str, name: str) -> None:
        self.theme_shown = activity
        self.theme_moment = name
        self.theme_selected = set()
        self.themes_view.refresh()  # its grid starts the search the first time

    async def look_for_moment(self, activity: str, name: str) -> None:
        """Look inside the theme's videos (a frame about every second) for the moment, and sort its shots: your
        ticks first, then the ones most like your words and your ticks."""
        if self.ai_busy:
            ui.notify("Still busy with the last request.", type="warning")
            return
        self.ai_busy = True
        done = {"share": 0.0, "step": "starting"}
        note = ui.notification("", spinner=True, timeout=None, multi_line=True)
        started = time.time()

        def tick() -> None:
            gone = int(time.time() - started)
            note.message = (f"🔍 Looking for \"{name}\" inside the {activity} shots: {done['step']}, "
                            f"{int(100 * done['share'])}% ({gone // 60}:{gone % 60:02d}). The first time takes "
                            "longest: the app looks at a frame about every second and remembers it.")

        tick()
        ticker = ui.timer(1.0, tick)
        try:
            try:
                words = (await run.io_bound(text_vectors, [f"a holiday photo of {name}"],
                                            app_dir() / "models" / "clip"))[0]
            except Exception:
                traceback.print_exc()
                words = None  # without the words model: your ticks still work
            found = await run.io_bound(self.trip.moment_matches, activity, name, words,
                                       lambda share, step: done.update(share=share, step=step))
        except Exception as exc:
            traceback.print_exc()
            ui.notify(f"Couldn't look for the moment: {exc}", type="negative", multi_line=True)
            found = []
        finally:
            self.ai_busy = False
            ticker.cancel()
            note.dismiss()
        self.moment_found[(activity, name)] = found
        if self.theme_shown == activity and self.theme_moment == name:
            self.themes_view.refresh()

    def moment_grid(self, activity: str, name: str, activities: list[dict]) -> None:
        emoji = {a["name"]: a["emoji"] for a in activities}
        found = self.moment_found.get((activity, name))
        saved = self.trip.moment(activity, name) or {"shots": []}
        with ui.row().classes("w-full items-center gap-2"):
            ui.label(f"{emoji.get(activity, '⭐')} {activity} › 🔍 {name}").classes("text-lg font-semibold")
            self.moment_count = ui.label(f"{len(saved['shots'])} marked").classes("text-sm text-gray-600")
            if self.moment_from_story is not None:
                ui.button("Done, back to my story", icon="arrow_back", on_click=self.back_to_story) \
                    .props("no-caps color=green").classes("ml-auto")
        ui.label(f"Tick the shots that show “{name}”: the picture is the second that fits best, and that is the part "
                 "the video keeps. After a few ticks, press Find the rest: the shots most like yours come first.") \
            .classes("text-sm text-gray-600")
        with ui.row().classes("w-full items-center gap-2"):
            ui.button("Find the rest", icon="search", on_click=lambda: self.look_for_moment(activity, name)) \
                .tooltip("Sort again: the shots most like the ones you ticked first")
            ui.button(f"Back to {activity}", icon="arrow_back", on_click=lambda: self.show_theme(activity)) \
                .props("flat no-caps")
            ui.button("Remove this moment", icon="delete_outline",
                      on_click=lambda: self.remove_moment(activity, name)).props("flat dense no-caps size=sm")
        if found is None:
            ui.label("Looking inside the videos…").classes("text-gray-500")
            if not self.ai_busy:
                ui.timer(0.1, lambda: self.look_for_moment(activity, name), once=True)
            return
        if not found:
            ui.label("No shots to look in. Analyse the trip first (Choose files, Analyse).").classes("text-gray-500")
            return
        marked = [f for f in found if f["ticked"]]
        suggested = [f for f in found if f["likely"] and not f["ticked"]]
        rest = [f for f in found if not f["likely"] and not f["ticked"]]
        self.nav_order[f"the {name} moments"] = [f["path"] for f in marked + suggested + rest]
        if marked and not suggested:
            with ui.row().classes("w-full items-center gap-3 p-2 rounded bg-green-50"):
                ui.label(f"That's all of them: {len(marked)} marked, nothing else looks like “{name}”.") \
                    .classes("text-sm text-green-800")
                if self.moment_from_story is not None:
                    ui.button("Back to my story", icon="arrow_back", on_click=self.back_to_story) \
                        .props("dense no-caps color=green")
                else:
                    ui.button(f"Back to {activity}", icon="arrow_back", on_click=lambda: self.show_theme(activity)) \
                        .props("dense flat no-caps")
        sections = [(f"✓ Marked ({len(marked)})", marked, ""),
                    (f"Suggested ({len(suggested)}): " + ("they look like your ticks" if marked else
                                                          "they fit your words best"), suggested, ""),
                    (f"Probably not ({len(rest)})", rest, "")]
        shown = 0
        for title, group, _ in sections:
            if not group:
                continue
            with ui.row().classes("w-full items-center gap-2 mt-2"):
                ui.label(title).classes("font-semibold text-gray-700")
                if group is not marked and any(f["kind"] == "video" and f.get("row") for f in group):
                    ui.button("Review one by one", icon="slideshow",
                              on_click=lambda g=group: self.open_moment_big(activity, name, g[0], g)) \
                        .props("dense flat no-caps color=purple") \
                        .tooltip("Each one big at the second found: This is it or Not this, then the next one")
                if group is suggested:
                    ui.button("Tick all suggested", icon="done_all",
                              on_click=lambda g=group: self.tick_all_moment(activity, name, g)).props("dense no-caps")
            room = max(0, self.theme_limit - shown)
            with ui.element("div").classes("w-full grid gap-2").style(
                    "grid-template-columns:repeat(auto-fill,minmax(170px,1fr))"):
                for f in group[:room]:
                    self.moment_tile(activity, name, f, group)
            shown += min(len(group), room)
        if len(found) > shown:
            ui.button(f"Show more ({len(found) - shown} left)", icon="expand_more",
                      on_click=self.more_themes).props("outline").classes("w-full")

    def moment_tile(self, activity: str, name: str, f: dict, group: list[dict] | None = None) -> None:
        with ui.element("div").classes("relative rounded p-1 border-2 bg-gray-50 cursor-pointer " +
                                       ("border-green-600" if f["ticked"] else "border-transparent")) as tile:
            if f["kind"] == "video" and f.get("row"):  # hover plays the seconds around the moment, again and again
                a, b = moments.window(f["t"], f["duration"] or 0, *(kept_part(f["row"]) or (0.0, f["duration"] or 0)))
                self.preview(f["row"] | {"thumb": f["thumb"]}, "width:100%;aspect-ratio:16/10", a, b, view="moment")
            elif f["thumb"]:
                ui.image(data_url(f["thumb"])).classes("w-full rounded").style("aspect-ratio:16/10") \
                    .props("fit=cover no-spinner")
            box = ui.element("div").classes("absolute top-2 left-2 w-7 h-7 rounded border-2 flex items-center "
                                            "justify-center text-base font-bold shadow cursor-pointer")
            with box:
                mark = ui.label("")

            def show() -> None:
                mark.set_text("✓" if f["ticked"] else ("?" if f["likely"] else ""))
                box.classes(replace="absolute top-2 left-2 w-7 h-7 rounded border-2 flex items-center justify-center "
                                    "text-base font-bold shadow cursor-pointer " +
                                    ("bg-green-600 border-green-600 text-white" if f["ticked"] else
                                     "bg-white border-gray-500 text-orange-600"))
                tile.classes(replace="relative rounded p-1 border-2 bg-gray-50 cursor-pointer " +
                                     ("border-green-600" if f["ticked"] else
                                      "border-orange-400 border-dashed" if f["likely"] else "border-transparent"))

            async def toggle() -> None:
                f["ticked"] = not f["ticked"]
                show()
                await run.io_bound(self.trip.tick_moment, activity, name, f["path"], f["t"], f["ticked"])
                saved = self.trip.moment(activity, name) or {"shots": []}
                self.moment_count.set_text(f"{len(saved['shots'])} marked")

            show()
            self.click_or_double(tile, toggle, lambda: self.shot_window(
                f["path"], f"the {name} moments", at=f["t"] if f["kind"] == "video" else None,
                moment={"activity": activity, "name": name, "t": f["t"]}))
            with ui.element("div").classes("absolute top-2 right-2").on("click.stop", lambda: None):
                ui.button(icon="swap_horiz", on_click=lambda: self.moment_theme_menu(activity, name, f)) \
                    .props("round dense size=sm color=white text-color=primary").classes("shadow") \
                    .tooltip("Wrong theme? Move it to another theme")
            with ui.row().classes("w-full items-center justify-between no-wrap"):
                when = ui.label("🎬 at " + fmt_length(f["t"]) if f["kind"] == "video" else "📷 photo") \
                    .classes("text-xs text-gray-600 mt-1")
                if f["kind"] == "video" and f.get("row"):
                    async def use(lo: float, hi: float) -> bool:
                        f["t"], f["ticked"] = round((lo + hi) / 2, 2), True
                        await run.io_bound(self.trip.tick_moment, activity, name, f["path"], f["t"], True)
                        when.set_text("🎬 at " + fmt_length(f["t"]))
                        show()
                        saved = self.trip.moment(activity, name) or {"shots": []}
                        self.moment_count.set_text(f"{len(saved['shots'])} marked")
                        ui.notify(f"Moment set at {fmt_length(f['t'])} and ticked.", type="positive")
                        return True

                    def big() -> None:
                        a, b = moments.window(f["t"], f["duration"] or 0, *(kept_part(f["row"]) or
                                                                            (0.0, f["duration"] or 0)))
                        self.part_dialog(f["row"], a, b, f"{name}: find the exact moment",
                                         [("This is the moment", "check", use, "no-caps color=green")])

                    with ui.element("div").on("click.stop", lambda: None):
                        ui.button(icon="open_in_full", on_click=big).props("flat dense round size=sm") \
                            .tooltip("Watch it big and choose the exact seconds")

    def moment_theme_menu(self, activity: str, name: str, f: dict) -> None:
        """Move a shot on a moment's page to another theme (it leaves this moment, which belongs to its theme)."""
        rows = {r["path"]: r for r in self.trip.theme_rows()}
        r = rows.get(f["path"])
        if not r:
            return

        async def move(dialog, new_theme: str, new: bool = False) -> None:
            if not new_theme:
                ui.notify("Type the new theme's name first.", type="warning")
                return
            dialog.close()
            if new:
                await self.add_activity(new_theme, quiet=True)
            await run.io_bound(self.trip.set_theme, [r["path"]], new_theme)
            if new_theme != activity and f.get("ticked"):
                await run.io_bound(self.trip.tick_moment, activity, name, f["path"], f["t"], False)
            ui.notify(f"Moved to {new_theme}." if new_theme != r["theme"] else f"Confirmed in {new_theme}.",
                      type="positive")
            self.themes_view.refresh()

        self.theme_menu(r, move)

    def open_moment_big(self, activity: str, name: str, f: dict, group: list[dict]) -> None:
        """Double-click a found moment: it opens big at the very second the app found, with Keep / Not this and
        the next one of the same row (Marked, Suggested or Probably not). A photo opens in the shot window."""
        if f["kind"] != "video" or not f.get("row"):
            self.shot_window(f["path"], f"the {name} moments")
            return
        items = [x for x in group if x["kind"] == "video" and x.get("row")]

        async def keep(x: dict, t: float) -> None:
            x["t"], x["ticked"] = t, True
            await run.io_bound(self.trip.tick_moment, activity, name, x["path"], t, True)

        async def drop(x: dict) -> None:
            if x["ticked"]:
                x["ticked"] = False
                await run.io_bound(self.trip.tick_moment, activity, name, x["path"], x["t"], False)
            x["likely"] = False

        self.suggestion_window(list(items), items.index(f) if f in items else 0, f"{name}", keep, drop,
                               self.themes_view.refresh)

    async def tick_all_moment(self, activity: str, name: str, group: list[dict]) -> None:
        for f in group:
            if not f["ticked"]:
                f["ticked"] = True
                await run.io_bound(self.trip.tick_moment, activity, name, f["path"], f["t"], True)
        ui.notify(f"{len(group)} ticked. Untick any that are wrong, or press Find the rest for more.", type="positive")
        self.themes_view.refresh()

    def remove_moment(self, activity: str, name: str) -> None:
        self.trip.remove_moment(activity, name)
        self.moment_found.pop((activity, name), None)
        self.show_theme(activity)

    def show_theme(self, name: str) -> None:
        self.theme_shown = name
        self.theme_moment = None
        self.theme_selected = set()
        self.theme_limit = self.THEME_PAGE
        self.themes_view.refresh()

    def more_themes(self) -> None:
        self.theme_limit += self.THEME_PAGE
        self.themes_view.refresh()

    def selected_words(self) -> str:
        return (f"{len(self.theme_selected)} selected" if self.theme_selected else
                "Tick the right ones, ⇄ moves one, or click shots to select several. ? marks a guess worth a look.")

    def pick_theme_shot(self, path: str) -> None:
        """Select or unselect a shot (only that tile changes: quick on a big trip)."""
        self.theme_selected ^= {path}
        tile = self.theme_tiles.get(path)
        if tile is not None:
            on = path in self.theme_selected
            tile.classes(add="border-blue-600 bg-blue-50" if on else "border-transparent bg-gray-50",
                         remove="border-transparent bg-gray-50" if on else "border-blue-600 bg-blue-50")
        self.theme_count.set_text(self.selected_words())
        self.theme_move.set_enabled(bool(self.theme_selected))

    async def suggest_themes(self) -> None:
        if self.trip and self.trip.has_fingerprints():
            await self.activity_vectors()
        await run.io_bound(self.trip.suggest_themes)
        self.themes_view.refresh()

    async def activity_vectors(self) -> None:
        """What each activity's words look like to CLIP (once; the words never change)."""
        activities = self.trip.activities()
        missing = [a for a in activities if a.get("vector") is None and a["describe"]]
        if not missing:
            return
        try:
            for a in missing:
                vectors = await run.io_bound(text_vectors, [f"a holiday photo of {d}" for d in a["describe"]],
                                             app_dir() / "models" / "clip")
                mean = vectors.mean(axis=0)
                a["vector"] = [round(float(x), 5) for x in mean / np.linalg.norm(mean)]
            self.trip.save_activities(activities)
        except Exception:
            traceback.print_exc()  # without the picture model: tags and your examples still work

    async def confirm_themes(self, paths: list[str]) -> None:
        await run.io_bound(self.trip.set_theme, paths, None)
        ui.notify(f"{len(paths)} shot{'s' * (len(paths) != 1)} confirmed. The app looked again at the rest, "
                  "comparing them with the shots you confirmed.", type="positive")
        self.theme_selected = set()
        self.themes_view.refresh()

    async def move_selected(self, activity: str) -> None:
        paths = sorted(self.theme_selected)
        if not paths:
            return
        await run.io_bound(self.trip.set_theme, paths, activity)
        ui.notify(f"Moved {len(paths)} to {activity}. The app looked again at the shots you haven't confirmed.",
                  type="positive")
        self.theme_selected = set()
        self.themes_view.refresh()

    async def add_activity(self, name: str, quiet: bool = False) -> None:
        name = " ".join((name or "").split())[:30]
        if not name:
            ui.notify("Type the activity's name first, e.g. Kitesurfing.", type="warning")
            return
        vector = None
        if self.trip.has_fingerprints():
            try:
                v = await run.io_bound(text_vectors, [f"a holiday photo of {name.lower()}"], app_dir() / "models" / "clip")
                vector = [round(float(x), 5) for x in v[0]]
            except Exception:
                traceback.print_exc()
        await run.io_bound(self.trip.add_activity, name, vector)
        if quiet:
            return
        self.theme_shown = name
        self.theme_selected = set()
        ui.notify(f"{name} added. The app put the shots that look like it here; move more in and it learns from them.",
                  type="positive", multi_line=True)
        self.themes_view.refresh()

    async def remove_activity(self, name: str) -> None:
        await run.io_bound(self.trip.remove_activity, name)
        self.theme_shown = ""
        ui.notify(f"{name} removed: its shots got a new suggestion.")
        self.themes_view.refresh()

    async def toggle_voice(self, button) -> None:
        """Dictation by the browser itself (Safari and Chrome on a Mac, free): your words appear in the box as you
        speak."""
        if not self.recording:
            self.voice_base = (self.wish_box.value or "").strip() if self.wish_box is not None else ""
            try:
                result = await ui.run_javascript("tmDictateStart()", timeout=10)
            except Exception as exc:
                result = str(exc)
            if result != "ok":
                self.voice_problem(result)
                return
            self.recording = True
            button.set_text("Stop listening")
            button.props("color=negative")
            ui.notify("Listening… Speak, and your words appear in the box. Press Stop listening when you're done.",
                      type="info")
        else:
            self.listening_stopped()
            ui.run_javascript("tmDictateStop()")

    VOICE_PROBLEMS = {
        "unsupported": "This browser can't turn speech into text. Open the app in Safari or Chrome, or use the "
                       "Mac's own dictation: click in the box and press the 🎤 key (or Fn twice).",
        "not-allowed": "The browser was not allowed to use the microphone. Allow it for this page (in Safari: Safari "
                       "menu > Settings for 127.0.0.1 > Microphone > Allow; in Chrome: the icon left of the address), "
                       "then try again.",
        "service-not-allowed": "Speech recognition is switched off on this Mac. Switch on System Settings > Keyboard "
                               "> Dictation, then try again (or press the 🎤 key in the box).",
        "audio-capture": "No microphone was found. Check System Settings > Sound > Input.",
        "network": "Speech recognition needs the internet in this browser. Check the connection, or use the Mac's "
                   "own dictation: click in the box and press the 🎤 key (or Fn twice).",
        "no-speech": "No words were heard. Is the right microphone on (System Settings > Sound > Input)?",
        "language-not-supported": "Speech recognition doesn't support this language here. Try Safari or Chrome.",
    }

    def listening_stopped(self) -> None:
        """The browser stopped listening: the button says Speak your idea again (what you said stays in the box)."""
        self.recording = False
        if self.talk_button is not None:
            self.talk_button.set_text("Speak your idea")
            self.talk_button.props("color=primary")

    def voice_problem(self, code: str) -> None:
        text = self.VOICE_PROBLEMS.get(code, f"Speech recognition stopped ({code}). Try again, or use the Mac's own "
                                              "dictation: click in the box and press the 🎤 key (or Fn twice).")
        ui.notify(text, type="warning", multi_line=True, timeout=15000)

    def dictation(self, e) -> None:
        args = e.args if isinstance(e.args, dict) else {}
        if args.get("error"):
            if self.recording and args["error"] != "aborted":
                self.voice_problem(args["error"])
            if args["error"] not in ("no-speech", "aborted"):
                self.listening_stopped()
            return
        if args.get("ended"):
            self.listening_stopped()
            return
        if self.wish_box is not None and "text" in args:
            self.wish_box.set_value((self.voice_base + " " + args["text"]).strip())

    def undo(self) -> None:
        if self.busy():
            return
        label = self.trip.undo()
        if label:
            ui.notify(f"Undone: {label}. Everything is back as it was before.", type="positive")
        self.sync_video_size()
        self.refresh_all()

    def sync_video_size(self) -> None:
        """The Video size in Make video follows the trip's (a wish for Instagram makes it upright)."""
        if not getattr(self, "resolution", None) or not self.trip:
            return
        if self.trip.video_size() == "phone" and self.resolution.value != PHONE:
            self.resolution.set_value(PHONE)
        elif self.trip.video_size() == "landscape" and self.resolution.value == PHONE:
            self.resolution.set_value("Full HD (1080p)")

    def video_size_changed(self, value: str) -> None:
        if self.trip:
            self.trip.set_video_size("phone" if value == PHONE else "landscape")

    def limits_now(self) -> AutoLimits:
        """Auto-edit's settings: the song's style sets the pace; the rest are yours (More settings)."""
        limits = AutoLimits(**asdict(self.auto_limits))
        music = self.music_path()
        song = self.song_maps.get(music) if music else None
        if song:
            style = musicmood.reading(song, music, app_dir() / "music-taps")["style"]
            for key, value in self.PACES[musicmood.STYLES[style]["pace"]].items():
                setattr(limits, key, value)
        return limits

    def length_now(self) -> object:
        """How long the video is, in seconds, or "song": as long as your (trimmed) track, unless you set your own
        length on the Storyboard. Nothing else changes it."""
        own = self.trip.own_length() if self.trip else None
        if own:
            return own
        return "song" if self.music_path() else 180

    def track_seconds(self, music: str | None) -> float | None:
        """How long the track is (the part you trimmed, if you did): the video's length unless you name another."""
        if not music:
            return None
        song = self.song_maps.get(music)
        if song:
            return song.duration
        if music not in self.track_lengths:
            try:
                self.track_lengths[music] = song_length(music)
            except Exception:
                return None
        return self.track_lengths[music]

    def track_words(self, music: str) -> str:
        seconds = self.track_seconds(music)
        what = "your trimmed track" if song_taps.trimmed_from(music, app_dir() / "music-taps") else "the whole song"
        return f"{fmt_length(seconds)}, {what}" if seconds else what

    def length_words(self) -> str:
        music = self.music_path()
        own = self.trip.own_length() if self.trip else None
        if own:
            track = f"; the track is {fmt_length(self.track_seconds(music) or 0)}" if music else ""
            return f"{fmt_length(own)} (your own length{track})"
        if music:
            return self.track_words(music)
        return "3:00 for now (choose a song in Music & audio: the video then lasts as long as its track)"

    def video_length_line(self, music: str) -> None:
        """In Music & audio: the track sets how long the video is (unless you set your own on the Storyboard)."""
        own = self.trip.own_length() if self.trip else None
        text = f"🎬 Video length: {self.length_words()}"
        if own:
            text += ". To go back to the track's length, switch off \"Use a different length\" on the Storyboard."
        else:
            text += (". Trim again to change it." if song_taps.trimmed_from(music, app_dir() / "music-taps")
                     else ". Trim the song above to make the video shorter or longer.")
        ui.label(text).classes("font-medium text-green-800 mt-2")

    def own_length_switched(self, on: bool) -> None:
        self.own_length_open = on
        if not on and self.trip.own_length():
            self.trip.set_own_length(None)
            ui.notify(f"Video length: {self.length_words()}.", type="positive")
        self.storyboard_view.refresh()
        self.music_tab_view.refresh()

    def set_own_length(self, text: str) -> None:
        m = re.fullmatch(r"\s*(\d{1,2}):([0-5]\d)\s*", text or "")
        if not m:
            ui.notify("Write the length as minutes:seconds, for example 1:05 or 0:45.", type="warning")
            return
        seconds = int(m.group(1)) * 60 + int(m.group(2))
        if seconds < 5:
            ui.notify("Make it at least 0:05.", type="warning")
            return
        self.trip.set_own_length(seconds)
        ui.notify(f"Video length: {self.length_words()}. Press Make a new cut to use it.", type="positive")
        self.storyboard_view.refresh()
        self.music_tab_view.refresh()

    def auto_edit_settings(self, limits: AutoLimits, pins: list[float] | None) -> None:
        with ui.row().classes("w-full gap-8"):
            for key, label, lo, hi in (("max_video", "Longest video clip", 1.0, 10.0),
                                       ("max_photo", "Longest photo", 0.5, 6.0),
                                       ("min_shot", "Shortest shot", 0.5, 3.0)):
                with ui.column().classes("min-w-48 flex-grow gap-0"):
                    ui.label(label).classes("text-sm text-gray-600")
                    ui.slider(min=lo, max=hi, step=0.5, value=getattr(limits, key),
                              on_change=lambda e, k=key: self.set_limit(k, e.value)).props("label-always").classes("mt-6")
        with ui.row().classes("w-full items-center gap-3"):
            beats = {k: v for k, v in CUT_ON.items() if k != "taps"}  # your pins always lead when there are some
            meaning = {
                "strong": "The hardest hits (drums, accents), keeping the pace of each part of the song.",
                "bass": "The kick drum and bass notes.",
                "changes": "Few cuts: shots held as long as you allow, cut on the biggest moments.",
                "beats": "A steady count of beats, lined up with the bars.",
            }
            choice = ui.select(beats, value=limits.cut_on if limits.cut_on in beats else "strong",
                               label="Without pins, cut on" if not pins else "After your last pin, cut on").classes("w-56")
            note = ui.label(meaning.get(choice.value, "")).classes("text-sm text-gray-600")
            choice.on_value_change(lambda e: (self.set_limit("cut_on", e.value), note.set_text(meaning.get(e.value, ""))))
        ui.switch("Speed effects: slow motion in calm, cinematic moments; slow shots sped up in lively music",
                  value=limits.speed_effects, on_change=lambda e: self.set_limit("speed_effects", e.value))
        ui.switch("Camera effects: zoom transitions, punch-ins and shakes on big hits, speed ramps, slow push-ins, "
                  "freeze-frames introducing the people you named",
                  value=limits.camera_effects, on_change=lambda e: self.set_limit("camera_effects", e.value))
        ui.label(
            "Cuts land on the beat: quick shots in the lively parts, longer ones in the calm parts, and a cut at every "
            "change in the music. Action clips go to lively music, calm and cinematic clips and photos to calm music, "
            "following your storyline, each at its best moment."
        ).classes("text-sm text-gray-600")

    def set_limit(self, key: str, value) -> None:
        setattr(self.auto_limits, key, value)
        save_settings(limits=asdict(self.auto_limits))

    @ui.refreshable
    def music_line(self) -> None:
        music = self.music_path()
        with ui.row().classes("w-full items-center gap-3"):
            ui.label("Music:").classes("font-medium")
            ui.label(Path(music).name if music else "none (choose a song in Music & audio)").classes("text-gray-700")
            ui.button("Music & audio", icon="graphic_eq", on_click=lambda: self.panels.set_value(self.music_tab)) \
                .props("flat dense no-caps")

    def music_changed(self) -> None:
        if self.trip and self.music_path():
            self.trip.set_song(self.music_path())  # remembered for the next start
        if hasattr(self, "music_credit"):
            credit = free_music.credit_for(self.music_input.value, app_dir() / "music") if self.music_input.value else None
            self.music_credit.text = f"Free song: {credit}. A credits file is saved next to your video." if credit else ""
        self.storyboard_view.refresh()
        self.music_tab_view.refresh()
        self.music_line.refresh()

    def music_path(self) -> str | None:
        music = (self.music_input.value or "").strip().strip("'\"") if hasattr(self, "music_input") else ""
        return music if music and Path(music).expanduser().is_file() else None

    async def run_auto_edit(self) -> None:
        music = self.music_path()
        s = self.settings_now()
        style = s.transition if TRANSITIONS.get(s.transition, "fade") not in (None, "mixed") else "Fade"
        if self.trip.progress.running:
            return
        if music and music not in self.song_maps:  # the song's style sets the pace: read the music first
            try:
                self.song_maps[music] = await run.io_bound(songmap.analyse, music, app_dir() / "music-maps")
            except Exception:
                traceback.print_exc()
        target = self.length_now()
        limits = self.limits_now()
        note = ui.notification("Making a first cut…", spinner=True, timeout=None)

        def work() -> object:
            length = None if target in ("song", "all") else float(target)
            report = lambda _f, message: setattr(note, "message", message)  # noqa: E731
            if music:
                return self.trip.auto_edit_to_music(music, limits, length, s.transition_seconds, style, report)
            return self.trip.auto_edit_steady(limits, length, s.transition_seconds, style, report)

        try:
            result = await run.io_bound(work)
        except Exception as exc:
            traceback.print_exc()
            note.dismiss()
            ui.notify(f"Auto-edit stopped: {exc}", type="negative", multi_line=True)
            return
        note.dismiss()
        beat = result.beat
        on_beat = f", cuts on the beat ({beat.bpm:.0f} BPM)" if music and beat and beat.clear else ""
        if music and result.pins:
            on_beat = f", cut exactly on your {result.pins} pins"
        if music:
            self.music_tab_view.refresh()  # the pins this cut skipped show grey there
        if music and result.kept_pins:
            total = len(song_taps.final_taps(music, app_dir() / "music-taps") or [])
            on_beat = (f", cut on {len(result.kept_pins)} of your {total} pins (too few shots for all of them; "
                       "Music & audio shows which it skipped)")
        song = self.song_maps.get(music) if music else None
        wanted = song.duration if song and target == "song" else (float(target) if target not in ("song", "all") else None)
        pool = self.trip.auto_edit_pool()
        photos = sum(it["kind"] == "photo" for it in pool)
        short = (f" There weren't enough shots for {fmt_length(wanted)}: your selection has {len(pool) - photos} "
                 f"videos and {photos} photos. Select more, or allow longer clips and photos."
                 if wanted and result.length < wanted - 3 else "")
        unused = len(result.left_out)
        short += (f" {unused} of your selected files didn't fit this length; they stay selected for next time."
                  if unused and not short else "")
        ui.notify(f"First cut ready: {len(result.shots)} shots, {fmt_length(result.length)}{on_beat}. "
                  f"Fine-tune it below, then make the video.{short}", type="positive", multi_line=True, timeout=10000)
        self.trip.set_edit_stage("fitted" if music else "rough")  # Auto-edit already cuts on the pins
        self.story_page = 1
        self.refresh_all()

    def story_row(self, n: int, it: dict, last: bool) -> None:
        # Drag by the handle (so the part sliders inside still work); drop anywhere on another clip.
        with ui.card().classes("w-full p-2") as card:
            card._props.update({
                "id": f"tm-shot-{n}",
                "ondragover": "event.preventDefault();this.style.outline='3px dashed #2563eb'",
                "ondragleave": "this.style.outline=''",
                "ondrop": "event.preventDefault();this.style.outline='';"
                          f"emitEvent('tm_board_drop',{{src:event.dataTransfer.getData('text/plain'),dst:{n}}})"})
            with ui.row().classes("w-full items-center no-wrap gap-3"):
                handle = ui.element("div").classes("flex flex-col items-center px-1 py-3 rounded cursor-grab "
                                                   "hover:bg-blue-50 select-none")
                handle._props.update({
                    "draggable": "true",
                    "ondragstart": f"event.dataTransfer.setData('text/plain','{n}');"
                                   f"event.dataTransfer.setDragImage(document.getElementById('tm-shot-{n}'),20,20);"
                                   f"document.getElementById('tm-shot-{n}').style.opacity=0.5",
                    "ondragend": f"document.getElementById('tm-shot-{n}').style.opacity=1"})
                with handle:
                    ui.icon("drag_indicator").classes("text-2xl text-gray-500")
                    ui.label(str(n)).classes("text-xs text-gray-500")
                handle.tooltip("Drag up or down to move this clip")
                box = None
                if it["thumb"]:
                    start, end = self.kept_part(it) if it["kind"] == "video" else (0, None)
                    with ui.element("div").style("flex:none").tooltip("Double-click: open it big (cut, split into shots, lock, theme)") as wrap:
                        box = self.preview(it, "width:160px;height:100px;flex:none", start, end, view="story")
                    self.click_or_double(wrap, None, lambda p=it["path"]: self.shot_window(p, "the Storyboard"))
                with ui.column().classes("flex-grow gap-1"):
                    place = ", ".join(x for x in (it["place"], it["country"]) if x)
                    icon = "🎬" if it["kind"] == "video" else "📷"
                    part = f" · part {it['path'].rsplit('#', 1)[1]} of this clip" if it.get("part_of") else ""
                    ui.label(f"{icon} {fmt_time(it['taken_at'])}" + (f" · {place}" if place else "") + part).classes("text-sm")
                    if it.get("ai_caption"):
                        ui.label(f"📝 {it['ai_caption']}").classes("text-xs italic text-gray-600")
                    self.title_control(it)
                    if it["kind"] == "video":
                        self.trim_control(it, box)
                        self.lock_area(it, "story")
                        with ui.row().classes("items-center gap-4"):
                            self.sound_control(it)
                            self.speed_control(it)
                        self.effects_control(it)
                    elif it.get("effects") and json.loads(it["effects"]).get("collage"):
                        self.collage_control(it)
                    else:
                        ui.label(f"Photo · {it['seconds']:.1f} s" if it["seconds"] else "Photo").classes("text-xs text-gray-500")
                with ui.column().classes("gap-1 w-56"):
                    if not last:
                        why = ui.label()
                        ui.select(
                            [DEFAULT_TRANSITION, *TRANSITIONS],
                            value=it["transition"] or DEFAULT_TRANSITION,
                            label="Transition after",
                            on_change=lambda e, p=it["path"], why=why: self.choose_transition(p, e.value, why),
                        ).props("dense options-dense").classes("w-full").move(target_index=0)
                        why.text = f"💡 {it['transition_why']}" if it.get("transition_why") else ""
                        why.classes("text-xs text-gray-500")
                    ui.button("Leave out", icon="remove_circle_outline", on_click=lambda p=it["path"]: self.leave_out(p)).props("flat dense size=sm")

    def title_control(self, it: dict) -> None:
        """The words over this shot (yours, or the automatic opening, story or end title), and a button to change them."""
        shown = getattr(self, "shown_titles", {}).get(it["path"])
        with ui.row().classes("items-center gap-2 no-wrap"):
            if shown:
                kind = {"opening": "Opening title", "story": "Story title", "ending": "End card"}.get(shown["auto"], "Your title")
                words = " · ".join(x for x in (shown["text"], shown["sub"]) if x)
                ui.label(f"🔤 {kind}: {words}").classes("text-xs text-indigo-800")
            ui.button("Edit title" if shown else "Add title", icon="title",
                      on_click=lambda: self.edit_title(it, shown)).props("flat dense size=sm no-caps")

    def edit_title(self, it: dict, shown: dict | None) -> None:
        own = bool(shown and shown["auto"] is None)
        with ui.dialog() as dialog, ui.card().classes("w-full max-w-lg"):
            ui.label("Words over this shot").classes("text-lg font-semibold")
            if shown and shown["auto"]:
                ui.label("This shot has an automatic title. What you write here shows instead. (Switch the automatic "
                         "titles off or change them in Make video.)").classes("text-xs text-gray-600")
            text = ui.input("Title", value=shown["text"] if shown else "").classes("w-full")
            sub = ui.input("Second line (optional)", value=shown["sub"] if shown else "").classes("w-full")
            with ui.row().classes("w-full gap-3"):
                style = ui.select({k: v["label"] for k, v in titles.STYLES.items()},
                                  value=(shown or {}).get("style") or "clean", label="Style").classes("w-48")
                where = ui.select({"": "Style's usual place", **titles.WHERE}, value=(shown or {}).get("where") or "",
                                  label="Where").classes("w-48")
            preview = ui.image().classes("w-full rounded bg-gray-800")

            def show() -> None:
                picture = titles.draw({"text": text.value, "sub": sub.value, "style": style.value, "where": where.value},
                                      640, 360)
                background = self.title_background(it, 640, 360)
                background.alpha_composite(picture)
                preview.set_source(pil_data_url(background.convert("RGB")))

            for box in (text, sub, style, where):
                box.on_value_change(lambda _: show())
            show()
            ui.label("The title fades in and out while this shot is on screen (up to 4 seconds).") \
                .classes("text-xs text-gray-500")
            with ui.row().classes("w-full justify-end gap-2"):
                if own:
                    ui.button("Remove my title", on_click=lambda: (self.save_title(it, "", "", "", ""), dialog.close())) \
                        .props("flat")
                ui.button("Cancel", on_click=dialog.close).props("flat")
                ui.button("Save", on_click=lambda: (self.save_title(it, text.value, sub.value, style.value, where.value),
                                                    dialog.close()))
        dialog.open()

    def title_background(self, it: dict, w: int, h: int):
        from PIL import Image, ImageOps

        try:
            with Image.open(it["thumb"]) as img:
                return ImageOps.fit(img.convert("RGB"), (w, h)).convert("RGBA")
        except (OSError, TypeError, AttributeError, ValueError):
            return Image.new("RGBA", (w, h), (60, 70, 80, 255))

    def save_title(self, it: dict, text: str, sub: str, style: str, where: str) -> None:
        self.trip.set_shot_title(it["path"], text, sub, style, where)
        self.storyboard_view.refresh()

    def choose_transition(self, path: str, value: str, why: ui.label) -> None:
        self.trip.set_transition(path, None if value == DEFAULT_TRANSITION else value)
        why.text = "Your choice"

    def kept_part(self, it: dict) -> tuple[float, float]:
        own = it["trim_start"] is not None
        item = MediaItem(path=Path(it["path"]), kind="video", taken_at=datetime.now(), duration=it["duration"],
                         trim_start=it["trim_start"] if own else it.get("keep_start"),
                         trim_end=it["trim_end"] if own else it.get("keep_end"))
        return clip_range(item, self.settings_now())

    def sound_control(self, it: dict, on_change=None) -> None:
        """The clip's own sound: off by default (the music plays alone), with a loudness of 0-200%."""
        if not it["has_audio"]:
            ui.label("🔇 This clip has no sound").classes("text-xs text-gray-400")
            return
        with ui.row().classes("items-center gap-3 no-wrap"):
            volume = round(100 * (1.0 if it["clip_volume"] is None else it["clip_volume"]))

            def switch(on: bool) -> None:
                self.trip.set_clip_sound(it["path"], on)
                it["clip_sound"] = int(on)
                loudness.set_visibility(on)
                if on_change:
                    on_change()

            def set_volume(value: int) -> None:
                self.trip.set_clip_sound(it["path"], True, value / 100)
                percent.text = f"{value}%"
                if on_change:
                    on_change()

            ui.checkbox("Clip sound", value=bool(it["clip_sound"]), on_change=lambda e: switch(e.value)).props("dense")
            with ui.row().classes("items-center gap-2 no-wrap") as loudness:
                ui.label("Volume").classes("text-xs text-gray-500")
                ui.slider(min=0, max=200, step=5, value=volume, on_change=lambda e: set_volume(int(e.value))).classes("w-40")
                percent = ui.label(f"{volume}%").classes("text-xs text-gray-600 w-10")
            loudness.set_visibility(bool(it["clip_sound"]))

    def trim_control(self, it: dict, box: str | None) -> None:
        start, end = self.kept_part(it)
        kept = {"min": round(start, 1), "max": round(end, 1)}
        label = ui.label().classes("text-xs text-gray-600")

        def show(lo: float, hi: float) -> None:
            speed = (it.get("speed") or 1.0) * source_factor(json.loads(it["effects"]) if it.get("effects") else None)
            frozen = held(json.loads(it["effects"]) if it.get("effects") else None)
            label.text = f"Keeping {fmt_length(lo)} to {fmt_length(hi)} ({hi - lo:.1f} s of {fmt_length(it['duration'])})" + (
                f" · {(hi - lo) / speed + frozen:.1f} s on screen" if abs(speed - 1) > 1e-3 or frozen else "")

        it["_show_trim"] = lambda: show(*self.kept_part(it))

        show(start, end)
        lo, hi = kept_part(it) or (0.0, it["duration"])  # a clip you cut: only the part you kept
        bar = ui.range(
            min=round(lo, 1), max=round(hi, 1), step=0.1, value={"min": round(start, 1), "max": round(end, 1)},
            on_change=lambda e: trimmed(e.value["min"], e.value["max"]),
        ).props("dense drag-range").classes("w-full")

        def trimmed(lo: float, hi: float) -> None:
            clip = self.clip_of(it)
            clash = next((n for x, y, n in self.trip.used_parts(clip, but=it["path"])
                          if lo < y - 0.05 and x + 0.05 < hi), None)
            if clash is not None:  # never the same seconds twice: back to where it was
                ui.notify(f"Shot {clash} already uses some of these seconds of the clip. Choose another part, so "
                          "nothing shows twice.", type="warning", multi_line=True)
                bar.set_value({"min": kept["min"], "max": kept["max"]})
                return
            show(lo, hi)
            self.trip.set_trim(it["path"], lo, hi)
            locked = next((k for k, (a, b) in enumerate(self.trip.locks(clip))
                           if abs(a - kept["min"]) < 0.15 and abs(b - kept["max"]) < 0.15), None)
            if locked is not None:  # a locked part you move stays locked
                self.trip.change_lock(clip, locked, lo, hi, checkpoint=False)
                ui.timer(0.05, lambda: self._locks_changed(clip), once=True)
            # Play the part now kept: from its start, or its last 2 seconds when the end was moved.
            from_ = lo if lo != kept["min"] or hi == kept["max"] else max(lo, hi - 2)
            kept.update(min=lo, max=hi)
            if box:
                ui.run_javascript(f"tmPlayPart('{box}', {lo}, {hi}, {from_})")

    def effects_control(self, it: dict) -> None:
        """The camera-style effect on a shot, if Auto-edit gave it one, with a button to take it off."""
        if not it.get("effects"):
            return
        with ui.row().classes("items-center gap-2") as row:
            ui.label(f"✨ {it.get('effects_why') or 'Camera effect'}").classes("text-xs text-purple-700")

            def remove() -> None:
                self.trip.clear_effects(it["path"])
                it["effects"] = None
                row.set_visibility(False)
                ui.notify("Effect taken off")

            ui.button("Remove", on_click=remove).props("flat dense size=sm no-caps")

    def collage_control(self, it: dict) -> None:
        """A collage (several photos on screen together): small previews of its other photos, and a way back."""
        parts = json.loads(it["effects"])["collage"]
        thumbs = {r["path"]: r["thumb"] for r in self.items if r["path"] in parts}
        with ui.row().classes("items-center gap-2"):
            ui.label(f"🧩 Collage of {len(parts) + 1} photos · {it['seconds'] or 0:.1f} s").classes("text-xs text-purple-700")
            for p in parts:
                if thumbs.get(p):
                    ui.image(data_url(thumbs[p])).style("width:48px;height:32px;object-fit:cover;flex:none")

            def split() -> None:
                self.trip.split_collage(it["path"])
                ui.notify("The photos now show one after another")
                self.storyboard_view.refresh()

            ui.button("Show them one by one", on_click=split).props("flat dense size=sm no-caps")

    def speed_control(self, it: dict) -> None:
        speeds = {0.5: "0.5× slow motion", 0.75: "0.75× slower", 1.0: "1× normal speed", 1.5: "1.5× faster", 2.0: "2× fast"}
        speed = min(speeds, key=lambda v: abs(v - (it.get("speed") or 1.0)))
        column = ui.column().classes("gap-0")
        with column:
            why = ui.label(f"💡 {it['speed_why']}" if it.get("speed_why") else "").classes("text-xs text-gray-500")

        def changed(value: float) -> None:
            self.trip.set_speed(it["path"], value)
            it["speed"] = value
            why.text = ""
            it["_show_trim"]()

        with column:
            ui.select(speeds, value=speed, on_change=lambda e: changed(e.value)).props("dense options-dense") \
                .classes("w-40").move(target_index=0)

    def board_drop(self, e) -> None:
        """A clip dragged onto another one in the Storyboard: it takes that place."""
        args = e.args if isinstance(e.args, dict) else {}
        try:
            src, dst = int(args.get("src")), int(args.get("dst"))
        except (TypeError, ValueError):
            return
        order = getattr(self, "board_order", [])
        if src == dst or not (1 <= src <= len(order) and 1 <= dst <= len(order)) or not self.trip:
            return
        self.move(order[src - 1], dst - src)

    def move(self, path: str, steps: int) -> None:
        self.trip.move(path, steps, self.photo_len.value)  # (the cuts stay on your pins)
        self.storyboard_view.refresh()
        self.editor_card.refresh()

    def reset_order(self) -> None:
        self.trip.reset_order()
        self.storyboard_view.refresh()

    def leave_out(self, path: str) -> None:
        self.trip.set_included([path], False)
        self.refresh_all()

    def go_to_story_page(self, page: int) -> None:
        self.story_page = page
        self.storyboard_view.refresh()

    # ---------- Make video ----------

    def build_video(self) -> None:
        self.video_summary()
        self.editor_card()
        self.looks_card()
        self.titles_card()
        with ui.card().classes("w-full"):
            ui.label("Style").classes("text-lg font-semibold")
            with ui.grid(columns=2).classes("w-full gap-4"):
                self.transition = ui.select(list(TRANSITIONS), value="Fade", label="Transition")
                self.resolution = ui.select(list(RESOLUTIONS), label="Video size",
                                            value=PHONE if self.trip and self.trip.video_size() == "phone" else "Full HD (1080p)",
                                            on_change=lambda e: self.video_size_changed(e.value))
                with ui.column():
                    ui.label("Transition length (seconds)").classes("text-sm text-gray-500")
                    self.t_len = ui.slider(min=0.3, max=2.0, step=0.1, value=0.8).props("label-always").classes("mt-8")
            # The song is chosen in the Music tab; this field only holds it.
            self.music_input = ui.input(on_change=self.music_changed).classes("hidden")
            self.music_line()
            self.music_credit = ui.label("").classes("text-xs text-gray-500")
            with ui.column().classes("w-full"):
                ui.label("Music volume").classes("text-sm text-gray-500")
                self.music_vol = ui.slider(min=0.1, max=1.0, step=0.05, value=0.5).props("label-always").classes("mt-8")

            with ui.expansion("Only for a video made without Auto-edit", icon="tune").classes("w-full").props("dense"):
                ui.label("Auto-edit sets each shot's length itself; these are used only for shots it has not timed.") \
                    .classes("text-xs text-gray-500")
                with ui.grid(columns=2).classes("w-full gap-4"):
                    with ui.column():
                        ui.label("Each photo shows for (seconds)").classes("text-sm text-gray-500")
                        self.photo_len = ui.slider(min=2, max=8, step=0.5, value=4).props("label-always").classes("mt-8")
                    with ui.column():
                        ui.label("Longest part of each video (seconds, 0 = whole video)").classes("text-sm text-gray-500")
                        self.clip_len = ui.slider(min=0, max=60, step=1, value=10).props("label-always").classes("mt-8")

        with ui.card().classes("w-full"):
            self.name_input = ui.input("Video name", placeholder="My trip").classes("w-full")
            self.make_button = ui.button("Make my video", icon="movie", on_click=self.make).props("size=lg")
            self.render_progress = ui.linear_progress(value=0, show_value=False).classes("w-full")
            self.render_status = ui.label("")
            self.result_box = ui.column().classes("w-full")
            self.render_progress.set_visibility(False)

    # ---------- the editor under the video: draft, fine-tune, finalize ----------

    DRAFT_SCALE = 3  # a draft is a third of the size (Full HD -> 640 x 360): quick to make
    AUTO_PAUSE = 3.0  # seconds without a change before the draft updates by itself

    def set_auto_update(self, on: bool) -> None:
        self.auto_update = on
        save_settings(auto_update=on)
        if not on:
            self.update_due = None

    async def auto_update_tick(self) -> None:
        """Every second: make the draft again once you have paused after a change (when switched on)."""
        if not self.update_due or time.time() < self.update_due or self.draft_busy or self.rendering["busy"]:
            return
        if not self.showing("Make video"):
            return  # it updates when you come back (the button says it is out of date)
        self.update_due = None
        await self.make_draft()

    def card_png(self, card: str, title: dict, w: int, h: int, picture: str | None = None) -> str:
        """A card's words (and bars) as a small picture for the quick player; with picture (a thumbnail): the
        whole card over it, to choose a look."""
        size = (w // self.DRAFT_SCALE // 2 * 2, h // self.DRAFT_SCALE // 2 * 2)
        words = {k: title.get(k) for k in ("text", "sub")}
        key = hashlib.sha1(json.dumps([card, words, size, picture, 2], sort_keys=True).encode()).hexdigest()[:16]
        folder = app_dir() / "drafts" / "titles"
        folder.mkdir(parents=True, exist_ok=True)
        out = folder / f"card-{key}.{'jpg' if picture is not None else 'png'}"
        if not out.exists():
            from PIL import Image
            if picture is not None:
                try:
                    pic = Image.open(picture) if picture else None
                except OSError:
                    pic = None
                titles.card_picture(card, words, pic, *size).save(out, quality=85)
            else:
                titles.card_layer(card, words, *size).save(out)
        return f"/drafts/titles/{out.name}"

    def card_live_html(self, board: list[dict], times: list[dict], w: int, h: int) -> str:
        """The opening and ending cards over the quick player, fading in and out at their time."""
        if not board or not times:
            return ""
        html = ""
        opening = self.trip.opening_card()
        if opening:
            length = max(1.0, min(opening["seconds"], times[0]["length"] - 0.3 if len(times) > 1 else times[0]["length"] - 0.5))
            css = titles.card_css(opening["card"])
            back = f"backdrop-filter:{css};-webkit-backdrop-filter:{css};" if css else ""
            html += (f'<div class="tm-live-title" data-start="0" data-len="{length:.3f}" data-fin="0" '
                     f'data-fout="{min(titles.CARD_FADE, length / 2):.3f}" style="position:absolute;inset:0;opacity:0;'
                     f'pointer-events:none;z-index:999;{back}">'
                     f'<img src="{self.card_png(opening["card"], opening, w, h)}" style="position:absolute;inset:0;'
                     'width:100%;height:100%"></div>')
        closing = self.trip.closing_card()
        if closing:
            end = times[-1]["start"] + times[-1]["length"]
            length = max(1.0, min(closing["seconds"], times[-1]["length"] - 0.3))
            card = closing["card"]
            css = titles.card_css(card) if card else ""
            back = (f"backdrop-filter:{css};-webkit-backdrop-filter:{css};" if css
                    else "background:#000;")
            words = {"text": closing["text"], "sub": closing["sub"], "style": "elegant", "where": "center"}
            picture = (self.card_png(card, closing, w, h) if card
                       else self.live_title_png(words, w, h) if closing["text"] or closing["sub"] else "")
            html += (f'<div class="tm-live-title" data-start="{end - length:.3f}" data-len="{length + 1:.3f}" '
                     f'data-fin="{min(titles.CARD_FADE, length / 2):.3f}" data-fout="0" style="position:absolute;'
                     f'inset:0;opacity:0;pointer-events:none;z-index:999;{back}">'
                     + (f'<img src="{picture}" style="position:absolute;inset:0;width:100%;height:100%">' if picture else "")
                     + '</div>')
        tail, words = self.trip.end_card()
        if tail > 0:
            end = times[-1]["start"] + times[-1]["length"]
            card = (words or {}).get("card")
            css = titles.card_css(card) if card else ""
            last = board[-1]
            back = (f"background:#000 url('{data_url(last['thumb'])}') center/cover;filter:{css};"
                    if css and last.get("thumb") else "background:#000;")
            picture = (self.card_png(card, words or {}, w, h) if card
                       else self.live_title_png(words, w, h) if words else "")
            html += (f'<div class="tm-live-title" data-start="{end - END_FADE:.3f}" data-len="{tail + END_FADE + 1:.3f}" '
                     f'data-fin="{END_FADE:.3f}" data-fout="0" style="position:absolute;inset:0;opacity:0;'
                     f'pointer-events:none;z-index:999;overflow:hidden">'
                     f'<div style="position:absolute;inset:-20px;{back}"></div>'
                     + (f'<img src="{picture}" style="position:absolute;inset:0;width:100%;height:100%">' if picture else "")
                     + '</div>')
        return html

    def cards_dialog(self) -> None:
        """Choose the opening and ending cards: their look (six, shown on your first shot), words and length."""
        if not self.trip:
            return
        t = self.trip.title_settings()
        board = self.trip.storyboard()
        first = board[0]["thumb"] if board and board[0].get("thumb") else ""
        last = board[-1]["thumb"] if board and board[-1].get("thumb") else first
        w, h = RESOLUTIONS[self.resolution.value] if getattr(self, "resolution", None) else (1920, 1080)
        state = {"card": t["card"] if t["card"] in titles.CARDS else "",
                 "end_card": t["end_card"] if t["end_card"] in titles.CARDS else ""}
        rooms = {float(v): f"{v:g} s" for v in (2, 3, 4, 5, 6, 8, 10, 12, 15)}

        def save(**values) -> None:
            self.trip.set_title_settings(**values)

        with ui.dialog() as dialog, ui.card().style("width:min(1100px,96vw);max-width:96vw;max-height:96vh") \
                .classes("gap-2 overflow-auto"):
            ui.label("Opening and ending cards").classes("text-lg font-semibold")
            ui.label("Each has its own look, words and length. Auto-edit keeps that many seconds of the song free at "
                     "the start and the end: one shot fills each, with the card over it (a blurred look shows the shot "
                     "behind the words, a black look hides it). Press Auto-edit again after changing a length.") \
                .classes("text-sm text-gray-600")

            draws: dict = {}

            def make_looks(which: str) -> None:  # (one refreshable each: refreshing one must not redraw the other)
                @ui.refreshable
                def draw() -> None:
                    looks(which)
                draws[which] = draw
                draw()

            def looks(which: str) -> None:
                words = ({"text": t["trip_title"], "sub": t["trip_sub"]} if which == "card" else
                         {"text": t["end_text"], "sub": t["end_sub"]})
                none = "No card (title over the shot)" if which == "card" else "Words on black"
                with ui.row().classes("w-full gap-2 no-wrap overflow-x-auto"):
                    for key, spec in [("", {"label": none}), *titles.CARDS.items()]:
                        picked = key == state[which]
                        box = ui.element("div").classes("cursor-pointer rounded overflow-hidden flex-none").style(
                            "width:150px;border:3px solid " + ("#2563eb" if picked else "#e5e7eb"))
                        box.on("click", lambda key=key: pick(which, key))
                        with box:
                            if key:
                                ui.image(self.card_png(key, words, w, h, first if which == "card" else last)) \
                                    .style("width:100%;aspect-ratio:16/9")
                            else:
                                ui.image(data_url(first) if first and which == "card" else "") \
                                    .style("width:100%;aspect-ratio:16/9;background:#000")
                            ui.label(("✓ " if picked else "") + spec["label"]).classes(
                                "text-xs px-2 py-1 " + ("font-semibold text-blue-700" if picked else ""))

            def pick(which: str, key: str) -> None:
                state[which] = key
                t[which] = key
                save(**{which: key})
                draws[which].refresh()

            # The opening
            with ui.column().classes("w-full gap-1 p-2 rounded bg-blue-50"):
                with ui.row().classes("w-full items-end gap-3"):
                    ui.label("🎬 Opening").classes("text-base font-semibold w-24")
                    ui.switch("On", value=t["opening"], on_change=lambda e: save(opening=e.value))
                    first_line = ui.input("Title", value=t["trip_title"]).classes("w-64")
                    second = ui.input("Second line", value=t["trip_sub"]).classes("w-64")

                    def words_changed() -> None:
                        t["trip_title"], t["trip_sub"] = first_line.value, second.value
                        save(trip_title=first_line.value, trip_sub=second.value)
                        draws["card"].refresh()
                    first_line.on("blur", words_changed)
                    second.on("blur", words_changed)
                    open_len = float(t["open_room"] or t["open_seconds"] or 3)
                    ui.select(rooms | ({open_len: f"{open_len:g} s"} if open_len not in rooms else {}),
                              value=open_len, label="Opening length",
                              on_change=lambda e: save(open_room=e.value)).classes("w-36") \
                        .tooltip("Auto-edit keeps this much at the start of the song: one shot (a photo or a long "
                                 "clip) with the card over it. Press Auto-edit again after changing it.")
                ui.label("Opening look").classes("text-xs text-gray-600")
                make_looks("card")
            # The ending
            with ui.column().classes("w-full gap-1 p-2 rounded bg-amber-50"):
                with ui.row().classes("w-full items-end gap-3"):
                    ui.label("🌅 Ending").classes("text-base font-semibold w-24")
                    ui.switch("On", value=t["ending"], on_change=lambda e: save(ending=e.value))
                    end = ui.input("Words", value=t["end_text"]).classes("w-64")
                    end_sub = ui.input("Second line", value=t["end_sub"]).classes("w-64")

                    def end_words() -> None:
                        t["end_text"], t["end_sub"] = end.value, end_sub.value
                        save(end_text=end.value, end_sub=end_sub.value)
                        draws["end_card"].refresh()
                    end.on("blur", end_words)
                    end_sub.on("blur", end_words)
                    end_len = float(t["end_room"] or 0)
                    ui.select({0.0: "No ending card"} | rooms | ({end_len: f"{end_len:g} s"} if end_len not in rooms
                                                                 and end_len else {}),
                              value=end_len, label="Ending length",
                              on_change=lambda e: save(end_room=e.value)).classes("w-36") \
                        .tooltip("Auto-edit keeps this much at the end of the song: the last shot, with the card "
                                 "fading in over it as the music ends. Press Auto-edit again after changing it.")
                ui.label("Ending look").classes("text-xs text-gray-600")
                make_looks("end_card")

            def done() -> None:
                dialog.close()
                self.storyboard_view.refresh()
                self.titles_card.refresh()
                self.editor_card.refresh()
            with ui.row().classes("w-full justify-end"):
                ui.button("Done", icon="check", on_click=done).props("no-caps color=primary")
        dialog.open()

    def live_title_png(self, title: dict, w: int, h: int) -> str:
        """The title as a see-through picture (a small one, like the draft), to lay over the playing draft."""
        size = (w // self.DRAFT_SCALE // 2 * 2, h // self.DRAFT_SCALE // 2 * 2)
        look = {k: title.get(k) for k in ("text", "sub", "style", "where")}
        key = hashlib.sha1(json.dumps([look, size], sort_keys=True).encode()).hexdigest()[:16]
        folder = app_dir() / "drafts" / "titles"
        folder.mkdir(parents=True, exist_ok=True)
        out = folder / f"{key}.png"
        if not out.exists():
            titles.draw(look, *size).save(out)
        return f"/drafts/titles/{out.name}"

    def _draft_files(self) -> tuple[Path, Path]:
        folder = app_dir() / "drafts"
        folder.mkdir(exist_ok=True)
        return folder / f"{self.trip.dir.name}.json", folder

    def draft_info(self) -> dict | None:
        """The last draft of this trip: {"file", "times" [{start, length}], "board" [(path, length)]}."""
        if not self.trip:
            return None
        info, folder = self._draft_files()
        try:
            data = json.loads(info.read_text())
        except (OSError, ValueError):
            return None
        return data if (folder / data["file"]).exists() else None

    def draft_board(self) -> list[tuple[str, float]]:
        photo = self.photo_len.value if getattr(self, "photo_len", None) else 4.0
        # (with what changes how a shot looks: turned, played backwards, cropped, its part)
        return [(r["path"], round(Trip.screen_length(r, photo), 2),
                 f"{r.get('rotation') or 0}|{r.get('reverse') or 0}|{r.get('crop') or ''}|{r.get('trim_start')}|{r.get('speed') or 1}"
                 f"|{r.get('clip_sound') or 0}|{r.get('clip_volume')}|{r.get('transition')}|{r.get('transition_len')}")
                for r in self.trip.storyboard()]

    @ui.refreshable
    def editor_card(self) -> None:
        if not self.trip or not self.items or not self.showing("Make video"):
            return
        draft = self.draft_info()
        board = self.trip.storyboard()
        now = self.draft_board()
        changed = bool(draft) and ([tuple(x) for x in draft["board"]] != now
                                   or draft.get("muted", []) != self.trip.muted_titles())
        rough = self.trip.edit_stage() != "fitted"
        with ui.card().classes("w-full bg-slate-50"):
            with ui.row().classes("w-full items-center gap-3"):
                ui.label("Edit, fit to the music, finalize").classes("text-lg font-semibold")
                ui.label("✂ Rough cut" if rough else "♪ Fitted to the music") \
                    .classes("text-sm " + ("text-orange-700" if rough else "text-green-700"))
            ui.label("1. Open the editor: the whole screen, every clip below as a box as long as it plays. Changes "
                     "play straight away. 2. Rough cut: put the clips in order, change their parts and speeds, "
                     "nothing snaps yet. 3. Fit to the music: one press puts every cut on your pins. "
                     "4. Finalize: the full-quality video.").classes("text-sm text-gray-600")
            with ui.row().classes("items-center gap-2"):
                ui.button("Open the editor", icon="fullscreen", on_click=lambda: self.open_editor(True)) \
                    .props("unelevated color=indigo no-caps").tooltip("Full screen (Esc or ✕ closes it)")
                self.draft_button = ui.button("Exact preview", icon="bolt", on_click=self.make_draft) \
                    .props("outline no-caps").tooltip("A small video made like the finished one (with the look and "
                                                      "the transitions); it takes a while")
                ui.button("Finalize the video", icon="movie", on_click=self.finalize).props("color=green") \
                    .tooltip("The full-quality video, saved in Movies › Trip Memory (clips the draft made are not reused: "
                             "they are smaller)")
                self.draft_status = ui.label("").classes("text-sm text-gray-600")
        if self.editor_full:  # (after the card: its status line and button are the editor's own)
            self.editor_window(draft, board, now, changed)

    ZOOMS = {0: "Whole video", 20: "Wide", 40: "Normal", 80: "Close", 160: "Very close"}  # pixels per second

    def open_editor(self, on: bool) -> None:
        self.editor_full = on
        ui.run_javascript(f"document.body.style.overflow = '{'hidden' if on else ''}'")
        if not on:
            ui.run_javascript("for (const k in (window.tmInst || {})) window.tmInst[k].stop()")
        self.editor_card.refresh()

    def set_zoom(self, value: int) -> None:
        self.editor_zoom = value
        self.editor_card.refresh()

    def set_stage(self, stage: str) -> None:
        if stage == self.trip.edit_stage():
            return
        if stage == "fitted":
            self.fit_to_music()
            return
        self.trip.set_edit_stage("rough")
        ui.notify("Rough cut: every clip stays as long as you make it. Fit to the music when the order is right.",
                  type="info")
        self.editor_card.refresh()

    def fit_to_music(self) -> None:
        """One press: every cut of the rough cut moves to the nearest of your pins it can reach, at its own
        speed (see the cut rules in Trip.reflow)."""
        music = self.music_path()
        pins = song_taps.final_taps(music, app_dir() / "music-taps") if music else None
        if not pins:
            ui.notify("Choose a song and put pins on it first (Music & audio): the cuts go on the pins.",
                      type="warning", multi_line=True)
            self.editor_card.refresh()
            return
        photo = self.photo_len.value if getattr(self, "photo_len", None) else 4.0
        seconds = self.t_len.value if getattr(self, "t_len", None) else 0.8
        result = self.trip.fit_to_music(pins, photo, seconds, self.transition_name())
        parts = [f"Fitted to the music: {result['moved']} cut{'s' if result['moved'] != 1 else ''} moved onto your pins"
                 if result["moved"] else "Fitted to the music: every cut was on a pin already"]
        if result["longer"] or result["shorter"]:
            parts.append(f"{result['longer']} clip{'s' if result['longer'] != 1 else ''} longer, "
                         f"{result['shorter']} shorter, all at their own speed")
        if result["gone"]:
            parts.append(f"{result['gone']} clip{'s' if result['gone'] > 1 else ''} past the end of the song "
                         "left the video")
        ui.notify(". ".join(parts) + ". Undo the fit takes it back.", type="positive", multi_line=True, timeout=8000)
        self.reload()
        self.storyboard_view.refresh()
        self.editor_card.refresh()

    def undo_fit(self) -> None:
        if self.trip.undo_label() != "fitting to the music":
            return
        self.trip.undo()
        ui.notify("Back to your rough cut, as it was before the fit.", type="info")
        self.reload()
        self.storyboard_view.refresh()
        self.editor_card.refresh()

    def instant_html(self, box: str, style: str) -> str:
        """The instant player: three layers (the clip playing, the one before it fading out, the next one getting
        ready) and a play button. tmInstant gives it a video's controls."""
        layer = ('<div class="tm-slot" style="position:absolute;inset:0;opacity:0">'
                 '<video muted playsinline preload="auto" style="position:absolute;inset:0;width:100%;height:100%;'
                 'object-fit:contain"></video><img alt="" style="position:absolute;inset:0;width:100%;height:100%;'
                 'object-fit:contain;display:none"><span class="tm-badge" style="position:absolute;right:6px;top:4px;'
                 'font-size:11px;color:#fde68a;text-shadow:0 0 3px #000"></span></div>')
        return (f'<div id="{box}" class="tm-instant" style="position:relative;overflow:hidden;background:#000;'
                f'isolation:isolate;cursor:pointer;{style}" onclick="tmInstantToggle(event, this)">{layer * 3}'
                '<div class="tm-ov" style="position:absolute;inset:0;z-index:890;opacity:0;pointer-events:none"></div>'
                '<div class="tm-iplay" style="position:absolute;left:10px;bottom:10px;z-index:900;width:38px;'
                'height:38px;border-radius:50%;background:rgba(15,23,42,.75);color:white;display:flex;'
                'align-items:center;justify-content:center;font-size:18px;pointer-events:none">▶</div></div>')

    def instant_data(self, board: list[dict], times: list[dict], loop: tuple[float, float] | None = None,
                     tail: float = 0.0) -> dict:
        """What the instant player plays: each clip straight from your file (its part, speed, turn, crop, sound),
        at its place in the video, and the song. tail: seconds after the last clip (the ending card)."""
        clips = []
        for k, (r, at) in enumerate(zip(board, times)):
            file = Path(r.get("part_of") or r["path"])
            thumb = data_url(r["thumb"]) if r.get("thumb") else ""
            try:
                src = self.video_src(file)
            except ValueError:
                src = ""
            crop = json.loads(r["crop"]) if r.get("crop") else None
            clip = {"k": "v" if r["kind"] == "video" else "p", "src": src, "fb": thumb,
                    "st": round(at["start"], 3), "len": round(at["length"], 3), "rot": r.get("rotation") or 0,
                    "crop": crop, "rev": bool(r.get("reverse")),
                    # (fades in over the end of the one before: as long as their transition overlaps them)
                    "fin": round(max(0.0, times[k - 1]["start"] + times[k - 1]["length"] - at["start"]), 3) if k else 0,
                    "tr": transitions.LOOKS.get((board[k - 1].get("transition") or self.transition_name()) if k else "",
                                                "fade")}
            fx = json.loads(r["effects"]) if r.get("effects") else {}
            if fx.get("collage") or fx.get("side"):
                clip["tiles"] = self.instant_tiles(r, fx)
            if r["kind"] == "video":
                clip.update(a=round(r["trim_start"] if r["trim_start"] is not None else (r["keep_start"] or 0.0), 3),
                            sp=r.get("speed") or 1.0, snd=bool(r.get("clip_sound")),
                            vol=1.0 if r.get("clip_volume") is None else float(r["clip_volume"]))
                if not src:
                    clip.update(k="p", src=thumb)
            elif file.suffix.lower() not in (".jpg", ".jpeg", ".png", ".webp", ".gif") or not src:
                clip["src"] = thumb  # (a HEIC photo: the browser shows its preview picture)
            clips.append(clip)
        music = self.music_path()
        total = max([t["start"] + t["length"] for t in times] + [0.1]) + tail
        return {"clips": clips, "total": round(total, 3), "music": serve_song(str(Path(music).expanduser())) if music else "",
                "mvol": self.music_vol.value if getattr(self, "music_vol", None) else 0.5,
                "fade": self.t_len.value if getattr(self, "t_len", None) else 0.8,
                "loop": list(loop) if loop else None}

    def instant_tiles(self, r: dict, fx: dict) -> list[dict]:
        """A collage's photos or the videos side by side, for the instant player: where each goes (shares of the
        picture), when a photo pops in, where a video starts."""
        from .render import collage_tiles, side_tiles

        w, h = RESOLUTIONS[self.resolution.value] if getattr(self, "resolution", None) else (1920, 1080)
        items = {it["path"]: it for it in self.items}
        side = bool(fx.get("side"))
        paths = [r["path"], *(fx.get("side") or fx.get("collage") or [])]
        places = side_tiles(len(paths), w, h) if side else \
            collage_tiles(len(paths), w, h, self.trip.collage_layouts().get(r["path"]) or "")
        framed = self.trip.tile_frames()
        appear = list(fx.get("appear") or [])
        starts = list(fx.get("starts") or [])
        out = []
        for k, (path, (x, y, tw, th)) in enumerate(zip(paths, places)):
            tile = {"x": round(x / w, 4), "y": round(y / h, 4), "w": round(tw / w, 4), "h": round(th / h, 4),
                    "f": framed.get(path) or {}}
            if k:
                it = items.get(path) or {}
                thumb = data_url(it.get("thumb")) if it.get("thumb") else ""
                try:
                    src = self.video_src(Path(path)) if side or Path(path).suffix.lower() in (
                        ".jpg", ".jpeg", ".png", ".webp", ".gif") else thumb
                except ValueError:
                    src = thumb
                tile.update(k="v" if side else "p", src=src or thumb, fb=thumb, rot=it.get("rotation") or 0,
                            at=appear[k] if k < len(appear) else 0.0,
                            a=float(starts[k - 1]) if side and k - 1 < len(starts) else 0.0)
            out.append(tile)
        return out

    def editor_window(self, draft: dict | None, board: list[dict], now: list, changed: bool) -> None:
        """The editor over the whole screen: the player and the clip (or title) you picked on top, the timeline
        below (titles, clips as long as they play, the music with your pins). Only the timeline scrolls.

        The player plays your clips straight from the files (instant: no waiting after a change); the exact
        preview, once made and while nothing changed, plays instead."""
        times = self.board_times(board, draft, changed)
        tail = self.trip.end_card()[0]  # (the ending card after the last clip)
        total = max([t["start"] + t["length"] for t in times] + [1.0]) + tail
        w, h = RESOLUTIONS[self.resolution.value] if getattr(self, "resolution", None) else (1920, 1080)
        resume = getattr(self, "draft_resume", 0.0) or 0.0
        before = [tuple(x) for x in draft["board"]] if draft else None
        exact = bool(draft) and not changed
        rough = self.trip.edit_stage() != "fitted"
        live = "".join(
            f'<img class="tm-live-title" data-id="{b["id"]}" data-start="{b["at"]:.3f}" data-len="{float(b["length"]):.3f}" '
            f'data-fin="{float(b.get("fade_in") or 0):.3f}" data-fout="{float(b.get("fade_out") or 0):.3f}" '
            f'src="{self.live_title_png(b, w, h)}" style="position:absolute;inset:0;width:100%;height:100%;'
            'opacity:0;pointer-events:none;z-index:1000">'
            for b in self.title_bars(board, times) if b["kind"] == "timeline")
        if not exact:  # (the exact preview has the cards in it)
            live += self.card_live_html(board, times, w, h)
        music = self.music_path()
        pins = [p for p in (song_taps.final_taps(music, app_dir() / "music-taps") or []) if p <= total + 0.01] \
            if music else []
        wave = f"/drafts/{draft['file']}" if exact else (serve_song(str(Path(music).expanduser())) if music else "")
        zoom = self.editor_zoom
        width = f"max(100%,{int(total * zoom)}px)" if zoom else "100%"
        with ui.element("div").classes("tm-editor").style(
                "position:fixed;inset:0;z-index:2000;background:#0f172a;color:#e2e8f0;display:flex;"
                "flex-direction:column;padding:10px 14px;gap:8px"):
            with ui.row().classes("w-full items-center gap-3 no-wrap"):
                ui.label("Editor").classes("text-lg font-semibold text-white")
                stage = ui.toggle({"rough": "✂ Rough cut", "fitted": "♪ Fit to the music"},
                                  value="rough" if rough else "fitted",
                                  on_change=lambda e: self.set_stage(e.value)) \
                    .props("dense no-caps toggle-color=orange color=grey-9 text-color=white")
                stage.tooltip("Rough cut: arrange the clips freely, nothing snaps. Fit to the music: every cut goes "
                              "on your pins, and stays there while you change things.")
                if self.trip.undo_label() == "fitting to the music":
                    ui.button("Undo the fit", icon="undo", on_click=self.undo_fit) \
                        .props("dense flat no-caps color=orange")
                ui.label("▶ Exact preview" if exact else "▶ Instant: straight from your files") \
                    .classes("text-sm " + ("text-green-400" if exact else "text-sky-300")) \
                    .tooltip("The instant player shows every change at once; the finished video adds the look and "
                             "the transitions you chose. Exact preview makes a small video like the finished one."
                             if not exact else "A small video made like the finished one")
                ui.space()
                self.draft_status = ui.label("").classes("text-xs text-slate-300")
                self.draft_button = ui.button("Exact preview", icon="bolt", on_click=self.make_draft) \
                    .props("dense outline no-caps color=white") \
                    .tooltip("Makes a small video like the finished one (with the look and transitions); it takes "
                             "a while. Not needed while editing.")
                ui.button("Exciting transitions", icon="auto_awesome", on_click=self.suggest_transitions) \
                    .props("dense outline no-caps color=amber") \
                    .tooltip("The app picks a transition for every cut from the music: fun ones (spin, bounce, zoom "
                             "punch) where it is lively, soft ones where it is calm. Click a ◆ to change one.")
                ui.button("Opening & ending", icon="style", on_click=self.cards_dialog) \
                    .props("dense outline no-caps color=white") \
                    .tooltip("A card with the trip's title at the start and words at the end, in one of six looks")
                ui.button("Add a title here", icon="title", on_click=self.add_timeline_title) \
                    .props("dense outline no-caps color=white") \
                    .tooltip("A new title on the clip you picked (or where the red line is)")
                ui.button("Finalize", icon="movie", on_click=self.finalize).props("dense no-caps color=green")
                ui.button(icon="close", on_click=lambda: self.open_editor(False)) \
                    .props("flat round dense color=white").tooltip("Close the editor (Esc)").classes("tm-editor-close")
            with ui.row().classes("w-full no-wrap gap-3").style("flex:1;min-height:0"):
                if exact:
                    player = (f'<video id="tm-draft" src="/drafts/{draft["file"]}#t={resume:.2f}" controls playsinline '
                              'preload="auto" style="width:100%;height:100%;background:#000;border-radius:6px" '
                              f'ontimeupdate="tmDraftNow(this.currentTime)" onseeked="tmDraftNow(this.currentTime)" '
                              f'onloadedmetadata="tmDraftNow(this.currentTime, true)"></video>')
                else:
                    player = self.instant_html("tm-draft", "width:100%;height:100%;border-radius:6px")
                ui.html(f'<div style="position:relative;height:100%;aspect-ratio:{w}/{h};max-width:100%;margin:0 auto">'
                        f'{player}{live}</div>', sanitize=False).style("flex:1;min-width:0;height:100%")
                with ui.column().classes("gap-2 no-wrap").style(
                        "width:380px;flex:none;height:100%;overflow-y:auto;background:#f8fafc;color:#0f172a;"
                        "border-radius:8px;padding:10px"):
                    if self.draft_pick is not None and self.draft_pick < len(board):
                        self.draft_clip(board)
                    elif getattr(self, "draft_title", None):
                        self.draft_title_editor(board, times)
                    else:
                        ui.label("Click a clip below to change it: its speed, its part, another shot, play it "
                                 "backwards or take it out. Click a title to change its words.") \
                            .classes("text-sm text-gray-600")
                        ui.label("Space plays and pauses · ← → move half a second · Esc closes · drag a clip onto the "
                                 "middle of another to swap them, onto its left edge (▼) to paste it in front.") \
                            .classes("text-xs text-gray-500")
            with ui.row().classes("w-full items-center gap-2 no-wrap"):
                ui.label("Zoom").classes("text-xs text-slate-400")
                ui.toggle(self.ZOOMS, value=zoom, on_change=lambda e: self.set_zoom(e.value)) \
                    .props("dense no-caps size=sm toggle-color=indigo color=grey-9 text-color=white")
                ui.html('<span id="tm-edclock" style="font:12px monospace;color:#94a3b8"></span>', sanitize=False)
                ui.space()
                ui.label(("Green lines: your pins (Fit to the music puts the cuts on them) · " if rough else "") +
                         "Orange: changed since the exact preview · ⏩/🐢: faster or slower").classes("text-xs text-slate-400")
            with ui.element("div").props('id="tm-tl-scroll" onscroll="window.tmTlScroll = this.scrollLeft"') \
                    .style("flex:none;max-height:42vh;overflow:auto;background:#1e293b;border-radius:8px;padding:6px"):
                with ui.element("div").style(f"position:relative;width:{width}") \
                        .props(f'id="tm-timeline" data-total="{total:.3f}"'):
                    step = next(x for x in (1, 2, 5, 10, 15, 30, 60, 1e9)
                                if (zoom and x * zoom >= 60) or (not zoom and total / x <= 30))
                    marks = "".join(f'<span style="position:absolute;left:{100 * k * step / total:.3f}%;top:0;'
                                    f'font-size:10px;color:#94a3b8;border-left:1px solid #64748b;padding-left:2px">'
                                    f'{fmt_length(k * step)}</span>' for k in range(int(total // step) + 1))
                    ui.html(f'<div style="position:relative;height:22px;cursor:ew-resize;background:#0f172a;'
                            f'border-radius:4px;touch-action:none" onpointerdown="tmScrubStart(event)">{marks}</div>',
                            sanitize=False).classes("w-full").tooltip("Press and drag: the video follows. Double-click: back to the start")
                    # (only its knob on the ruler takes the mouse: clicks on the clips under the line reach them)
                    ui.html('<div id="tm-playhead" style="position:absolute;top:0;bottom:0;left:0;width:14px;'
                            'margin-left:-6px;z-index:6;pointer-events:none"><div style="position:absolute;left:6px;'
                            'top:0;bottom:0;width:2px;background:#ef4444;box-shadow:0 0 4px #ef4444"></div>'
                            '<div onpointerdown="tmScrubStart(event)" style="position:absolute;left:0;top:0;width:14px;'
                            'height:14px;border-radius:50% 50% 50% 0;transform:rotate(-45deg);background:#ef4444;'
                            'pointer-events:auto;cursor:ew-resize;touch-action:none"></div></div>', sanitize=False)
                    bars = sorted(self.title_bars(board, times), key=lambda b: b["at"])
                    ends: list[float] = []  # titles that overlap in time go on rows of their own
                    for bar in bars:
                        lane = next((k for k, end in enumerate(ends) if end <= bar["at"] + 0.01), len(ends))
                        ends[lane:lane + 1] = [bar["at"] + float(bar["length"])]
                        bar["lane"] = lane
                    with ui.element("div").style(f"position:relative;height:{8 + 34 * max(1, len(ends))}px;"
                                                 "background:#312e81;border-radius:4px;margin-bottom:4px") \
                            .props(f'data-total="{total:.3f}" data-snaps="'
                                   + ",".join(f"{x:.3f}" for x in sorted({round(t["start"], 3) for t in times} | set(pins)))
                                   + '"'):
                        for bar in bars:
                            self.title_bar(bar, total)
                    with ui.element("div").style("position:relative;height:80px"):
                        if rough and pins:  # where Fit to the music will put the cuts
                            ui.html("".join(f'<i style="position:absolute;top:0;bottom:0;left:{100 * p / total:.3f}%;'
                                            'width:1px;background:rgba(34,197,94,.55);z-index:4"></i>' for p in pins),
                                    sanitize=False).style("position:absolute;inset:0;pointer-events:none")
                        for n, r in enumerate(board):
                            was = before[n] if before and n < len(before) else None
                            self.draft_tile(n, r, times[n]["length"], times[n]["start"], total,
                                            changed=before is not None and (was is None or tuple(was) != tuple(now[n])))
                        for n in range(len(board) - 1):  # ◆ between two clips: its transition
                            self.transition_mark(n, board, times, total)
                        end = ui.element("div").classes("absolute rounded border-2 border-dashed border-slate-500 "
                                                         "flex items-center justify-center text-slate-400")
                        end.style("left:100%;top:0;bottom:0;width:22px;margin-left:-24px")
                        end._props.update({
                            "ondragover": "event.preventDefault();this.style.borderColor='#60a5fa'",
                            "ondragleave": "this.style.borderColor=''",
                            "ondrop": "event.preventDefault();this.style.borderColor='';"
                                      f"emitEvent('tm_draft_drop',{{src:event.dataTransfer.getData('text/plain'),"
                                      f"dst:{len(board)},mode:'insert'}})"})
                        with end:
                            ui.label("⇥")
                        end.tooltip("Drop a clip here to put it at the end")
                    ticks = "".join(f'<i style="position:absolute;top:0;bottom:0;left:{100 * p / total:.3f}%;'
                                    'width:2px;background:#22c55e"></i>' for p in pins)
                    ui.html(f'<div style="position:relative;height:34px;margin-top:4px;background:#0f172a;'
                            f'border-radius:4px;overflow:hidden;cursor:ew-resize;touch-action:none" '
                            f'onpointerdown="tmScrubStart(event)" '
                            f'title="The music, with your pins (green): the cuts · drag to scrub">'
                            f'<canvas id="tm-edwave" data-src="{wave}" data-secs="{total:.3f}" '
                            f'style="position:absolute;inset:0;width:100%;height:100%"></canvas>{ticks}</div>',
                            sanitize=False).classes("w-full")
        if not exact:
            data = json.dumps(self.instant_data(board, times, tail=tail))
            ui.run_javascript(f"setTimeout(() => tmInstant('tm-draft', {data}, tmDraftNow), 50)")
        ui.run_javascript("setTimeout(tmEditorReady, 60)")

    def editor_times(self, board: list[dict]) -> list[dict]:
        """Where each clip is in the video: the draft's own times while it is up to date."""
        draft = self.draft_info()
        changed = bool(draft) and ([tuple(x) for x in draft["board"]] != self.draft_board())
        return self.board_times(board, draft, changed)

    def board_times(self, board: list[dict], draft: dict | None, changed: bool) -> list[dict]:
        """Where each clip is in the video: from the draft while it is up to date, else worked out."""
        if draft and not changed and len(draft["times"]) == len(board):
            return draft["times"]
        photo = self.photo_len.value if getattr(self, "photo_len", None) else 4.0
        seconds = self.t_len.value if getattr(self, "t_len", None) else 0.8
        lengths = [Trip.screen_length(r, photo) for r in board]
        styles = [style_between(board[k].get("transition"), k + 1, self.transition_name(), seconds,
                                board[k].get("transition_len")) for k in range(len(board) - 1)]  # (each shot's own transition, as the video is made)
        return shot_times(lengths, styles) if lengths else []

    def transition_name(self) -> str:
        """The video's transition (Make video › More settings), for shots without their own."""
        return self.transition.value if getattr(self, "transition", None) else "Fade"

    def title_bars(self, board: list[dict], times: list[dict]) -> list[dict]:
        """Every title on the video: the timeline ones and the ones still sitting on a shot."""
        place = {r["path"]: n for n, r in enumerate(board)}
        bars = []
        for t in self.trip.timeline_titles():
            if t.get("link"):
                if t["link"] not in place:
                    continue  # its clip is out of the video
                start = times[place[t["link"]]]["start"] + float(t.get("offset") or 0)
            else:
                start = float(t.get("start") or 0)
            bars.append(t | {"at": start, "kind": "timeline"})
        for path, t in self.trip.shot_titles(board).items():
            if path in place:
                a, b = titles.timing(times[place[path]]["length"])
                bars.append(t | {"id": f"shot:{path}", "at": times[place[path]]["start"] + a, "length": b - a,
                                 "kind": "shot", "path": path})
        return bars

    def title_bar(self, bar: dict, total: float) -> None:
        picked = bar["id"] == getattr(self, "draft_title", None)
        words = " / ".join(x for x in (bar.get("text"), bar.get("sub")) if x) or "(no words)"
        left, width = 100 * bar["at"] / total, max(0.8, 100 * float(bar["length"]) / total)
        el = ui.element("div").classes("absolute rounded text-[11px] px-1 overflow-hidden whitespace-nowrap select-none "
                                       + ("cursor-grab" if bar["kind"] == "timeline" else "cursor-pointer"))
        style = ("background:#e0e7ff;color:#3730a3;border:1px solid #6366f1" if bar["kind"] == "timeline"
                 else "background:#f8fafc;color:#475569;border:1px dashed #94a3b8")
        el.style(f"left:{left:.3f}%;width:{width:.3f}%;top:{4 + 34 * bar.get('lane', 0)}px;height:30px;{style};"
                 "padding:0 12px;line-height:28px;touch-action:none"
                 + (";box-shadow:0 0 0 2px #60a5fa" if picked else ""))
        el._props["data-id"] = bar["id"]
        if bar["kind"] == "timeline":
            el._props["onpointerdown"] = "tmTitleDown(event, this)"
        else:
            el.on("click", lambda b=bar: self.shot_title_to_timeline(b))
            el.on("dblclick", lambda b=bar: self.title_menu(SimpleNamespace(args={"id": b["id"]})))
        link = " 🔗" if bar.get("link") else ""
        with el:
            ui.label(words + link)
            if bar["kind"] == "timeline":  # grips: drag an end to trim the title from that side
                for side in ("left", "right"):
                    ui.element("div").style(
                        f"position:absolute;{side}:0;top:0;bottom:0;width:10px;cursor:ew-resize;"
                        "background:repeating-linear-gradient(90deg,#6366f1 0 2px,transparent 2px 4px);"
                        f"border-radius:{'4px 0 0 4px' if side == 'left' else '0 4px 4px 0'};opacity:.8")
        el.tooltip(f"{words} · {clock(bar['at'])} → {clock(bar['at'] + float(bar['length']))}" + (" · moves with its clip" if bar.get("link") else "")
                   + ("" if bar["kind"] == "timeline" else " · on its shot: click to move it onto the timeline"))

    def shot_title_to_timeline(self, bar: dict) -> None:
        board = self.trip.storyboard()
        r = next((x for x in board if x["path"] == bar["path"]), None)
        if r is None:
            return
        new = self.trip.title_to_timeline(bar["path"], self.board_times(board, self.draft_info(), True)[
            board.index(r)]["length"])
        if new:
            self.draft_title = new["id"]
            ui.notify("This title is on the timeline now: drag it, stretch it, or change it below.", type="positive")
        self.editor_card.refresh()

    def title_click(self, e) -> None:
        self.draft_title = (e.args or {}).get("id") if isinstance(e.args, dict) else None
        self.editor_card.refresh()

    def title_menu(self, e) -> None:
        """Double-click on a title: a small window with its words, style and timing (double-click it to close)."""
        key = (e.args or {}).get("id") if isinstance(e.args, dict) else None
        if not key:
            return
        if key.startswith("shot:"):  # a title still sitting on its shot: onto the timeline first
            board = self.trip.storyboard()
            new = self.trip.title_to_timeline(key[5:], next((t["length"] for r, t in zip(board, self.editor_times(board))
                                                            if r["path"] == key[5:]), 3.0))
            if not new:
                return
            key = new["id"]
        for old in list(self.window_box):
            if isinstance(old, ui.dialog) and not old.value:
                old.delete()
        with self.window_box, ui.dialog().props("seamless position=top") as dialog, \
                ui.card().classes("w-[44rem] max-w-full mt-16").style("box-shadow:0 10px 30px rgba(0,0,0,.5)") as card:
            card.on("dblclick", dialog.close)
            board = self.trip.storyboard()
            self.draft_title_editor(board, self.editor_times(board), key=key, dialog=dialog)
            ui.label("Double-click here to close").classes("text-xs text-gray-400")
        dialog.on("hide", lambda: self.editor_card.refresh())
        dialog.open()

    def title_move(self, e) -> None:
        args = e.args if isinstance(e.args, dict) else {}
        try:
            key, start, length = args["id"], max(0.0, float(args["start"])), max(0.3, float(args["length"]))
        except (KeyError, TypeError, ValueError):
            return
        board = self.trip.storyboard()
        times = self.editor_times(board)
        place = {r["path"]: n for n, r in enumerate(board)}
        found = self.trip.timeline_titles()
        for t in found:
            if t["id"] == key:
                t["length"] = round(length, 2)
                if t.get("link") in place:
                    t["offset"] = round(start - times[place[t["link"]]]["start"], 2)
                else:
                    t["start"] = round(start, 2)
        self.trip.save_timeline_titles(found)
        self.draft_title = key
        self.editor_card.refresh()

    async def add_timeline_title(self) -> None:
        try:
            at = float(await ui.run_javascript("(document.getElementById('tm-draft') || {}).currentTime || 0", timeout=3))
        except Exception:
            at = 0.0
        board = self.trip.storyboard()
        draft = self.draft_info()
        changed = bool(draft) and ([tuple(x) for x in draft["board"]] != self.draft_board())
        times = self.board_times(board, draft, changed)  # (the draft's own times while it is up to date)
        if self.draft_pick is not None and self.draft_pick < len(board):  # the clip you picked: on it
            n, offset = self.draft_pick, 0.0
        else:  # where the player is
            n = max([k for k, t in enumerate(times) if t["start"] <= at] or [0])
            offset = max(0.0, at - times[n]["start"]) if times else 0.0
        new = {"id": f"t{time.time_ns()}", "text": "Your words", "sub": "", "style": "clean", "where": "lower",
               "length": 3.0, "fade_in": titles.FADE_IN, "fade_out": titles.FADE_OUT}
        if board:
            length = times[n]["length"] if n < len(times) else 3.0
            new |= {"link": board[n]["path"], "offset": round(min(offset, max(0.0, length - 0.5)), 2),
                    "length": round(min(3.0, max(1.0, length)), 2)}
        else:
            new["start"] = round(at, 2)
        self.trip.save_timeline_titles(self.trip.timeline_titles() + [new])
        self.draft_title = new["id"]
        self.editor_card.refresh()

    def draft_title_editor(self, board: list[dict], times: list[dict], key: str | None = None, dialog=None) -> None:
        key = key or getattr(self, "draft_title", None)
        found = self.trip.timeline_titles()
        t = next((x for x in found if x["id"] == key), None)
        if t is None:
            return
        place = {r["path"]: n for n, r in enumerate(board)}

        def save(**values) -> None:
            for x in found:
                if x["id"] == key:
                    x.update(values)
            self.trip.save_timeline_titles(found)

        def save_and_draw(**values) -> None:
            save(**values)
            self.editor_card.refresh()

        def set_link(on: bool) -> None:
            start = (times[place[t["link"]]]["start"] + float(t.get("offset") or 0)) if t.get("link") in place \
                else float(t.get("start") or 0)
            if on and board:
                n = max([k for k, x in enumerate(times) if x["start"] <= start] or [0])
                t.pop("start", None)
                save_and_draw(link=board[n]["path"], offset=round(start - times[n]["start"], 2), start=None)
            else:
                save_and_draw(link=None, offset=None, start=round(start, 2))

        def remove() -> None:
            self.trip.save_timeline_titles([x for x in found if x["id"] != key])
            self.draft_title = None
            ui.notify("Title taken away.")
            if dialog is not None:
                dialog.close()
            self.editor_card.refresh()

        at = (times[place[t["link"]]]["start"] + float(t.get("offset") or 0)) if t.get("link") in place \
            else float(t.get("start") or 0)
        with ui.card().classes("w-full mt-2 bg-indigo-50"):
            with ui.row().classes("items-center gap-2"):
                ui.label("Title").classes("font-semibold")
                ui.label(f"{clock(at)} → {clock(at + float(t.get('length') or 0))} in the video").classes("text-sm")
                ui.button("Watch it", icon="play_arrow", on_click=lambda: ui.run_javascript(
                    f"const v = document.getElementById('tm-draft'); v.currentTime = {max(0, at - 1):.2f}; v.play()")) \
                    .props("flat dense no-caps")
                linked = t.get("link") in place
                ui.switch(f"Moves with clip {place[t['link']] + 1}" if linked else "Moves with its clip", value=linked,
                          on_change=lambda e: set_link(e.value)) \
                    .tooltip("On: the title stays on its clip when you move the clip. Off: it stays at this time")
                ui.space()
                ui.button("Remove title", icon="delete", on_click=remove).props("flat dense no-caps color=negative")
            with ui.row().classes("w-full items-end gap-2"):
                words = ui.input("Words", value=t.get("text") or "").classes("min-w-48 flex-grow")
                words.on("blur", lambda: save_and_draw(text=words.value))
                sub = ui.input("Second line", value=t.get("sub") or "").classes("min-w-40")
                sub.on("blur", lambda: save_and_draw(sub=sub.value))
                ui.select({k: v["label"] for k, v in titles.STYLES.items()}, value=t.get("style") or "clean",
                          label="Style", on_change=lambda e: save_and_draw(style=e.value)).classes("w-44")
                ui.select({"": "Style's own place", **titles.WHERE}, value=t.get("where") or "", label="Where",
                          on_change=lambda e: save_and_draw(where=e.value)).classes("w-36")
            with ui.row().classes("items-end gap-2"):
                for label, field, top in (("Shows for (s)", "length", 30), ("Fade in (s)", "fade_in", 5),
                                          ("Fade out (s)", "fade_out", 5)):
                    ui.number(label, value=float(t.get(field) or 0), min=0 if field != "length" else 0.3, max=top,
                              step=0.1, format="%.1f",
                              on_change=lambda e, f=field: e.value is not None and save(**{f: round(float(e.value), 2)})) \
                        .on("blur", lambda: self.editor_card.refresh()) \
                        .props("dense outlined").classes("w-32")
                ui.button("Done", icon="check", on_click=lambda: (setattr(self, "draft_title", None),
                                                                  dialog.close() if dialog is not None else None,
                                                                  self.editor_card.refresh())).props("flat dense no-caps")
            ui.label("Changes show over the playing draft straight away; Finalize puts the titles into the video.") \
                .classes("text-xs text-gray-500")

    def resolved_titles(self, board: list[dict] | None = None) -> list[dict]:
        """The timeline titles for making the video: linked ones by their clip's number in the video."""
        board = self.trip.storyboard() if board is None else board
        place = {r["path"]: n for n, r in enumerate(board)}
        out = []
        for t in self.trip.timeline_titles():
            keep = {k: t.get(k) for k in ("text", "sub", "style", "where", "length", "fade_in", "fade_out")}
            if t.get("link"):
                if t["link"] in place:
                    out.append(keep | {"shot": place[t["link"]], "offset": float(t.get("offset") or 0)})
            else:
                out.append(keep | {"start": float(t.get("start") or 0)})
        return out

    async def finalize(self) -> None:
        ui.run_javascript(f"getHtmlElement({self.make_button.id}).scrollIntoView({{block: 'center'}})")
        await self.make()

    def draft_tile(self, n: int, r: dict, length: float, start: float, total: float, changed: bool = False) -> None:
        locked = bool(r.get("locks") and r["locks"] != "[]")
        picked = n == self.draft_pick
        tile = ui.element("div").classes("relative cursor-pointer select-none")
        thumb = data_url(r["thumb"]) if r.get("thumb") else ""
        # Each clip a closed box: its own frame and rounded ends, a gap to the next one.
        edge = "#fb923c" if changed else "#e2e8f0"
        tile.style(f"position:absolute;top:4px;bottom:4px;left:calc({100 * start / total:.3f}% + 2px);"
                   f"width:calc({100 * length / total:.3f}% - 4px);min-width:6px;border-radius:7px;"
                   f"border:2px solid {edge};box-sizing:border-box;"
                   f"background:#334155 url('{thumb}') left center/auto 100% repeat-x;"
                   + ("box-shadow:0 0 0 3px #3b82f6,0 6px 14px rgba(59,130,246,.6);transform:translateY(-3px);z-index:3"
                      if picked else "box-shadow:0 2px 4px rgba(0,0,0,.5)"))
        tile._props.update({
            "id": f"tm-ed-{n}", "draggable": "true",
            "data-start": f"{start:.3f}", "data-length": f"{length:.3f}",
            "ondragstart": f"event.dataTransfer.setData('text/plain','{n}')",
            "ondragover": "tmClipOver(event, this)", "ondragleave": "tmClipLeave(this)",
            "ondrop": f"tmClipDrop(event, this, {n})"})
        self.click_or_double(tile, lambda n=n, start=start: self.pick_draft_clip(n, start),
                             lambda p=r["path"]: self.shot_window(p, "Make video"))
        tile.tooltip(f"{n + 1}: {length:.1f} s" + (" · 🔒 locked moment" if locked else "") +
                     " · double-click: open it big (theme, cut, lock)")
        with tile:
            speed = r.get("speed") or 1.0
            pace = "" if r["kind"] != "video" or abs(speed - 1) < 0.01 else \
                f" {'⏩' if speed > 1 else '🐢'}{speed:g}×"
            fx = json.loads(r["effects"]) if r.get("effects") else {}
            many = f" 🧩{len(fx['collage']) + 1}" if fx.get("collage") else f" 🎞{len(fx['side']) + 1}" if fx.get("side") else ""
            ui.label(f"{n + 1}" + many + (" 🔒" if locked else "") + (" ⇆" if r.get("reverse") else "") + pace).classes(
                "absolute left-1 top-0 text-[10px] font-bold text-white whitespace-nowrap").style("text-shadow:0 0 3px #000")
            ui.label(f"{length:.1f}s").classes("absolute left-1 bottom-0 text-[10px] text-white") \
                .style("text-shadow:0 0 3px #000")

    TRANSITION_MARKS = {"Soft": "◆", "Move": "➜", "Shape": "◯", "Fun": "✦"}

    def transition_of(self, r: dict) -> str:
        return r.get("transition") or self.transition_name()

    def transition_mark(self, n: int, board: list[dict], times: list[dict], total: float) -> None:
        """The cut between clip n and the next: a small mark you click to choose its transition."""
        name = self.transition_of(board[n])
        end = times[n]["start"] + times[n]["length"]
        cut = (times[n + 1]["start"] + end) / 2
        group = next((g for g, names in transitions.GROUPS.items() if name in names), "")
        mark = "│" if name == transitions.HARD_CUT else self.TRANSITION_MARKS.get(group, "◆")
        own = board[n].get("transition") is not None
        el = ui.element("div").classes("tm-trmark absolute flex items-center justify-center cursor-pointer select-none")
        el.style(f"left:{100 * cut / total:.3f}%;top:-2px;width:20px;height:20px;margin-left:-10px;z-index:5;"
                 f"border-radius:50%;font-size:12px;font-weight:700;color:#0f172a;"
                 f"background:{'#fbbf24' if own else '#e2e8f0'};box-shadow:0 0 0 2px #0f172a")
        with el:
            ui.label(mark)
        el.on("click.stop", lambda n=n: self.transition_picker(n))
        el.tooltip(f"{name}" + (f" · {board[n]['transition_len']:.1f} s" if board[n].get("transition_len") else "")
                   + (f" · {board[n]['transition_why']}" if board[n].get("transition_why") else "")
                   + " · click to change")

    def transition_picker(self, n: int) -> None:
        """Every transition, moving, in groups: click one to see it big between your two clips; its length; use it
        here or for every cut."""
        board = self.trip.storyboard()
        if n + 1 >= len(board):
            return
        a_row, b_row = board[n], board[n + 1]
        current = self.transition_of(a_row)
        seconds = self.t_len.value if getattr(self, "t_len", None) else 0.8
        pick = {"name": current, "len": float(a_row.get("transition_len") or
                                              transitions.seconds_of(current, seconds) if current != transitions.HARD_CUT else seconds)}
        a_pic, b_pic = data_url(a_row["thumb"]) if a_row.get("thumb") else "", data_url(b_row["thumb"]) if b_row.get("thumb") else ""
        times = self.editor_times(board)
        len_a, len_b = (times[n]["length"], times[n + 1]["length"]) if len(times) > n + 1 else (4.0, 4.0)

        def demo(fam: str, size: str, cls: str = "tm-trdemo", length: float = 0.9) -> str:
            return (f'<div class="{cls}" data-fam="{fam}" data-len="{length:.2f}" style="position:relative;overflow:hidden;background:#000;'
                    f'border-radius:6px;{size}"><img src="{a_pic}" style="position:absolute;inset:0;width:100%;'
                    f'height:100%;object-fit:cover"><div class="tm-b" style="position:absolute;inset:0">'
                    f'<img src="{b_pic}" style="width:100%;height:100%;object-fit:cover"></div>'
                    f'<div class="tm-ov" style="position:absolute;inset:0;opacity:0"></div></div>')

        with self.window_box, ui.dialog() as dialog, ui.card().classes("w-[64rem] max-w-full gap-2"):
            ui.label(f"Transition from clip {n + 1} to clip {n + 2}").classes("text-lg font-semibold")

            @ui.refreshable
            def big() -> None:
                name = pick["name"]
                fam = "" if name == transitions.HARD_CUT else transitions.LOOKS.get(name, "fade")
                with ui.row().classes("w-full items-center gap-4 no-wrap"):
                    ui.html(demo(fam or "cut", "width:320px;height:180px;flex:none",
                                 length=min(pick["len"], OVERLAP_SHARE * min(len_a, len_b))), sanitize=False)
                    with ui.column().classes("gap-1 flex-1"):
                        ui.label(name).classes("text-xl font-semibold")
                        if name != transitions.HARD_CUT and transitions.STYLES.get(name, transitions.Style("fade")).xfade:
                            ui.label(f"How long: {pick['len']:g} s").classes("text-sm font-medium")
                            with ui.row().classes("gap-1"):
                                for v in (0.3, 0.5, 1.0, 1.5, 2.0, 3.0):
                                    ui.button(f"{v:g} s", on_click=lambda v=v: (pick.update(len=v), big.refresh())) \
                                        .props("dense no-caps size=sm " + ("unelevated color=amber-8"
                                                                           if abs(pick["len"] - v) < 0.01 else "outline"))
                            ui.slider(min=0.2, max=3.0, step=0.1, value=pick["len"]).props("dense label") \
                                .classes("w-72").on("change", lambda e: (pick.update(len=float(e.args)), big.refresh()))
                            most = OVERLAP_SHARE * min(len_a, len_b)
                            if pick["len"] > most + 0.05:
                                ui.label(f"Here it lasts {most:.1f} s at most: a transition takes at most half of each "
                                         f"clip, and these are {len_a:.1f} s and {len_b:.1f} s long. Make the clips "
                                         "longer (or join pins) for a longer one.").classes("text-xs text-orange-700")
                        else:
                            ui.label("A clean cut, right on the pin." if name == transitions.HARD_CUT else
                                     "A quick effect around a clean cut.").classes("text-sm text-gray-600")
                        if self.trip.edit_stage() == "fitted":
                            ui.label("The cut stays on its pin; the clips grow into the transition.") \
                                .classes("text-xs text-gray-500")
                ui.run_javascript("tmTrDemos()")

            big()
            with ui.element("div").classes("w-full max-h-[46vh] overflow-auto"):
                options = [("Clean cut", [transitions.HARD_CUT])] + list(transitions.GROUPS.items())
                for group, names in options:
                    ui.label(group).classes("text-sm font-semibold text-gray-700 mt-1")
                    with ui.row().classes("gap-2"):
                        for name in names:
                            fam = "cut" if name == transitions.HARD_CUT else transitions.LOOKS.get(name, "fade")
                            on = name == current
                            with ui.column().classes("gap-0 items-center cursor-pointer rounded p-1 " +
                                                     ("bg-amber-100" if on else "hover:bg-gray-100")) as tile:
                                ui.html(demo(fam, "width:112px;height:63px"), sanitize=False)
                                ui.label(name).classes("text-xs")
                            tile.on("click", lambda name=name: (
                                pick.update(name=name, len=round(transitions.seconds_of(name, seconds), 2)
                                            if name != transitions.HARD_CUT else pick["len"]), big.refresh()))

            def use(every: bool) -> None:
                name = pick["name"]
                quick = transitions.STYLES.get(name, transitions.Style("fade"))
                length = round(pick["len"], 2) if name != transitions.HARD_CUT and quick.xfade else None
                rows = board[:-1] if every else [a_row]
                for r in rows:
                    self.trip.set_transition(r["path"], name, length, "Your choice")
                dialog.close()
                self.draft_changed(f"{name} " + ("for every cut." if every else f"between clips {n + 1} and {n + 2}."))

            with ui.row().classes("w-full items-center gap-2"):
                ui.button("Video's default", on_click=lambda: (
                    [self.trip.set_transition(a_row["path"], None)], dialog.close(),
                    self.draft_changed("Back to the video's transition."))).props("flat no-caps") \
                    .tooltip("Make video › More settings › Transition")
                ui.space()
                ui.button("Close", on_click=dialog.close).props("flat no-caps")
                ui.button("Use it for every cut", on_click=lambda: use(True)).props("outline no-caps")
                ui.button("Use it here", icon="check", on_click=lambda: use(False)).props("unelevated no-caps color=amber-8")
        dialog.open()
        ui.run_javascript("setTimeout(tmTrDemos, 100)")

    async def suggest_transitions(self) -> None:
        """A transition for every cut, from the music (lively: fun ones; calm: soft ones), the days and stories."""
        board = self.trip.storyboard()
        if len(board) < 2:
            return
        times = self.editor_times(board)
        music = self.music_path()
        song = None
        if music:
            try:
                song = await run.io_bound(songmap.analyse, music, app_dir() / "music-maps")
            except Exception:
                traceback.print_exc()
        beats = song.beat_times() if song is not None else np.zeros(0)
        accents = np.asarray(getattr(song, "accents", None) or [], float) if song is not None else np.zeros(0)

        def energy(t: float) -> float:
            return float(song.energy_at(t)) if song is not None else 0.5

        def hit(t: float) -> float:
            if not len(beats) or len(accents) != len(beats):
                return 0.0
            k = int(np.argmin(np.abs(beats - t)))
            return float(accents[k]) if abs(beats[k] - t) < 0.08 else 0.0

        cuts = []
        for k in range(len(board) - 1):
            at = (times[k]["start"] + times[k]["length"] + times[k + 1]["start"]) / 2
            cuts.append(transitions.Cut(before=board[k], after=board[k + 1], energy_before=energy(max(0, at - 1)),
                                        energy_after=energy(at + 1), hit=hit(at),
                                        room=0.4 * min(times[k]["length"], times[k + 1]["length"])))
        seconds = self.t_len.value if getattr(self, "t_len", None) else 0.8
        chosen = transitions.exciting(cuts, seconds)
        for r, (name, why) in zip(board, chosen):
            style = transitions.STYLES.get(name, transitions.Style("fade"))
            length = 0.5 if name in transitions.FUN and style.xfade and style.quick is None else None
            self.trip.set_transition(r["path"], name, length, why)
        fun = sum(1 for name, _ in chosen if name in transitions.FUN)
        self.draft_changed(f"A transition for every cut: {fun} fun ones where the music is lively, soft ones where "
                           "it is calm, clean cuts on the beat. Hover a ◆ to see why; click it to change it.")

    def pick_draft_clip(self, n: int, start: float | None) -> None:
        self.draft_pick = None if self.draft_pick == n else n
        self.draft_title = None  # (the red line went where you clicked on the clip: see tmClipClick)
        self.editor_card.refresh()

    def draft_clip(self, board: list[dict]) -> None:
        n = self.draft_pick
        if n is None or n >= len(board):
            return
        r = board[n]
        photo = self.photo_len.value if getattr(self, "photo_len", None) else 4.0
        length = Trip.screen_length(r, photo)
        theme = self.theme_of(r)
        with ui.card().classes("w-full mt-2"):
            with ui.row().classes("items-center gap-2"):
                ui.label(f"Clip {n + 1} · {theme} · {length:.1f} s").classes("font-semibold")
                ui.label(("🎬 " if r["kind"] == "video" else "📷 ") + Path(r["part_of"] or r["path"]).name) \
                    .classes("text-xs text-gray-500")
            with ui.row().classes("items-center gap-2"):
                ui.button("Find an alternative", icon="auto_awesome",
                          on_click=lambda: self.draft_swap_dialog(r, theme)).props("outline no-caps color=purple") \
                    .tooltip("The app suggests shots that could take its place (photos or videos)")
                if r["kind"] == "video":
                    ui.button("Choose the part", icon="content_cut",
                              on_click=lambda: self.draft_part(r, length)).props("outline no-caps")
                ui.button("Take it out", icon="remove_circle_outline",
                          on_click=lambda: self.draft_take_out(r)).props("outline no-caps color=negative") \
                    .tooltip("The clip before it plays on for its time, so the later cuts stay put")
                ui.button("Open it big", icon="open_in_full",
                          on_click=lambda: self.shot_window(r["path"], "Make video", whole=True)).props("flat no-caps") \
                    .tooltip("Theme, cut, lock, split (or double-click the clip)")
            self.together_panel(r)
            if r["kind"] == "video":
                self.sound_control(r, on_change=self.sound_changed)
                self.draft_speed(r, length)
                with ui.row().classes("items-center gap-2"):
                    ui.switch("⇆ Play backwards", value=bool(r.get("reverse")),
                              on_change=lambda e: self.draft_reverse(r, e.value)).props("dense")
                self.lock_area(r, "story")

    def together_panel(self, r: dict) -> None:
        """A collage or videos side by side: every shot on this cut as a box; take one off, swap it or add one."""
        key, members = self.trip.together(r["path"])
        if not key and r["kind"] != "photo" and not self._upright(r):
            return
        thumbs = {it["path"]: it for it in self.items}
        room = (3 if key == "collage" or (not key and r["kind"] == "photo") else 2) - len(members)
        with ui.column().classes("w-full gap-1 p-2 rounded bg-purple-50"):
            if key:
                ui.label(("🧩 Collage: " if key == "collage" else "🎞 Side by side: ")
                         + f"{len(members) + 1} {'photos' if key == 'collage' else 'videos'} on this cut") \
                    .classes("text-sm font-medium text-purple-900")
            with ui.row().classes("gap-2 items-start"):
                for n, path in enumerate([r["path"], *members]):
                    it = thumbs.get(path) or {}
                    with ui.column().classes("gap-0 items-center"):
                        ui.image(data_url(it.get("thumb"))).style("width:64px;height:44px;object-fit:cover;"
                                                                  "border-radius:4px;border:2px solid #a855f7")
                        if n:
                            with ui.row().classes("gap-0"):
                                ui.button(icon="swap_horiz", on_click=lambda p=path: self.together_pick(r, key, members, p)) \
                                    .props("flat dense round size=xs color=purple").tooltip("Swap it for another")
                                ui.button(icon="close", on_click=lambda p=path: self.together_set(
                                    r, key, [m for m in members if m != p], "Taken off this cut")) \
                                    .props("flat dense round size=xs color=negative").tooltip("Take it off this cut")
                        else:
                            ui.label("first").classes("text-[10px] text-gray-500")
                if room > 0 and (key or r["kind"] == "photo" or self._upright(r)):
                    kind = key or ("collage" if r["kind"] == "photo" else "side")
                    ui.button("+ Add one", on_click=lambda: self.together_pick(r, kind, members, None)) \
                        .props("outline dense no-caps size=sm color=purple") \
                        .tooltip("Another photo into the collage" if kind == "collage"
                                 else "Another upright video next to this one")
            if key:
                ui.button("Edit it big", icon="dashboard_customize", on_click=lambda: self.collage_window(r["path"])) \
                    .props("unelevated dense no-caps size=sm color=purple") \
                    .tooltip("Move and zoom each picture in its place, swap places, layout (or double-click the clip)")
                ui.label("The photos pop in one after another on this cut." if key == "collage" else
                         "They play together; the sound is the first one's.").classes("text-xs text-gray-600")

    def collage_window(self, path: str) -> None:
        """A collage or videos side by side, big, as it will look: move and zoom each picture in its place, swap
        places, take one out or add one, choose the layout; for videos the moment each one plays, and turn them."""
        from .render import LAYOUTS, collage_tiles, side_tiles

        key, members = self.trip.together(path)
        if not key:
            self.shot_window(path, whole=True)
            return
        w, h = RESOLUTIONS[self.resolution.value] if getattr(self, "resolution", None) else (1920, 1080)
        state = {"path": path, "dirty": False, "n": 0}
        for old in list(self.window_box):  # (windows closed before)
            old.delete()

        def close() -> None:
            dialog.close()
            if state["dirty"]:
                self.draft_changed("Collage changed." if key == "collage" else "Videos side by side changed.")

        with self.window_box, ui.dialog().props("maximized transition-show=none transition-hide=none") as dialog, \
                ui.card().classes("w-full h-full gap-2 p-3 tm-bigwin"):

            @ui.refreshable
            def body() -> None:
                nonlocal key, members
                lead = state["path"]
                key, members = self.trip.together(lead)
                if not key:
                    ui.label("Nothing else on this cut any more.").classes("text-gray-600")
                    return
                rows = {it["path"]: it for it in self.trip.items(parts=True)}
                order = [lead, *members]
                layouts = LAYOUTS.get(len(order), {}) if key == "collage" else {}
                layout = self.trip.collage_layouts().get(lead) or ""
                places = (collage_tiles(len(order), w, h, layout) if key == "collage" else side_tiles(len(order), w, h))
                framed = self.trip.tile_frames()
                lead_row = rows.get(lead) or {}
                fx = json.loads(lead_row.get("effects") or "{}") if lead_row.get("effects") else {}
                starts = [lead_row.get("trim_start") or 0.0, *(fx.get("starts") or [])]
                length = max(0.5, (lead_row.get("trim_end") or 0) - (lead_row.get("trim_start") or 0)) \
                    if key == "side" else 0.0
                room = (3 if key == "collage" else 2) - len(members)
                with ui.row().classes("w-full items-center gap-2 no-wrap"):
                    ui.label(("🧩 Collage: " if key == "collage" else "🎞 Side by side: ")
                             + f"{len(order)} {'photos' if key == 'collage' else 'videos'}").classes("text-lg font-semibold")
                    ui.label("Drag a picture to move it · scroll, pinch or − + to make it bigger or smaller · drag ⠿ onto another place to swap")\
                        .classes("text-sm text-gray-600")
                    ui.space()
                    for name, label in layouts.items():
                        ui.button(label, on_click=lambda n=name: set_layout(n)).props(
                            "dense no-caps size=sm " + ("unelevated color=purple" if name == layout else "outline color=purple"))
                    if room > 0:
                        ui.button("+ Add one", on_click=lambda: self.together_pick(
                            rows.get(lead) or {"path": lead}, key, members, None, after=added)) \
                            .props("outline dense no-caps size=sm color=purple")
                    ui.button("Done", icon="check", on_click=close).props("unelevated no-caps color=primary")
                state["n"] += 1
                stage_id = f"tm-cstage-{state['n']}"
                with ui.element("div").classes("tm-fill w-full flex items-center justify-center").style("min-height:0"):
                    below = 150 if key == "side" else 100  # (the title row, and the moment sliders for videos)
                    stage = ui.element("div").props(f"id={stage_id}").style(
                        f"position:relative;aspect-ratio:{w}/{h};width:min(100%, calc((100vh - {below}px) * {w / h:.4f}));"
                        "background:#0c0c0c;border-radius:6px;overflow:hidden;flex:none")
                    with stage:
                        for k, (p, (x, y, tw, th)) in enumerate(zip(order, places)):
                            it = rows.get(p) or {}
                            f = framed.get(p) or {}
                            video = key == "side"
                            try:
                                src = self.video_src(Path(it.get("part_of") or p))
                            except ValueError:
                                src = data_url(it.get("thumb")) if it.get("thumb") else ""
                            if not video and (it.get("rotation") or 0) and it.get("thumb"):
                                src = data_url(it["thumb"])  # (turned by you: its preview is turned too)
                            a = float(starts[k]) if video and k < len(starts) else 0.0
                            tile = ui.element("div").classes("tm-ctile").props(
                                f'data-k={k} data-x={f.get("x", 0.5)} data-y={f.get("y", 0.5)} data-z={f.get("z", 1)} '
                                f'data-a={a:.3f} data-len={length:.3f}').style(
                                f"position:absolute;left:{100 * x / w:.3f}%;top:{100 * y / h:.3f}%;width:{100 * tw / w:.3f}%;"
                                f"height:{100 * th / h:.3f}%;overflow:hidden;cursor:grab;touch-action:none")
                            with tile:
                                if it.get("thumb"):  # (shows round the picture when you make it smaller)
                                    ui.html(f'<img class="tm-cbg" src="{data_url(it["thumb"])}" draggable="false" '
                                            'style="position:absolute;inset:0;width:100%;height:100%;object-fit:cover;'
                                            'filter:blur(14px) brightness(.55);pointer-events:none">')
                                turn = it.get("rotation") or 0 if video else 0
                                rot = f"rotate:{turn}deg;" if turn else ""
                                if video:
                                    ui.html(f'<video class="tm-cm" src="{src}" muted playsinline preload="auto" '
                                            f'style="position:absolute;left:0;top:0;z-index:1;width:100%;height:100%;object-fit:cover;pointer-events:none;{rot}"></video>')
                                else:
                                    ui.html(f'<img class="tm-cm" src="{src}" draggable="false" '
                                            'style="position:absolute;left:0;top:0;z-index:1;width:100%;height:100%;object-fit:cover;pointer-events:none">')
                                with ui.row().classes("tm-cbtn absolute gap-1 items-center").style(
                                        "top:6px;left:6px;right:6px;z-index:2"):
                                    ui.html('<span class="tm-chandle" draggable="true" title="Drag onto another place to '
                                            'swap them" style="cursor:move;background:#fffd;border-radius:4px;'
                                            'padding:0 6px;font-size:18px">⠿</span>')
                                    ui.label("first" if k == 0 else f"{k + 1}").classes(
                                        "text-xs px-1 rounded bg-white/80")
                                    ui.html('<span class="tm-cz text-xs px-1 rounded empty:hidden" style="background:#fffc"></span>')
                                    ui.space()
                                    ui.html('<button data-zoom="0.8" title="Zoom out" style="background:#fffd;'
                                            'border-radius:4px;width:24px">−</button>')
                                    ui.html('<button data-zoom="1.25" title="Zoom in" style="background:#fffd;'
                                            'border-radius:4px;width:24px">+</button>')
                                    ui.button(icon="center_focus_strong", on_click=lambda p=p: reset(p)) \
                                        .props("dense round size=xs color=white text-color=black") \
                                        .tooltip("Back to the middle, not zoomed")
                                    if video:
                                        ui.button(icon="rotate_left", on_click=lambda p=p: turn_it(p, -90)) \
                                            .props("dense round size=xs color=white text-color=black").tooltip("Turn left")
                                        ui.button(icon="rotate_right", on_click=lambda p=p: turn_it(p, 90)) \
                                            .props("dense round size=xs color=white text-color=black").tooltip("Turn right")
                                    if k:
                                        ui.button(icon="swap_horiz", on_click=lambda p=p: self.together_pick(
                                            rows.get(lead) or {"path": lead}, key, members, p, after=added)) \
                                            .props("dense round size=xs color=white text-color=purple") \
                                            .tooltip("Swap it for another")
                                        ui.button(icon="close", on_click=lambda k=k: take_off(k)) \
                                            .props("dense round size=xs color=white text-color=negative") \
                                            .tooltip("Take it off this cut")
                    stage.on("tmframe", lambda e, order=order: framed_one(order, e.args.get("detail") or e.args), ["detail"])
                    stage.on("tmswap", lambda e: swap(e.args.get("detail") or e.args), ["detail"])
                if key == "side":  # which moment each video plays
                    with ui.row().classes("w-full gap-4 no-wrap"):
                        for k, p in enumerate(order):
                            it = rows.get(p) or {}
                            top = max(0.0, (it.get("duration") or 0) - length)
                            with ui.column().classes("gap-0 flex-1"):
                                ui.label(f"{'First' if k == 0 else f'Video {k + 1}'}: plays from "
                                         f"{fmt_length(float(starts[k]) if k < len(starts) else 0)}") \
                                    .classes("text-xs text-gray-600")
                                ui.slider(min=0, max=round(top, 2) or 0.01, step=0.1,
                                          value=min(top, float(starts[k]) if k < len(starts) else 0.0)) \
                                    .props("dense color=purple").classes("w-full") \
                                    .on("change", lambda e, k=k: moment(k, float(e.args)))
                ui.run_javascript(f"setTimeout(() => tmCollageEdit('{stage_id}'), 50)")

            def framed_one(order: list[str], d: dict) -> None:
                k = int(d.get("k", 0))
                if 0 <= k < len(order):
                    self.trip.set_tile_frame(order[k], float(d["x"]), float(d["y"]), float(d["z"]))
                    state["dirty"] = True

            def reset(p: str) -> None:
                self.trip.set_tile_frame(p, 0.5, 0.5, 1.0)
                state["dirty"] = True
                body.refresh()

            def set_layout(name: str) -> None:
                self.trip.set_collage_layout(state["path"], name)
                state["dirty"] = True
                body.refresh()

            def swap(d: dict) -> None:
                state["path"] = self.trip.swap_together(state["path"], int(d["a"]), int(d["b"]))
                state["dirty"] = True
                body.refresh()

            def take_off(k: int) -> None:
                _, now = self.trip.together(state["path"])
                self.trip.set_together(state["path"], key, [m for n, m in enumerate(now) if n != k - 1])
                state["dirty"] = True
                body.refresh()

            def turn_it(p: str, degrees: int) -> None:
                self.trip.rotate(p, degrees)
                state["dirty"] = True
                body.refresh()

            def moment(k: int, start: float) -> None:
                self.trip.set_side_start(state["path"], k, start)
                state["dirty"] = True
                body.refresh()

            def added() -> None:
                state["dirty"] = True
                body.refresh()

            body()
        dialog.open()

    def _upright(self, r: dict) -> bool:
        rows = [dict(r)]
        Trip._mark_upright(rows)
        return bool(rows[0]["upright"]) and self.trip.video_size() != "phone"

    def together_set(self, r: dict, key: str, members: list[str], message: str) -> None:
        self.trip.set_together(r["path"], key, members)
        self.draft_changed(message)

    def together_pick(self, r: dict, key: str, members: list[str], replacing: str | None,
                      after: Callable[[], None] | None = None) -> None:
        """Choose a shot for this cut's collage (photos) or to play next to it (upright videos): best first (same
        theme, nearest in time, not in the video yet), 8 at a time. Videos already in the video can lend another
        moment, but never the same video twice on one cut."""
        board = self.trip.storyboard()
        busy = {x["part_of"] or x["path"] for x in board}
        for x in board:
            busy |= set(self.trip.together(x["path"])[1])
        here = {r.get("part_of") or r["path"], *members}  # (on this cut already: never twice)
        theme = self.theme_of(r)
        kind = "photo" if key == "collage" else "video"
        others = [it for it in self.items if it["path"] not in here and it["thumb"] and it["kind"] == kind
                  and (it["path"] not in busy or key == "side")]
        if key == "side":
            Trip._mark_upright(others)
            others = [it for it in others if it["upright"]]

        def when(it: dict) -> float:
            try:
                return abs(datetime.fromisoformat(it["taken_at"]).timestamp()
                           - datetime.fromisoformat(r["taken_at"]).timestamp())
            except (TypeError, ValueError, KeyError):
                return 1e12

        others.sort(key=lambda it: (it["path"] in busy, not it["chosen"], self.theme_of(it) != theme,
                                    it.get("story") != r.get("story"), when(it)))
        state = {"page": 0}
        with self.window_box, ui.dialog() as dialog, ui.card().classes("w-[60rem] max-w-full"):
            ui.label(("Swap this photo" if replacing else "Add a photo to the collage") if key == "collage" else
                     ("Swap this video" if replacing else "Add an upright video next to it")).classes("text-lg font-semibold")
            ui.label(f"Best first: not in the video yet, {theme}, the nearest in time. Hover a video to play it."
                     + (" Videos already in the video lend another moment." if key == "side" else "")) \
                .classes("text-sm text-gray-600")

            def use(it: dict) -> None:
                dialog.close()
                new = [it["path"] if m == replacing else m for m in members] if replacing else [*members, it["path"]]
                if after:  # (from the collage window: it shows the change itself)
                    self.trip.set_together(r["path"], key, new)
                    after()
                else:
                    self.together_set(r, key, new, "Swapped." if replacing else "Added to this cut.")

            @ui.refreshable
            def grid() -> None:
                if not others:
                    ui.label(("Every other photo is in the video already." if key == "collage" else "No other upright videos in this trip.")) \
                        .classes("text-sm text-gray-600 py-6")
                    return
                pages = -(-len(others) // 8)
                k = state["page"] % pages
                with ui.element("div").classes("w-full grid grid-cols-4 gap-2"):
                    for it in others[k * 8:k * 8 + 8]:
                        with ui.element("div").classes("rounded p-1 cursor-pointer hover:bg-purple-50") as tile:
                            self.preview(it, "width:100%;aspect-ratio:16/10", start=self.hover_start(it), view="swap")
                            ui.label(f"{self.theme_of(it)} · {fmt_time(it['taken_at'])}").classes("text-xs text-gray-600")
                            if it["path"] in busy:
                                ui.label("in the video: another moment of it").classes("text-xs text-orange-700")
                        tile.on("click", lambda it=it: use(it))
                with ui.row().classes("w-full items-center"):
                    ui.label(f"{k * 8 + 1}-{min(len(others), k * 8 + 8)} of {len(others)}").classes("text-xs text-gray-500")
                    ui.space()
                    if pages > 1:
                        ui.button("8 more" if k + 1 < pages else "Start again", icon="refresh",
                                  on_click=lambda: (state.update(page=state["page"] + 1), grid.refresh())) \
                            .props("outline no-caps color=purple")

            grid()
            with ui.row().classes("justify-end w-full"):
                ui.button("Cancel", on_click=dialog.close).props("flat")
        dialog.open()

    def sound_changed(self) -> None:
        """A clip's sound on or off, or louder: the instant player plays it so straight away."""
        self.editor_card.refresh()

    SPEEDS = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]

    def draft_speed(self, r: dict, length: float) -> None:
        """Faster or slower: the clip keeps its place and length (the cuts stay on your pins), it shows more of the
        video when faster and less, in slow motion, when slower."""
        speed = r.get("speed") or 1.0
        with ui.column().classes("w-full gap-1 p-2 rounded bg-indigo-50"):
            ui.label(f"Speed: {speed:g}×" + (" (slow motion)" if speed < 0.99 else " (faster)" if speed > 1.01 else
                                              " (normal)")).classes("text-sm font-medium text-indigo-900")
            with ui.row().classes("gap-1"):
                for v in self.SPEEDS:
                    ui.button(f"{v:g}×", on_click=lambda v=v: self.draft_set_speed(r, length, v)) \
                        .props("dense no-caps size=sm " + ("unelevated color=indigo" if abs(v - speed) < 0.01
                                                           else "outline color=indigo"))
            ui.slider(min=0.25, max=3.0, step=0.05, value=speed) \
                .props("dense label color=indigo").classes("w-full") \
                .on("change", lambda e: self.draft_set_speed(r, length, float(e.args)))
            ui.label("Only if you want it: every clip plays at normal speed unless you pick another speed here. "
                     + ("It keeps its place, so the cuts stay on your pins." if self.trip.edit_stage() == "fitted" else
                        "Rough cut: the same part plays, so the clip gets shorter or longer."))\
                .classes("text-xs text-gray-600")

    def draft_set_speed(self, r: dict, length: float, speed: float) -> None:
        speed = round(max(0.25, min(3.0, speed)), 2)
        self.trip.set_speed(r["path"], speed)
        if self.trip.edit_stage() == "fitted":  # its place, at the speed you chose (shorter if the video ends)
            self.trip.fit_length(r["path"], length)  # (in the rough cut the same part plays, faster or slower)
        self.draft_changed(f"Clip {self.draft_pick + 1} plays at {speed:g}×.")

    def draft_reverse(self, r: dict, on: bool) -> None:
        self.trip.set_reverse(self.clip_of(r), on)
        self.draft_changed("Clip plays backwards." if on else "Clip plays forwards again.")

    def reflow_board(self, force: bool = False) -> int:
        """The clips back on the music's pins at their own speed (see Trip.reflow): how many no longer fit.
        In the rough cut nothing snaps to the music (each clip stays as long as you made it) until you press
        Fit to the music."""
        if not force and self.trip.edit_stage() != "fitted":
            return 0
        music = self.music_path()
        pins = song_taps.final_taps(music, app_dir() / "music-taps") if music else None
        photo = self.photo_len.value if getattr(self, "photo_len", None) else 4.0
        seconds = self.t_len.value if getattr(self, "t_len", None) else 0.8
        return self.trip.reflow(pins or [], None, photo, seconds, self.transition_name())

    def draft_changed(self, message: str) -> None:
        gone = self.reflow_board()
        if gone:
            message += (f" {gone} clip{'s' if gone > 1 else ''} at the end no longer fit{'' if gone > 1 else 's'} "
                        "the music and left the video.")
        ui.notify(message, type="positive")
        self.reload()
        self.storyboard_view.refresh()
        self.editor_card.refresh()

    def draft_drop(self, e) -> None:
        args = e.args if isinstance(e.args, dict) else {}
        try:
            src, dst = int(args.get("src")), int(args.get("dst"))
        except (TypeError, ValueError):
            return
        board = self.trip.storyboard() if self.trip else []
        if args.get("mode") == "insert":
            if not (0 <= src < len(board) and 0 <= dst <= len(board)) or dst in (src, src + 1):
                return
            self.trip.insert_shot(board[src]["path"], dst, self.photo_len.value)
            self.draft_pick = dst - 1 if dst > src else dst
            self.draft_changed(f"Clip {src + 1} pasted " + (f"in front of clip {dst + 1}" if dst < len(board)
                                                             else "at the end") + "; every cut stayed in place.")
            return
        if src == dst or not (0 <= src < len(board) and 0 <= dst < len(board)):
            return
        self.trip.swap_shots(board[src]["path"], board[dst]["path"], self.photo_len.value)
        self.draft_pick = dst
        self.draft_changed(f"Swapped clips {src + 1} and {dst + 1}; each took the other's length.")

    def draft_take_out(self, r: dict) -> None:
        self.trip.take_out(r["path"], self.photo_len.value)
        self.draft_pick = None
        self.draft_changed("Taken out; the clip before it plays on.")

    def draft_swap_dialog(self, r: dict, theme: str) -> None:
        """Find an alternative: the app's best 4 suggestions for this clip's place (look alike, same time, same
        theme, sharp, long enough), 4 more at a time; or only this theme. One click swaps it in."""
        photo = self.photo_len.value if getattr(self, "photo_len", None) else 4.0
        length = Trip.screen_length(r, photo)
        old = r["path"]
        found = self.trip.alternatives(old, length, theme)
        state = {"page": 0, "only": False}

        with self.window_box, ui.dialog() as dialog, ui.card().classes("w-[64rem] max-w-full"):
            with ui.row().classes("w-full items-center gap-3"):
                ui.label("Find an alternative").classes("text-lg font-semibold")
                ui.label(f"for clip {Path(r.get('part_of') or old).name} · {theme} · {length:.1f} s") \
                    .classes("text-sm text-gray-500")
                ui.space()
                ui.switch(f"Only {theme}", on_change=lambda e: (state.update(only=e.value, page=0), grid.refresh())) \
                    .props("dense")
            ui.label("Picked for you: shots that look like it, were taken around then and aren't in the video yet. "
                     "Hover a video to play it; click one to swap it in (same place, same length).") \
                .classes("text-sm text-gray-600")

            def use(it: dict) -> None:
                dialog.close()
                self.trip.replace_shot(old, it["path"], photo)
                self.draft_changed(f"Swapped for {Path(it['path']).name}.")

            @ui.refreshable
            def grid() -> None:
                rows = [x for x in found if not state["only"] or self.theme_of(x) == theme]
                if not rows:
                    ui.label("No other shots to suggest: every one is in the video already." if not state["only"] else
                             f"Every {theme} shot is in the video already. Switch off 'Only {theme}' for others.") \
                        .classes("text-sm text-gray-600 py-6")
                    return
                pages = max(1, -(-len(rows) // 4))
                k = state["page"] % pages
                with ui.element("div").classes("w-full grid grid-cols-4 gap-3"):
                    for n, it in enumerate(rows[k * 4:k * 4 + 4]):
                        with ui.element("div").classes("rounded-lg p-1 cursor-pointer border-2 border-transparent "
                                                       "hover:border-purple-500 hover:bg-purple-50") as tile:
                            self.preview(it, "width:100%;aspect-ratio:16/10", start=self.hover_start(it), view="swap")
                            ui.label(("🎬 " if it["kind"] == "video" else "📷 ") + self.theme_of(it)) \
                                .classes("text-sm font-medium mt-1 truncate")
                            ui.label(f"{'⭐ ' if k == 0 and n == 0 else ''}{it['why']}") \
                                .classes("text-xs text-purple-700 truncate")
                            ui.label(fmt_time(it["taken_at"])).classes("text-xs text-gray-500")
                        tile.on("click", lambda it=it: use(it))
                with ui.row().classes("w-full items-center gap-2 mt-2"):
                    ui.label(f"Ideas {k * 4 + 1}-{min(len(rows), k * 4 + 4)} of {len(rows)}") \
                        .classes("text-xs text-gray-500")
                    ui.space()
                    if k > 0:
                        ui.button("Back", icon="chevron_left",
                                  on_click=lambda: (state.update(page=state["page"] - 1), grid.refresh())) \
                            .props("flat no-caps")
                    if pages > 1:
                        ui.button("4 more ideas" if k + 1 < pages else "Start again", icon="refresh",
                                  on_click=lambda: (state.update(page=state["page"] + 1), grid.refresh())) \
                            .props("outline no-caps color=purple")

            grid()
            with ui.row().classes("justify-end w-full"):
                ui.button("Cancel", on_click=dialog.close).props("flat")
        dialog.open()

    def draft_part(self, r: dict, length: float) -> None:
        lo, hi = self.kept_part(r)

        def use(a: float, b: float, locked: bool = False) -> bool:
            # The part you chose, at its own speed: it ends on the nearest pin, the clips after it move along.
            self.trip.set_trim(r["path"], round(a, 3), round(b, 3))
            if locked:
                self.trip.add_lock(self.clip_of(r), a, b)
                self._locks_changed(self.clip_of(r))
            self.draft_changed(f"This clip now shows {fmt_length(a)}–{fmt_length(b)}"
                               + (", ending on the nearest pin" if self.trip.edit_stage() == "fitted" else "")
                               + "; the clips after it moved along.")
            return True

        self.part_dialog(r, lo, hi, f"Choose the part of clip {self.draft_pick + 1} (now {length:.1f} s; it ends on the nearest pin)",
                         [("Use this part", "check", use, "outline no-caps"),
                          ("Use and lock", "lock", lambda a, b: use(a, b, True), "no-caps color=amber-9")],
                         taken=self.trip.used_parts(self.clip_of(r), but=r["path"]))

    def render_settings(self, scale: int = 1) -> tuple[RenderSettings | None, str | None]:
        """The settings for making the video (scale 3: a third of the size, for a draft), or (None, problem)."""
        width, height = RESOLUTIONS[self.resolution.value]
        music = (self.music_input.value or "").strip().strip("'\"") or None
        if music and not Path(music).expanduser().is_file():
            return None, "The music file was not found."
        settings = RenderSettings(
            width=width // scale // 2 * 2, height=height // scale // 2 * 2,
            photo_seconds=self.photo_len.value, max_clip_seconds=self.clip_len.value,
            transition=self.transition.value, transition_seconds=self.t_len.value,
            music_path=str(Path(music).expanduser()) if music else None, music_volume=self.music_vol.value,
            look=self.trip.look(),
        )
        settings.end_black, settings.end_title = self.trip.end_card()
        settings.opening = self.trip.opening_card()
        settings.closing = self.trip.closing_card()
        return settings, None

    async def make_draft(self) -> None:
        if self.draft_busy or self.rendering["busy"] or not self.trip:
            return
        settings, problem = self.render_settings(self.DRAFT_SCALE)
        if problem:
            ui.notify(problem, type="warning")
            return
        width, height = RESOLUTIONS[self.resolution.value]
        items = await run.io_bound(self.trip.media_items, False, height > width)
        if not items:
            ui.notify("Nothing is in the Storyboard yet.", type="warning")
            return
        board = self.draft_board()
        settings.titles = []  # timeline titles show live over the draft (see editor_card)
        info, folder = self._draft_files()
        name = f"{self.trip.dir.name}-{int(time.time())}.mp4"
        state = {"message": "Starting…"}
        status = self.draft_status
        timer = ui.timer(0.5, lambda: None if status.is_deleted else status.set_text(state["message"]))
        self.draft_busy = True
        self.draft_button.disable()
        try:
            timing = folder / f"{name}.times.json"
            await run.io_bound(render, items, settings, folder / name,
                               lambda f, m: state.update(message=f"{m} ({100 * f:.0f}%)"), app_dir() / "clip-cache", timing)
            old = self.draft_info()
            info.write_text(json.dumps({"file": name, "times": json.loads(timing.read_text()), "board": board,
                                        "muted": self.trip.muted_titles()}))
            timing.unlink(missing_ok=True)
            if old:
                (folder / old["file"]).unlink(missing_ok=True)
        except Exception as exc:
            traceback.print_exc()
            timer.cancel()
            status.set_text(f"Something went wrong: {exc}")
            return
        finally:
            self.draft_busy = False
        timer.cancel()
        try:  # carry on watching from where you were
            self.draft_resume = float(await ui.run_javascript(
                "(document.getElementById('tm-draft') || {}).currentTime || 0", timeout=2) or 0)
        except Exception:
            self.draft_resume = 0.0
        self.editor_card.refresh()

    @ui.refreshable
    def looks_card(self) -> None:
        """One colour mood for the whole video, chosen from small previews of one of your own shots."""
        if not self.trip or not self.items or not self.showing("Make video"):
            return
        board = self.trip.storyboard()
        sample = next((it["thumb"] for it in board if it.get("thumb")), None)
        chosen = self.trip.look()
        with ui.card().classes("w-full"):
            ui.label("Look").classes("text-lg font-semibold")
            ui.label("One colour mood for the whole video. Click one; changing it later is quick (the clips are "
                     "not prepared again).").classes("text-sm text-gray-600")
            with ui.row().classes("gap-3"):
                for name, (label, _) in looks.LOOKS.items():
                    on = name == chosen
                    with ui.column().classes("items-center gap-1 cursor-pointer p-1 rounded " +
                                             ("ring-4 ring-blue-500" if on else "")) \
                            .on("click", lambda n=name: self.set_look(n)):
                        image = ui.image("").style("width:150px;height:84px;object-fit:cover;background:#e5e7eb")
                        ui.label(("✓ " if on else "") + label).classes("text-sm" + (" font-semibold" if on else ""))
                    if sample:
                        self.show_look_preview(image, sample, name)

    def show_look_preview(self, image, sample: str, name: str) -> None:
        key = (sample, name)
        if key in self.look_previews:
            image.set_source(self.look_previews[key])
            return

        async def make() -> None:
            self.look_previews[key] = await run.io_bound(looks.preview, sample, name)
            image.set_source(self.look_previews[key])

        ui.timer(0.05, make, once=True)

    def set_look(self, name: str) -> None:
        self.trip.set_look(name)
        ui.notify(f"Look: {looks.LOOKS[name][0]}")
        self.looks_card.refresh()

    @ui.refreshable
    def titles_card(self) -> None:
        if not self.trip or not self.items:
            return
        t = self.trip.title_settings()
        styles = {k: v["label"] for k, v in titles.STYLES.items()}

        def save(**values) -> None:
            self.trip.set_title_settings(**values)
            self.storyboard_view.refresh()

        with ui.card().classes("w-full"):
            ui.label("Titles").classes("text-lg font-semibold")
            ui.label("Words that fade in over the video. Add your own on any shot in the Storyboard (Add title).") \
                .classes("text-sm text-gray-600")
            with ui.row().classes("items-center gap-2"):
                card, end_look = titles.CARDS.get(t["card"]), titles.CARDS.get(t["end_card"])
                ui.button("Opening & ending cards", icon="style", on_click=self.cards_dialog).props("outline no-caps")
                ui.label(f"Opening: {card['label'] if card else 'title over the first shot'} · Ending: "
                         f"{end_look['label'] if end_look else 'words on black'}").classes("text-sm text-gray-600")
            with ui.row().classes("w-full items-end gap-3"):
                ui.switch("Opening title", value=t["opening"], on_change=lambda e: save(opening=e.value)).classes("w-44")
                first = ui.input("Title", value=t["trip_title"]).classes("w-64")
                first.on("blur", lambda: first.value != t["trip_title"] and save(trip_title=first.value))
                second = ui.input("Second line", value=t["trip_sub"]).classes("w-64")
                second.on("blur", lambda: save(trip_sub=second.value))
                ui.select(styles, value=t["style"], label="Style", on_change=lambda e: save(style=e.value)).classes("w-48")
            with ui.row().classes("w-full items-end gap-3"):
                ui.switch("A title for each story", value=t["stories"], on_change=lambda e: save(stories=e.value)) \
                    .classes("w-44")
                ui.label("The story's name (the AI's theme or your own) with its place and day, on its first shot.") \
                    .classes("text-xs text-gray-500 w-64")
                ui.select(styles, value=t["story_style"], label="Style",
                          on_change=lambda e: save(story_style=e.value)).classes("w-48")
            with ui.row().classes("w-full items-end gap-3"):
                ui.switch("End card", value=t["ending"], on_change=lambda e: save(ending=e.value)).classes("w-44")
                end = ui.input("Words", value=t["end_text"]).classes("w-64")
                end.on("blur", lambda: save(end_text=end.value))
                end_sub = ui.input("Second line", value=t["end_sub"]).classes("w-64")
                end_sub.on("blur", lambda: save(end_sub=end_sub.value))
                end_room = float(t["end_room"] or 0)
                ui.select({0.0: "No room (black after)"} | {float(v): f"{v} s at the end" for v in (4, 6, 8, 10, 12)}
                          | ({end_room: f"{end_room:g} s at the end"} if end_room and end_room not in (4, 6, 8, 10, 12)
                             else {}), value=end_room, label="Ending room",
                          on_change=lambda e: save(end_room=e.value)).classes("w-48") \
                    .tooltip("Auto-edit keeps these seconds at the end of the song for the end card, over the last "
                             "shot (Opening & ending: its look). Press Auto-edit again after changing it.")

    @ui.refreshable
    def video_summary(self) -> None:
        if not self.items:
            ui.label("Open a trip first, then pick what goes in the video in the Library.").classes("text-gray-500")
            return
        chosen = self.trip.storyboard()  # exactly what the Storyboard shows and the video is made of
        videos = sum(it["kind"] == "video" for it in chosen)
        ui.label(
            f"The video will use {len(chosen)} items ({videos} videos, {len(chosen) - videos} photos) in the Storyboard order. "
            "Change the selection in the Library and the order in the Storyboard."
        ).classes("text-gray-700")

    def find_music(self) -> None:
        """Search free, openly licensed songs online and use one."""
        state = {"words": "happy summer", "songs": [], "busy": False, "error": ""}

        with ui.dialog().props("maximized=false") as dialog, ui.card().classes("w-full max-w-3xl"):
            ui.label("Find free music").classes("text-lg font-semibold")
            ui.label(
                "Creative Commons songs from Openverse (Jamendo, Freesound and others) and ccMixter. Only songs you may "
                "use in your own video are shown. Most ask you to credit the artist: the app saves a credits file next "
                "to your video."
            ).classes("text-sm text-gray-600")
            with ui.row().classes("w-full items-center no-wrap"):
                words = ui.input("Search (mood, style, instrument…)", value=state["words"]).classes("flex-grow")
                ui.button("Search", icon="search", on_click=lambda: search(words.value))
            with ui.row().classes("gap-1"):
                for mood in free_music.MOODS:
                    ui.button(mood, on_click=lambda m=mood: pick(m)).props("outline dense rounded size=sm no-caps")

            @ui.refreshable
            def results() -> None:
                if state["busy"]:
                    with ui.row().classes("items-center"):
                        ui.spinner(size="lg")
                        ui.label("Searching the free music libraries… this can take up to a minute.").classes("text-gray-500")
                    return
                if state["error"]:
                    ui.label(state["error"]).classes("text-red-600")
                    return
                if not state["songs"]:
                    ui.label("No songs found yet. Try other words.").classes("text-gray-500")
                    return
                with ui.scroll_area().classes("w-full h-96"):
                    for song in state["songs"]:
                        with ui.row().classes("w-full items-center no-wrap gap-2 py-1 border-b"):
                            with ui.column().classes("gap-0 flex-grow"):
                                ui.label(f"{song.title} · {song.artist}").classes("text-sm font-medium")
                                ui.label(f"{fmt_length(song.seconds)} · {song.licence} · {song.source}").classes(
                                    "text-xs text-gray-500")
                            ui.audio(song.url, autoplay=False).props("preload=none").classes("w-56")
                            ui.button("Use this", on_click=lambda sg=song: use(sg)).props("dense")

            results()

            async def search(text: str) -> None:
                state.update(words=text, busy=True, error="")
                results.refresh()
                try:
                    state["songs"] = await run.io_bound(free_music.search, text)
                except Exception as exc:
                    traceback.print_exc()
                    state["error"] = f"Could not search for music (is the internet connected?): {exc}"
                state["busy"] = False
                results.refresh()

            async def pick(mood: str) -> None:
                words.value = mood
                await search(mood)

            def get_from_page(song) -> None:
                name = free_music.remember_credit(song, app_dir() / "music")
                since = time.time()
                ui.navigate.to(song.page, new_tab=True)
                waiting = ui.notification(
                    f"{song.source or 'This site'} only allows downloading on its own page, now open in a new tab. "
                    "Click its download button there; the app picks the song up from your Downloads folder by itself.",
                    spinner=True, timeout=None, multi_line=True, close_button=True,
                )

                async def check() -> None:
                    found = free_music.find_download(name, since)
                    if found is None:
                        if time.time() - since > 15 * 60:
                            watcher.cancel()
                            waiting.dismiss()
                        return
                    watcher.cancel()
                    await asyncio.sleep(1.5)  # let the browser finish writing it
                    path = await run.io_bound(free_music.adopt, found, song, app_dir() / "music")
                    waiting.dismiss()
                    self.music_input.value = str(path)
                    dialog.close()
                    ui.notify(f"Got it: using {song.title} by {song.artist}.", type="positive")

                watcher = ui.timer(2.0, check)

            async def use(song) -> None:
                note = ui.notification(f"Downloading {song.title}…", spinner=True, timeout=None)
                try:
                    path = await run.io_bound(free_music.download, song, app_dir() / "music")
                except Exception as exc:
                    traceback.print_exc()
                    note.dismiss()
                    if "403" in str(exc) and song.page:
                        # Some sites (like ccMixter) only allow downloads from their own web page: open it, and
                        # pick the song up from Downloads as soon as it arrives.
                        get_from_page(song)
                    else:
                        ui.notify(f"Could not download it: {exc}", type="negative", multi_line=True)
                    return
                note.dismiss()
                self.music_input.value = str(path)
                dialog.close()
                ui.notify(f"Using {song.title} by {song.artist}.", type="positive")

            with ui.row().classes("w-full justify-end"):
                ui.button("Close", on_click=dialog.close).props("flat")
        dialog.open()
        ui.timer(0.1, lambda: search(state["words"]), once=True)

    def music_site(self, name: str, url: str) -> None:
        """Open a free music library in the browser; a song you download there is picked up from Downloads."""
        if url.startswith("itms") and sys.platform == "darwin":
            subprocess.Popen(["open", url])  # (straight into the Music app, no browser tab)
        else:
            ui.navigate.to(url, new_tab=True)
        since = time.time()
        if getattr(self, "music_watch", None):
            self.music_watch.cancel()
        ui.notify("The Music app opens on the iTunes Store. Buy a song there: it shows up here by itself." if
                  url.startswith("itms") else
                  f"{name} is open in your browser. Listen, then click Download on a song: it shows up here.",
                  multi_line=True, timeout=8000)

        def check() -> None:
            found = new_song_since(since)
            if found is None:
                if time.time() - since > 30 * 60:
                    self.music_watch.cancel()
                return
            if found.stat().st_size == 0 or time.time() - found.stat().st_mtime < 1.5:
                return  # (the browser is still writing it)
            self.music_watch.cancel()
            self.offer_download(found)

        with self.window_box:
            self.music_watch = ui.timer(2.0, check)

    def offer_download(self, found: Path) -> None:
        with self.window_box, ui.dialog() as dialog, ui.card().classes("gap-3"):
            ui.label("A new song in your Downloads").classes("text-lg font-semibold")
            ui.label(found.name).classes("font-medium")
            url = f"/downloaded/{hashlib.sha1(str(found).encode()).hexdigest()[:12]}{found.suffix.lower()}"
            app.add_media_file(local_file=found, url_path=url)  # (just this file, to listen to it here)
            ui.audio(url).classes("w-96")

            def use() -> None:
                folder = app_dir() / "music"
                folder.mkdir(parents=True, exist_ok=True)
                path = folder / found.name
                if not path.exists():
                    import shutil
                    shutil.copyfile(found, path)  # (a copy: moving or deleting Downloads later changes nothing)
                self.music_input.value = str(path)
                dialog.close()
                ui.notify(f"Using {found.stem}. If the site asks for credit, copy it into your video's description.",
                          type="positive", multi_line=True)

            with ui.row().classes("w-full justify-end"):
                ui.button("Not this one", on_click=dialog.close).props("flat")
                ui.button("Use this song", icon="check", on_click=use).props("color=primary")
        dialog.open()

    async def browse_music(self) -> None:
        path = await run.io_bound(choose_with_finder, "music")
        if path:
            self.music_input.value = path

    async def update_video(self) -> None:
        """Your changes in the Storyboard (a longer clip, another part, a new order) into the video, without a new
        cut: only the changed clips are made again."""
        if not self.trip or not self.trip.storyboard():
            ui.notify("Make the first cut first.", type="warning")
            return
        self.panels.set_value(self.video_tab)
        await self.make()

    async def make(self) -> None:
        r = self.rendering
        if r["busy"] or not self.trip:
            return
        width, height = RESOLUTIONS[self.resolution.value]
        # A phone video shows a narrow upright slice of each shot: find where the people are, to keep them in it.
        items = await run.io_bound(self.trip.media_items, False, height > width)
        if not items:
            ui.notify("Nothing is ticked for the video yet. Pick items in the Library.", type="warning")
            return
        music = (self.music_input.value or "").strip().strip("'\"") or None
        if music and not Path(music).expanduser().is_file():
            ui.notify("The music file was not found.", type="warning")
            return
        settings = RenderSettings(
            width=width,
            height=height,
            photo_seconds=self.photo_len.value,
            max_clip_seconds=self.clip_len.value,
            transition=self.transition.value,
            transition_seconds=self.t_len.value,
            music_path=str(Path(music).expanduser()) if music else None,
            music_volume=self.music_vol.value,
            look=self.trip.look(),
        )
        settings.end_black, settings.end_title = self.trip.end_card()
        settings.opening = self.trip.opening_card()
        settings.closing = self.trip.closing_card()
        settings.titles = self.resolved_titles()
        title = (self.name_input.value or "").strip() or self.trip.folder.name or "My trip"
        safe = "".join(c for c in title if c.isalnum() or c in " -_").strip() or "My trip"
        out = OUTPUT_DIR / f"{safe} {datetime.now():%Y-%m-%d %H%M}.mp4"

        def on_progress(fraction: float, message: str) -> None:
            r["progress"], r["message"] = fraction, message

        r.update(busy=True, progress=0.0, message="Starting…")
        self.render_status.text = ""
        self.make_button.disable()
        self.render_progress.set_visibility(True)
        self.result_box.clear()
        try:
            result = await run.io_bound(render, items, settings, out, on_progress, app_dir() / "clip-cache")
            self.render_status.text = f"Saved to {result}"
            credit = free_music.credit_for(settings.music_path, app_dir() / "music") if settings.music_path else None
            if credit:
                credits = result.with_name(result.stem + " - music credits.txt")
                credits.write_text(f"Music: {credit}\n")
                self.render_status.text += f" (music credit saved in {credits.name})"
            with self.result_box:
                ui.video(f"/videos/{result.name}").classes("w-full")
                ui.button("Show in Finder", icon="folder", on_click=lambda: reveal(result)).props("outline")
        except Exception as exc:  # show any problem on screen instead of failing silently
            traceback.print_exc()
            self.render_status.text = f"Something went wrong: {exc}"
        finally:
            r["busy"] = False
            self.make_button.enable()


_served: dict[str, str] = {}

RED_LINE = ('<div class="tm-red" style="position:absolute;top:0;bottom:0;left:0;width:2px;margin-left:-1px;'
            'background:#ef4444;box-shadow:0 0 4px #ef4444;pointer-events:none;z-index:5"><div style="position:absolute;'
            'top:-2px;left:-5px;width:12px;height:12px;border-radius:50%;background:#ef4444"></div></div>')


# Plays a video preview while the mouse is over it, and stops it when the mouse leaves.
HOVER_SCRIPT = """
<script>
// Hover a video preview to play it. It plays only from data-start to data-end, then starts again.
function tmVideo(box) {
  const v = box.querySelector('video');
  if (!v.src) {
    v.preload = 'auto';  // fetch it now so we can jump to the start of the part
    v.src = v.dataset.src;
    v.addEventListener('timeupdate', () => {
      // (only while playing: a paused video showing the last frame of the part stays there)
      if (!v.paused && v.currentTime >= parseFloat(box.dataset.end) - 0.05) v.currentTime = parseFloat(box.dataset.start);
    });
    v.addEventListener('ended', () => { v.currentTime = parseFloat(box.dataset.start); v.play().catch(() => {}); });
  }
  const rot = parseInt(box.dataset.rot || '0');
  if (rot === 90 || rot === 270) {  // turned sideways: swap the frame's width and height
    const w = box.clientWidth, h = box.clientHeight;
    // (set inset first: it is shorthand for top/left/right/bottom and would undo them)
    Object.assign(v.style, {inset: 'auto', width: h + 'px', height: w + 'px', left: (w - h) / 2 + 'px', top: (h - w) / 2 + 'px'});
  }
  v.style.transform = rot ? `rotate(${rot}deg)` : '';
  v.style.display = 'block';
  return v;
}
function tmPlayFrom(box, from) {
  const v = tmVideo(box);
  // With sound if allowed; the browser only allows sound after a first click on the page.
  if (box.dataset.sound === '1') v.muted = false;  // try with sound again (it may have been refused before your first click)
  const go = () => { v.currentTime = from; v.play().catch(() => { v.muted = true; v.play().catch(() => {}); }); };
  if (v.readyState >= 1) go(); else v.addEventListener('loadedmetadata', go, {once: true});
}
function tmStop(box) {
  const v = box.querySelector('video');
  v.pause();
  v.style.display = 'none';
}
let tmPlaying = null;
// A title bar under the draft: drag it to move it, drag an end to change its length, click to change it.
function tmTitleDown(e, el) {
  const row = el.parentElement, total = parseFloat(row.dataset.total), W = row.clientWidth;
  const box = el.getBoundingClientRect(), x0 = e.clientX;
  const mode = (x0 - box.left < 12) ? 'left' : (box.right - x0 < 12) ? 'right' : 'move';
  e.preventDefault();
  const l0 = el.offsetLeft, w0 = el.offsetWidth;
  let moved = false;
  el.setPointerCapture(e.pointerId);
  const move = (ev) => {
    const dx = ev.clientX - x0;
    if (Math.abs(dx) > 3) moved = true;
    let l = l0, w = w0;
    if (mode === 'move') l = Math.max(0, Math.min(W - w0, l0 + dx));
    else if (mode === 'left') { l = Math.max(0, Math.min(l0 + w0 - 6, l0 + dx)); w = w0 + (l0 - l); }
    else w = Math.max(6, Math.min(W - l0, w0 + dx));
    // Near a cut or a pin (8 pixels), the moved edge snaps to it.
    const snaps = (row.dataset.snaps || '').split(',').filter(x => x).map(x => W * parseFloat(x) / total);
    const snap = (x) => { let best = x, d = 8; for (const p of snaps) if (Math.abs(p - x) < d) { d = Math.abs(p - x); best = p; } return best; };
    if (mode === 'move') { const s1 = snap(l), s2 = snap(l + w) - w; l = Math.abs(s1 - l) <= Math.abs(s2 - l) ? s1 : s2; }
    else if (mode === 'left') { const e0 = l + w; l = Math.min(e0 - 6, snap(l)); w = e0 - l; }
    else w = Math.max(6, snap(l + w) - l);
    el.style.left = (100 * l / W) + '%'; el.style.width = (100 * w / W) + '%';
    const a = total * l / W, len = total * w / W;
    el.dataset.times = `${tmClock2(a)} → ${tmClock2(a + len)}`;
    let tip = el.querySelector('.tm-times');
    if (!tip) { tip = document.createElement('div'); tip.className = 'tm-times';
      tip.style.cssText = 'position:absolute;left:0;top:-18px;background:#1e293b;color:white;font-size:11px;padding:0 4px;border-radius:3px;white-space:nowrap;z-index:7';
      el.appendChild(tip); el.style.overflow = 'visible'; }
    tip.textContent = el.dataset.times;
    const img = document.querySelector(`.tm-live-title[data-id="${el.dataset.id}"]`);
    if (img) { img.dataset.start = a; img.dataset.len = len; }
    const v = document.getElementById('tm-draft');
    if (v) { v.pause(); tmSeekSoft(v, mode === 'right' ? Math.max(0, a + len - 0.1) : a + Math.min(0.5, len / 2)); }
  };
  const up = () => {
    el.removeEventListener('pointermove', move); el.removeEventListener('pointerup', up);
    if (!moved) {  // a click picks the title; a double-click opens its small window
      const now = Date.now(), id = el.dataset.id;
      if (window.tmTitleTap && window.tmTitleTap.id === id && now - window.tmTitleTap.at < 400) {
        clearTimeout(window.tmTitleTap.timer); window.tmTitleTap = null;
        emitEvent('tm_title_menu', {id: id}); return;
      }
      window.tmTitleTap = {id: id, at: now, timer: setTimeout(() => { window.tmTitleTap = null;
        emitEvent('tm_title_click', {id: id}); }, 400)};
      return;
    }
    emitEvent('tm_title_move', {id: el.dataset.id, start: total * el.offsetLeft / W, length: total * el.offsetWidth / W});
  };
  el.addEventListener('pointermove', move); el.addEventListener('pointerup', up);
}
// The post-it window: Shift sticks a post-it at the video's current second; the strip scrubs through it.
document.addEventListener('keydown', (e) => {
  const v = document.getElementById('tm-pi');
  if (!v || e.key !== 'Shift' || e.repeat || ['INPUT', 'TEXTAREA'].includes(e.target.tagName)) return;
  emitEvent('tm_postit', {t: v.currentTime});
});
function tmPostitScrub(e, strip, a, b) {
  const v = document.getElementById('tm-pi');
  if (!v) return;
  const r = strip.getBoundingClientRect();
  v.pause();
  v.currentTime = a + (b - a) * Math.max(0, Math.min(1, (e.clientX - r.left) / r.width));
}
// The draft under the editor: the clip playing now is underlined, the red line shows where the player is,
// and the timeline titles show over the video at their time, fading in and out.
function tmDraftNow(t, loaded) {
  const v = document.getElementById('tm-draft');
  if (loaded && v && window.tmDraftT !== undefined && Math.abs(v.currentTime - window.tmDraftT) > 0.2) {
    v.currentTime = Math.min(window.tmDraftT, (v.duration || 1e9) - 0.05); return;  // carry on from where you were
  }
  if (!loaded) window.tmDraftT = t;
  const clock = document.getElementById('tm-edclock');
  if (clock && v) clock.textContent = `${tmClock2(t)} / ${tmClock2(v.duration || 0)}`;
  const sc = document.getElementById('tm-tl-scroll'), head = document.getElementById('tm-playhead');
  if (sc && head && v && !v.paused) {  // the timeline moves along as the video plays
    const x = head.offsetLeft;
    if (x < sc.scrollLeft + 40 || x > sc.scrollLeft + sc.clientWidth * 0.7) sc.scrollLeft = Math.max(0, x - sc.clientWidth * 0.3);
  }
  document.querySelectorAll('[id^=tm-ed-]').forEach(el => {
    const a = parseFloat(el.dataset.start), b = a + parseFloat(el.dataset.length);
    el.style.boxShadow = (!isNaN(a) && t >= a && t < b) ? 'inset 0 -5px 0 #f59e0b' : '';
  });
  const line = document.getElementById('tm-playhead'), tl = document.getElementById('tm-timeline');
  if (line && tl) line.style.left = (100 * t / parseFloat(tl.dataset.total)) + '%';
  document.querySelectorAll('.tm-live-title').forEach(img => {
    const a = parseFloat(img.dataset.start), len = parseFloat(img.dataset.len);
    const fin = Math.min(parseFloat(img.dataset.fin), len / 2), fout = Math.min(parseFloat(img.dataset.fout), len / 2);
    let o = 0;
    if (t >= a && t <= a + len) o = Math.min(1, fin > 0 ? (t - a) / fin : 1, fout > 0 ? (a + len - t) / fout : 1);
    img.style.opacity = o;
  });
}
// The full-screen editor: the timeline scrolls back to where it was, the music's waveform is drawn under the clips.
function tmEditorReady() {
  const sc = document.getElementById('tm-tl-scroll');
  if (sc && window.tmTlScroll) sc.scrollLeft = window.tmTlScroll;
  const cv = document.getElementById('tm-edwave');
  if (!cv) return;
  const draw = (peaks) => {
    const W = cv.width = cv.clientWidth * (window.devicePixelRatio || 1), H = cv.height = cv.clientHeight * (window.devicePixelRatio || 1);
    const g = cv.getContext('2d'); g.fillStyle = '#38bdf8'; g.globalAlpha = 0.75;
    const secs = parseFloat(cv.dataset.secs), N = secs > 0 ? Math.round(secs / 0.02) : peaks.length;  // (the song: as far as the video goes)
    for (let x = 0; x < W; x++) {
      const a = Math.floor(N * x / W), b = Math.max(a + 1, Math.floor(N * (x + 1) / W));
      let m = 0; for (let i = a; i < b && i < peaks.length; i++) if (peaks[i] > m) m = peaks[i];
      g.fillRect(x, H / 2 - m * H / 2, 1, Math.max(1, m * H));
    }
  };
  window.tmWaves = window.tmWaves || {};
  const src = cv.dataset.src;
  if (!src) return;
  if (window.tmWaves[src]) { draw(window.tmWaves[src]); return; }
  const Off = window.OfflineAudioContext || window.webkitOfflineAudioContext;
  fetch(src).then(r => r.arrayBuffer()).then(buf => new Promise((ok, bad) => {
    const p = new Off(1, 2, 44100).decodeAudioData(buf, ok, bad); if (p && p.then) p.then(ok, bad);
  })).then(audio => {
    const data = audio.getChannelData(0), per = Math.max(1, Math.round(audio.sampleRate * 0.02));
    const peaks = new Float32Array(Math.ceil(data.length / per));
    let top = 0;
    for (let i = 0; i < peaks.length; i++) {
      let m = 0; const end = Math.min(data.length, (i + 1) * per);
      for (let j = i * per; j < end; j++) { const v = Math.abs(data[j]); if (v > m) m = v; }
      peaks[i] = m; if (m > top) top = m;
    }
    if (top > 0) for (let i = 0; i < peaks.length; i++) peaks[i] /= top;
    window.tmWaves[src] = peaks; draw(peaks);
  }).catch(e => console.warn('draft waveform', e));
}
document.addEventListener('keydown', (e) => {  // keys in the editor: Space, ← →, Esc
  const ed = document.querySelector('.tm-editor');
  if (!ed || document.querySelector('.q-dialog') || ['INPUT', 'TEXTAREA'].includes(e.target.tagName)) return;
  const v = document.getElementById('tm-draft');
  if (e.key === 'Escape') { e.preventDefault(); const x = ed.querySelector('.tm-editor-close'); if (x) x.click(); }
  else if (!v) return;
  else if (e.code === 'Space') { e.preventDefault(); e.stopPropagation(); if (v.paused) v.play(); else v.pause(); }
  else if (e.key === 'ArrowLeft' || e.key === 'ArrowRight') {
    e.preventDefault(); e.stopPropagation();
    v.currentTime = Math.max(0, v.currentTime + (e.key === 'ArrowLeft' ? -0.5 : 0.5)); tmDraftNow(v.currentTime);
  }
}, true);
// Scrubbing: press on the ruler, the red line or the music and drag; the video shows the frame under the cursor.
// A new frame is asked for only once the last one is shown, so it keeps up with the mouse.
function tmSeekSoft(v, t) {
  // One seek at a time (smooth dragging), but never stuck: some browsers (Safari) don't always say a seek ended.
  v._want = t;
  if (v._busy && Date.now() - (v._busyAt || 0) < 250) return;
  v._busy = true;
  const go = () => {
    const want = v._want; v._want = null;
    if (want === null || want === undefined) { v._busy = false; return; }
    v._busyAt = Date.now();
    v.currentTime = want;
    v.addEventListener('seeked', () => { if (v._want !== null && v._want !== undefined) go(); else v._busy = false; }, {once: true});
    setTimeout(() => { if (v._busy && Date.now() - v._busyAt >= 240) go(); }, 250);
  };
  go();
}
function tmScrubStart(e) {
  const tl = document.getElementById('tm-timeline'), v = document.getElementById('tm-draft');
  if (!tl || !v) return;
  e.preventDefault();
  const total = parseFloat(tl.dataset.total), head = document.getElementById('tm-playhead');
  const was = !v.paused; v.pause();
  const at = (ev) => {
    const r = tl.getBoundingClientRect();
    const t = Math.max(0, Math.min(total - 0.04, total * (ev.clientX - r.left) / r.width));
    if (head) head.style.left = (100 * t / total) + '%';
    window.tmDraftT = t;
    const clock = document.getElementById('tm-edclock');
    if (clock) clock.textContent = `${tmClock2(t)} / ${tmClock2(v.duration || 0)}`;
    tmSeekSoft(v, t);
  };
  at(e);
  const move = (ev) => at(ev);
  const up = () => { document.removeEventListener('pointermove', move); document.removeEventListener('pointerup', up);
                     if (was) v.play(); };
  document.addEventListener('pointermove', move); document.addEventListener('pointerup', up);
}
// A click on a clip in the editor: the red line and the video go to that very moment of it.
function tmClipClick(e, el) {
  const v = document.getElementById('tm-draft'), r = el.getBoundingClientRect();
  if (!v) return;
  const t = parseFloat(el.dataset.start) + parseFloat(el.dataset.length) * Math.max(0, Math.min(1, (e.clientX - r.left) / r.width));
  v.pause(); window.tmDraftT = t; tmSeekSoft(v, t);
  const tl = document.getElementById('tm-timeline'), head = document.getElementById('tm-playhead');
  if (tl && head) head.style.left = (100 * t / parseFloat(tl.dataset.total)) + '%';
}
document.addEventListener('click', (e) => {  // (on every clip box of the editor's timeline)
  const el = e.target.closest && e.target.closest('.tm-editor [id^=tm-ed-][data-start]');
  if (el) tmClipClick(e, el);
}, true);
// A double-click on the timeline (the ruler, the red line or the empty space, not on a clip): back to the start.
document.addEventListener('dblclick', (e) => {
  if (!e.target.closest || !e.target.closest('.tm-editor #tm-timeline')) return;
  if (e.target.closest('[id^=tm-ed-][data-start]')) return;  // (a clip: its own window opens)
  const v = document.getElementById('tm-draft'), head = document.getElementById('tm-playhead');
  if (v) { v.pause(); window.tmDraftT = 0; tmSeekSoft(v, 0); }
  if (head) head.style.left = '0%';
  const clock = document.getElementById('tm-edclock');
  if (clock && v) clock.textContent = `${tmClock2(0)} / ${tmClock2(v.duration || 0)}`;
  const sc = document.getElementById('tm-tl-scroll');
  if (sc) sc.scrollTo({left: 0, behavior: 'smooth'});
}, true);
function tmSeekRuler(e, el, total) {
  const v = document.getElementById('tm-draft'), r = el.getBoundingClientRect();
  if (v) { v.currentTime = total * (e.clientX - r.left) / r.width; tmDraftNow(v.currentTime); }
}
// The instant player: plays the clips straight from your files, one after another, with the song. It looks like a
// video to the rest of the page (currentTime, paused, duration, play, pause, a 'seeked' event), so the scrubbing,
// the keys and the red line work on it as on the draft. Three layers: the clip now, the one before it (while the
// next one fades in) and the next one getting ready.
window.tmInst = window.tmInst || {};
// The collage window: drag a picture to move it in its place, scroll (or pinch) to zoom, drag the ⠿ handle onto
// another place to swap them. Each change goes to the app as a 'tmframe' or 'tmswap' event on the stage.
function tmCollageEdit(stageId) {
  const stage = document.getElementById(stageId);
  if (!stage || stage._ready) return;
  stage._ready = true;
  const send = (name, detail) => stage.dispatchEvent(new CustomEvent(name, {bubbles: false, detail: detail}));
  const media = (tile) => tile.querySelector('.tm-cm');
  const show = (tile) => {
    const m = media(tile), x = +tile.dataset.x, y = +tile.dataset.y, z = +tile.dataset.z;
    m.style.objectPosition = `${100 * x}% ${100 * y}%`;
    m.style.transformOrigin = `${100 * x}% ${100 * y}%`;
    m.style.scale = z;
    const label = tile.querySelector('.tm-cz'); if (label) label.textContent = Math.abs(z - 1) > 0.01 ? `${z.toFixed(1)}×` : '';
  };
  const room = (tile) => {  // how far the picture can move (px on screen) left-right and up-down
    const m = media(tile), tw = tile.clientWidth, th = tile.clientHeight, z = +tile.dataset.z;
    const nw = m.naturalWidth || m.videoWidth || tw, nh = m.naturalHeight || m.videoHeight || th;
    const s = Math.max(tw / nw, th / nh);
    return [nw * s * z - tw, nh * s * z - th];
  };
  const save = (tile) => send('tmframe', {k: +tile.dataset.k, x: +tile.dataset.x, y: +tile.dataset.y, z: +tile.dataset.z});
  stage.querySelectorAll('.tm-ctile').forEach(tile => {
    show(tile);
    let drag = null, timer = null;
    tile.addEventListener('pointerdown', e => {
      if (e.target.closest('.tm-cbtn') || e.target.closest('.tm-chandle')) return;
      e.preventDefault(); tile.setPointerCapture(e.pointerId);
      drag = {x: e.clientX, y: e.clientY, fx: +tile.dataset.x, fy: +tile.dataset.y, room: room(tile)};
      tile.style.cursor = 'grabbing';
    });
    tile.addEventListener('pointermove', e => {
      if (!drag) return;
      const [rx, ry] = drag.room;
      if (+tile.dataset.z < 0.999) {  // smaller than its place: the picture follows the mouse anywhere in it
        const z = +tile.dataset.z, fw = tile.clientWidth * (1 - z), fh = tile.clientHeight * (1 - z);
        if (fw > 1) tile.dataset.x = Math.min(1, Math.max(0, drag.fx + (e.clientX - drag.x) / fw));
        if (fh > 1) tile.dataset.y = Math.min(1, Math.max(0, drag.fy + (e.clientY - drag.y) / fh));
      } else {
        if (rx > 1) tile.dataset.x = Math.min(1, Math.max(0, drag.fx - (e.clientX - drag.x) / rx));
        if (ry > 1) tile.dataset.y = Math.min(1, Math.max(0, drag.fy - (e.clientY - drag.y) / ry));
      }
      show(tile);
    });
    const end = () => { if (!drag) return; drag = null; tile.style.cursor = 'grab'; save(tile); };
    tile.addEventListener('pointerup', end); tile.addEventListener('pointercancel', end);
    tile.addEventListener('wheel', e => {
      e.preventDefault();
      const z = Math.min(4, Math.max(0.4, +tile.dataset.z * Math.exp(-e.deltaY * (e.ctrlKey ? 0.01 : 0.002))));
      tile.dataset.z = z; show(tile);
      clearTimeout(timer); timer = setTimeout(() => save(tile), 350);
    }, {passive: false});
    tile.querySelectorAll('[data-zoom]').forEach(b => b.addEventListener('click', e => {
      e.stopPropagation();
      tile.dataset.z = Math.min(4, Math.max(0.4, +tile.dataset.z * (+b.dataset.zoom))); show(tile); save(tile);
    }));
    const handle = tile.querySelector('.tm-chandle');
    if (handle) handle.addEventListener('dragstart', e => { e.dataTransfer.setData('text/plain', tile.dataset.k); });
    tile.addEventListener('dragover', e => { e.preventDefault(); tile.style.outline = '3px solid #a855f7'; });
    tile.addEventListener('dragleave', () => { tile.style.outline = ''; });
    tile.addEventListener('drop', e => {
      e.preventDefault(); tile.style.outline = '';
      const from = +e.dataTransfer.getData('text/plain');
      if (!isNaN(from) && from !== +tile.dataset.k) send('tmswap', {a: from, b: +tile.dataset.k});
    });
    // A video plays its moment over and over.
    const v = tile.querySelector('video.tm-cm');
    if (v) {
      const a = +tile.dataset.a, len = +tile.dataset.len;
      v.addEventListener('loadedmetadata', () => { v.currentTime = a; v.play().catch(() => {}); show(tile); });
      v.addEventListener('timeupdate', () => { if (v.currentTime > a + len || v.currentTime < a - 0.3) v.currentTime = a; });
    }
    const im = tile.querySelector('img.tm-cm');
    if (im) im.addEventListener('load', () => show(tile));
  });
}
// How you framed a picture in its place of a collage (see render.frame_crop): the visible part at x% y%,
// zoomed in z times around that point, cut to its place.
function tmFrameCss(f) {
  f = f || {};
  const x = f.x === undefined ? 0.5 : f.x, y = f.y === undefined ? 0.5 : f.y, z = Math.min(4, Math.max(0.4, f.z || 1)), k = Math.max(0, 1 - 1 / z);
  return `object-position:${100 * x}% ${100 * y}%;transform-origin:${100 * x}% ${100 * y}%;scale:${z};` +
    `clip-path:inset(${100 * y * k}% ${100 * (1 - x) * k}% ${100 * (1 - y) * k}% ${100 * x * k}%);`;
}
function tmInstant(id, data, onTime) {
  const el = document.getElementById(id);
  if (window.tmInst[id]) window.tmInst[id].stop();
  if (!el) return null;
  const clips = data.clips, total = data.total, fade = data.fade || 0, loop = data.loop;
  const slots = [...el.querySelectorAll('.tm-slot')];
  slots.forEach(s => { s._i = -1; });
  let music = null;
  if (data.music) { music = new Audio(data.music); music.preload = 'auto'; music.volume = Math.min(1, data.mvol || 0.5); }
  let playing = false, t = loop ? loop[0] : 0, t0 = 0, p0 = 0, raf = 0, dead = false;
  const now = () => playing ? Math.min(total, t0 + (performance.now() - p0) / 1000) : t;
  const button = el.querySelector('.tm-iplay');
  const look = (s, c) => {  // turned and cropped like in the video
    const m = c.k === 'v' ? s.querySelector('video') : s.querySelector('img');
    const W = el.clientWidth || 16, H = el.clientHeight || 9;
    const mw = m.videoWidth || m.naturalWidth || W, mh = m.videoHeight || m.naturalHeight || H;
    let tf = '';
    const rot = ((c.rot || 0) % 360 + 360) % 360;
    if (rot) {
      const side = rot === 90 || rot === 270;
      const fit0 = Math.min(W / mw, H / mh), fit1 = side ? Math.min(W / mh, H / mw) : fit0;
      tf += `rotate(${rot}deg) scale(${fit1 / fit0})`;
    }
    if (c.crop && c.crop.zoom > 1) {
      m.style.transformOrigin = `${100 * c.crop.x}% ${100 * c.crop.y}%`;
      tf += ` scale(${c.crop.zoom})`;
    } else m.style.transformOrigin = '50% 50%';
    return tf;
  };
  const place = (s, c, i) => {
    s._i = i;
    const v = s.querySelector('video'), im = s.querySelector('img');
    (s._extra || []).forEach(x => { if (x.el.pause) x.el.pause(); x.el.remove(); });
    s._extra = [];
    [v, im].forEach(m => { if (m._css === undefined) m._css = m.style.cssText; });
    // Behind a picture that doesn't fill the screen (an upright clip): itself, big and blurred, as in the video.
    let bg = s.querySelector('.tm-bgblur');
    if (!bg) {
      bg = document.createElement('div');
      bg.className = 'tm-bgblur';
      bg.style.cssText = 'position:absolute;inset:-30px;background:center/cover no-repeat;filter:blur(22px) brightness(.75)';
      s.insertBefore(bg, s.firstChild);
    }
    bg.style.backgroundImage = c.fb ? `url("${c.fb}")` : 'none';
    if (c.tiles) {  // a collage or videos side by side: the first in its place, the others around it
      const t0 = c.tiles[0], box = `left:${100 * t0.x}%;top:${100 * t0.y}%;width:${100 * t0.w}%;height:${100 * t0.h}%;`;
      [v, im].forEach(m => { const hide = m.style.display; m.style.cssText = 'position:absolute;object-fit:cover;' + box + tmFrameCss(t0.f); m.style.display = hide; });
      c.tiles.slice(1).forEach(t => {
        const el = document.createElement(t.k === 'v' ? 'video' : 'img');
        el.style.cssText = `position:absolute;left:${100 * t.x}%;top:${100 * t.y}%;width:${100 * t.w}%;height:${100 * t.h}%;object-fit:cover;opacity:0;transition:opacity .15s` + ';' + tmFrameCss(t.f);
        if (t.rot) el.style.rotate = t.rot + 'deg';
        if (t.k === 'v') { el.muted = true; el.playsInline = true; el.preload = 'auto'; el.onloadedmetadata = () => { el._ready = true; }; }
        el.onerror = () => { el.onerror = null; if (t.fb && t.k !== 'v') el.src = t.fb; };
        el.src = t.src;
        s.appendChild(el);
        s._extra.push({el: el, t: t});
      });
    } else {
      [v, im].forEach(m => { const hide = m.style.display; m.style.cssText = m._css; m.style.display = hide; });
    }
    if (c.k === 'v') {
      im.style.display = 'none'; v.style.display = '';
      if (v._src !== c.src) {
        v._src = c.src; v._ready = false; v.src = c.src;
        v.onloadedmetadata = () => { v._ready = true; v._tf = null; tick(true); };
      }
    } else {
      v.pause(); v.style.display = 'none'; im.style.display = '';
      if (im._src !== c.src) {
        im._src = c.src; im.onerror = () => { im.onerror = null; if (c.fb && im.src !== c.fb) im.src = c.fb; };
        im.onload = () => tick(true); im.src = c.src;
      }
    }
    s.querySelector('.tm-badge').textContent = c.rev ? '⇆ backwards in the finished video' : '';
  };
  let shown = -1;
  const cover = el.querySelector('.tm-ov');
  const tick = (hard) => {
    const T = Math.min(now(), total - 0.02);  // (at the very end: its last picture)
    if (cover) cover.style.opacity = 0;
    for (let i = 0; i < clips.length; i++) {  // who goes on which layer
      const c = clips[i];
      if (!(T >= c.st - 2.5 && T < c.st + c.len)) continue;
      const s = slots[i % slots.length];
      if (s._i === i) continue;
      const o = s._i >= 0 ? clips[s._i] : null;
      if (o && T >= o.st && T < o.st + o.len) continue;  // (still showing: it waits)
      place(s, c, i);
    }
    slots.forEach(s => {
      const i = s._i;
      if (i < 0) { s.style.opacity = 0; return; }
      const c = clips[i], v = s.querySelector('video'), im = s.querySelector('img');
      const on = T >= c.st && T < c.st + c.len;
      const q = on && c.fin > 0.05 ? (T - c.st) / c.fin : 1;
      tmTrLook(s, on ? c.tr : '', q, cover);
      s.style.zIndex = i + 1;
      const local = Math.max(0, Math.min(c.len, T - c.st));
      if (c.k === 'v') {
        v.style.transform = look(s, c);
        const want = c.a + local * c.sp;
        if (!v._ready) return;
        if (on && playing) {
          if (Math.abs(v.playbackRate - c.sp) > 0.001) v.playbackRate = c.sp;
          v.muted = !c.snd; v.volume = Math.min(1, c.vol);
          if (Math.abs(v.currentTime - want) > 0.3 && !v.seeking) v.currentTime = want;
          if (v.paused) v.play().catch(() => {});
        } else {
          if (!v.paused) v.pause();
          const at = on ? want : c.a;
          if ((hard || Math.abs(v.currentTime - at) > 0.05) && !v.seeking) v.currentTime = at;
        }
      } else {
        im.style.transform = look(s, c) + ` scale(${1 + 0.05 * local / Math.max(0.1, c.len)})`;  // a gentle push in
      }
      (s._extra || []).forEach(({el, t}) => {
        if (t.k === 'v') {
          el.style.opacity = 1;
          if (!el._ready) return;
          const want = t.a + local;
          if (on && playing) {
            if (Math.abs(el.currentTime - want) > 0.3 && !el.seeking) el.currentTime = want;
            if (el.paused) el.play().catch(() => {});
          } else {
            if (!el.paused) el.pause();
            if (Math.abs(el.currentTime - want) > 0.05 && !el.seeking) el.currentTime = want;
          }
        } else {
          const shown = local >= (t.at || 0);
          el.style.opacity = shown ? 1 : 0;
          el.style.transform = shown ? 'scale(1)' : 'scale(.82)';
        }
      });
    });
    if (button) button.textContent = playing ? '❚❚' : '▶';
    if (onTime && (hard || T !== shown)) onTime(T);
    shown = T;
  };
  const frame = () => {
    if (dead) return;
    if (!el.isConnected) { stop(); return; }
    if (playing) {
      let T = now();
      if (loop && T >= loop[1]) { t = loop[0]; t0 = t; p0 = performance.now(); if (music) music.currentTime = t; T = t; }
      if (T >= total - 0.01) { playing = false; t = total; if (music) music.pause(); }
      if (music && !music.paused && T < (music.duration || 1e9) && Math.abs(music.currentTime - T) > 0.25) music.currentTime = T;
    }
    tick(false);
    raf = requestAnimationFrame(frame);
  };
  const stop = () => {
    dead = true; cancelAnimationFrame(raf); playing = false;
    if (music) { music.pause(); music.src = ''; }
    slots.forEach(s => s.querySelectorAll('video').forEach(v => v.pause()));
    if (window.tmInst[id] && window.tmInst[id].el === el) delete window.tmInst[id];
  };
  const seek = (x) => {
    t = Math.max(0, Math.min(total, x));
    if (playing) { t0 = t; p0 = performance.now(); }
    if (music && t < (music.duration || 1e9)) music.currentTime = t;
    tick(true);
    let done = false;
    const fire = () => { if (!done) { done = true; el.dispatchEvent(new Event('seeked')); } };
    const s = slots.find(s => s._i >= 0 && t >= clips[s._i].st && t < clips[s._i].st + clips[s._i].len);
    const v = s && clips[s._i].k === 'v' ? s.querySelector('video') : null;
    if (v && v.seeking) v.addEventListener('seeked', fire, {once: true});
    setTimeout(fire, v && v.seeking ? 250 : 0);
  };
  Object.defineProperty(el, 'currentTime', {configurable: true, get: now, set: seek});
  Object.defineProperty(el, 'paused', {configurable: true, get: () => !playing});
  Object.defineProperty(el, 'duration', {configurable: true, get: () => total});
  el.play = () => {
    if (playing) return Promise.resolve();
    if (t >= total - 0.05) t = loop ? loop[0] : 0;
    playing = true; t0 = t; p0 = performance.now();
    if (music && t < (music.duration || 1e9)) { music.currentTime = t; music.play().catch(() => {}); }
    tick(true);
    return Promise.resolve();
  };
  el.pause = () => { if (!playing) return; t = now(); playing = false; if (music) music.pause(); tick(true); };
  window.tmInst[id] = {el: el, stop: stop};
  if (id === 'tm-draft' && window.tmDraftT !== undefined) t = Math.min(total, window.tmDraftT);  // where you were
  tick(true);
  raf = requestAnimationFrame(frame);
  return el;
}
// A transition in the instant player, roughly as the finished video draws it: the incoming clip's layer, q from
// 0 (the transition starts) to 1 (done); fam '' hides the layer.
function tmTrLook(s, fam, q, cover) {
  if (!fam) { s.style.opacity = 0; s.style.transform = ''; s.style.clipPath = ''; s.style.filter = ''; return; }
  q = Math.max(0, Math.min(1, q));
  let o = 1, tf = '', clip = '', filt = '';
  if (q < 1) {
    const e = 1 - Math.pow(1 - q, 3), back = 1 + 2.70158 * Math.pow(q - 1, 3) + 1.70158 * Math.pow(q - 1, 2);
    const p = (x) => (100 * x).toFixed(2) + '%';
    switch (fam) {
      case 'black': case 'white': case 'grey':
        o = q < 0.5 ? 0 : 1;
        if (cover) { cover.style.background = fam === 'black' ? '#000' : fam === 'white' ? '#fff' : '#808080';
                     cover.style.opacity = 1 - Math.abs(2 * q - 1); }
        break;
      case 'blur': o = q; filt = `blur(${(1 - q) * 14}px)`; break;
      case 'slide-l': tf = `translateX(${p(1 - e)})`; break;
      case 'slide-r': tf = `translateX(-${p(1 - e)})`; break;
      case 'slide-u': tf = `translateY(${p(1 - e)})`; break;
      case 'slide-d': tf = `translateY(-${p(1 - e)})`; break;
      case 'wipe-l': clip = `inset(0 0 0 ${p(1 - q)})`; break;
      case 'wipe-u': clip = `inset(${p(1 - q)} 0 0 0)`; break;
      case 'squeeze': tf = `scaleX(${Math.max(0.01, e)})`; break;
      case 'circle': clip = `circle(${p(0.75 * q)} at 50% 50%)`; break;
      case 'diag': clip = `polygon(0 0, ${p(2 * q)} 0, 0 ${p(2 * q)})`; break;
      case 'doors': clip = `inset(0 ${p(0.5 * (1 - q))})`; break;
      case 'box': clip = `inset(${p(0.5 * (1 - q))})`; break;
      case 'zoom': tf = `scale(${0.3 + 0.7 * e})`; o = Math.min(1, q * 2); break;
      case 'reveal': tf = `scale(${1 + 0.6 * (1 - e)})`; break;
      case 'spin': tf = `rotate(${(1 - e) * 360}deg) scale(${Math.max(0.01, e)})`; break;
      case 'bounce': tf = `translateX(${p(1 - back)})`; break;
      case 'drop': tf = `translateY(-${p(1 - back)})`; break;
      default: o = q;
    }
  }
  s.style.opacity = o; s.style.transform = tf; s.style.clipPath = clip; s.style.filter = filt;
}
// The transition picker: every little preview plays its transition again and again (A, then B coming in).
function tmTrDemos() {
  if (window.tmTrLoop) return;
  const loop = () => {
    const demos = document.querySelectorAll('.q-dialog .tm-trdemo');
    if (!demos.length) { window.tmTrLoop = null; return; }
    const now = performance.now() / 1000;
    demos.forEach(d => {
      const len = parseFloat(d.dataset.len) || 0.9, t = now % (len + 1.3), q = Math.max(0, Math.min(1, (t - 0.6) / len));
      const b = d.querySelector('.tm-b'), ov = d.querySelector('.tm-ov');
      if (ov) ov.style.opacity = 0;
      if (d.dataset.fam === 'cut') { tmTrLook(b, 'fade', q > 0 ? 1 : 0, ov); return; }
      tmTrLook(b, d.dataset.fam, q, ov);
    });
    window.tmTrLoop = requestAnimationFrame(loop);
  };
  window.tmTrLoop = requestAnimationFrame(loop);
}
function tmInstantToggle(e, el) {
  if (el.paused) el.play(); else el.pause();
}
function tmClock2(s) { return `${Math.floor(s / 60)}:${(s % 60).toFixed(1).padStart(4, '0')}`; }
// Dragging a clip: over its left edge, an arrow shows where it will be pasted in front; over the middle, a swap.
function tmClipMode(e, el) { return (e.clientX - el.getBoundingClientRect().left) < Math.max(14, el.offsetWidth * 0.3) ? 'insert' : 'swap'; }
function tmClipOver(e, el) {
  e.preventDefault();
  const mode = tmClipMode(e, el);
  el.style.outline = mode === 'swap' ? '3px dashed #2563eb' : '';
  let mark = el.querySelector('.tm-paste');
  if (mode === 'insert' && !mark) {
    mark = document.createElement('div');
    mark.className = 'tm-paste';
    mark.style.cssText = 'position:absolute;left:-4px;top:-14px;bottom:0;width:4px;background:#2563eb;z-index:6;pointer-events:none';
    mark.innerHTML = '<span style="position:absolute;top:-6px;left:-6px;color:#2563eb;font-size:14px">▼</span>';
    el.appendChild(mark);
  } else if (mode === 'swap' && mark) mark.remove();
}
function tmClipLeave(el) { el.style.outline = ''; const m = el.querySelector('.tm-paste'); if (m) m.remove(); }
function tmClipDrop(e, el, n) {
  e.preventDefault();
  const mode = tmClipMode(e, el);
  tmClipLeave(el);
  emitEvent('tm_draft_drop', {src: e.dataTransfer.getData('text/plain'), dst: n, mode: mode});
}
// Dragging a clip or a theme near the top or bottom of the window scrolls the page, faster the closer you get.
let tmDragY = null, tmDragTimer = null;
let tmDragX = null;
document.addEventListener('dragover', (e) => { tmDragY = e.clientY; tmDragX = e.clientX; });
// Where you were on the page when you double-clicked a shot (the big window stops the page scrolling, so it is
// read here, before the window opens): closing the window brings you back to it.
document.addEventListener('dblclick', (e) => {
  if (document.querySelector('.q-dialog')) return;
  window.tmShotScroll = window.scrollY;
  const el = e.target.closest && e.target.closest('[data-tile], .tm-hover, [id]');
  window.tmShotAnchor = el ? {sel: el.dataset.tile ? `[data-tile="${el.dataset.tile}"]` : `#${CSS.escape(el.id)}`,
                              top: el.getBoundingClientRect().top} : null;
}, true);
// After the shot window closes and the page is drawn again: the shot you opened is back at the same place.
function tmBackToShot() {
  const a = window.tmShotAnchor;
  const el = a && [...document.querySelectorAll(a.sel)].find(x => x.offsetParent !== null);  // (the one on show)
  if (el) window.scrollBy(0, el.getBoundingClientRect().top - a.top);
  else if (window.tmShotScroll != null) window.scrollTo(0, window.tmShotScroll);
}
document.addEventListener('dragstart', () => {
  clearInterval(tmDragTimer);
  tmDragTimer = setInterval(() => {
    if (tmDragY === null) return;
    const edge = 120, h = window.innerHeight;
    if (tmDragY < edge) window.scrollBy(0, -Math.ceil(30 * (edge - tmDragY) / edge));
    else if (tmDragY > h - edge) window.scrollBy(0, Math.ceil(30 * (tmDragY - (h - edge)) / edge));
    // The editor's clip strip: near its left or right end it scrolls sideways (towards the start or the end).
    const sc = document.getElementById('tm-tl-scroll');
    if (sc && tmDragX !== null && sc.offsetParent !== null) {
      const r = sc.getBoundingClientRect(), side = Math.min(140, r.width / 4);
      if (tmDragY > r.top - 80 && tmDragY < r.bottom + 80) {
        if (tmDragX < r.left + side) sc.scrollLeft -= Math.ceil(28 * Math.min(1, (r.left + side - tmDragX) / side));
        else if (tmDragX > r.right - side) sc.scrollLeft += Math.ceil(28 * Math.min(1, (tmDragX - (r.right - side)) / side));
      }
    }
  }, 16);
});
const tmDragDone = () => { clearInterval(tmDragTimer); tmDragTimer = null; tmDragY = null; tmDragX = null; };
document.addEventListener('dragend', tmDragDone);
document.addEventListener('drop', tmDragDone);
// After you cut a clip: its pictures everywhere on the page play only the kept part and show its new length.
function tmShowCut(id, start, end, thumb, words) {
  for (const box of document.querySelectorAll(`.tm-hover[id$="-${id}"]`)) {
    if (box.id.startsWith('tm-cut-')) continue;
    box.dataset.start = start; box.dataset.end = end;
    const img = box.querySelector('img'); if (img && thumb) img.src = thumb;
    const len = box.querySelector('.tm-len'); if (len) len.textContent = '▶ ' + words;
  }
  for (const e of document.querySelectorAll('.tm-len-' + id)) e.textContent = words;
}
function tmPlayPart(id, start, end, from, sound) {
  const box = document.getElementById(id);
  if (!box) return;
  if (tmPlaying && tmPlaying !== box) tmStop(tmPlaying);
  box.dataset.start = start; box.dataset.end = end;
  tmPlaying = box;
  if (sound) tmVideo(box).muted = false;  // cutting a clip: hear it
  tmPlayFrom(box, from);
}
// The photo editor: the photo zoomed (z) with the point x, y (fractions of it) in the middle; drag to move it.
function tmCropShow(id, z, x, y) {
  const box = document.getElementById(id);
  if (!box) return;
  const img = box.querySelector('img');
  box.dataset.z = z; box.dataset.x = x; box.dataset.y = y;
  img.style.transform = `translate(${(0.5 - x) * box.clientWidth * z}px, ${(0.5 - y) * box.clientHeight * z}px) scale(${z})`;
}
function tmCropDown(e, box) {
  const z = parseFloat(box.dataset.z || '1');
  if (z <= 1.001) return;
  const x0 = e.clientX, y0 = e.clientY, sx = parseFloat(box.dataset.x), sy = parseFloat(box.dataset.y);
  const W = box.clientWidth * z, H = box.clientHeight * z;
  box.setPointerCapture(e.pointerId);
  box.style.cursor = 'grabbing';
  let x = sx, y = sy;
  const move = (ev) => {
    x = Math.min(1 - 0.5 / z, Math.max(0.5 / z, sx - (ev.clientX - x0) / W));
    y = Math.min(1 - 0.5 / z, Math.max(0.5 / z, sy - (ev.clientY - y0) / H));
    tmCropShow(box.id, z, x, y);
  };
  const up = () => {
    box.removeEventListener('pointermove', move); box.removeEventListener('pointerup', up);
    box.style.cursor = 'grab';
    if (x !== sx || y !== sy) box.dispatchEvent(new CustomEvent('tmcrop', {bubbles: true, detail: {x: x, y: y}}));
  };
  box.addEventListener('pointermove', move); box.addEventListener('pointerup', up);
}
// ← → walk through the shots in the big window; hold the key to keep going (one shot every 0.35 s).
function tmNavKey(e) {
  const all = document.querySelectorAll(`.q-dialog [data-nav="${e.key === 'ArrowLeft' ? -1 : 1}"]`);
  const b = all[all.length - 1];
  if (!b) return false;
  e.preventDefault(); e.stopPropagation();
  const now = Date.now();
  if (now - (window.tmNavAt || 0) < (e.repeat ? 350 : 150)) return true;
  window.tmNavAt = now;
  b.click();
  return true;
}
// The photo's zoom slider doesn't keep the arrows: they still walk to the next photo.
document.addEventListener('keydown', (e) => {
  if ((e.key === 'ArrowLeft' || e.key === 'ArrowRight') && e.target.closest && e.target.closest('.tm-zoom')) {
    e.target.blur && e.target.blur();
    tmNavKey(e);
  }
}, true);
// Zoom on a video in the big window: a frame over the picture shows what the video keeps (outside darkened);
// drag the frame to move it. z: zoom (1 = all), x, y: its middle (0..1) in the turned picture.
function tmVZoom(id, z, x, y) {
  const box = document.getElementById(id);
  if (!box) return;
  let f = box.querySelector('.tm-vzoom');
  if (z <= 1.001) { if (f) f.remove(); return; }
  const img = box.querySelector('img');
  const iw = (img && img.naturalWidth) || 16, ih = (img && img.naturalHeight) || 9;
  const W = box.clientWidth, H = box.clientHeight, k = Math.min(W / iw, H / ih);
  const cw = iw * k, ch = ih * k, l0 = (W - cw) / 2, t0 = (H - ch) / 2;
  x = Math.min(Math.max(x, 0.5 / z), 1 - 0.5 / z); y = Math.min(Math.max(y, 0.5 / z), 1 - 0.5 / z);
  if (!f) {
    f = document.createElement('div');
    f.className = 'tm-vzoom';
    f.title = 'What the video keeps: drag to move it';
    Object.assign(f.style, {position: 'absolute', border: '2px solid #facc15', boxShadow: '0 0 0 9999px rgba(0,0,0,.55)',
                            cursor: 'grab', zIndex: 5, touchAction: 'none'});
    f.addEventListener('dblclick', (e) => e.stopPropagation());
    f.addEventListener('pointerdown', (e) => {
      e.preventDefault(); e.stopPropagation();
      f.setPointerCapture(e.pointerId);
      const sx = e.clientX, sy = e.clientY, ox = parseFloat(f.dataset.x), oy = parseFloat(f.dataset.y);
      const move = (m) => tmVZoom(id, parseFloat(f.dataset.z), ox + (m.clientX - sx) / parseFloat(f.dataset.cw),
                                  oy + (m.clientY - sy) / parseFloat(f.dataset.ch));
      const up = () => {
        f.removeEventListener('pointermove', move); f.removeEventListener('pointerup', up);
        box.dispatchEvent(new CustomEvent('tmvzoom', {bubbles: true, detail: {x: parseFloat(f.dataset.x), y: parseFloat(f.dataset.y)}}));
      };
      f.addEventListener('pointermove', move); f.addEventListener('pointerup', up);
    });
    box.appendChild(f);
    if (img && !img.complete) img.addEventListener('load', () => tmVZoom(id, parseFloat(f.dataset.z), parseFloat(f.dataset.x), parseFloat(f.dataset.y)), {once: true});
  }
  Object.assign(f.dataset, {z: z, x: x, y: y, cw: cw, ch: ch});
  Object.assign(f.style, {width: cw / z + 'px', height: ch / z + 'px', left: l0 + cw * (x - 0.5 / z) + 'px',
                          top: t0 + ch * (y - 0.5 / z) + 'px'});
}
// The shot window: Space plays and pauses the video, Shift puts a highlight at the second it is at.
document.addEventListener('keydown', (e) => {
  if ((e.key === 'ArrowLeft' || e.key === 'ArrowRight') && !e.target.closest('.q-range, .q-slider')
      && !document.querySelector('.q-menu') && !['INPUT', 'TEXTAREA'].includes(e.target.tagName)
      && tmNavKey(e)) return;
  if (e.key === 'Shift' && !e.repeat && !['INPUT', 'TEXTAREA'].includes(e.target.tagName)) {
    const star = document.querySelector('.q-dialog [data-photostar]');
    if (star) { star.click(); return; }
  }
  const holder = document.querySelector('.q-dialog [data-shotwin]');
  const strip = document.querySelector('.q-dialog [data-scrub-for][data-press]');
  const box = (holder && holder.querySelector('.tm-hover')) || (strip && document.getElementById(strip.dataset.scrubFor));
  if (!box || e.repeat || ['INPUT', 'TEXTAREA'].includes(e.target.tagName)) return;
  if (e.code === 'Space') {
    e.preventDefault(); e.stopPropagation();
    if (tmPlaying && tmPlaying !== box) tmStop(tmPlaying);
    tmPlaying = box;
    const v = tmVideo(box);
    if (v.paused) {  // from the red line
      const r = parseFloat(box.dataset.red);
      if (!isNaN(r) && Math.abs(v.currentTime - r) > 0.1) v.currentTime = r;
      const at = isNaN(r) ? v.currentTime : r, f1 = parseFloat(box.dataset.full1);
      if (!isNaN(f1) && (at < parseFloat(box.dataset.start) - 0.05 || at >= parseFloat(box.dataset.end) - 0.1)) {
        box.dataset.start = at; box.dataset.end = f1;  // outside the green part: it plays on from where you are
      }
      v.play().catch(() => {});
    } else v.pause();
  } else if (e.key === 'Shift' && holder) {
    const v = box.querySelector('video');
    const t = v && v.src ? v.currentTime : parseFloat(box.dataset.start || '0');
    box.dispatchEvent(new CustomEvent('tmhighlight', {bubbles: true, detail: {t: t}}));
  }
}, true);
document.addEventListener('keyup', (e) => {  // (so Space never also presses the button that has the focus)
  if (e.code === 'Space' && document.querySelector('.q-dialog [data-shotwin], .q-dialog [data-scrub-for][data-press]') &&
      !['INPUT', 'TEXTAREA'].includes(e.target.tagName)) { e.preventDefault(); e.stopPropagation(); }
}, true);
function tmHighlightsSet(ts) {  // the yellow lines on the strip, drawn again
  document.querySelectorAll('.q-dialog .tm-hl').forEach(el => el.remove());
  ts.forEach(tmHighlightMark);
}
function tmHighlightMark(t) {
  const box = document.querySelector('.q-dialog [data-shotwin] .tm-hover');
  const strip = box && document.querySelector(`[data-scrub-for="${box.id}"]`);
  if (!strip) return;
  const a = parseFloat(box.dataset.full0), b = parseFloat(box.dataset.full1);
  const line = document.createElement('div');
  line.className = 'tm-hl';
  line.style.cssText = `position:absolute;top:0;bottom:0;width:3px;left:${100 * (t - a) / ((b - a) || 1)}%;background:#facc15;box-shadow:0 0 3px #000;pointer-events:none`;
  strip.appendChild(line);
}
// Dragging a green handle: the video stops and shows the frame where the handle is (Play plays the part).
function tmShowAt(id, start, end, at) {
  const box = document.getElementById(id);
  if (!box) return;
  if (tmPlaying && tmPlaying !== box) tmStop(tmPlaying);
  tmPlaying = box;
  box.dataset.start = start; box.dataset.end = end;
  const v = tmVideo(box);
  v.pause();
  tmRedSet(box, at);
  if (v.readyState >= 1) tmSeekSoft(v, at); else v.addEventListener('loadedmetadata', () => { v.currentTime = at; }, {once: true});
}
// The red line on a clip's strip (the big window, cutting a clip), as in the editor: it shows where the video is.
// Press and drag on the strip and the video follows; a click puts the line there; Space or Play plays from it.
function tmRedSet(box, t) {
  const strip = document.querySelector(`[data-scrub-for="${box.id}"]`), red = strip && strip.querySelector('.tm-red');
  box.dataset.red = t;
  if (!red) return;
  const a = parseFloat(box.dataset.full0), b = parseFloat(box.dataset.full1);
  red.style.left = (100 * Math.max(0, Math.min(1, (t - a) / ((b - a) || 1)))) + '%';
}
setInterval(() => {  // (while it plays, the red line moves along)
  document.querySelectorAll('[data-scrub-for][data-press]').forEach(strip => {
    const box = document.getElementById(strip.dataset.scrubFor), v = box && box.querySelector('video');
    if (v && v.src && !v.paused) tmRedSet(box, v.currentTime);
  });
}, 50);
document.addEventListener('pointerdown', (e) => {
  const strip = e.target.closest && e.target.closest('[data-scrub-for][data-press]');
  const box = strip && document.getElementById(strip.dataset.scrubFor);
  if (!box || e.button !== 0) return;
  e.preventDefault();
  if (tmPlaying && tmPlaying !== box) tmStop(tmPlaying);
  tmPlaying = box;
  const v = tmVideo(box);
  v.pause();
  const at = (ev) => {
    const r = strip.getBoundingClientRect(), a = parseFloat(box.dataset.full0), b = parseFloat(box.dataset.full1);
    const t = a + Math.max(0, Math.min(1, (ev.clientX - r.left) / r.width)) * (b - a);
    tmRedSet(box, t);
    if (v.readyState >= 1) tmSeekSoft(v, t); else v.addEventListener('loadedmetadata', () => { v.currentTime = t; }, {once: true});
  };
  at(e);
  const move = (ev) => at(ev);
  const up = () => { document.removeEventListener('pointermove', move); document.removeEventListener('pointerup', up); };
  document.addEventListener('pointermove', move); document.addEventListener('pointerup', up);
}, true);
// Just moving the mouse along the strip (no press) shows the video at that frame, the red line with it; Space
// then plays from there and stops again.
document.addEventListener('mousemove', (e) => {
  if (e.buttons) return;
  const strip = e.target.closest && e.target.closest('[data-scrub-for][data-press]');
  const box = strip && document.getElementById(strip.dataset.scrubFor);
  if (!box) return;
  if (tmPlaying && tmPlaying !== box) tmStop(tmPlaying);
  tmPlaying = box;
  const v = tmVideo(box);
  const r = strip.getBoundingClientRect(), a = parseFloat(box.dataset.full0), b = parseFloat(box.dataset.full1);
  if (isNaN(a) || isNaN(b)) return;
  const t = a + Math.max(0, Math.min(1, (e.clientX - r.left) / r.width)) * (b - a);
  v.pause();
  tmRedSet(box, t);
  if (v.readyState >= 1) tmSeekSoft(v, t); else v.addEventListener('loadedmetadata', () => { v.currentTime = t; }, {once: true});
});
function tmPlayRed(id, start, end) {
  const box = document.getElementById(id);
  if (!box) return;
  const r = parseFloat(box.dataset.red), f1 = parseFloat(box.dataset.full1);
  if (!isNaN(r) && !isNaN(f1) && (r < start || r >= end - 0.1)) { tmPlayPart(id, r, f1, r); return; }
  tmPlayPart(id, start, end, (r >= start && r < end - 0.1) ? r : start);
}
// Cutting a clip: the mouse along the picture strip (or across the picture) moves the video to that moment.
function tmScrubAt(box, frac, play) {
  const a = parseFloat(box.dataset.full0), b = parseFloat(box.dataset.full1);
  const t = a + Math.min(1, Math.max(0, frac)) * (b - a);
  const head = document.getElementById(box.id + '-head');
  if (head) { head.style.display = 'block'; head.style.left = (100 * Math.min(1, Math.max(0, frac))) + '%'; }
  if (play) { tmPlayPart(box.id, t, b, t); return; }
  if (tmPlaying && tmPlaying !== box) tmStop(tmPlaying);
  tmPlaying = box;
  const v = tmVideo(box);
  v.pause();
  if (v.readyState >= 1) v.currentTime = t; else v.addEventListener('loadedmetadata', () => { v.currentTime = t; }, {once: true});
}
function tmScrubTarget(e) {
  const strip = e.target.closest && e.target.closest('[data-scrub-for]');
  if (strip && strip.dataset.press) return [null, null];  // (it has a red line: press and drag, see below)
  if (strip) return [document.getElementById(strip.dataset.scrubFor), strip];
  const box = e.target.closest && e.target.closest('.tm-hover');
  return box && box.dataset.scrub === '1' ? [box, box] : [null, null];
}
document.addEventListener('mousemove', (e) => {
  const [box, over] = tmScrubTarget(e);
  if (!box) return;
  const r = over.getBoundingClientRect();
  tmScrubAt(box, (e.clientX - r.left) / r.width, false);
});
document.addEventListener('click', (e) => {
  const [box, over] = tmScrubTarget(e);
  if (!box) return;
  const r = over.getBoundingClientRect();
  tmScrubAt(box, (e.clientX - r.left) / r.width, true);
});
// Skimming: moving the mouse left and right over a video goes through it (a thin line shows where); keep the
// mouse still and it plays from there.
let tmSkimTimer = null;
document.addEventListener('mousemove', (e) => {
  const box = e.target.closest && e.target.closest('.tm-hover');
  if (!box || box.dataset.skim0 === undefined || box.dataset.scrub === '1') return;
  const a = parseFloat(box.dataset.skim0), b = parseFloat(box.dataset.skim1);
  if (!(b > a)) return;
  const r = box.getBoundingClientRect(), frac = Math.min(1, Math.max(0, (e.clientX - r.left) / r.width));
  const t = a + frac * (b - a);
  if (box.dataset.hs === undefined) { box.dataset.hs = box.dataset.start; box.dataset.he = box.dataset.end; }
  let line = box.querySelector('.tm-skim');
  if (!line) {
    line = document.createElement('div');
    line.className = 'tm-skim';
    line.style.cssText = 'position:absolute;top:0;bottom:0;width:2px;background:#facc15;box-shadow:0 0 3px #000;pointer-events:none;z-index:3';
    box.appendChild(line);
  }
  line.style.left = (100 * frac) + '%';
  if (tmPlaying && tmPlaying !== box) tmStop(tmPlaying);
  tmPlaying = box;
  const v = tmVideo(box);
  v.pause();
  if (v.readyState >= 1) v.currentTime = t; else v.addEventListener('loadedmetadata', () => { v.currentTime = t; }, {once: true});
  clearTimeout(tmSkimTimer);
  tmSkimTimer = setTimeout(() => {  // the mouse rests: play on from there (to the end, then from the start)
    if (tmPlaying !== box) return;
    box.dataset.start = a; box.dataset.end = b;
    v.play().catch(() => {});
  }, 350);
});
document.addEventListener('mouseenter', (e) => {
  const box = e.target.closest && e.target.closest('.tm-hover');
  if (!box || e.target !== box || box.dataset.scrub === '1') return;
  if (tmPlaying && tmPlaying !== box) tmStop(tmPlaying);
  tmPlaying = box;
  tmPlayFrom(box, parseFloat(box.dataset.start));
}, true);
document.addEventListener('mouseleave', (e) => {
  const box = e.target.closest && e.target.closest('.tm-hover');
  if (!box || e.target !== box || box.dataset.scrub === '1') return;
  clearTimeout(tmSkimTimer);
  const line = box.querySelector('.tm-skim');
  if (line) line.remove();
  if (box.dataset.hs !== undefined) {  // hovering plays its usual part again next time
    box.dataset.start = box.dataset.hs; box.dataset.end = box.dataset.he;
    delete box.dataset.hs; delete box.dataset.he;
  }
  tmStop(box);
}, true);
// Choose files is one long list: when you scroll near its bottom, press "Loading more" for you.
setInterval(() => {
  const more = document.querySelector('.tm-more');
  if (!more || more.dataset.pressed) return;
  if (more.getBoundingClientRect().top < window.innerHeight + 1200) {
    more.dataset.pressed = '1';
    more.click();
  }
}, 400);
</script>
"""


DICTATE_SCRIPT = """
<script>
// "Speak your idea": the browser's own speech recognition (Safari and Chrome), words straight into the box.
window.tmDictate = {rec: null, on: false, done: ''};
function tmDictateStart() {
  const Rec = window.SpeechRecognition || window.webkitSpeechRecognition;
  if (!Rec) return 'unsupported';
  try {
    const rec = new Rec();
    rec.continuous = true;
    rec.interimResults = true;
    rec.lang = navigator.language || 'en-GB';
    tmDictate.done = '';
    rec.onresult = (e) => {
      let now = '';
      for (let i = e.resultIndex; i < e.results.length; i++) {
        if (e.results[i].isFinal) tmDictate.done += e.results[i][0].transcript + ' ';
        else now += e.results[i][0].transcript;
      }
      emitEvent('tm_dictation', {text: (tmDictate.done + now).replace(/\\s+/g, ' ').trim()});
    };
    rec.onerror = (e) => {
      if (e.error !== 'no-speech') tmDictate.on = false;
      emitEvent('tm_dictation', {error: e.error || 'unknown'});
    };
    rec.onend = () => {
      // Browsers stop listening after a pause: keep listening until you press Stop.
      if (tmDictate.on) { try { rec.start(); return; } catch (err) {} }
      emitEvent('tm_dictation', {ended: true});
    };
    tmDictate.rec = rec;
    tmDictate.on = true;
    rec.start();
    return 'ok';
  } catch (e) {
    return e.message || String(e);
  }
}
function tmDictateStop() {
  tmDictate.on = false;
  if (tmDictate.rec) tmDictate.rec.stop();
}
</script>
"""


TAP_SCRIPT = """
<script>
// The Music tab: play the song, press the RIGHT Shift key wherever you want a cut. Pins can be dragged,
// selected and deleted; on playback every pin you pass makes a click so you hear where it is.
window.tm = window.tm || {};
tm.tap = {pins: [], sel: -1, view: 0, span: 0, until: -1, ctx: null, raf: 0, drag: -1, osc: [], srcEl: null,
          wave: null, waveSrc: '', follow: true, trim: null, tdrag: -1, skip: true, fade: [0, 0], fdrag: -1, gain: null,
          ai: new Set(),  // "Finish my pins": the app's suggestions (orange) until you keep them
          unused: new Set()};  // too few shots: the pins the last cut skipped (grey)
const TM_WAVE_STEP = 0.002;  // the waveform keeps the loudest moment of every 2 ms

function tmTapAudio() { return document.getElementById('tm-song'); }
function tmTapVisible() { const tl = document.getElementById('tm-tl'); return tl && tl.offsetParent !== null; }

// The Music tab's panel is only on the page while it is open, so nothing here assumes it exists.
// trim: whether the song can still be trimmed; then two green handles on the line mark the part you keep.
function tmTapInit(pins, span, trim) {
  const T = tm.tap;
  T.pins = (pins || []).slice().sort((a, b) => a - b);
  T.sel = -1; T.view = 0; T.span = span || 0; T.trim = trim ? [0, -1] : null; T.tdrag = -1; T.skip = true; T.fade = [0, 0]; T.fdrag = -1;
  tmTapUnschedule();
  tmTapLoadWave();
  tmTapDraw();
}

// The song's waveform, so you can see the hits (the spikes) and put pins exactly on them.
function tmTapLoadWave() {
  const T = tm.tap, a = tmTapAudio();
  if (!a || !a.src || T.waveSrc === a.src) return;
  T.waveSrc = a.src; T.wave = null;
  const Off = window.OfflineAudioContext || window.webkitOfflineAudioContext;
  fetch(a.src).then(r => r.arrayBuffer()).then(buf => new Promise((ok, bad) => {
    const p = new Off(1, 2, 44100).decodeAudioData(buf, ok, bad); if (p && p.then) p.then(ok, bad);
  })).then(audio => {
    const data = audio.getChannelData(0), per = Math.max(1, Math.round(audio.sampleRate * TM_WAVE_STEP));
    const peaks = new Float32Array(Math.ceil(data.length / per));
    for (let i = 0; i < peaks.length; i++) {
      let m = 0; const end = Math.min(data.length, (i + 1) * per);
      for (let j = i * per; j < end; j++) { const v = Math.abs(data[j]); if (v > m) m = v; }
      peaks[i] = m;
    }
    let top = 0; for (let i = 0; i < peaks.length; i++) if (peaks[i] > top) top = peaks[i];
    if (top > 0) for (let i = 0; i < peaks.length; i++) peaks[i] /= top;
    T.wave = peaks; tmTapDraw();
  }).catch(e => console.warn('waveform', e));
}

function tmTapWaveSvg(v0, span) {
  const peaks = tm.tap.wave; if (!peaks) return '';
  const cols = 800, top = [], bottom = [];
  for (let c = 0; c < cols; c++) {
    const a = Math.floor((v0 + span * c / cols) / TM_WAVE_STEP), b = Math.max(a + 1, Math.floor((v0 + span * (c + 1) / cols) / TM_WAVE_STEP));
    let m = 0; for (let i = Math.max(0, a); i < Math.min(peaks.length, b); i++) if (peaks[i] > m) m = peaks[i];
    const h = (m * 34).toFixed(1);
    top.push(`${c},${(36 - h)}`); bottom.push(`${c},${(36 + +h)}`);
  }
  return `<svg viewBox="0 0 ${cols} 72" preserveAspectRatio="none" style="position:absolute;left:0;top:6px;width:100%;height:calc(100% - 34px);pointer-events:none">` +
         `<polygon points="${top.join(' ')} ${bottom.reverse().join(' ')}" fill="#cbd5e1"/></svg>`;
}
function tmTapOnPlay() {
  const T = tm.tap, a = tmTapAudio(); T.follow = true;
  if (a && T.trim && T.skip) { const [k0, k1] = tmTrimPart(); if (a.currentTime < k0 || a.currentTime >= k1 - 0.05) a.currentTime = k0; }
  tmTapCtx(); tmTapUnschedule(); cancelAnimationFrame(T.raf); tmTapLoop(); tmTapDraw();
}
// Listen on the whole page: the line itself is rebuilt whenever the screen refreshes or the tab opens.
document.addEventListener('pointerdown', ev => { if (ev.target.closest && ev.target.closest('#tm-tl')) tmTapPointerDown(ev); });
window.addEventListener('pointermove', ev => tmTapPointerMove(ev));
window.addEventListener('pointerup', () => {
  const T = tm.tap;
  if (T.drag >= 0) { T.drag = -1; T.pins.sort((a, b) => a - b); tmTapDraw(); }
  if (T.tdrag >= 0) { const i = T.tdrag; T.tdrag = -1; tmTrimHear(i); }  // hear where you just put the handle
  if (T.fdrag >= 0) { const i = T.fdrag; T.fdrag = -1; tmTrimHear(i); }  // and hear the fade you just set
});
// Scroll up/down (or pinch) on the line to zoom in and out around the mouse; scroll sideways to move along.
document.addEventListener('wheel', ev => {
  const tl = ev.target.closest && ev.target.closest('#tm-tl'); if (!tl) return;
  ev.preventDefault();
  const T = tm.tap, d = tmTapLength(); if (!d) return;
  const [v0, span] = tmTapWindow();
  if (Math.abs(ev.deltaX) > Math.abs(ev.deltaY)) {
    T.span = span; T.view = Math.max(0, Math.min(d - span, v0 + ev.deltaX / tl.getBoundingClientRect().width * span));
  } else {
    const at = tmTapTimeAt(ev), wanted = Math.max(1, Math.min(d, span * Math.exp(ev.deltaY * 0.004)));
    T.span = wanted >= d ? 0 : wanted;
    T.view = T.span ? Math.max(0, Math.min(d - T.span, at - (at - v0) * wanted / span)) : 0;
  }
  const a = tmTapAudio(); T.follow = !(a && !a.paused && (a.currentTime < T.view || a.currentTime > T.view + (T.span || d)));
  tmTapDraw();
}, {passive: false});
// When the Music tab opens, draw the pins.
setInterval(() => { const tl = document.getElementById('tm-tl'); if (tl && !tl.childElementCount) tmTapDraw(); }, 300);

// The song plays through the same sound path as the clicks, so both reach your ears with the same delay.
function tmTapCtx() {
  const T = tm.tap, a = tmTapAudio();
  if (!T.ctx) T.ctx = new (window.AudioContext || window.webkitAudioContext)();
  if (T.ctx.state === 'suspended') T.ctx.resume();
  if (a && T.srcEl !== a) {
    try {  // through a volume control, so the fades can be heard as you set them
      if (!T.gain) { T.gain = T.ctx.createGain(); T.gain.connect(T.ctx.destination); }
      T.ctx.createMediaElementSource(a).connect(T.gain); T.srcEl = a;
    } catch (e) { console.warn(e); }
  }
}
// How long sound takes from the page to your ears (the sound card's own delay, as the browser reports it).
function tmTapLatency() {
  const c = tm.tap.ctx; if (!c || tm.tap.srcEl !== tmTapAudio()) return 0;
  return (c.outputLatency || 0) + (c.baseLatency || 0);
}

// A short, bright "tick", easy to tell apart from the music, at an exact moment on the sound card's clock.
function tmTapClick(high, when) {
  const T = tm.tap, ctx = T.ctx; if (!ctx) return;
  const at = Math.max(ctx.currentTime, when || 0);
  const o = ctx.createOscillator(), g = ctx.createGain();
  o.type = 'square'; o.frequency.value = high ? 2200 : 1500;
  g.gain.setValueAtTime(0.25, at);
  g.gain.exponentialRampToValueAtTime(0.001, at + 0.06);
  o.connect(g); g.connect(ctx.destination); o.start(at); o.stop(at + 0.07);
  if (!high) { T.osc.push(o); o.onended = () => { T.osc = T.osc.filter(x => x !== o); }; }
}
function tmTapUnschedule() {
  const T = tm.tap;
  T.osc.forEach(o => { try { o.stop(); } catch (e) {} });
  T.osc = []; T.until = -1;
}

// Clicks are booked a quarter of a second ahead on the sound card's clock, so they land exactly on the pins
// (not whenever the screen happens to redraw).
function tmTapLoop() {
  const T = tm.tap, a = tmTapAudio();
  if (!a || a.paused) return;
  const [k0, k1] = tmTrimPart();
  if (T.trim && T.skip && T.tdrag < 0) {
    if (a.currentTime < k0 - 0.05) { a.currentTime = k0; T.until = -1; }
    else if (a.currentTime >= k1) { a.pause(); a.currentTime = k1; tmTapDraw(); return; }
  }
  const now = a.currentTime, rate = a.playbackRate || 1;
  tmFadeVolume(now);
  if (T.until < 0 || T.until < now - 0.3 || T.until > now + 1) T.until = now;  // just started, or jumped
  const ahead = now + 0.25;
  if (T.ctx) T.pins.forEach(p => { if (p > T.until && p <= ahead && p >= k0 && p <= k1) tmTapClick(false, T.ctx.currentTime + (p - now) / rate); });
  T.until = ahead;
  if (T.span && T.follow && (now < T.view || now > T.view + T.span * 0.9)) T.view = Math.max(0, now - T.span * 0.1);
  tmTapDraw();
  T.raf = requestAnimationFrame(tmTapLoop);
}

function tmTapLength() { const a = tmTapAudio(); return a && isFinite(a.duration) ? a.duration : 0; }
function tmTapWindow() {
  const d = tmTapLength(), T = tm.tap;
  return T.span && T.span < d ? [T.view, T.span] : [0, d || 1];
}

function tmTapDraw() {
  const tl = document.getElementById('tm-tl'); if (!tl) return;
  const T = tm.tap, a = tmTapAudio(), [v0, span] = tmTapWindow();
  const x = t => ((t - v0) / span * 100).toFixed(3) + '%';
  let html = tmTapWaveSvg(v0, span);
  (T.moods || []).forEach(([s0, s1, col]) => {  // how the music feels: a coloured strip along the bottom
    const a0 = Math.max(s0, v0), a1 = Math.min(s1, v0 + span); if (a1 <= a0) return;
    html += `<div style="position:absolute;left:${x(a0)};width:${((a1 - a0) / span * 100).toFixed(3)}%;bottom:22px;height:8px;background:${col};pointer-events:none"></div>`;
  });
  const d = tmTapLength();
  if (T.trim && T.trim[1] < 0 && d) T.trim[1] = d;
  const [k0, k1] = tmTrimPart();
  if (T.trim && d) {  // the parts you cut off, greyed out
    if (k0 > v0) html += `<div style="position:absolute;left:0;width:${x(Math.min(k0, v0 + span))};top:0;bottom:0;background:#64748b55;pointer-events:none"></div>`;
    if (k1 < v0 + span) html += `<div style="position:absolute;left:${x(Math.max(k1, v0))};right:0;top:0;bottom:0;background:#64748b55;pointer-events:none"></div>`;
  }
  // Time marks along the bottom: finer as you zoom in (down to a tenth of a second).
  const steps = [0.1, 0.25, 0.5, 1, 2, 5, 10, 30, 60];
  const stepS = steps.find(st => span / st <= 14) || 60;
  for (let k = Math.ceil(v0 / stepS - 1e-9); k * stepS <= v0 + span; k++) {
    const s = k * stepS, m = Math.floor(s / 60);
    const sec = stepS < 1 ? (s % 60).toFixed(stepS < 0.25 ? 1 : 2).padStart(stepS < 0.25 ? 4 : 5, '0') : String(Math.round(s % 60)).padStart(2, '0');
    html += `<div style="position:absolute;left:${x(s)};bottom:0;height:10px;border-left:1px solid #94a3b8"></div>` +
            `<div style="position:absolute;left:${x(s)};bottom:10px;font-size:10px;color:#64748b;transform:translateX(-50%)">${m}:${sec}</div>`;
  }
  // Each pin's own time next to its head (when there is room), so you can read exactly where it is.
  const width = tl.clientWidth || 1000, fine = span <= 20;
  let lastLabel = -1e9;
  T.pins.forEach((p, i) => {
    if (p < v0 || p > v0 + span) return;
    const c = i === T.sel ? '#dc2626' : T.trim && (p < k0 || p > k1) ? '#94a3b8' : T.ai.has(p.toFixed(3)) ? '#f59e0b' :
              T.unused.has(p.toFixed(3)) ? '#cbd5e1' : '#2563eb',
          px = (p - v0) / span * width;
    if (px - lastLabel > (fine ? 70 : 52) || i === T.sel) {
      const m = Math.floor(p / 60), sec = (p % 60).toFixed(fine ? 2 : 1).padStart(fine ? 5 : 4, '0');
      html += `<div style="position:absolute;left:${x(p)};top:-1px;margin-left:8px;font-size:10px;font-weight:600;color:${c};` +
              `background:#f1f5f9cc;padding:0 2px;pointer-events:none">${m}:${sec}</div>`;
      lastLabel = px;
    }
    html += `<div data-pin="${i}" title="${p.toFixed(3)} s" style="position:absolute;left:${x(p)};top:0;bottom:24px;width:14px;` +
            `margin-left:-7px;cursor:ew-resize;display:flex;justify-content:center"><div style="width:3px;background:${c};height:100%"></div>` +
            `<div style="position:absolute;top:-2px;width:12px;height:12px;border-radius:6px;background:${c}"></div></div>`;
  });
  if (T.trim && d) {  // the fades: a purple slope from silent to full volume, with a dot to drag
    const [f0, f1] = T.fade, cols = [];
    if (f0 > 0) cols.push([k0, k0 + f0, 0]);
    if (f1 > 0) cols.push([k1 - f1, k1, 1]);
    cols.forEach(([s0, s1, out]) => {
      const a0 = Math.max(s0, v0), a1 = Math.min(s1, v0 + span); if (a1 <= a0) return;
      const y = t => { const v = out ? (s1 - t) / (s1 - s0) : (t - s0) / (s1 - s0); return (100 - 100 * v).toFixed(2); };
      html += `<svg viewBox="0 0 100 100" preserveAspectRatio="none" style="position:absolute;left:${x(a0)};width:${((a1 - a0) / span * 100).toFixed(3)}%;top:0;bottom:24px;height:calc(100% - 24px);pointer-events:none">` +
              `<polygon points="0,0 100,0 100,${y(a1)} 0,${y(a0)}" fill="#7c3aed33"/>` +
              `<line x1="0" y1="${y(a0)}" x2="100" y2="${y(a1)}" stroke="#7c3aed" stroke-width="2" vector-effect="non-scaling-stroke"/></svg>`;
    });
    [k0 + f0, k1 - f1].forEach((t, i) => {
      if (t < v0 - 0.001 || t > v0 + span + 0.001) return;
      html += `<div data-fade="${i}" title="Drag to fade ${i ? 'out' : 'in'} (${T.fade[i].toFixed(1)} s)" style="position:absolute;left:${x(t)};top:2px;width:16px;height:16px;` +
              `margin-left:-8px;border-radius:8px;background:#7c3aed;border:2px solid white;cursor:ew-resize;z-index:4"></div>`;
      if (T.fade[i] > 0) html += `<div style="position:absolute;left:${x(t)};top:20px;${i ? 'transform:translateX(-100%);margin-left:-4px' : 'margin-left:4px'};font-size:10px;font-weight:700;color:#6d28d9;background:#f1f5f9cc;padding:0 2px;white-space:nowrap;pointer-events:none">fade ${i ? 'out' : 'in'} ${T.fade[i].toFixed(1)} s</div>`;
    });
  }
  if (T.trim && d) [k0, k1].forEach((t, i) => {  // the two trim handles: a green bar with a grip to drag
    if (t < v0 - 0.001 || t > v0 + span + 0.001) return;
    const m = Math.floor(t / 60), sec = (t % 60).toFixed(1).padStart(4, '0');
    html += `<div data-trim="${i}" title="Drag to trim the ${i ? 'end' : 'start'}" style="position:absolute;left:${x(t)};top:0;bottom:0;width:18px;` +
            `margin-left:${i ? -16 : -2}px;cursor:ew-resize;display:flex;justify-content:${i ? 'flex-end' : 'flex-start'};z-index:2">` +
            `<div style="width:4px;background:#16a34a;height:100%"></div>` +
            `<div style="position:absolute;top:50%;${i ? 'right' : 'left'}:0;transform:translateY(-50%);width:14px;height:36px;border-radius:4px;background:#16a34a;color:white;font-size:10px;display:flex;align-items:center;justify-content:center">${i ? ']' : '['}</div>` +
            `<div style="position:absolute;bottom:26px;${i ? 'right' : 'left'}:6px;font-size:10px;font-weight:700;color:#15803d;background:#f1f5f9cc;padding:0 2px;white-space:nowrap;pointer-events:none">${i ? 'End' : 'Start'} ${m}:${sec}</div></div>`;
  });
  if (a) html += `<div style="position:absolute;left:${x(a.currentTime)};top:0;bottom:0;width:2px;background:#f97316;pointer-events:none;z-index:3"></div>`;
  tl.innerHTML = html;
  const tinfo = document.getElementById('tm-trim-info');
  if (tinfo) tinfo.textContent = T.trim && d ? (k0 < 0.05 && k1 > d - 0.05 ? `Keeping the whole song (${tmClock(d)}): drag the green handles to trim it` :
    `Keeping ${tmClock(k0)} to ${tmClock(k1)}: ${tmClock(k1 - k0)} of ${tmClock(d)} (cutting ${tmClock(k0)} at the start, ${tmClock(d - k1)} at the end)`) +
    ` · fade in ${T.fade[0] > 0 ? T.fade[0].toFixed(1) + ' s' : 'none'}, fade out ${T.fade[1] > 0 ? T.fade[1].toFixed(1) + ' s' : 'none'}` : '';
  const info = document.getElementById('tm-tap-info');
  if (info) {
    const n = T.pins.length;
    const t = a ? a.currentTime : 0;
    const last = n ? T.pins[n - 1] : 0;
    info.textContent = `${Math.floor(t / 60)}:${String(Math.floor(t % 60)).padStart(2, '0')} · ${n} pin${n === 1 ? '' : 's'}` +
      (n ? ` → ${n} shots up to ${Math.floor(last / 60)}:${String(Math.floor(last % 60)).padStart(2, '0')}` +
           (d - last > 1 ? ', then the app cuts the rest' : '') : '') +
      (T.sel >= 0 ? ` · selected cut at ${T.pins[T.sel].toFixed(3)} s (← → move it 10 ms, with Alt 1 ms; Delete removes it)` : '') +
      ` · showing ${span < 10 ? span.toFixed(1) : Math.round(span)} s (scroll on the line to zoom, sideways to move)`;
  }
}

function tmTapTimeAt(ev) {
  const tl = document.getElementById('tm-tl'), r = tl.getBoundingClientRect(), [v0, span] = tmTapWindow();
  return Math.max(0, Math.min(tmTapLength(), v0 + (ev.clientX - r.left) / r.width * span));
}

function tmTapPointerDown(ev) {
  const T = tm.tap, fade = ev.target.closest('[data-fade]'), handle = ev.target.closest('[data-trim]'), pin = ev.target.closest('[data-pin]');
  if (fade) { T.fdrag = parseInt(fade.dataset.fade); const a = tmTapAudio(); if (a && !a.paused) a.pause(); ev.preventDefault(); }
  else if (handle) { T.tdrag = parseInt(handle.dataset.trim); const a = tmTapAudio(); if (a && !a.paused) a.pause(); ev.preventDefault(); }
  else if (pin) { T.sel = T.drag = parseInt(pin.dataset.pin); ev.preventDefault(); }
  else { T.sel = -1; const a = tmTapAudio(); if (a) { a.currentTime = tmTapTimeAt(ev); tmTapUnschedule(); } }
  tmTapDraw();
}

function tmTapPointerMove(ev) {
  const T = tm.tap;
  if (T.fdrag >= 0) {  // a fade dot: how far in from its handle the music reaches full volume
    const t = tmTapTimeAt(ev), [k0, k1] = tmTrimPart(), room = (k1 - k0) / 2;
    T.fade[T.fdrag] = Math.round(Math.max(0, Math.min(room, T.fdrag ? k1 - t : t - k0)) * 10) / 10;
    tmTapDraw(); return;
  }
  if (T.tdrag >= 0) {  // a trim handle: keep at least 2 seconds between them
    const t = tmTapTimeAt(ev), d = tmTapLength();
    if (T.tdrag === 0) T.trim[0] = Math.max(0, Math.min(t, T.trim[1] - 2));
    else T.trim[1] = Math.min(d, Math.max(t, T.trim[0] + 2));
    const room = (T.trim[1] - T.trim[0]) / 2; T.fade = T.fade.map(f => Math.min(f, room));
    const a = tmTapAudio(); if (a) a.currentTime = T.trim[T.tdrag];
    tmTapDraw(); return;
  }
  if (T.drag < 0) return;
  T.pins[T.drag] = tmTapTimeAt(ev);
  tmTapDraw();
}

document.addEventListener('keydown', ev => {
  if (!tmTapVisible()) return;
  const T = tm.tap, a = tmTapAudio();
  if (ev.code === 'ShiftRight' && a && !ev.repeat) {
    // Where the song was when you HEARD it: take off the sound card's delay and the time since the key went down.
    const waited = Math.min(0.2, Math.max(0, (performance.now() - ev.timeStamp) / 1000));
    const t = Math.max(0, a.currentTime - (a.paused ? 0 : tmTapLatency() + waited));
    T.pins.push(t); T.pins.sort((x, y) => x - y);
    tmTapClick(true);
    tmTapDraw();
    ev.preventDefault();
  } else if ((ev.key === 'ArrowLeft' || ev.key === 'ArrowRight') && T.sel >= 0 &&
             !['INPUT', 'TEXTAREA'].includes(document.activeElement.tagName)) {
    const by = (ev.key === 'ArrowLeft' ? -1 : 1) * (ev.altKey ? 0.001 : 0.01), p = T.pins[T.sel] + by;
    T.pins[T.sel] = Math.max(0, p); T.pins.sort((x, y) => x - y); T.sel = T.pins.indexOf(Math.max(0, p));
    tmTapUnschedule(); tmTapDraw(); ev.preventDefault();
  } else if ((ev.key === 'Delete' || ev.key === 'Backspace') && T.sel >= 0 && !['INPUT', 'TEXTAREA'].includes(document.activeElement.tagName)) {
    T.pins.splice(T.sel, 1); T.sel = -1; tmTapUnschedule(); tmTapDraw(); ev.preventDefault();
  }
});

function tmTapPlay() { const a = tmTapAudio(); if (!a) return; tmTapCtx(); if (a.paused) a.play(); else a.pause(); }
function tmTapSeek(by) { const a = tmTapAudio(); if (!a) return; a.currentTime = Math.max(0, a.currentTime + by); tmTapUnschedule(); tmTapDraw(); }
function tmTapStart() { const a = tmTapAudio(); if (!a) return; a.currentTime = tm.tap.trim && tm.tap.skip ? tmTrimPart()[0] : 0; tmTapUnschedule(); tm.tap.view = 0; tmTapDraw(); }
function tmTapDelete() { const T = tm.tap; if (T.sel >= 0) { T.pins.splice(T.sel, 1); T.sel = -1; tmTapUnschedule(); tmTapDraw(); } }
function tmTapClear() { tm.tap.pins = []; tm.tap.ai = new Set(); tm.tap.sel = -1; tmTapDraw(); }
function tmTapZoom(span) {
  const T = tm.tap, a = tmTapAudio(), at = T.sel >= 0 ? T.pins[T.sel] : a ? a.currentTime : 0;
  T.span = span; T.follow = true; T.view = Math.max(0, at - span * (T.sel >= 0 ? 0.5 : 0.1)); tmTapDraw();
}
function tmClock(s) { return `${Math.floor(s / 60)}:${(s % 60).toFixed(1).padStart(4, '0')}`; }
// The part of the song you keep (the whole song when it can't be trimmed here).
function tmTrimPart() {
  const T = tm.tap, d = tmTapLength() || 1e9;
  return T.trim ? [T.trim[0], T.trim[1] < 0 ? d : T.trim[1]] : [0, d];
}
function tmTrimGet() { const d = tmTapLength(); return tm.tap.trim && d ? [...tmTrimPart(), ...tm.tap.fade] : null; }
// The volume right now: rising over the fade in, falling over the fade out (full volume outside the kept part).
function tmFadeVolume(t) {
  const T = tm.tap; if (!T.gain || !T.ctx) return;
  const [k0, k1] = tmTrimPart(), [f0, f1] = T.trim ? T.fade : [0, 0];
  let v = 1;
  if (t >= k0 && t <= k1) {
    if (f0 > 0 && t < k0 + f0) v = Math.min(v, (t - k0) / f0);
    if (f1 > 0 && t > k1 - f1) v = Math.min(v, (k1 - t) / f1);
  }
  T.gain.gain.setTargetAtTime(Math.max(0, v), T.ctx.currentTime, 0.015);
}
function tmMoodsSet(list) { if (!window.tm || !tm.tap) return; tm.tap.moods = list; tmTapDraw(); }
function tmFadeSet(fin, fout) {
  const T = tm.tap; if (!T.trim) return;
  const [k0, k1] = tmTrimPart(), room = (k1 - k0) / 2;
  T.fade = [Math.min(fin, room), Math.min(fout, room)]; tmTapDraw();
}
// Hear a trim handle: the start plays from there on; the end plays the last 5 seconds and stops at it.
function tmTrimHear(i) {
  const T = tm.tap, a = tmTapAudio(); if (!a || !T.trim) return;
  const [k0, k1] = tmTrimPart(), skip = T.skip;
  a.currentTime = i ? Math.max(k0, k1 - Math.max(5, T.fade[1] + 3)) : k0; tmTapUnschedule();
  tmFadeVolume(a.currentTime);
  tmTapCtx(); T.skip = true; a.play();  // the end preview always stops at the end handle
  if (!skip) { const back = () => { T.skip = false; a.removeEventListener('pause', back); }; a.addEventListener('pause', back); }
  tmTapDraw();
}
function tmTapGet() { return tm.tap.pins.filter(p => !tm.tap.ai.has(p.toFixed(3))).sort((a, b) => a - b); }  // yours
function tmTapSuggest(pins) {  // the app's suggestions join the line in orange; nothing is saved until you keep them
  const T = tm.tap; tmTapDropAi();
  pins.forEach(p => { T.pins.push(p); T.ai.add(p.toFixed(3)); });
  T.pins.sort((a, b) => a - b); T.sel = -1; tmTapUnschedule(); tmTapDraw();
}
function tmTapDropAi() { const T = tm.tap; T.pins = T.pins.filter(p => !T.ai.has(p.toFixed(3))); T.ai = new Set(); T.sel = -1; tmTapUnschedule(); tmTapDraw(); }
function tmTapKeepAi() { tm.tap.ai = new Set(); tmTapDraw(); }
function tmTapSet(pins) { tm.tap.pins = pins.slice().sort((a, b) => a - b); tm.tap.ai = new Set(); tm.tap.sel = -1; tmTapUnschedule(); tmTapDraw(); }
</script>
"""


_sizes: dict[str, int] = {}


def _file_size(path: str) -> int:
    if path not in _sizes:
        try:
            _sizes[path] = Path(path).stat().st_size
        except OSError:
            _sizes[path] = 0
    return _sizes[path]


def serve_song(path: str) -> str:
    """Let the page play a song from anywhere on this computer (for the Music tab)."""
    stat = Path(path).stat()
    key = hashlib.sha1(f"{path}|{stat.st_mtime}".encode()).hexdigest()[:12]
    if key not in _served:
        _served[key] = f"/song/{key}{Path(path).suffix.lower()}"
        app.add_media_file(local_file=path, url_path=_served[key])
    return _served[key]


def serve_trip_folder(trip: Trip) -> str:
    """Let the page stream the trip's own videos (for hover previews). Only on this computer."""
    key = trip.dir.name
    if key not in _served:
        _served[key] = f"/trip/{key}"
        app.add_media_files(_served[key], trip.folder)
        trip.shots_dir.mkdir(parents=True, exist_ok=True)
        app.add_media_files(_served[key] + "-shots", trip.shots_dir)  # good parts you made into shots of their own
    return _served[key]


def reveal(path: Path | None) -> None:
    if path and sys.platform == "darwin":
        subprocess.run(["open", "-R", str(path)])


def main() -> None:
    # The page talks to the app many times a second; with a big trip being analysed in the background, give it
    # its turn more often (the work threads hand over every 2 ms instead of 5), so it never loses the connection.
    sys.setswitchinterval(0.002)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    app.add_media_files("/videos", OUTPUT_DIR)
    (app_dir() / "drafts").mkdir(exist_ok=True)
    app.add_media_files("/drafts", app_dir() / "drafts")
    app.add_static_files("/data", app_dir())
    ui.add_body_html(HOVER_SCRIPT, shared=True)
    ui.add_body_html(TAP_SCRIPT, shared=True)
    ui.add_body_html(DICTATE_SCRIPT, shared=True)
    ui.page("/")(lambda: Page().build())
    print("\nTrip Memory is running. Your browser should open by itself.")
    print("If it does not, open http://localhost:8765 . Close this window to stop the app.\n")
    # A busy moment (drawing many pictures on a MacBook Air) must not count as the page being closed, which reloads it.
    ui.run(title="Trip Memory", host="127.0.0.1", port=8765, reload=False, show=True, favicon="🎬", reconnect_timeout=120,
           unocss="wind4")  # UnoCSS styles only what changed; Tailwind re-checked the whole page on every tick


if __name__ in {"__main__", "__mp_main__"}:
    main()
