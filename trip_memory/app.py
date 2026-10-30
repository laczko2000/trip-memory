"""The Trip Memory screen. Runs locally and opens in your web browser."""

from __future__ import annotations

import asyncio
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
from nicegui import app, run, ui

from .catalog import Trip, app_dir, kept_part
from .media import MediaItem
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
from . import taps as song_taps
from .beats import song_length
from .autoedit import CUT_ON, shot_value
from .autoedit import Limits as AutoLimits
from .render import TRANSITIONS, RenderSettings, clip_range, held, render, source_factor
from .scenes import SCENES, text_vectors
from .tags import ALL_TAGS, EDITABLE_TAGS, TAG_GROUP

OUTPUT_DIR = Path.home() / "Movies" / "Trip Memory"
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
        self.lock_views: dict[str, list[tuple]] = {}  # clip -> the places on screen that show its locked moments
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
        ui.on("tm_board_drop", self.board_drop)
        ui.add_css(".thumb { height: 150px; object-fit: cover; }")
        with ui.column().classes("w-full max-w-6xl mx-auto p-4 gap-3"):
            with ui.row().classes("items-baseline gap-2"):
                ui.label("Trip Memory").classes("text-3xl font-bold")
                ui.label(f"version {app_version()}").classes("text-sm text-gray-500")
            self.build_folder_row()
            with ui.tabs().classes("w-full") as tabs:
                # The path to a video, left to right; then the extra tabs for looking things up.
                self.choose_tab = ui.tab("Choose files", icon="checklist")
                self.panels = None
                themes_tab = self.themes_tab = ui.tab("Themes", icon="category")
                music_tab = ui.tab("Music & audio", icon="graphic_eq")
                story_tab = self.storyboard_tab = ui.tab("Storyboard", icon="view_timeline")
                video_tab = ui.tab("Make video", icon="movie")
                ui.separator().props("vertical").classes("mx-2")
                library_tab = ui.tab("Library", icon="photo_library").classes("opacity-70")
                people_tab = ui.tab("People", icon="people").classes("opacity-70")
                stories_tab = ui.tab("Stories", icon="auto_stories").classes("opacity-70")
            self.library_tab = library_tab
            self.music_tab = music_tab
            self.people_tab = people_tab
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
                if self.dirty and time.time() - self.last_refresh > 8:
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
        self.reload()
        self.sync_video_size()
        self.choose_view.refresh()
        self.filters_view.refresh()
        self.gallery.refresh()
        self.people_view.refresh()
        self.video_summary.refresh()
        self.titles_card.refresh()
        self.looks_card.refresh()
        self.storyboard_view.refresh()
        self.themes_view.refresh()
        self.stories_view.refresh()
        self.director_card.refresh()

    # ---------- Choose files ----------

    def tab_opened(self) -> None:
        # Tabs are only built while open (a big trip would otherwise make the page very slow), so build this one now.
        self.stale = False
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
        with ui.card().classes("w-full bg-amber-50"):
            ui.label("Step 1: sort out").classes("text-lg font-semibold")
            self.sort_label = ui.label(self.sort_text(total, chosen, size, removed)).classes("text-sm")
            ui.label(
                f"Go through quickly and untick what you don't want. {doubles} files are doubles or triples of another "
                "shot; Show → Only doubles and triples lists just those. Then remove the unticked files from the trip: "
                "they disappear from every tab and are never analysed, so everything after this is faster. "
                "Your original files on the Mac are not touched."
            ).classes("text-sm text-gray-700")
            with ui.row().classes("items-center gap-2"):
                ui.select(stories.STRICTNESS_NAMES, value=self.trip.look_alike_level(), label="Doubles are",
                          on_change=lambda e: self.set_look_alike(e.value)).classes("w-72") \
                    .tooltip("How alike shots must be to count as doubles of each other")
                ui.button("Keep only the ⭐ of every set", icon="star", on_click=self.suggest).props("outline") \
                    .tooltip("Untick every double or triple except the best one (sharpest, best exposed)")
                self.remove_button = ui.button(f"Remove {total - chosen} unticked from the trip", icon="delete_sweep",
                                               on_click=self.confirm_remove).props("color=negative")
                self.remove_button.set_visibility(total > chosen)
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
        if self.trip.look_alike_level() == level:
            return
        note = ui.notification("Finding doubles again…", spinner=True, timeout=None)
        try:
            await run.io_bound(self.trip.set_look_alike_level, level)
        finally:
            note.dismiss()
        self.reload()
        sets = len(self.all_items) - len({it["similar_set"] for it in self.all_items})
        ui.notify(f"{sets} files are now doubles of another shot. Click Keep only the ⭐ of every set to untick them.",
                  multi_line=True, timeout=8000)
        self.refresh_all()

    def confirm_remove(self) -> None:
        n = sum(1 for it in self.all_items if not it["chosen"])
        with ui.dialog() as dialog, ui.card():
            ui.label(f"Remove {n} unticked files from this trip?").classes("text-lg font-semibold")
            ui.label("They disappear from every tab (Library, People, Stories, Storyboard) and are never analysed. "
                     "The files themselves stay on your Mac, untouched. You can bring them back with See removed.") \
                .classes("text-sm")
            with ui.row().classes("justify-end w-full"):
                ui.button("Cancel", on_click=dialog.close).props("flat")
                ui.button(f"Remove {n}", icon="delete_sweep", on_click=lambda: (dialog.close(), self.remove_unchosen())) \
                    .props("color=negative")
        dialog.open()

    def remove_unchosen(self) -> None:
        n = self.trip.remove_unchosen()
        ui.notify(f"Removed {n} files from the trip.", type="positive")
        self.refresh_all()

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
        with ui.row().classes("w-full items-center gap-3"):
            ui.label("Step 2:").classes("text-lg font-semibold")
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
        if self.choose_focus_card is not None:
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
                        with ui.row().classes("gap-1"):
                            for it in members:
                                self.choose_tile(it, in_set=True)

    def choose_tile(self, it: dict, in_set: bool) -> None:
        with ui.column().classes("w-36 gap-0" + ("" if it["chosen"] else " opacity-40")) as tile:
            if it["path"] == self.choose_focus_path:
                self.choose_focus_card = tile
            self.preview(it, "width:144px;height:96px", start=self.hover_start(it))
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
                ui.image(thumb).props("fit=contain").style(size).classes("rounded bg-gray-800")
            return box
        file = Path(it.get("part_of") or it["path"])  # a part of a long clip plays from the clip itself
        src = f"{self.media_route}/{quote(str(file.relative_to(self.trip.folder)))}"
        end = (it["duration"] or 0) if end is None else end
        kept = kept_part(it)
        if kept:  # a clip you cut: only the part you kept plays
            end = min(end, kept[1])
            start = start if kept[0] <= start < end else kept[0]
        length = f"✂ {fmt_length(kept[1] - kept[0])}" if kept else fmt_length(it["duration"])
        ui.html(
            f'<div id="{box}" class="tm-hover" data-sound="{int(sound)}" data-start="{start:.2f}" data-end="{end:.2f}" data-rot="{it.get("rotation") or 0}" '
            f'style="position:relative;overflow:hidden;border-radius:4px;background:#222;{size}">'
            f'<img src="{thumb}" style="width:100%;height:100%;object-fit:contain">'
            f'<video data-src="{src}" {"" if sound else "muted "}playsinline preload="none" '
            f'style="display:none;position:absolute;inset:0;width:100%;height:100%;object-fit:contain;background:#222"></video>'
            f'<span class="tm-len" style="position:absolute;left:4px;bottom:2px;color:white;font-size:11px;'
            f'text-shadow:0 0 3px black">▶ {length}</span></div>',
            sanitize=False,
        ).style("width:100%" if "100%" in size else "")  # else a tall picture squeezes a full-width frame
        return box

    @staticmethod
    def hover_start(it: dict) -> float:
        start, end = kept_part(it) or (0.0, it["duration"] or 0)
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
        # Move the mouse along it and the video follows; click to play from there.
        ui.html(f'<div data-scrub-for="{box}" style="position:relative;height:40px;margin:4px 10px 0;border-radius:4px;'
                f'overflow:hidden;background:#e2e8f0;cursor:col-resize">'
                f'<img id="{box}-strip" style="width:100%;height:100%;object-fit:fill;display:none;pointer-events:none">'
                f'<div id="{box}-l" style="{shade};left:0;width:0;pointer-events:none"></div>'
                f'<div id="{box}-r" style="{shade};right:0;width:0;pointer-events:none"></div>'
                f'<div id="{box}-head" style="position:absolute;top:0;bottom:0;width:2px;background:#facc15;display:none;'
                f'pointer-events:none"></div></div>', sanitize=False).classes("w-full")
        # While you cut, the picture above follows the mouse too (instead of playing by itself).
        ui.timer(0.1, lambda: ui.run_javascript(
            f"const b = document.getElementById('{box}'); if (b) {{ b.dataset.scrub = '1'; "
            f"b.dataset.full0 = {start}; b.dataset.full1 = {end}; }}"), once=True)
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
            # Show the handle you just moved: its frame, then a moment of what follows (or leads up to the end).
            at = lo if lo != part["min"] else max(lo, hi - 1.5)
            part.update(min=lo, max=hi)
            show()
            ui.run_javascript(f"tmPlayPart('{box}', {lo}, {hi}, {at})")

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
                    taken: list[tuple[float, float, int]] | None = None) -> None:
        """A big window to choose a part of a video: it plays that part (silently, again and again), the strip below
        shows the whole clip (move the mouse along it to look), and the green handles set the part.
        actions: (label, icon, callback(start, end), props) buttons; a callback returning True closes the window."""
        full0, full1 = kept_part(it) or (0.0, float(it["duration"] or 0))
        part = {"min": round(max(full0, a), 1), "max": round(min(full1, b), 1)}
        with ui.dialog() as dialog, ui.card().classes("w-[60rem] max-w-full gap-2"):
            ui.label(title).classes("text-lg font-semibold")
            box = self.preview(it, "width:100%;height:min(60vh,520px)", part["min"], part["max"], view="part")
            shade = "position:absolute;top:0;bottom:0;background:rgba(15,23,42,.65)"
            ui.html(f'<div data-scrub-for="{box}" style="position:relative;height:48px;margin:4px 10px 0;'
                    f'border-radius:4px;overflow:hidden;background:#e2e8f0;cursor:col-resize">'
                    f'<img id="{box}-strip" style="width:100%;height:100%;object-fit:fill;display:none;'
                    f'pointer-events:none">'
                    f'<div id="{box}-l" style="{shade};left:0;width:0;pointer-events:none"></div>'
                    f'<div id="{box}-r" style="{shade};right:0;width:0;pointer-events:none"></div>'
                    f'<div id="{box}-head" style="position:absolute;top:0;bottom:0;width:2px;background:#facc15;'
                    f'display:none;pointer-events:none"></div>'
                    + "".join(f'<div title="Used by shot {n}" style="position:absolute;top:0;bottom:0;'
                              f'left:{100 * (x - full0) / ((full1 - full0) or 1):.2f}%;'
                              f'width:{100 * (y - x) / ((full1 - full0) or 1):.2f}%;background:rgba(220,38,38,.55);'
                              f'border-left:2px solid #dc2626;border-right:2px solid #dc2626;pointer-events:none">'
                              f'<span style="color:white;font:600 11px sans-serif;padding:2px">shot {n}</span></div>'
                              for x, y, n in taken or [])
                    + '</div>', sanitize=False).classes("w-full")
            if taken:
                ui.label("Red: parts other shots in your video already use (never shown twice).") \
                    .classes("text-xs text-red-700 px-2")
            ui.range(min=full0, max=full1, step=0.1, value=dict(part),
                     on_change=lambda e: moved(e.value["min"], e.value["max"])).props("color=green dense drag-range") \
                .classes("w-full px-2")
            label = ui.label().classes("text-sm font-medium text-green-800 px-2")
            ui.label("Move the mouse along the strip to look through the clip (click plays from there). Drag the "
                     "green bar's middle to move the same length to another moment, or its ends to make it longer "
                     "or shorter.").classes("text-xs text-gray-500 px-2")

            def show() -> None:
                label.text = (f"{fmt_length(part['min'])}–{fmt_length(part['max'])} · "
                              f"{part['max'] - part['min']:.1f} s")
                length = full1 - full0 or 1
                left, right = 100 * (part["min"] - full0) / length, 100 * (full1 - part["max"]) / length
                ui.run_javascript(f"for (const [s, w] of [['l', {left:.2f}], ['r', {right:.2f}]]) {{"
                                  f"const e = document.getElementById('{box}-' + s); if (e) e.style.width = w + '%'; }}")

            def moved(lo: float, hi: float) -> None:
                at = lo if lo != part["min"] else max(lo, hi - 1.5)
                part.update(min=lo, max=hi)
                show()
                ui.run_javascript(f"tmPlayPart('{box}', {lo}, {hi}, {at})")

            async def start() -> None:
                ui.run_javascript(f"const b = document.getElementById('{box}'); if (b) {{ b.dataset.full0 = {full0}; "
                                  f"b.dataset.full1 = {full1}; }} tmPlayPart('{box}', {part['min']}, {part['max']}, "
                                  f"{part['min']})")
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
                    f"tmPlayPart('{box}', {part['min']}, {part['max']}, {part['min']})")).props("outline dense no-caps")
                ui.space()
                ui.button("Close", on_click=dialog.close).props("flat no-caps")
                for text, icon, callback, props in actions:
                    ui.button(text, icon=icon, on_click=lambda c=callback: act(c)).props(props)
            ui.timer(0.3, start, once=True)
        dialog.on("hide", lambda: ui.run_javascript(f"const b = document.getElementById('{box}'); if (b) tmStop(b);"))
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
        self.refresh_all()  # last: this rebuilds the tile the button sits on

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
        with ui.grid().classes("w-full gap-3").style(f"grid-template-columns:repeat({self.library_columns},minmax(0,1fr))"):
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
        ui.html('<div id="tm-tl" style="position:relative;width:100%;height:150px;background:#f1f5f9;border-radius:8px;'
                'touch-action:none;user-select:none;cursor:pointer;overflow:hidden"></div>', sanitize=False).classes("w-full")
        ui.html('<div id="tm-trim-info" style="font-size:14px;font-weight:600;color:#15803d"></div>', sanitize=False)
        ui.html('<div id="tm-tap-info" style="font-size:13px;color:#475569"></div>', sanitize=False)
        self.suggesting = False  # the line starts afresh with only your own pins
        self.finish_row()
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
        themes.TIME: "The whole trip in the order it happened.",
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
            elif line["order"] != themes.TIME:
                self.storyline_blocks_view(line)
            if line["order"] != themes.STORY:  # my story ends with its last part, nothing after it
                with ui.row().classes("items-center gap-3"):
                    ui.switch("Finale: end with the most exciting shots", value=line["finale"],
                              on_change=lambda e: self.set_finale(e.value))
            with ui.row().classes("w-full items-center gap-3"):
                ui.button("Make the first cut", icon="auto_fix_high", on_click=self.run_auto_edit) \
                    .tooltip("Replaces the storyboard's order, trims and transitions; Undo brings them back")
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
                                look = ui.input(value=entry.get("moment", ""), placeholder="e.g. kite jump") \
                                    .props("dense borderless").classes("text-xs w-28")
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
                    tile.on("click", toggle)
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
                                               self.video_seconds())
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
        rows = self.trip.theme_rows()
        if not rows:
            ui.label("Nothing selected yet: tick files in Choose files first.").classes("text-gray-500")
            return
        if all(r["theme_auto"] is None for r in rows):
            ui.timer(0.1, self.suggest_themes, once=True)
            ui.label("Looking at your shots…").classes("text-gray-500")
            return
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
        if not self.trip.has_fingerprints():
            ui.label("Tip: run Analyse in Choose files first. The app then judges each picture itself; until then it "
                     "only guesses from tags.").classes("text-sm text-orange-700")
        saved_moments = self.trip.moments()
        with ui.row().classes("w-full no-wrap items-start gap-4"):
            with ui.column().classes("w-56 gap-1 flex-none"):
                for a in activities:
                    count = sum(r["theme"] == a["name"] for r in rows)
                    todo = sum(r["theme"] == a["name"] and not r["theme_ok"] for r in rows)
                    on = a["name"] == self.theme_shown
                    with ui.row().classes("w-full items-center no-wrap gap-1 px-2 py-1 rounded cursor-pointer" +
                                          (" bg-blue-100 text-blue-800 font-semibold" if on else " hover:bg-gray-100")) \
                            .on("click", lambda n=a["name"]: self.show_theme(n)):
                        ui.label(f"{a['emoji']} {a['name']}").classes("flex-grow text-sm")
                        ui.label(f"{count}" + (f" · {todo}?" if todo else "")).classes("text-xs text-gray-500")
                    for mo in [m for m in saved_moments if m["activity"] == a["name"]]:
                        here = on and self.theme_moment == mo["name"]
                        with ui.row().classes("w-full items-center no-wrap gap-1 pl-6 pr-2 py-0.5 rounded cursor-pointer" +
                                              (" bg-purple-100 text-purple-800 font-semibold" if here
                                               else " hover:bg-gray-100")) \
                                .on("click", lambda a=a["name"], m=mo["name"]: self.open_moment(a, m)):
                            ui.label(f"↳ 🔍 {mo['name']}").classes("flex-grow text-xs")
                            ui.label(f"{len(mo['shots'])} ✓").classes("text-xs text-gray-500")
                with ui.row().classes("items-center gap-1 mt-2 no-wrap"):
                    new = ui.input(placeholder="Add an activity").props("dense outlined").classes("w-36")
                    new.on("keydown.enter", lambda: self.add_activity(new.value))
                    ui.button(icon="add", on_click=lambda: self.add_activity(new.value)).props("flat dense round")
            with ui.column().classes("flex-grow gap-2 min-w-0"):
                self.theme_grid(rows, activities)

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
                .on("click", lambda p=r["path"]: self.pick_theme_shot(p)) as tile:
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
            if r["kind"] == "video":
                with ui.element("div").classes("absolute top-11 right-2").on("click.stop", lambda: None):
                    self.lock_area(r, "compact")
            with ui.column().classes("gap-0 text-xs leading-tight mt-1"):
                ui.label(("🎬 " if r["kind"] == "video" else "") + (who or "—")) \
                    .classes("font-semibold text-gray-800 truncate")
                ui.label(where).classes("text-gray-500 truncate")
                ui.label(when).classes("text-gray-500")

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

    def theme_menu(self, r: dict) -> None:
        """A big window: the shot, and every theme as a button. One click moves it there (and confirms it)."""
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
                    ui.button(f"{a['emoji']} {a['name']}", on_click=lambda n=a["name"]: self.move_one(dialog, r, n)) \
                        .props(("unelevated" if now else "outline") + " no-caps").classes("h-12 text-base w-full")
            with ui.row().classes("w-full items-center gap-2"):
                new = ui.input(placeholder="Or a new theme, e.g. Kitesurfing").props("dense outlined").classes("flex-grow")
                ui.button("Add and move here", icon="add",
                          on_click=lambda: self.move_one(dialog, r, " ".join((new.value or "").split())[:30], True)) \
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
            field = ui.input("Moment").props("outlined autofocus").classes("w-full")

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
                if group is suggested:
                    ui.button("Tick all suggested", icon="done_all",
                              on_click=lambda g=group: self.tick_all_moment(activity, name, g)).props("dense no-caps")
            room = max(0, self.theme_limit - shown)
            with ui.element("div").classes("w-full grid gap-2").style(
                    "grid-template-columns:repeat(auto-fill,minmax(170px,1fr))"):
                for f in group[:room]:
                    self.moment_tile(activity, name, f)
            shown += min(len(group), room)
        if len(found) > shown:
            ui.button(f"Show more ({len(found) - shown} left)", icon="expand_more",
                      on_click=self.more_themes).props("outline").classes("w-full")

    def moment_tile(self, activity: str, name: str, f: dict) -> None:
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
            tile.on("click", toggle)
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
                    box = self.preview(it, "width:160px;height:100px;flex:none", start, end, view="story")
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
        except (OSError, TypeError):
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

    def sound_control(self, it: dict) -> None:
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

            def set_volume(value: int) -> None:
                self.trip.set_clip_sound(it["path"], True, value / 100)
                percent.text = f"{value}%"

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
        self.trip.move(path, steps)
        self.storyboard_view.refresh()

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
                ui.select({0.0: "No black ending", 2.0: "2 s of black", 3.0: "3 s of black", 4.0: "4 s of black",
                           5.0: "5 s of black"}, value=float(t["end_black"] or 0), label="Ending",
                          on_change=lambda e: save(end_black=e.value)).classes("w-48") \
                    .tooltip("The last shot fades to black, the end card fades in on the black, and the music fades out.")

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

    async def browse_music(self) -> None:
        path = await run.io_bound(choose_with_finder, "music")
        if path:
            self.music_input.value = path

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
        title = (self.name_input.value or "").strip() or self.trip.folder.name or "My trip"
        safe = "".join(c for c in title if c.isalnum() or c in " -_").strip() or "My trip"
        out = OUTPUT_DIR / f"{safe} {datetime.now():%Y-%m-%d %H%M}.mp4"

        def on_progress(fraction: float, message: str) -> None:
            r["progress"], r["message"] = fraction, message

        r.update(busy=True, progress=0.0, message="Starting…")
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
      if (v.currentTime >= parseFloat(box.dataset.end) - 0.05) v.currentTime = parseFloat(box.dataset.start);
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
// Dragging a clip or a theme near the top or bottom of the window scrolls the page, faster the closer you get.
let tmDragY = null, tmDragTimer = null;
document.addEventListener('dragover', (e) => { tmDragY = e.clientY; });
document.addEventListener('dragstart', () => {
  clearInterval(tmDragTimer);
  tmDragTimer = setInterval(() => {
    if (tmDragY === null) return;
    const edge = 120, h = window.innerHeight;
    if (tmDragY < edge) window.scrollBy(0, -Math.ceil(30 * (edge - tmDragY) / edge));
    else if (tmDragY > h - edge) window.scrollBy(0, Math.ceil(30 * (tmDragY - (h - edge)) / edge));
  }, 16);
});
const tmDragDone = () => { clearInterval(tmDragTimer); tmDragTimer = null; tmDragY = null; };
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
    return _served[key]


def reveal(path: Path | None) -> None:
    if path and sys.platform == "darwin":
        subprocess.run(["open", "-R", str(path)])


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    app.add_media_files("/videos", OUTPUT_DIR)
    app.add_static_files("/data", app_dir())
    ui.add_body_html(HOVER_SCRIPT, shared=True)
    ui.add_body_html(TAP_SCRIPT, shared=True)
    ui.add_body_html(DICTATE_SCRIPT, shared=True)
    ui.page("/")(lambda: Page().build())
    print("\nTrip Memory is running. Your browser should open by itself.")
    print("If it does not, open http://localhost:8765 . Close this window to stop the app.\n")
    # A busy moment (drawing many pictures on a MacBook Air) must not count as the page being closed, which reloads it.
    ui.run(title="Trip Memory", host="127.0.0.1", port=8765, reload=False, show=True, favicon="🎬", reconnect_timeout=30,
           unocss="wind4")  # UnoCSS styles only what changed; Tailwind re-checked the whole page on every tick


if __name__ in {"__main__", "__mp_main__"}:
    main()
