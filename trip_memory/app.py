"""The Trip Memory screen. Runs locally and opens in your web browser."""

from __future__ import annotations

import asyncio
import hashlib
import json
import subprocess
import sys
import time
import traceback
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from urllib.parse import quote

from nicegui import app, run, ui

from .catalog import Trip, app_dir
from .media import MediaItem
from . import director
from . import music as free_music
from . import songmap
from . import taps as song_taps
from .beats import song_length
from .autoedit import CUT_ON, STORYLINES
from .autoedit import Limits as AutoLimits
from .render import TRANSITIONS, RenderSettings, clip_range, render, source_factor
from .scenes import SCENES
from .tags import ALL_TAGS, EDITABLE_TAGS, TAG_GROUP

OUTPUT_DIR = Path.home() / "Movies" / "Trip Memory"
RESOLUTIONS = {"Full HD (1080p)": (1920, 1080), "HD (720p, faster)": (1280, 720)}
PAGE_SIZE = 48
STORIES_PER_PAGE = 8
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


def data_url(path: str | None) -> str:
    """Web address for a thumbnail stored in the app's data folder."""
    if not path:
        return ""
    return "/data/" + str(Path(path).relative_to(app_dir()))


def fmt_time(iso: str) -> str:
    return datetime.fromisoformat(iso).astimezone().strftime("%a %d %b, %H:%M")


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


class Page:
    """Everything on the screen for one browser tab."""

    def __init__(self) -> None:
        self.trip: Trip | None = None
        self.all_items: list[dict] = []  # every file in the folder
        self.items: list[dict] = []  # the files you chose for this trip
        self.choose_filters = {"day": ALL, "kind": ALL}
        self.choose_page = 1
        self.story_page = 1
        self.media_route: str | None = None
        self.auto_length: object = None
        self.auto_limits = AutoLimits(**load_settings().get("limits", {}))
        self.song_maps: dict = {}
        self.dirty = False
        self.last_refresh = 0.0
        self.filters = {"day": ALL, "place": ALL, "scene": ALL, "tag": ALL, "look": ALL, "person": ALL, "show": ALL}
        self.page = 1
        self.rendering = {"busy": False, "progress": 0.0, "message": "", "result": None}
        self.ollama_found: tuple[list[str], list[str]] | None = None  # (models that see pictures, all models)
        self.ollama_checked = False

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
        return {
            "day": day_opts,
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
        ui.add_css(".thumb { height: 150px; object-fit: cover; }")
        with ui.column().classes("w-full max-w-6xl mx-auto p-4 gap-3"):
            ui.label("Trip Memory").classes("text-3xl font-bold")
            self.build_folder_row()
            with ui.tabs().classes("w-full") as tabs:
                self.choose_tab = ui.tab("Choose files", icon="checklist")
                library_tab = ui.tab("Library", icon="photo_library")
                people_tab = ui.tab("People", icon="people")
                music_tab = ui.tab("Music", icon="graphic_eq")
                stories_tab = ui.tab("Stories", icon="auto_stories")
                story_tab = ui.tab("Storyboard", icon="view_timeline")
                video_tab = ui.tab("Make video", icon="movie")
            self.library_tab = library_tab
            self.people_tab = people_tab
            with ui.tab_panels(tabs, value=self.choose_tab).classes("w-full") as self.panels:
                with ui.tab_panel(self.choose_tab):
                    self.choose_view()
                with ui.tab_panel(library_tab):
                    self.build_library()
                with ui.tab_panel(people_tab):
                    self.people_view()
                with ui.tab_panel(music_tab):
                    self.music_tab_view()
                with ui.tab_panel(stories_tab):
                    self.director_card()
                    self.stories_view()
                with ui.tab_panel(story_tab):
                    self.storyboard_view()
                with ui.tab_panel(video_tab):
                    self.build_video()
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
            else:
                self.progress_bar.set_visibility(False)
                self.progress_label.text = (
                    f"{self.trip.folder.name}: {len(self.all_items)} files in the folder, {len(self.items)} chosen."
                )
            self.notes_label.text = "\n".join(p.notes[:8])
        r = self.rendering
        if r["busy"]:
            self.render_progress.value = r["progress"]
            self.render_status.text = r["message"]

    def refresh_all(self) -> None:
        self.reload()
        self.choose_view.refresh()
        self.filters_view.refresh()
        self.gallery.refresh()
        self.people_view.refresh()
        self.video_summary.refresh()
        self.storyboard_view.refresh()
        self.stories_view.refresh()
        self.director_card.refresh()

    # ---------- Choose files ----------

    def choose_shown(self) -> list[dict]:
        f = self.choose_filters
        return [
            it
            for it in self.all_items
            if (f["day"] == ALL or it["day"] == f["day"]) and (f["kind"] == ALL or it["kind"] == f["kind"])
        ]

    @ui.refreshable
    def choose_view(self) -> None:
        if not self.all_items:
            ui.label("Open a trip folder to see its files here.").classes("text-gray-500")
            return
        days = sorted({it["day"] for it in self.all_items})
        day_opts = {ALL: "All days", **{d: f"Day {n} · {datetime.fromisoformat(d):%a %d %b}" for n, d in enumerate(days, 1)}}
        if self.choose_filters["day"] not in day_opts:
            self.choose_filters["day"] = ALL
        shown = self.choose_shown()
        chosen = len(self.items)
        with ui.row().classes("w-full items-center gap-3"):
            ui.label(f"{chosen} of {len(self.all_items)} files chosen").classes("text-lg font-semibold")
            ui.button(
                f"Analyse the {chosen} chosen files", icon="auto_awesome", on_click=self.analyse_chosen
            ).props("" if chosen else "disable")
            ui.button("Suggest best shots", icon="star", on_click=self.suggest).props("outline").tooltip(
                "Tick only the best shot of each set of similar shots"
            )
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
            ui.button("Tick all shown", icon="check_box", on_click=lambda: self.choose_shown_files(True)).props("flat dense")
            ui.button("Untick all shown", icon="check_box_outline_blank", on_click=lambda: self.choose_shown_files(False)).props("flat dense")

        story_ids = sorted({it["story"] or 0 for it in shown})
        pages = max(1, -(-len(story_ids) // STORIES_PER_PAGE))
        self.choose_page = min(self.choose_page, pages)
        for number in story_ids[(self.choose_page - 1) * STORIES_PER_PAGE : self.choose_page * STORIES_PER_PAGE]:
            self.story_block(number, [it for it in shown if (it["story"] or 0) == number])
        if pages > 1:
            ui.pagination(1, pages, direction_links=True, value=self.choose_page,
                          on_change=lambda e: self.go_to_choose_page(e.value))

    def story_block(self, number: int, items: list[dict]) -> None:
        first, last = datetime.fromisoformat(items[0]["taken_at"]).astimezone(), datetime.fromisoformat(items[-1]["taken_at"]).astimezone()
        places = [it["place"] or it["country"] for it in items if it["place"] or it["country"]]
        place = max(set(places), key=places.count) if places else "Unknown place"
        picked = sum(it["chosen"] for it in items)
        with ui.card().classes("w-full p-3 mt-2"):
            with ui.row().classes("w-full items-center gap-2"):
                ui.label(f"Story {number + 1} · {first:%a %d %b, %H:%M}–{last:%H:%M} · {place}").classes("font-semibold")
                ui.label(f"{picked} of {len(items)} chosen").classes("text-sm text-gray-500")
                paths = [it["path"] for it in items]
                ui.button("Tick story", on_click=lambda p=paths: self.choose_paths(p, True)).props("flat dense size=sm")
                ui.button("Untick story", on_click=lambda p=paths: self.choose_paths(p, False)).props("flat dense size=sm")
            sets: dict[int, list[dict]] = {}
            for it in items:
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
        with ui.column().classes("w-36 gap-0" + ("" if it["chosen"] else " opacity-40")):
            self.preview(it, "width:144px;height:96px", start=(it["duration"] or 0) * 0.3)
            with ui.row().classes("items-center gap-0 no-wrap"):
                ui.checkbox(
                    value=bool(it["chosen"]),
                    on_change=lambda e, path=it["path"]: self.choose_paths([path], e.value),
                ).props("dense size=xs")
                rotate_buttons(lambda degrees, path=it["path"]: self.rotate(path, degrees))
                star = "⭐ " if in_set and it["best"] else ""
                ui.label(f"{star}{datetime.fromisoformat(it['taken_at']).astimezone():%H:%M:%S}").classes("text-xs text-gray-600")

    def preview(self, it: dict, size: str, start: float = 0, end: float | None = None, view: str = "",
                sound: bool = False) -> str:
        """A preview picture; for a video, hovering plays it (only the part from start to end).

        Returns the element id so the storyboard can change the part being played."""
        box = f"tm-{view}-{hashlib.sha1(it['path'].encode()).hexdigest()[:12]}"
        thumb = data_url(it["thumb"])
        if it["kind"] != "video" or not self.media_route:
            if thumb:
                ui.image(thumb).props("fit=contain").style(size).classes("rounded bg-gray-800")
            return box
        src = f"{self.media_route}/{quote(str(Path(it['path']).relative_to(self.trip.folder)))}"
        end = (it["duration"] or 0) if end is None else end
        ui.html(
            f'<div id="{box}" class="tm-hover" data-start="{start:.2f}" data-end="{end:.2f}" data-rot="{it.get("rotation") or 0}" '
            f'style="position:relative;overflow:hidden;border-radius:4px;background:#222;{size}">'
            f'<img src="{thumb}" style="width:100%;height:100%;object-fit:contain">'
            f'<video data-src="{src}" {"" if sound else "muted "}playsinline preload="none" '
            f'style="display:none;position:absolute;inset:0;width:100%;height:100%;object-fit:contain;background:#222"></video>'
            f'<span style="position:absolute;left:4px;bottom:2px;color:white;font-size:11px;text-shadow:0 0 3px black">'
            f'▶ {fmt_length(it["duration"])}</span></div>',
            sanitize=False,
        ).style("width:100%" if "100%" in size else "")  # else a tall picture squeezes a full-width frame
        return box

    def suggest(self) -> None:
        self.trip.suggest_selection()
        ui.notify("Ticked the best shot of each set of similar shots.")
        self.refresh_all()

    def keep_best(self, members: list[dict]) -> None:
        self.trip.set_chosen([m["path"] for m in members if not m["best"]], False)
        self.trip.set_chosen([m["path"] for m in members if m["best"]], True)
        self.refresh_all()

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
        self.trip.set_chosen(paths, chosen)
        self.refresh_all()

    def set_choose_filter(self, key: str, value: str) -> None:
        self.choose_filters[key] = value
        self.choose_page = 1
        self.story_page = 1
        self.media_route: str | None = None
        self.choose_view.refresh()

    def go_to_choose_page(self, page: int) -> None:
        self.choose_page = page
        self.choose_view.refresh()

    def choose_shown_files(self, chosen: bool) -> None:
        self.trip.set_chosen([it["path"] for it in self.choose_shown()], chosen)
        self.refresh_all()

    # ---------- Library ----------

    def build_library(self) -> None:
        self.filters_view()
        self.gallery()

    @ui.refreshable
    def filters_view(self) -> None:
        if not self.items:
            return
        opts = self.options()
        with ui.row().classes("w-full items-end gap-3"):
            for key, label in (("day", "Day"), ("place", "Place"), ("scene", "Scene"), ("tag", "Tag"),
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
        if not self.items:
            if self.trip:
                ui.label("Nothing to show yet.").classes("text-gray-500")
            return
        shown = self.shown()
        in_video = sum(it["included"] for it in self.items)
        ui.label(f"Showing {len(shown)} of {len(self.items)} · {in_video} in the video").classes("text-sm text-gray-500")
        pages = max(1, -(-len(shown) // PAGE_SIZE))
        self.page = min(self.page, pages)
        with ui.grid().classes("w-full gap-3 grid-cols-2 sm:grid-cols-3 md:grid-cols-4"):
            for it in shown[(self.page - 1) * PAGE_SIZE : self.page * PAGE_SIZE]:
                self.card(it)
        if pages > 1:
            ui.pagination(1, pages, direction_links=True, value=self.page, on_change=lambda e: self.go_to_page(e.value))

    def go_to_page(self, page: int) -> None:
        self.page = page
        self.gallery.refresh()

    def card(self, it: dict) -> None:
        with ui.card().tight().classes("w-full" + ("" if it["included"] else " opacity-50")) as card:
            if it["thumb"]:
                self.preview(it, "width:100%;height:150px", start=(it["duration"] or 0) * 0.3)
            else:
                ui.label("Analysing…").classes("w-full thumb flex items-center justify-center bg-gray-100 text-gray-400")
            with ui.column().classes("p-2 gap-1 w-full"):
                icon = "🎬 " + fmt_length(it["duration"]) if it["kind"] == "video" else "📷"
                ui.label(f"{icon} · {fmt_time(it['taken_at'])}").classes("text-xs text-gray-600")
                if it.get("ai_caption"):
                    ui.label(f"📝 {it['ai_caption']}" + " ★" * (it["ai_score"] or 0)).classes("text-xs italic") \
                        .tooltip("The AI director's description, and how much it stands out (1 to 5 stars)")
                cut_note = ui.label("✂ Selected, but not in the current Auto-edit cut").classes("text-xs text-orange-700")
                cut_note.visible = bool(it["included"] and it["cut_out"])
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
                            ui.button("Restore", on_click=lambda pid=person["id"]: self.restore(pid)).props("flat dense size=sm")

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

    def restore(self, person_id: int) -> None:
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
        saved = song_taps.load(music, app_dir() / "music-taps")
        ui.label(
            "Press Play and tap the RIGHT Shift key wherever you want the video to cut. Each tap puts a pin on the "
            "line below. Play it back to check: every pin makes a click as it passes. Drag a pin to move it, click it "
            "and press Delete to remove it. Then let the app snap the pins onto the exact hits, and mark them final."
        ).classes("text-sm text-gray-600")
        ui.html(f'<audio id="tm-song" preload="auto" src="{serve_song(music)}" onloadedmetadata="tmTapDraw()" '
                f'onplay="tmTapOnPlay()" onpause="tmTapUnschedule(); tmTapDraw()"></audio>', sanitize=False)
        with ui.row().classes("w-full items-center gap-2"):
            ui.button("Play / pause", icon="play_arrow", on_click=lambda: ui.run_javascript("tmTapPlay()"))
            ui.button("5 s back", icon="replay_5", on_click=lambda: ui.run_javascript("tmTapSeek(-5)")).props("outline")
            ui.button("From the start", icon="skip_previous", on_click=lambda: ui.run_javascript("tmTapStart()")).props("outline")
            ui.select({0: "Whole song", 30: "30 seconds", 10: "10 seconds"}, value=0, label="Show",
                      on_change=lambda e: ui.run_javascript(f"tmTapZoom({e.value})")).classes("w-36")
            ui.button("Delete selected pin", icon="delete", on_click=lambda: ui.run_javascript("tmTapDelete()")).props("flat")
            ui.button("Clear all", icon="clear_all", on_click=self.clear_taps).props("flat")
        ui.html('<div id="tm-tl" style="position:relative;width:100%;height:96px;background:#f1f5f9;border-radius:8px;'
                'touch-action:none;user-select:none;cursor:pointer;overflow:hidden"></div>', sanitize=False).classes("w-full")
        ui.html('<div id="tm-tap-info" style="font-size:13px;color:#475569"></div>', sanitize=False)
        with ui.row().classes("w-full items-center gap-3"):
            ui.button("Snap to the music", icon="center_focus_strong", on_click=self.snap_taps).props("outline")
            ui.button("Save for later", icon="save", on_click=lambda: self.save_taps(final=False)).props("outline")
            ui.button("These are final", icon="check_circle", on_click=lambda: self.save_taps(final=True))
            state = ("✓ Final: Auto-edit cuts here" if saved["final"] else "Saved, not final yet") if saved["taps"] else ""
            ui.label(state).classes("text-sm " + ("text-green-700" if saved["final"] else "text-gray-500"))
        if saved["taps"]:
            self.tap_lengths(music, saved["taps"])
        ui.timer(0.2, lambda: ui.run_javascript(f"tmTapInit({json.dumps(saved['taps'])}, 0)"), once=True)

    def tap_lengths(self, music: str, taps: list[float]) -> None:
        """The shots your cuts make, checked against your limits."""
        song = self.song_maps.get(music)
        try:
            duration = song.duration if song else song_length(music)
        except Exception:
            duration = max(taps) + 1
        lengths = song_taps.shot_lengths(taps, max(taps))  # up to your last pin; after it the app cuts by itself
        limits = self.auto_limits
        longest = max(limits.max_video, limits.max_photo)
        with ui.row().classes("w-full gap-1"):
            for n, length in enumerate(lengths, 1):
                colour = "orange" if length > longest * 1.03 else "red" if length < limits.min_shot else "blue-grey"
                ui.chip(f"{n}: {length:.1f} s", color=colour, text_color="white").props("dense square")
        too_long = sum(length > longest * 1.03 for length in lengths)
        too_short = sum(length < limits.min_shot for length in lengths)
        rest = duration - max(taps)
        notes = [f"After your last pin ({fmt_length(max(taps))}) the app cuts the remaining {fmt_length(rest)} by itself."
                 if rest > longest else ""]
        if too_long:
            notes.append(f"{too_long} orange shot(s) are longer than your longest clip ({longest:g} s); Auto-edit splits "
                         "them on the beat. Raise the limit in the Storyboard to keep them whole.")
        if too_short:
            notes.append(f"{too_short} red shot(s) are shorter than your shortest shot ({limits.min_shot:g} s); they're "
                         "kept, since you placed them.")
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

    async def save_taps(self, final: bool) -> None:
        music = self.music_path()
        pins = await ui.run_javascript("tmTapGet()", timeout=5)
        if not music:
            return
        song_taps.save(music, app_dir() / "music-taps", [float(p) for p in pins or []], final)
        if final and pins:
            self.set_limit("cut_on", "taps")
            ui.notify(f"{len(pins)} cuts saved. Auto-edit (in the Storyboard) now cuts exactly there.", type="positive")
        else:
            ui.notify("Saved." if pins else "Saved: no pins.")
        self.music_tab_view.refresh()
        self.storyboard_view.refresh()

    async def clear_taps(self) -> None:
        ui.run_javascript("tmTapClear()")

    # ---------- Stories ----------

    @ui.refreshable
    def director_card(self) -> None:
        if not self.trip:
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
                else:
                    self.claude_settings(ai)
            with ui.row().classes("items-center gap-2"):
                left = len(chosen) - described
                ui.button(f"Describe clips ({left} to go)" if left else "Describe clips", icon="visibility",
                          on_click=lambda: self.describe_clips(again=False)).props("" if left else "outline")
                if described:
                    ui.button("Describe all again", on_click=lambda: self.describe_clips(again=True)).props("flat dense no-caps")
                ui.button("Suggest storylines", icon="lightbulb", on_click=self.suggest_storylines)
            if chosen and not described:
                ui.label("Tip: describe the clips first, so the storylines know what is in each story.") \
                    .classes("text-xs text-gray-500")
            saved = self.trip.saved_storylines()
            if saved:
                used = self.trip.storyline_used()
                names = {s["key"]: s["name"] for s in self.trip.stories()}
                with ui.row().classes("w-full gap-3 items-stretch"):
                    for i, line in enumerate(saved["storylines"]):
                        on = i == used
                        with ui.card().classes("p-3 flex-1 min-w-60" + (" ring-2 ring-indigo-500" if on else "")):
                            ui.label(line["title"]).classes("font-semibold")
                            ui.label(line["idea"]).classes("text-sm text-gray-600")
                            ui.label(" → ".join(saved["names"].get(k) or names.get(k, "?") for k in line["stories"])) \
                                .classes("text-xs text-gray-500")
                            ui.button("Using this" if on else "Use this", icon="check" if on else None,
                                      on_click=lambda i=i: self.use_storyline(i)).props("dense" + (" outline" if on else ""))

    def director_ready(self, ai: dict) -> bool:
        if ai.get("provider") == "claude":
            return bool(ai.get("api_key"))
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
        ui.button("Check again", icon="refresh", on_click=self.check_ollama).props("flat dense no-caps")

    async def check_ollama(self) -> None:
        self.ollama_found = await run.io_bound(director.ollama_models)
        self.director_card.refresh()

    def claude_settings(self, ai: dict) -> None:
        ui.markdown(
            "Very good descriptions and storylines. Costs a little per use, roughly 1 cent per clip described, "
            "paid to Anthropic. **One-time setup:** at [console.anthropic.com](https://console.anthropic.com) add some "
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
        model = ui.input("Model", value=ai.get("claude_model") or director.CLAUDE_MODEL).classes("w-72")
        model.on("blur", lambda: self.save_ai(refresh=False, claude_model=model.value.strip() or director.CLAUDE_MODEL))

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

    async def describe_clips(self, again: bool) -> None:
        if not self.trip or self.trip.progress.running:
            return
        ai = self.make_director()
        if ai:
            await self.run_step(lambda notify: self.trip.describe_clips(ai, notify, again))

    async def suggest_storylines(self) -> None:
        if not self.trip or self.trip.progress.running:
            return
        ai = self.make_director()
        if not ai:
            return
        note = ui.notification("The AI is thinking about storylines…", spinner=True, timeout=None)
        try:
            await run.io_bound(self.trip.suggest_storylines, ai)
        except director.DirectorError as exc:
            ui.notify(str(exc), type="warning", multi_line=True, timeout=10000)
        except Exception as exc:
            traceback.print_exc()
            ui.notify(f"Stopped: {exc}", type="negative")
        finally:
            note.dismiss()
        self.director_card.refresh()

    def use_storyline(self, index: int) -> None:
        self.trip.apply_storyline(index)
        ui.notify("Stories put in this order. Run Make a first cut in the Storyboard to use it.", type="positive")
        self.refresh_all()

    @ui.refreshable
    def stories_view(self) -> None:
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
                    start, end = fmt_time(story["start"]), datetime.fromisoformat(story["end"]).astimezone().strftime("%H:%M")
                    ui.label(f"{start}–{end}" + (f" · {place}" if place else "") +
                             f" · {len(members)} files, {included} in the video").classes("text-xs text-gray-600")
                    if story["renamed"]:
                        ui.label(f"The app's name: {story['auto_name']}").classes("text-xs text-gray-400")
                ui.switch("In the video", value=story["enabled"],
                          on_change=lambda e, k=story["key"]: self.set_story_enabled(k, e.value))

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
        if not self.trip:
            ui.label("Open a trip first.").classes("text-gray-500")
            return
        self.auto_edit_card()
        board = self.trip.storyboard()
        if not board:
            ui.label("Nothing is in the video yet. Tick files in the Library, or let Auto-edit choose.").classes("text-gray-500")
            return
        with ui.row().classes("w-full items-center gap-4"):
            ui.label(f"{len(board)} clips in this order").classes("text-lg font-semibold")
            ui.button("Back to date order", icon="restart_alt", on_click=self.reset_order).props("flat dense")
        ui.label(
            "Use the arrows to change the order. For videos, drag the two ends of the bar to choose which part to keep. "
            "The transition setting is for the cut after that clip."
        ).classes("text-sm text-gray-600")
        for n, it in enumerate(board, 1):  # one long list you scroll through
            self.story_row(n, it, last=n == len(board))

    def auto_edit_card(self) -> None:
        music = self.music_path()
        limits = self.auto_limits
        with ui.card().classes("w-full bg-blue-50"):
            ui.label("✨ Auto-edit: the music leads, the shots follow").classes("text-lg font-semibold")
            # 1. The song
            with ui.row().classes("w-full items-center gap-3"):
                ui.label("1. Song:").classes("font-medium")
                ui.label(Path(music).name if music else "none chosen yet").classes("text-gray-700")
                if sys.platform == "darwin":
                    ui.button("Choose song…", icon="music_note", on_click=self.browse_music).props("outline dense")
                ui.button("Find free music", icon="library_music", on_click=self.find_music).props("flat dense")
            if music:
                self.music_map_view(music)
            else:
                ui.label(
                    "Without a song the cuts follow a steady pace. With one, the app studies its beats and calm and "
                    "lively parts first, and cuts to them."
                ).classes("text-sm text-gray-600")
            # 2. Your limits
            ui.label("2. Your limits").classes("font-medium mt-2")
            with ui.row().classes("w-full gap-8"):
                for key, label, lo, hi in (("max_video", "Longest video clip", 1.0, 10.0),
                                           ("max_photo", "Longest photo", 0.5, 6.0),
                                           ("min_shot", "Shortest shot", 0.5, 3.0)):
                    with ui.column().classes("min-w-48 flex-grow gap-0"):
                        ui.label(label).classes("text-sm text-gray-600")
                        ui.slider(min=lo, max=hi, step=0.5, value=getattr(limits, key),
                                  on_change=lambda e, k=key: self.set_limit(k, e.value)).props("label-always").classes("mt-6")
            with ui.row().classes("w-full items-center gap-3"):
                ui.select(STORYLINES, value=limits.storyline if limits.storyline in STORYLINES else "teaser",
                          label="Storyline", on_change=lambda e: self.set_limit("storyline", e.value)).classes("w-56")
                ui.select(CUT_ON, value=limits.cut_on if limits.cut_on in CUT_ON else "strong", label="Cut on",
                          on_change=lambda e: self.set_limit("cut_on", e.value)).classes("w-56")
                ui.label({
                    "strong": "Cuts on the hardest hits (drums, accents), keeping the pace of each part of the song.",
                    "bass": "Cuts on the kick drum and bass notes.",
                    "changes": "Few cuts: shots held as long as you allow, cut on the biggest moments.",
                    "beats": "Cuts on a steady count of beats, lined up with the bars.",
                    "taps": "Cuts exactly on the pins you tapped in the Music tab.",
                }.get(limits.cut_on, "")).classes("text-sm text-gray-600")
            ui.switch("Speed effects: slow motion in calm, cinematic moments; slow shots sped up in lively music",
                      value=limits.speed_effects, on_change=lambda e: self.set_limit("speed_effects", e.value))
            ui.switch("Camera effects: zoom transitions, punch-ins on big hits, speed ramps, slow push-ins on still shots",
                      value=limits.camera_effects, on_change=lambda e: self.set_limit("camera_effects", e.value))
            ui.label({
                "teaser": "Storyline: opens with a quick teaser of the most exciting shots, then tells each story; every "
                          "story starts with a wide, calm view, look-alike shots never come back to back, and the "
                          "video ends on a calm shot.",
                "stories": "Storyline: each story starts with a wide, calm view, look-alike shots never come back to "
                           "back, and the video ends on a calm shot.",
                "date": "Storyline: everything in the order it was taken.",
            }.get(limits.storyline, "")).classes("text-sm text-gray-600")
            # 3. Length, then go
            with ui.row().classes("w-full items-center gap-4 mt-2"):
                ui.label("3.").classes("font-medium")
                lengths = {60: "1 minute", 120: "2 minutes", 180: "3 minutes", 300: "5 minutes", 600: "10 minutes",
                           "all": "Use everything"}
                if music:
                    lengths = {"song": "As long as the song", **lengths}
                    lengths.pop("all")
                if self.auto_length not in lengths:
                    self.auto_length = "song" if music else 180
                ui.select(lengths, value=self.auto_length, label="Length",
                          on_change=lambda e: setattr(self, "auto_length", e.value)).classes("w-48")
                ui.button("Make a first cut", icon="auto_fix_high", on_click=self.confirm_auto_edit)
            ui.label(
                "Cuts land on the beat: quick shots in the lively parts, longer ones in the calm parts, and a cut at every "
                "change in the music. Action clips go to lively music, calm and cinematic clips and photos to calm music, "
                "following your storyline, each at its best moment."
            ).classes("text-sm text-gray-600")

    def set_limit(self, key: str, value) -> None:
        setattr(self.auto_limits, key, value)
        save_settings(limits=asdict(self.auto_limits))
        if key == "cut_on":
            self.storyboard_view.refresh()

    def music_map_view(self, music: str) -> None:
        """A small picture of the song: one bar per musical bar, tall when lively, with its parts named."""
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
                self.storyboard_view.refresh()

            ui.timer(0.1, study, once=True)
            return
        colours = {"Calm": "#93c5fd", "Medium": "#60a5fa", "Lively": "#f97316"}
        width = max(1, len(song.bar_energy))
        bars = []
        bar_len = song.period * songmap.BEATS_PER_BAR
        starts = song.bar_starts()
        for k, e in enumerate(song.bar_energy):
            t = starts[k] if k < len(starts) else song.duration
            level = next((s.level for s in song.sections if s.start <= t < s.end), "Medium")
            h = 8 + 52 * e
            bars.append(f'<rect x="{k}" y="{60 - h:.1f}" width="0.8" height="{h:.1f}" fill="{colours[level]}"/>')
        # The strong beats: small marks along the top, taller for a harder hit.
        first_bar = starts[0] if len(starts) else song.first_beat
        for t, a in zip(song.beat_times(), song.accents):
            if a >= 0.6 and bar_len:
                x = (t - first_bar) / bar_len
                bars.append(f'<line x1="{x:.2f}" y1="0" x2="{x:.2f}" y2="{3 + 6 * a:.1f}" stroke="#be123c" stroke-width="0.3"/>')
        for s in song.sections[1:]:
            x = (s.start - song.first_beat) / bar_len if bar_len else 0
            bars.append(f'<line x1="{x:.2f}" y1="0" x2="{x:.2f}" y2="60" stroke="#1e3a8a" stroke-width="0.25"/>')
        ui.html(f'<svg viewBox="0 0 {width} 60" preserveAspectRatio="none" style="width:100%;height:60px">{"".join(bars)}</svg>',
                sanitize=False).classes("w-full")
        parts = " → ".join(f"{s.level} {fmt_length(s.end - s.start)}" for s in song.sections)
        beat = f"{song.bpm:.0f} beats a minute" if song.clear_beat else "no clear beat (steady pace)"
        strong = sum(a >= 0.6 for a in song.accents)
        beat += f" · {strong} strong beats (red marks)" if strong else ""
        ui.label(f"{fmt_length(song.duration)} · {beat} · {parts}").classes("text-xs text-gray-600")

    def music_changed(self) -> None:
        if hasattr(self, "music_credit"):
            credit = free_music.credit_for(self.music_input.value, app_dir() / "music") if self.music_input.value else None
            self.music_credit.text = f"Free song: {credit}. A credits file is saved next to your video." if credit else ""
        self.auto_length = None  # back to the default: as long as the song, if there is one
        self.storyboard_view.refresh()
        self.music_tab_view.refresh()

    def music_path(self) -> str | None:
        music = (self.music_input.value or "").strip().strip("'\"") if hasattr(self, "music_input") else ""
        return music if music and Path(music).expanduser().is_file() else None

    def confirm_auto_edit(self) -> None:
        with ui.dialog() as dialog, ui.card():
            ui.label("Make a first cut?").classes("text-lg font-semibold")
            ui.label("This replaces what's in the storyboard now: which files are in, their order, trims and transitions. "
                     "Your files, people and tags are not touched.")
            with ui.row().classes("justify-end w-full"):
                ui.button("Cancel", on_click=dialog.close).props("flat")
                async def make_it() -> None:
                    dialog.close()
                    await self.run_auto_edit()

                ui.button("Make it", on_click=make_it)
        dialog.open()

    async def run_auto_edit(self) -> None:
        music = self.music_path()
        s = self.settings_now()
        style = s.transition if TRANSITIONS.get(s.transition, "fade") not in (None, "mixed") else "Fade"
        target = self.auto_length
        limits = AutoLimits(**asdict(self.auto_limits))
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
        if music and limits.cut_on == "taps":
            on_beat = (f", cut exactly on your {result.pins} pins" if result.pins else
                       ". This song has no final pins, so it cut on the beat: in the Music tab, place pins and click "
                       "These are final")
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
        with ui.card().classes("w-full p-2"):
            with ui.row().classes("w-full items-center no-wrap gap-3"):
                with ui.column().classes("gap-0 items-center"):
                    ui.button(icon="arrow_upward", on_click=lambda p=it["path"]: self.move(p, -1)).props("flat dense round size=sm")
                    ui.label(str(n)).classes("text-xs text-gray-500")
                    ui.button(icon="arrow_downward", on_click=lambda p=it["path"]: self.move(p, 1)).props("flat dense round size=sm")
                box = None
                if it["thumb"]:
                    start, end = self.kept_part(it) if it["kind"] == "video" else (0, None)
                    box = self.preview(it, "width:160px;height:100px;flex:none", start, end, view="story", sound=True)
                with ui.column().classes("flex-grow gap-1"):
                    place = ", ".join(x for x in (it["place"], it["country"]) if x)
                    icon = "🎬" if it["kind"] == "video" else "📷"
                    ui.label(f"{icon} {fmt_time(it['taken_at'])}" + (f" · {place}" if place else "")).classes("text-sm")
                    if it.get("ai_caption"):
                        ui.label(f"📝 {it['ai_caption']}").classes("text-xs italic text-gray-600")
                    if it["kind"] == "video":
                        self.trim_control(it, box)
                        with ui.row().classes("items-center gap-4"):
                            self.sound_control(it)
                            self.speed_control(it)
                        self.effects_control(it)
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

    def choose_transition(self, path: str, value: str, why: ui.label) -> None:
        self.trip.set_transition(path, None if value == DEFAULT_TRANSITION else value)
        why.text = "Your choice"

    def kept_part(self, it: dict) -> tuple[float, float]:
        item = MediaItem(path=Path(it["path"]), kind="video", taken_at=datetime.now(), duration=it["duration"],
                         trim_start=it["trim_start"], trim_end=it["trim_end"])
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
            label.text = f"Keeping {fmt_length(lo)} to {fmt_length(hi)} ({hi - lo:.1f} s of {fmt_length(it['duration'])})" + (
                f" · {(hi - lo) / speed:.1f} s on screen" if abs(speed - 1) > 1e-3 else "")

        it["_show_trim"] = lambda: show(*self.kept_part(it))

        show(start, end)
        ui.range(
            min=0, max=round(it["duration"], 1), step=0.1, value={"min": round(start, 1), "max": round(end, 1)},
            on_change=lambda e: trimmed(e.value["min"], e.value["max"]),
        ).props("dense").classes("w-full")

        def trimmed(lo: float, hi: float) -> None:
            show(lo, hi)
            self.trip.set_trim(it["path"], lo, hi)
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
        with ui.card().classes("w-full"):
            ui.label("Style").classes("text-lg font-semibold")
            with ui.grid(columns=2).classes("w-full gap-4"):
                self.transition = ui.select(list(TRANSITIONS), value="Fade", label="Transition")
                self.resolution = ui.select(list(RESOLUTIONS), value="Full HD (1080p)", label="Video size")
                with ui.column():
                    ui.label("Transition length (seconds)").classes("text-sm text-gray-500")
                    self.t_len = ui.slider(min=0.3, max=2.0, step=0.1, value=0.8).props("label-always").classes("mt-8")
                with ui.column():
                    ui.label("Each photo shows for (seconds)").classes("text-sm text-gray-500")
                    self.photo_len = ui.slider(min=2, max=8, step=0.5, value=4).props("label-always").classes("mt-8")
                with ui.column():
                    ui.label("Longest part of each video (seconds, 0 = whole video)").classes("text-sm text-gray-500")
                    self.clip_len = ui.slider(min=0, max=60, step=1, value=10).props("label-always").classes("mt-8")
            with ui.row().classes("w-full items-center"):
                self.music_input = ui.input(
                    "Music (optional)", placeholder="/Users/you/Music/song.mp3",
                    on_change=self.music_changed,  # Auto-edit uses the music's beat
                ).classes("flex-grow")
                if sys.platform == "darwin":
                    ui.button("Browse…", on_click=self.browse_music).props("outline")
                ui.button("Find free music", icon="library_music", on_click=self.find_music).props("outline")
            self.music_credit = ui.label("").classes("text-xs text-gray-500")
            with ui.column().classes("w-full"):
                ui.label("Music volume").classes("text-sm text-gray-500")
                self.music_vol = ui.slider(min=0.1, max=1.0, step=0.05, value=0.5).props("label-always").classes("mt-8")

        with ui.card().classes("w-full"):
            self.name_input = ui.input("Video name", placeholder="My trip").classes("w-full")
            self.make_button = ui.button("Make my video", icon="movie", on_click=self.make).props("size=lg")
            self.render_progress = ui.linear_progress(value=0, show_value=False).classes("w-full")
            self.render_status = ui.label("")
            self.result_box = ui.column().classes("w-full")
            self.render_progress.set_visibility(False)

    @ui.refreshable
    def video_summary(self) -> None:
        if not self.items:
            ui.label("Open a trip first, then pick what goes in the video in the Library.").classes("text-gray-500")
            return
        chosen = [it for it in self.items if it["included"] and not it["cut_out"]]
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
        items = self.trip.media_items()
        if not items:
            ui.notify("Nothing is ticked for the video yet. Pick items in the Library.", type="warning")
            return
        music = (self.music_input.value or "").strip().strip("'\"") or None
        if music and not Path(music).expanduser().is_file():
            ui.notify("The music file was not found.", type="warning")
            return
        width, height = RESOLUTIONS[self.resolution.value]
        settings = RenderSettings(
            width=width,
            height=height,
            photo_seconds=self.photo_len.value,
            max_clip_seconds=self.clip_len.value,
            transition=self.transition.value,
            transition_seconds=self.t_len.value,
            music_path=str(Path(music).expanduser()) if music else None,
            music_volume=self.music_vol.value,
        )
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
  const go = () => { v.currentTime = from; v.play().catch(() => { v.muted = true; v.play().catch(() => {}); }); };
  if (v.readyState >= 1) go(); else v.addEventListener('loadedmetadata', go, {once: true});
}
function tmStop(box) {
  const v = box.querySelector('video');
  v.pause();
  v.style.display = 'none';
}
let tmPlaying = null;
function tmPlayPart(id, start, end, from) {
  const box = document.getElementById(id);
  if (!box) return;
  if (tmPlaying && tmPlaying !== box) tmStop(tmPlaying);
  box.dataset.start = start; box.dataset.end = end;
  tmPlaying = box;
  tmPlayFrom(box, from);
}
document.addEventListener('mouseenter', (e) => {
  const box = e.target.closest && e.target.closest('.tm-hover');
  if (!box || e.target !== box) return;
  if (tmPlaying && tmPlaying !== box) tmStop(tmPlaying);
  tmPlaying = box;
  tmPlayFrom(box, parseFloat(box.dataset.start));
}, true);
document.addEventListener('mouseleave', (e) => {
  const box = e.target.closest && e.target.closest('.tm-hover');
  if (!box || e.target !== box) return;
  tmStop(box);
}, true);
</script>
"""


TAP_SCRIPT = """
<script>
// The Music tab: play the song, press the RIGHT Shift key wherever you want a cut. Pins can be dragged,
// selected and deleted; on playback every pin you pass makes a click so you hear where it is.
window.tm = window.tm || {};
tm.tap = {pins: [], sel: -1, view: 0, span: 0, until: -1, ctx: null, raf: 0, drag: -1, osc: [], srcEl: null};

function tmTapAudio() { return document.getElementById('tm-song'); }
function tmTapVisible() { const tl = document.getElementById('tm-tl'); return tl && tl.offsetParent !== null; }

// The Music tab's panel is only on the page while it is open, so nothing here assumes it exists.
function tmTapInit(pins, span) {
  const T = tm.tap;
  T.pins = (pins || []).slice().sort((a, b) => a - b);
  T.sel = -1; T.view = 0; T.span = span || 0; tmTapUnschedule();
  tmTapDraw();
}
function tmTapOnPlay() {
  const T = tm.tap, a = tmTapAudio();
  tmTapCtx(); tmTapUnschedule(); cancelAnimationFrame(T.raf); tmTapLoop(); tmTapDraw();
}
// Listen on the whole page: the line itself is rebuilt whenever the screen refreshes or the tab opens.
document.addEventListener('pointerdown', ev => { if (ev.target.closest && ev.target.closest('#tm-tl')) tmTapPointerDown(ev); });
window.addEventListener('pointermove', ev => tmTapPointerMove(ev));
window.addEventListener('pointerup', () => { if (tm.tap.drag >= 0) { tm.tap.drag = -1; tm.tap.pins.sort((a, b) => a - b); tmTapDraw(); } });
// When the Music tab opens, draw the pins.
setInterval(() => { const tl = document.getElementById('tm-tl'); if (tl && !tl.childElementCount) tmTapDraw(); }, 300);

// The song plays through the same sound path as the clicks, so both reach your ears with the same delay.
function tmTapCtx() {
  const T = tm.tap, a = tmTapAudio();
  if (!T.ctx) T.ctx = new (window.AudioContext || window.webkitAudioContext)();
  if (T.ctx.state === 'suspended') T.ctx.resume();
  if (a && T.srcEl !== a) {
    try { T.ctx.createMediaElementSource(a).connect(T.ctx.destination); T.srcEl = a; } catch (e) { console.warn(e); }
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
  const now = a.currentTime, rate = a.playbackRate || 1;
  if (T.until < 0 || T.until < now - 0.3 || T.until > now + 1) T.until = now;  // just started, or jumped
  const ahead = now + 0.25;
  if (T.ctx) T.pins.forEach(p => { if (p > T.until && p <= ahead) tmTapClick(false, T.ctx.currentTime + (p - now) / rate); });
  T.until = ahead;
  if (T.span && (now < T.view || now > T.view + T.span * 0.9)) T.view = Math.max(0, now - T.span * 0.1);
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
  let html = '';
  // Second marks along the bottom.
  const stepS = span > 120 ? 30 : span > 40 ? 10 : span > 15 ? 5 : 1;
  for (let s = Math.ceil(v0 / stepS) * stepS; s <= v0 + span; s += stepS) {
    const m = Math.floor(s / 60), sec = String(Math.round(s % 60)).padStart(2, '0');
    html += `<div style="position:absolute;left:${x(s)};bottom:0;height:10px;border-left:1px solid #94a3b8"></div>` +
            `<div style="position:absolute;left:${x(s)};bottom:10px;font-size:10px;color:#64748b;transform:translateX(-50%)">${m}:${sec}</div>`;
  }
  T.pins.forEach((p, i) => {
    if (p < v0 || p > v0 + span) return;
    const c = i === T.sel ? '#dc2626' : '#2563eb';
    html += `<div data-pin="${i}" title="${p.toFixed(2)} s" style="position:absolute;left:${x(p)};top:0;bottom:24px;width:14px;` +
            `margin-left:-7px;cursor:ew-resize;display:flex;justify-content:center"><div style="width:3px;background:${c};height:100%"></div>` +
            `<div style="position:absolute;top:-2px;width:12px;height:12px;border-radius:6px;background:${c}"></div></div>`;
  });
  if (a) html += `<div style="position:absolute;left:${x(a.currentTime)};top:0;bottom:0;width:2px;background:#f97316;pointer-events:none"></div>`;
  tl.innerHTML = html;
  const info = document.getElementById('tm-tap-info');
  if (info) {
    const d = tmTapLength(), n = T.pins.length;
    const t = a ? a.currentTime : 0;
    info.textContent = `${Math.floor(t / 60)}:${String(Math.floor(t % 60)).padStart(2, '0')} · ${n} cut${n === 1 ? '' : 's'}` +
      (d && n ? ` → ${n + 1} shots, about ${(d / (n + 1)).toFixed(1)} s each` : '') +
      (T.sel >= 0 ? ` · selected cut at ${T.pins[T.sel].toFixed(2)} s (Delete key removes it)` : '');
  }
}

function tmTapTimeAt(ev) {
  const tl = document.getElementById('tm-tl'), r = tl.getBoundingClientRect(), [v0, span] = tmTapWindow();
  return Math.max(0, Math.min(tmTapLength(), v0 + (ev.clientX - r.left) / r.width * span));
}

function tmTapPointerDown(ev) {
  const T = tm.tap, pin = ev.target.closest('[data-pin]');
  if (pin) { T.sel = T.drag = parseInt(pin.dataset.pin); ev.preventDefault(); }
  else { T.sel = -1; const a = tmTapAudio(); if (a) { a.currentTime = tmTapTimeAt(ev); tmTapUnschedule(); } }
  tmTapDraw();
}

function tmTapPointerMove(ev) {
  const T = tm.tap;
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
  } else if ((ev.key === 'Delete' || ev.key === 'Backspace') && T.sel >= 0 && !['INPUT', 'TEXTAREA'].includes(document.activeElement.tagName)) {
    T.pins.splice(T.sel, 1); T.sel = -1; tmTapUnschedule(); tmTapDraw(); ev.preventDefault();
  }
});

function tmTapPlay() { const a = tmTapAudio(); if (!a) return; tmTapCtx(); if (a.paused) a.play(); else a.pause(); }
function tmTapSeek(by) { const a = tmTapAudio(); if (!a) return; a.currentTime = Math.max(0, a.currentTime + by); tmTapUnschedule(); tmTapDraw(); }
function tmTapStart() { const a = tmTapAudio(); if (!a) return; a.currentTime = 0; tmTapUnschedule(); tm.tap.view = 0; tmTapDraw(); }
function tmTapDelete() { const T = tm.tap; if (T.sel >= 0) { T.pins.splice(T.sel, 1); T.sel = -1; tmTapUnschedule(); tmTapDraw(); } }
function tmTapClear() { tm.tap.pins = []; tm.tap.sel = -1; tmTapDraw(); }
function tmTapZoom(span) { const a = tmTapAudio(); tm.tap.span = span; tm.tap.view = a ? Math.max(0, a.currentTime - span * 0.1) : 0; tmTapDraw(); }
function tmTapGet() { return tm.tap.pins.slice().sort((a, b) => a - b); }
function tmTapSet(pins) { tm.tap.pins = pins.slice().sort((a, b) => a - b); tm.tap.sel = -1; tmTapUnschedule(); tmTapDraw(); }
</script>
"""


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
    ui.page("/")(lambda: Page().build())
    print("\nTrip Memory is running. Your browser should open by itself.")
    print("If it does not, open http://localhost:8765 . Close this window to stop the app.\n")
    ui.run(title="Trip Memory", host="127.0.0.1", port=8765, reload=False, show=True, favicon="🎬")


if __name__ in {"__main__", "__mp_main__"}:
    main()
