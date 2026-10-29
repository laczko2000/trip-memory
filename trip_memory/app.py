"""The Trip Memory screen. Runs locally and opens in your web browser."""

from __future__ import annotations

import json
import subprocess
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from urllib.parse import quote

from nicegui import app, run, ui

from .catalog import Trip, app_dir
from .media import MediaItem
from .render import TRANSITIONS, RenderSettings, clip_range, render
from .scenes import SCENES

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
        self.chapters = bool(load_settings().get("chapters", True))
        self.dirty = False
        self.last_refresh = 0.0
        self.filters = {"day": ALL, "place": ALL, "scene": ALL, "person": ALL, "show": ALL}
        self.page = 1
        self.rendering = {"busy": False, "progress": 0.0, "message": "", "result": None}

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
        return {
            "day": day_opts,
            "place": {ALL: "All places", **{p: p for p in places}},
            "scene": {ALL: "All scenes", **{s: s for s in scenes}},
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
                story_tab = ui.tab("Storyboard", icon="view_timeline")
                video_tab = ui.tab("Make video", icon="movie")
            self.library_tab = library_tab
            with ui.tab_panels(tabs, value=self.choose_tab).classes("w-full") as self.panels:
                with ui.tab_panel(self.choose_tab):
                    self.choose_view()
                with ui.tab_panel(library_tab):
                    self.build_library()
                with ui.tab_panel(people_tab):
                    self.people_view()
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
            thumb = data_url(it["thumb"])
            if it["kind"] == "video" and self.media_route:
                start = (it["duration"] or 0) * 0.3
                src = f"{self.media_route}/{quote(str(Path(it['path']).relative_to(self.trip.folder)))}#t={start:.1f}"
                ui.html(
                    f'<div class="tm-hover" style="position:relative;width:144px;height:96px">'
                    f'<img src="{thumb}" style="width:144px;height:96px;object-fit:cover;border-radius:4px">'
                    f'<video data-src="{src}" muted loop playsinline preload="none" '
                    f'style="display:none;position:absolute;inset:0;width:144px;height:96px;object-fit:cover;border-radius:4px"></video>'
                    f'<span style="position:absolute;left:4px;bottom:2px;color:white;font-size:11px;text-shadow:0 0 3px black">'
                    f'▶ {fmt_length(it["duration"])}</span></div>',
                    sanitize=False,
                )
            elif thumb:
                ui.image(thumb).classes("w-36 h-24 object-cover rounded")
            with ui.row().classes("items-center gap-0 no-wrap"):
                ui.checkbox(
                    value=bool(it["chosen"]),
                    on_change=lambda e, path=it["path"]: self.choose_paths([path], e.value),
                ).props("dense size=xs")
                star = "⭐ " if in_set and it["best"] else ""
                ui.label(f"{star}{datetime.fromisoformat(it['taken_at']).astimezone():%H:%M:%S}").classes("text-xs text-gray-600")

    def suggest(self) -> None:
        self.trip.suggest_selection()
        ui.notify("Ticked the best shot of each set of similar shots.")
        self.refresh_all()

    def keep_best(self, members: list[dict]) -> None:
        self.trip.set_chosen([m["path"] for m in members if not m["best"]], False)
        self.trip.set_chosen([m["path"] for m in members if m["best"]], True)
        self.refresh_all()

    def choose_paths(self, paths: list[str], chosen: bool) -> None:
        self.trip.set_chosen(paths, chosen)
        self.refresh_all()

    def set_choose_filter(self, key: str, value: str) -> None:
        self.choose_filters[key] = value
        self.choose_page = 1
        self.story_page = 1
        self.media_route: str | None = None
        self.chapters = bool(load_settings().get("chapters", True))
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
            for key, label in (("day", "Day"), ("place", "Place"), ("scene", "Scene"), ("person", "Person"), ("show", "Show")):
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
        with ui.card().tight().classes("w-full" + ("" if it["included"] else " opacity-50")):
            if it["thumb"]:
                ui.image(data_url(it["thumb"])).classes("w-full thumb")
            else:
                ui.label("Analysing…").classes("w-full thumb flex items-center justify-center bg-gray-100 text-gray-400")
            with ui.column().classes("p-2 gap-1 w-full"):
                icon = "🎬 " + fmt_length(it["duration"]) if it["kind"] == "video" else "📷"
                ui.label(f"{icon} · {fmt_time(it['taken_at'])}").classes("text-xs text-gray-600")
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
                if it["people"]:
                    with ui.row().classes("gap-1"):
                        for name in it["people"]:
                            ui.badge(name).props("outline")
                ui.checkbox(
                    "In the video",
                    value=bool(it["included"]),
                    on_change=lambda e, path=it["path"]: self.toggle(path, e.value),
                ).props("dense")

    def toggle(self, path: str, included: bool) -> None:
        self.trip.set_included([path], included)
        for it in self.items:
            if it["path"] == path:
                it["included"] = int(included)
        self.video_summary.refresh()
        self.storyboard_view.refresh()

    # ---------- People ----------

    @ui.refreshable
    def people_view(self) -> None:
        if not self.trip:
            ui.label("Open a trip first.").classes("text-gray-500")
            return
        people = self.trip.people()
        if not people:
            ui.label("No faces found yet.").classes("text-gray-500")
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
            keep_clip_sound=self.keep_sound.value,
            music_volume=self.music_vol.value,
        )

    @ui.refreshable
    def storyboard_view(self) -> None:
        if not self.trip:
            ui.label("Open a trip first.").classes("text-gray-500")
            return
        board = self.trip.storyboard()
        if not board:
            ui.label("Nothing is in the video yet. Tick files in the Library.").classes("text-gray-500")
            return
        with ui.row().classes("w-full items-center gap-4"):
            ui.label(f"{len(board)} clips in this order").classes("text-lg font-semibold")
            ui.switch("Day title cards (\"Day 2 · Lisbon\")", value=self.chapters, on_change=lambda e: self.set_chapters(e.value))
            ui.button("Back to date order", icon="restart_alt", on_click=self.reset_order).props("flat dense")
        ui.label(
            "Use the arrows to change the order. For videos, drag the two ends of the bar to choose which part to keep. "
            "The transition setting is for the cut after that clip."
        ).classes("text-sm text-gray-600")
        pages = max(1, -(-len(board) // STORY_PAGE_SIZE))
        self.story_page = min(self.story_page, pages)
        first = (self.story_page - 1) * STORY_PAGE_SIZE
        previous_day = board[first - 1]["day"] if first else None
        for n, it in enumerate(board[first : first + STORY_PAGE_SIZE], first + 1):
            if self.chapters and it["day"] != previous_day:
                ui.label(f"🎬 Day title: {datetime.fromisoformat(it['day']):%a %d %b}").classes("text-sm text-blue-700 mt-2")
            previous_day = it["day"]
            self.story_row(n, it, last=n == len(board))
        if pages > 1:
            ui.pagination(1, pages, direction_links=True, value=self.story_page, on_change=lambda e: self.go_to_story_page(e.value))

    def story_row(self, n: int, it: dict, last: bool) -> None:
        with ui.card().classes("w-full p-2"):
            with ui.row().classes("w-full items-center no-wrap gap-3"):
                with ui.column().classes("gap-0 items-center"):
                    ui.button(icon="arrow_upward", on_click=lambda p=it["path"]: self.move(p, -1)).props("flat dense round size=sm")
                    ui.label(str(n)).classes("text-xs text-gray-500")
                    ui.button(icon="arrow_downward", on_click=lambda p=it["path"]: self.move(p, 1)).props("flat dense round size=sm")
                if it["thumb"]:
                    ui.image(data_url(it["thumb"])).classes("w-32 h-20 object-cover rounded")
                with ui.column().classes("flex-grow gap-1"):
                    place = ", ".join(x for x in (it["place"], it["country"]) if x)
                    icon = "🎬" if it["kind"] == "video" else "📷"
                    ui.label(f"{icon} {fmt_time(it['taken_at'])}" + (f" · {place}" if place else "")).classes("text-sm")
                    if it["kind"] == "video":
                        self.trim_control(it)
                    else:
                        ui.label("Photo").classes("text-xs text-gray-500")
                with ui.column().classes("gap-1 w-56"):
                    if not last:
                        ui.select(
                            [DEFAULT_TRANSITION, *TRANSITIONS],
                            value=it["transition"] or DEFAULT_TRANSITION,
                            label="Transition after",
                            on_change=lambda e, p=it["path"]: self.trip.set_transition(p, None if e.value == DEFAULT_TRANSITION else e.value),
                        ).props("dense options-dense").classes("w-full")
                    ui.button("Leave out", icon="remove_circle_outline", on_click=lambda p=it["path"]: self.leave_out(p)).props("flat dense size=sm")

    def trim_control(self, it: dict) -> None:
        item = MediaItem(path=Path(it["path"]), kind="video", taken_at=datetime.now(), duration=it["duration"],
                         trim_start=it["trim_start"], trim_end=it["trim_end"])
        start, end = clip_range(item, self.settings_now())
        label = ui.label().classes("text-xs text-gray-600")

        def show(lo: float, hi: float) -> None:
            label.text = f"Keeping {fmt_length(lo)} to {fmt_length(hi)} ({hi - lo:.1f} s of {fmt_length(it['duration'])})"

        show(start, end)
        ui.range(
            min=0, max=round(it["duration"], 1), step=0.1, value={"min": round(start, 1), "max": round(end, 1)},
            on_change=lambda e, p=it["path"]: (show(e.value["min"], e.value["max"]), self.trip.set_trim(p, e.value["min"], e.value["max"])),
        ).props("dense").classes("w-full")

    def set_chapters(self, on: bool) -> None:
        self.chapters = on
        save_settings(chapters=on)
        self.storyboard_view.refresh()

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
                self.keep_sound = ui.switch("Keep the sound from the videos", value=True)
            with ui.row().classes("w-full items-center"):
                self.music_input = ui.input("Music (optional)", placeholder="/Users/you/Music/song.mp3").classes("flex-grow")
                if sys.platform == "darwin":
                    ui.button("Browse…", on_click=self.browse_music).props("outline")
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
        chosen = [it for it in self.items if it["included"]]
        videos = sum(it["kind"] == "video" for it in chosen)
        ui.label(
            f"The video will use {len(chosen)} items ({videos} videos, {len(chosen) - videos} photos) in the Storyboard order. "
            "Change the selection in the Library and the order in the Storyboard."
        ).classes("text-gray-700")

    async def browse_music(self) -> None:
        path = await run.io_bound(choose_with_finder, "music")
        if path:
            self.music_input.value = path

    async def make(self) -> None:
        r = self.rendering
        if r["busy"] or not self.trip:
            return
        items = self.trip.media_items(chapters=self.chapters)
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
            keep_clip_sound=self.keep_sound.value,
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
document.addEventListener('mouseenter', (e) => {
  const box = e.target.closest && e.target.closest('.tm-hover');
  if (!box || e.target !== box) return;
  const v = box.querySelector('video');
  if (!v.src) v.src = v.dataset.src;
  v.style.display = 'block';
  v.play().catch(() => {});
}, true);
document.addEventListener('mouseleave', (e) => {
  const box = e.target.closest && e.target.closest('.tm-hover');
  if (!box || e.target !== box) return;
  const v = box.querySelector('video');
  v.pause();
  v.style.display = 'none';
}, true);
</script>
"""


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
    ui.page("/")(lambda: Page().build())
    print("\nTrip Memory is running. Your browser should open by itself.")
    print("If it does not, open http://localhost:8765 . Close this window to stop the app.\n")
    ui.run(title="Trip Memory", host="127.0.0.1", port=8765, reload=False, show=True, favicon="🎬")


if __name__ in {"__main__", "__mp_main__"}:
    main()
