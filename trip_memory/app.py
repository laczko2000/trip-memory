"""The Trip Memory screen. Runs locally and opens in your web browser."""

from __future__ import annotations

import subprocess
import sys
import traceback
from datetime import datetime
from pathlib import Path

from nicegui import app, run, ui

from .media import MediaItem, scan_folder
from .render import TRANSITIONS, RenderSettings, render

OUTPUT_DIR = Path.home() / "Movies" / "Trip Memory"
RESOLUTIONS = {"Full HD (1080p)": (1920, 1080), "HD (720p, faster)": (1280, 720)}


class State:
    folder: str = ""
    items: list[MediaItem] = []
    problems: list[str] = []
    progress: float = 0.0
    message: str = ""
    busy: bool = False
    result: Path | None = None
    error: str = ""


state = State()


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


def fmt_date(dt: datetime) -> str:
    return dt.astimezone().strftime("%d %b %Y, %H:%M")


def fmt_length(item: MediaItem) -> str:
    if item.kind == "photo":
        return "photo"
    m, s = divmod(round(item.duration), 60)
    return f"{m}:{s:02d}"


def summary(items: list[MediaItem]) -> str:
    videos = sum(i.kind == "video" for i in items)
    photos = len(items) - videos
    if not items:
        return "No videos or photos found in this folder."
    first, last = items[0].taken_at.astimezone(), items[-1].taken_at.astimezone()
    span = first.strftime("%d %b %Y") if first.date() == last.date() else f"{first:%d %b} to {last:%d %b %Y}"
    return f"Found {videos} videos and {photos} photos, taken {span}."


def build_page() -> None:
    ui.colors(primary="#2563eb")
    with ui.column().classes("w-full max-w-4xl mx-auto p-4 gap-4"):
        ui.label("Trip Memory").classes("text-3xl font-bold")
        ui.label("Turn a folder of holiday videos and photos into one memory video.").classes("text-gray-500")

        # 1. Folder
        with ui.card().classes("w-full"):
            ui.label("1. Choose your trip folder").classes("text-lg font-semibold")
            with ui.row().classes("w-full items-center"):
                folder_input = ui.input("Folder", placeholder="/Users/you/Pictures/Lisbon 2025").classes("flex-grow")
                if sys.platform == "darwin":
                    async def browse_folder() -> None:
                        path = await run.io_bound(choose_with_finder, "folder")
                        if path:
                            folder_input.value = path
                            await scan()

                    ui.button("Browse…", on_click=browse_folder).props("outline")
                scan_button = ui.button("Find videos and photos")
            summary_label = ui.label("")
            problems_label = ui.label("").classes("text-orange-600 text-sm whitespace-pre-line")
            table = ui.table(
                columns=[
                    {"name": "n", "label": "#", "field": "n", "align": "left"},
                    {"name": "kind", "label": "Type", "field": "kind", "align": "left"},
                    {"name": "name", "label": "File", "field": "name", "align": "left"},
                    {"name": "date", "label": "Taken", "field": "date", "align": "left"},
                    {"name": "length", "label": "Length", "field": "length", "align": "left"},
                ],
                rows=[],
                pagination=10,
            ).classes("w-full")
            table.set_visibility(False)

        async def scan() -> None:
            folder = (folder_input.value or "").strip().strip("'\"")
            if not folder or not Path(folder).expanduser().is_dir():
                ui.notify("Please choose a folder that exists.", type="warning")
                return
            state.folder = str(Path(folder).expanduser())
            summary_label.text = "Looking through the folder…"
            scan_button.disable()
            try:
                state.items, state.problems = await run.io_bound(scan_folder, state.folder)
            finally:
                scan_button.enable()
            summary_label.text = summary(state.items)
            problems_label.text = "\n".join(state.problems[:10])
            table.rows = [
                {"n": i + 1, "kind": item.kind, "name": item.name, "date": fmt_date(item.taken_at), "length": fmt_length(item)}
                for i, item in enumerate(state.items)
            ]
            table.set_visibility(bool(state.items))

        scan_button.on_click(scan)

        # 2. Style
        with ui.card().classes("w-full"):
            ui.label("2. Choose the style").classes("text-lg font-semibold")
            with ui.grid(columns=2).classes("w-full gap-4"):
                transition = ui.select(list(TRANSITIONS), value="Fade", label="Transition")
                resolution = ui.select(list(RESOLUTIONS), value="Full HD (1080p)", label="Video size")
                with ui.column():
                    ui.label("Transition length (seconds)").classes("text-sm text-gray-500")
                    t_len = ui.slider(min=0.3, max=2.0, step=0.1, value=0.8).props("label-always").classes("mt-8")
                with ui.column():
                    ui.label("Each photo shows for (seconds)").classes("text-sm text-gray-500")
                    photo_len = ui.slider(min=2, max=8, step=0.5, value=4).props("label-always").classes("mt-8")
                with ui.column():
                    ui.label("Longest part of each video (seconds, 0 = whole video)").classes("text-sm text-gray-500")
                    clip_len = ui.slider(min=0, max=60, step=1, value=10).props("label-always").classes("mt-8")
                keep_sound = ui.switch("Keep the sound from the videos", value=True)
            with ui.row().classes("w-full items-center"):
                music_input = ui.input("Music (optional)", placeholder="/Users/you/Music/song.mp3").classes("flex-grow")
                if sys.platform == "darwin":
                    async def browse_music() -> None:
                        path = await run.io_bound(choose_with_finder, "music")
                        if path:
                            music_input.value = path

                    ui.button("Browse…", on_click=browse_music).props("outline")
            with ui.column().classes("w-full"):
                ui.label("Music volume").classes("text-sm text-gray-500")
                music_vol = ui.slider(min=0.1, max=1.0, step=0.05, value=0.5).props("label-always").classes("mt-8")

        # 3. Make it
        with ui.card().classes("w-full"):
            ui.label("3. Make the video").classes("text-lg font-semibold")
            name_input = ui.input("Video name", placeholder="My trip").classes("w-full")
            make_button = ui.button("Make my video", icon="movie").props("size=lg")
            progress = ui.linear_progress(value=0, show_value=False).classes("w-full")
            status = ui.label("")
            result_box = ui.column().classes("w-full")
            progress.set_visibility(False)

        def refresh() -> None:
            progress.value = state.progress
            status.text = state.error or state.message

        ui.timer(0.5, refresh)

        async def make() -> None:
            if state.busy:
                return
            if not state.items:
                ui.notify("First choose a folder and find the videos and photos.", type="warning")
                return
            music = (music_input.value or "").strip().strip("'\"") or None
            if music and not Path(music).expanduser().is_file():
                ui.notify("The music file was not found.", type="warning")
                return
            width, height = RESOLUTIONS[resolution.value]
            settings = RenderSettings(
                width=width,
                height=height,
                photo_seconds=photo_len.value,
                max_clip_seconds=clip_len.value,
                transition=transition.value,
                transition_seconds=t_len.value,
                keep_clip_sound=keep_sound.value,
                music_path=str(Path(music).expanduser()) if music else None,
                music_volume=music_vol.value,
            )
            title = (name_input.value or "").strip() or Path(state.folder).name or "My trip"
            safe = "".join(c for c in title if c.isalnum() or c in " -_").strip() or "My trip"
            out = OUTPUT_DIR / f"{safe} {datetime.now():%Y-%m-%d %H%M}.mp4"

            def on_progress(fraction: float, message: str) -> None:
                state.progress, state.message = fraction, message

            state.busy, state.error, state.progress = True, "", 0.0
            make_button.disable()
            progress.set_visibility(True)
            result_box.clear()
            try:
                state.result = await run.io_bound(render, state.items, settings, out, on_progress)
                state.message = f"Saved to {state.result}"
                with result_box:
                    ui.video(f"/videos/{state.result.name}").classes("w-full")
                    ui.button("Show in Finder", icon="folder", on_click=lambda: reveal(state.result)).props("outline")
            except Exception as exc:  # show any problem on screen instead of failing silently
                traceback.print_exc()
                state.error = f"Something went wrong: {exc}"
            finally:
                state.busy = False
                make_button.enable()

        make_button.on_click(make)


def reveal(path: Path | None) -> None:
    if path and sys.platform == "darwin":
        subprocess.run(["open", "-R", str(path)])


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    app.add_media_files("/videos", OUTPUT_DIR)
    ui.page("/")(build_page)
    print("\nTrip Memory is running. Your browser should open by itself.")
    print("If it does not, open http://localhost:8765 . Close this window to stop the app.\n")
    ui.run(title="Trip Memory", host="127.0.0.1", port=8765, reload=False, show=True, favicon="🎬")


if __name__ in {"__main__", "__mp_main__"}:
    main()
