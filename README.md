# Trip Memory

Turn a folder of holiday videos and photos into one memory video, on your own Mac.
Nothing is uploaded anywhere.

## Start the app

1. Double-click **Start Trip Memory.command**.
   - The very first time, macOS may say it can't check the file. Right-click it, choose **Open**, then **Open** again.
   - The first start downloads what the app needs (a few minutes). Later starts take seconds.
2. Your browser opens the Trip Memory page. If not, go to <http://localhost:8765>.
3. To stop the app, close the Terminal window.

## Make a video

1. **Choose your trip folder** with *Browse…*. The app lists every video and photo, sorted by when it was taken.
2. **Choose the style:** transition, how long photos show, how much of each video to keep, and optional music.
3. **Make my video.** When it's done you can watch it right there. The file is saved in
   *Movies → Trip Memory*.

## What it does today (milestone 2)

- Reads videos (MOV, MP4 and more) and photos (JPG, HEIC, PNG and more), including subfolders.
- Sorts everything by the date and time stored in the files.
- Vertical videos and photos get a soft blurred background instead of black bars.
- Photos get a slow zoom so they don't look static.
- iPhone HDR videos are converted so colours don't look washed out.
- Joins everything with the transition you pick, fades in and out, and mixes in music.

## For the engineer

- Python engine in `trip_memory/` (`media.py` finds and dates files, `render.py` builds the video with FFmpeg,
  `app.py` is the NiceGUI screen).
- Run with `uv run trip-memory`. Tests: `uv run pytest`.
- FFmpeg comes from the system if installed, otherwise the `static-ffmpeg` package downloads a free build once.
