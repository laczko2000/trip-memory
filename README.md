# Trip Memory

Turn a folder of holiday videos and photos into one memory video, on your own Mac.
Nothing is uploaded anywhere.

## Start the app

1. Double-click **Start Trip Memory.command**.
   - The very first time, macOS may say it can't check the file. Right-click it, choose **Open**, then **Open** again.
   - The first start downloads what the app needs (a few minutes). Later starts take seconds.
2. Your browser opens the Trip Memory page. If not, go to <http://localhost:8765>.
3. To stop the app, close the Terminal window.

## Use it

1. **Choose your trip folder** with *Browse…*. The app looks through every video and photo and works out:
   - **when** each was taken (sorted by day),
   - **where**: the place name from the phone's GPS, looked up offline,
   - **what**: a suggested scene type such as Beach, City, Food or Group photo,
   - **who**: faces grouped by person.

   The first time, it downloads the free AI models (about 1 GB). Each trip is analysed once and remembered;
   adding new files later only analyses the new ones.
2. **Library tab:** browse the thumbnails, filter by day, place, scene or person, correct a scene if it's wrong,
   and tick or untick what goes into the video.
3. **People tab:** type a name under each face group and press Enter. Give two groups the same name to merge
   them; use the crossed-out eye to hide a group.
4. **Make video tab:** choose transition, photo length, music, then **Make my video**. The file is saved in
   *Movies → Trip Memory*.

Everything runs on your Mac. Nothing is uploaded.

## What's new

- **0.3 (milestone 3):** trip catalogue with places, scene suggestions, people, filters and picking.
- **0.2 (milestone 2):** basic stitcher: date order, blurred background for vertical shots, slow zoom on photos,
  HDR fix, transitions, music.

## For the engineer

- Python engine in `trip_memory/`: `media.py` finds and dates files, `catalog.py` keeps the per-trip SQLite
  catalogue (in ~/Library/Application Support/TripMemory), `faces.py` (InsightFace buffalo_l via onnxruntime),
  `scenes.py` (CLIP via fastembed), `render.py` builds the video with FFmpeg, `app.py` is the NiceGUI screen.
- Face models are InsightFace's, licensed for non-commercial use only.
- Run with `uv run trip-memory`. Tests: `uv run pytest`.
- FFmpeg comes from the system if installed, otherwise the `static-ffmpeg` package downloads a free build once.
