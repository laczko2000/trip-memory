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

1. **Choose your trip folder** with *Browse…*. The app takes a quick first look (dates, places, small previews).
2. **Choose files tab:** files are grouped into **stories** (taken close together in time and place). Shots
   that look nearly the same sit together in a blue box; the app ticks the ⭐ best one (sharpest, best
   exposed) for a new trip. Tick or untick a whole story, keep only the ⭐ of a set, or pick by hand.
   Hover over a video to play it. The ↻ button under a picture turns it 90° clockwise (click again to keep
   turning); the turn is used everywhere, including the final video. **Suggest best shots** re-applies the suggestion. Then click
   **Analyse the chosen files**. Only those files are analysed and used.
   The analysis works out:
   - **when** each was taken (sorted by day),
   - **where**: the place name from the phone's GPS, looked up offline. Photos without GPS borrow the place
     of a photo taken within 3 hours, or at least the country from the same day (marked "guessed"),
   - **what**: a suggested scene type such as Beach, City, Food or Group photo,
   - **who**: faces grouped by person.
   - **which way up**: a photo or video that is sideways or upside down but clearly shows faces is turned
     upright automatically (you'll see a note; use ↻ if it guessed wrong).

   The first time, it downloads the free AI models (about 1 GB). Each trip is analysed once and remembered;
   adding new files later only analyses the new ones.
3. **Library tab:** browse the thumbnails, filter by day, place, scene or person, correct a scene if it's wrong,
   and tick or untick what goes into the video.
4. **People tab:** for each face group, pick a name you already used from the drop-down, or type a new one
   and press Enter. Picking an existing name joins the groups. **Unknown** hides strangers. Faces seen only
   once or twice are listed separately so you can check them, then **Mark all of these as Unknown**.
5. **Storyboard tab:** change the order with the arrows, trim videos by dragging the ends of the bar, pick a
   transition for any cut, and switch day title cards ("Day 2 · Lisbon") on or off.
6. **Make video tab:** choose transition, photo length, music, then **Make my video**. The file is saved in
   *Movies → Trip Memory*.

Everything runs on your Mac. Nothing is uploaded.

## What's new

- **0.7:** rotate button in Choose files and the Library, automatic upright fix for shots with faces.
- **0.6:** Choose files grouped into stories and sets of similar shots, best-shot suggestion, hover-to-play.
- **0.5:** storyboard with your own order, trims, per-cut transitions and day title cards.
- **0.4.1:** faster video making: uses the Mac's built-in video encoder, prepares several clips at once,
  and remembers prepared clips so making the video again after a change is much quicker.
- **0.4:** choose which files of a folder to analyse and use.
- **0.3.2:** missing places are guessed from photos taken around the same time.
- **0.3.1:** name drop-down, one-click Unknown, review list for faces seen only once or twice.
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
