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
   Hover over a video to play it. The ↺ and ↻ buttons under a picture turn it a quarter turn anti-clockwise or
   clockwise; the turn is used everywhere, including the final video. **Suggest best shots** re-applies the suggestion. Then click
   **Analyse the chosen files**. Only those files are analysed and used.
   The analysis works out:
   - **when** each was taken (sorted by day),
   - **where**: the place name from the phone's GPS, looked up offline. Photos without GPS borrow the place
     of a photo taken within 3 hours, or at least the country from the same day (marked "guessed"),
   - **what**: a suggested scene type such as Beach, City, Food or Group photo,
   - **who**: faces grouped by person, and how many people (solo, couple, group).
   - **how it moves** (videos): *Still*, *Calm*, *Lively* or *Fast action*, plus *Camera moving* and *Shaky*.
   - **tags**: several per file, for the kind of place (beach, old town, underwater…), the activity (walking,
     snorkelling, eating & drinking…) and the light (daylight, sunset, night). Videos are checked every few
     seconds, so a clip that starts on the beach and ends in the water gets both.
   - **look-alike groups**: everything that looks alike across the whole trip, e.g. all the snorkelling.
   - **which way up**: a photo or video that is sideways or upside down but clearly shows faces is turned
     upright automatically (you'll see a note; use ↻ if it guessed wrong).

   The first time, it downloads the free AI models (about 1 GB). Each trip is analysed once and remembered;
   adding new files later only analyses the new ones.
3. **Library tab:** browse the thumbnails, filter by day, place, scene, tag, look-alike group or person, correct a
   scene or the tags if they're wrong, click 🔗 to see everything that looks alike,
   and tick or untick what goes into the video.
4. **People tab:** for each face group, pick a name you already used from the drop-down, or type a new one
   and press Enter. Picking an existing name joins the groups. **Unknown** hides strangers. Faces seen only
   once or twice are listed separately so you can check them, then **Mark all of these as Unknown**.
5. **Storyboard tab:** **✨ Auto-edit** makes a first cut where the music leads: (1) choose the song, and the
   app studies it (beats, bars, calm and lively parts, where it changes) and shows a music map; (2) set your
   limits: longest video clip, longest photo, shortest shot; (3) pick a length and click *Make a first cut*.
   Cuts land on the beat: quick shots in lively parts, longer ones in calm parts, a cut at every change in the
   music. Action clips go to lively music, calm and cinematic clips and photos to calm music, in the order they
   were taken, each at its best moment. Then fine-tune it: change the order with the arrows, trim videos by dragging the ends of the bar (the preview
   plays only the part you keep; hover over it to watch again), and pick a
   transition for any cut.
6. **Make video tab:** choose transition, photo length, music (your own file, or **Find free music** to search
   free Creative Commons songs online and download one), then **Make my video**. The file is saved in
   *Movies → Trip Memory*.

Everything runs on your Mac. Nothing is uploaded.

## What's new

- **0.23:** camera-style effects. New *Zoom through* (the shot rushes in, the next one lands zoomed and settles) and *Zoom out reveal* transitions, suggested at story changes, where the music lifts and on big hits. Inside shots: punch-in zooms on big hits and speed ramps (fast through the middle) in lively music, a slow push-in on still shots in calm music. Slow motion now blends frames so it no longer stutters. Each effect shows its reason in the Storyboard with a Remove button; switch them all off with *Camera effects* in Auto-edit.
- **0.22:** smarter shot picking. Every video is scored five times a second for sharpness, steadiness, camera swing, brightness jumps, movement and sound. Auto-edit never uses the moments when the phone is being raised or lowered, blurred moments or wild swings, picks the best steady stretch of each clip, and prefers clips that have one. (The first Auto-edit after updating measures your videos again, which takes a little longer once.)
- **0.21.4:** previews made before a file was turned are made again when the trip opens, so the Library shows each file the way it plays and goes into the video. Turned files show a small ↻ 90° label.
- **0.21.3:** fixed finished videos stopping half way when the new quick transitions were used: every transition now starts and ends exactly on a frame.
- **0.21.2:** if a finished video ever comes out shorter than planned, the app now tries again by itself (on the main processor, then with every clip re-made in one plain format) and saves a report next to the video.
- **0.21.1:** the Storyboard is one long list you scroll through, with no pages.
- **0.21:** speed effects. Auto-edit plays calm, cinematic moments in calm music in slow motion (0.5× for high-frame-rate videos such as iPhone slo-mo, 0.75× otherwise, so it stays smooth), and speeds slow shots up 1.5× in lively music, with the reason shown. Change any clip's speed in the Storyboard (0.5× to 2×), or switch *Speed effects* off in Auto-edit.
- **0.20:** *Analyse* now reads how the camera moves in each video: Static camera, Pan left/right, Tilt up/down, Zoom in/out, Whip pan, Turn around, Rolling, Fast camera. They show as tags (filter by them in the Library). Auto-edit uses them: a whip pan or fast camera counts as action, a pan carries on into the next shot with a whip in the same direction, and a zoom in leads into a Zoom punch. Click *Analyse* once so existing videos get read.
- **0.19:** a *Storyline* choice in Auto-edit. *Teaser, then the stories* (the default) opens with a quick teaser of the most exciting shot from each of the busiest stories. Every story starts with a wide, calm view, look-alike shots never come back to back, and the video ends on a calm shot. Also *Story by story* (no teaser) and *Plain date order*.
- **0.18:** Auto-edit now suggests a transition for every cut, with the reason under it (for example "Next story: Deer", "The music lifts here", "A fast camera move: a whip pan"). New quick effects: Flash, Whip left/right/up, Zoom punch, Blur, Pixelate, plus Dip to white, Smooth wipe and Radial. Change any of them in the Storyboard. Also fixed: ticking *In the video* in the Library now adds the file straight away, in time order, and counts it in the video length.
- **0.17:** a new *Music* tab: play your song and tap the right Shift key wherever you want a cut. Play it back to hear a click at every pin, drag pins to move them, click one and press delete to remove it. *Snap to the music* moves each pin onto the exact hit, *These are final* makes Auto-edit cut exactly there (Cut on: My taps). After your last pin the app cuts the rest by itself.
- **0.16:** photos glide smoothly instead of flickering, and move more: each one zooms in, zooms out or pans sideways (tall photos only zoom), so a row of photos feels varied.
- **0.15:** the app now hears which beats hit hardest (drums, bass, cymbals), shown as red marks over the song. A new *Cut on* choice in Auto-edit: *Strong beats* (the default), *Bass hits*, *Big moments only* (few, long shots) or *Every few beats* (the old steady count).
- **0.14:** a new *Stories* tab. Each story is named by what it shows, where and when (for example "Fish & aquarium · Lisbon · Sat 5 Jul, morning"); rename it, move it up or down, or switch it off. Auto-edit and the Storyboard tell the stories in that order. Click *Analyse* once more so existing trips get the new names.
- **0.13.3:** Auto-edit uses only your selection (ticked in Choose files and "In the video" in the Library) and never unticks it; files that don't fit a cut stay selected for next time.
- **0.13.2:** fixed the finished video stopping after the first clip on the Mac.
- **0.13.1:** Auto-edit uses every file you ticked (it used to keep only one of each set of look-alike shots, which could leave very few), and stretches the shots further when there are few.
- **0.13:** videos are silent by default so the music plays alone; tick *Clip sound* on a clip in the Storyboard to bring its sound back, with a volume from 0 to 200%.
- **0.12.1:** no more automatic day title cards (titles may come back later, placed where you choose).
- **0.12:** music-led Auto-edit: song first with a music map, your limits for clip, photo and shot length, quick cuts in lively parts and long ones in calm parts, action clips matched to lively music. New *Cinematic* tag.
- **0.11.3:** after you download a ccMixter song on its page, the app finds it in Downloads and selects it by itself.
- **0.11.2:** when a music site only allows downloads from its own page (ccMixter), the app opens that page for you and still remembers the song's credit.
- **0.11.1:** free music search also asks ccMixter and waits longer, so a slow library no longer stops it.
- **0.11:** Find free music: search and download Creative Commons songs (via Openverse); a credits file is saved next to the video.
- **0.10:** ✨ Auto-edit in the Storyboard: a first cut at the length you want, best moments, cuts on the music's beat.
- **0.9.1:** Storyboard previews play with sound.
- **0.9:** movement tags for videos, many tags per file (place, activity, light, people), look-alike groups. Click **Analyse** again on a trip you already did to add them; people are not redone.
- **0.8.1:** fixed hover-to-play for videos you have turned.
- **0.8:** hover-to-play in the Library and Storyboard too. In the Storyboard the preview plays only the part you keep, and moving the trim bar plays the new part straight away.
- **0.7.1:** turn buttons both ways, the whole picture is shown in previews, and a message confirms each turn.
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
  `scenes.py` (CLIP via fastembed), `tags.py` (CLIP tag groups, look-alike groups), `motion.py` (optical-flow movement), `beats.py` (tempo and beat grid), `songmap.py` (beats, bars, energy, sections), `autoedit.py` (music-led first cut), `music.py` (Openverse music search), `render.py` builds the video with FFmpeg, `app.py` is the NiceGUI screen.
- Face models are InsightFace's, licensed for non-commercial use only.
- Run with `uv run trip-memory`. Tests: `uv run pytest`.
- FFmpeg comes from the system if installed, otherwise the `static-ffmpeg` package downloads a free build once.
