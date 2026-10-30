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

Everything runs on your Mac. Nothing is uploaded, unless you choose Claude for the AI director (then small
copies of the chosen pictures, and three frames of each video, are sent to Anthropic to be described).

## What's new

- **0.45:** trim, fade and pin on one line. The Music tab's separate trim slider is gone: the song's trim handles (green) now sit at the two ends of the pin line itself, with the cut-off parts greyed out. Drag the small purple dot at the top of each handle inwards to fade the music in or out (or press *Fades 1 s / 3 s / 6 s*); the purple slope shows the volume. Letting go of a handle or a dot plays it straight away (the start from there on, the end its last seconds), and Play skips the grey parts and plays the fades, so you hear exactly the track you will get. The line under it says what you keep and how much you cut. Pins in the grey parts are greyed and don't click. *Use this part as the track* makes the track with your fades.
- **0.44:** the length you write is understood. The app reads it from your own words itself (the AI is not trusted with numbers): "1:20", "80 seconds", "1 min 20", "two and a half minutes", "the whole song", "half the song". When your words could mean two lengths ("1.2 minutes": 1:12 or 1:20?), it asks. Any length works: the fixed 1, 2, 3 minute list is gone; type a length in the *Length* box (in the wish's line, or in Auto-edit) and press *Set length*.
- **0.43:** speak your idea. In *The video you want*, press *Speak your idea* and talk as long as you like; press it again and your words are written into the box. This runs on your Mac with Whisper, free and offline; the very first time it downloads its speech model (about 500 MB, once). *Make it short and clear* lets the AI tidy a long idea into a short brief that keeps every wish (and your corrections); change it if you like, then *Make this video* or *Put it in the box*. The first time, the browser asks to use the microphone: allow it.
- **0.42:** audio first. The Music tab now starts with *1. Trim the song*: drag the two ends to cut off a long intro or the ending, listen to the start and end points, and *Use this part as the track*. That part (with a short fade in and out) becomes the video's song, and pins you had already placed in it come along. *Trim again from the whole song* goes back to the full song. Then *2. Pin the cuts* on the whole track. When the video is shorter than the track (1, 2 or 2.5 minutes), it ends on your pin closest to that length, with the music fading out, so your key cuts are kept.
- **0.41:** one AI workflow. The Storyboard's *The video you want* is the one place where the AI shapes the whole video: describe it in as many words as you like, or leave it empty to be surprised. The AI offers 3 versions (they replace *Suggest storylines*, which has left the Stories tab) and cuts the first one straight away; *Use this* on another version cuts that one. A line at the top says what the video follows now (version, length, pace, look, landscape or upright) and what you set yourself: story names you typed, stories you switched on or off, and your titles are never overwritten by the AI. *Undo* takes back the last AI answer or first cut. While you follow a wish, Auto-edit has a single *Make a new cut* button and always uses the wish's length; *Stop following the wish* gives you Auto-edit's own settings back. A wish for Instagram or TikTok makes the video upright, and the Video size is now remembered per trip.
- **0.40:** a Stop button for every AI step (describing clips, planning stories, storylines and *Tell the app the video you want*). It appears while the AI works, at the top and in the AI director card. What the AI finished before you pressed it is kept (for example the clips it already described); the rest is left as it was, and *Describe clips* carries on from there next time. A stopped storyline or wish changes nothing.
- **0.39:** collage montage. In the lively parts of the song, a quick run of 3 or 4 photos from one story becomes a collage: the photos pop in one after another, on the same beats where they would have been cut, and stay on screen together (side by side, one big and two small, or 2×2; stacked in a phone video). At most one every 12 seconds. In the Storyboard a collage shows its photos, and *Show them one by one* turns it back into separate shots. It comes with *Camera effects* in Auto-edit's More settings.
- **0.38:** phone format. *Make video → Video size → Phone, upright 9:16 (Instagram, TikTok)* makes a vertical 1080×1920 video. Every shot fills the screen: wide shots show the upright slice with the people in it (the app finds the faces in each shot once, on this Mac), and shots without people show their middle. Titles are sized for the narrow screen. Also fixed: the chosen Look is now used when you make the video.
- **0.37:** looks. *Make video → Look* gives the whole video one colour mood: Natural, Warm summer, Vivid, Film (softer colours, grain and a vignette), Cool blue hour, or Black and white. Each is shown on one of your own shots, so you see what you get, and changing it later is quick because the clips aren't prepared again. The AI assistant can pick one too ("a warm, vintage film of the beach days").
- **0.36:** dynamic effects. On the song's hardest hits, an action video gets a camera shake with a flash of light (other shots keep the punch-in). The first good video of each person you named ends on a one-second freeze-frame with their name written on it, like a film introducing its cast (in calmer music, at most one every 20 seconds, never on a shot that already has a story title). Both come with *Camera effects* in Auto-edit's More settings, and each can be taken off a shot in the Storyboard. (The whip pans, zoom punches and flashes between shots were already there.)
- **0.35:** a cleaner, simpler path. The tabs now go in working order: Choose files → Stories → Music → Storyboard → Make video, with Library and People at the end. The song is chosen in one place, the Music tab; the Storyboard and Make video show it with a link there. Your pins are saved as you place them (no more *Save for later* / *These are final*) and Auto-edit always cuts on them; the automatic beat finding is only for a song without pins. In the Storyboard, *Tell the app the video you want* comes first, and Auto-edit's sliders and switches fold into *More settings*. Make video keeps only what rendering needs (photo and video lengths for a hand-made video fold away; *Find free music* is hidden). The Storyboard's reset is now *Back to story order*, and Make video's count matches the Storyboard exactly.
- **0.34:** the AI assistant. At the top of the Storyboard, *Tell the app the video you want*: type something like "2 minutes, beach and kids, energetic" (or click an example) and press *Make this video*. The AI picks the stories and their order, who and what to show more of, the pace (calm, medium or lively), the length, slow motion, camera effects and an opening title; then Auto-edit makes the first cut, with your song if you chose one. It shows what it understood, and the settings below stay yours to change. *Stop showing more of these* takes back its favourites. Works with Ollama on this Mac or with Claude; describe the clips first so it knows what is in them.
- **0.33:** Stories, Storyboard and the video now agree. Shots a story's plan leaves out stay out of the Storyboard, Auto-edit and the video (put one back with its role menu in the plan). Changing a plan reorders the Storyboard to match. Using an AI storyline starts the Storyboard again from it, story by story (Auto-edit switches to *Story by story*, so no teaser pulls shots out of their stories), and its story names replace the last storyline's names in the titles; a name you typed yourself always wins.
- **0.32:** a proper ending. The last shot fades to black, a few seconds of black follow with the end card ("Until next time" and the trip's name) fading in on it, and the music fades out gently over the ending. Choose how long in *Make video → Titles → Ending* (2 to 5 seconds, 3 by default, or no black ending).
- **0.31:** video doubles. Videos are now compared on frames from their start, middle and end (not one picture), counted from when the previous take ended, so repeated takes of the same scene land in one set. A video's doubles are other videos (a photo of the scene is kept separately). The ⭐ goes to the sharpest, steadiest, longest take, and *Keep only the ⭐ of every set* works for videos too. The first opening after updating looks at every video again once.
- **0.30:** titles. *Make video → Titles* sets an opening title over the first shot (the trip's name and dates, which you can change), a title on each story's first shot (its name, the AI's theme or your own, with place and day), and an end card; each can be switched off. In the Storyboard, *Add title* puts your own words on any shot, with a live preview: style (Clean, Bold, Handwritten, Elegant, Place and date caption) and where (middle, bottom, top, bottom left). Titles fade in and out while their shot is on screen (up to 4 seconds) and use the Mac's built-in fonts.
- **0.29:** the story director. In Stories, *Plan the stories* (or *Plan this story*) has the AI read each story's shots in time order and plan it like a short scene: a theme (it becomes the story's name unless you named it), an idea, and the shots in order as *opening*, *build-up*, *quick montage on the beat*, *highlight*, *slow moment* and *ending*, leaving out dull or repeated shots. Open a story's plan to see it; move shots earlier or later, change what a shot does, or leave it out. Auto-edit follows the plans: each story in its plan's order, montages cut quickly in the lively parts of the song, slow moments in the calm parts (slow motion or a slow zoom), highlights always used, and the story's ending closes the video. Storylines use the plans too. Planning reads text only, so it's quick even on this Mac (describe the clips first).
- **0.28:** a faster AI director. Look-alike shots are described once: the AI looks at the ⭐ of each set and its doubles get the same description (the button says how many it will look at). On this Mac, the AI stops writing as soon as the short description is done, and the settings suggest a smaller, about 3× faster model (`ollama pull moondream`). With Claude you can choose *Fast and cheap* (Haiku 5.5, a few cents a trip, the new default) or *Most detailed* (Opus 5.5); storylines always use the best model. While describing, it shows about how many minutes are left.
- **0.27:** link stories in Choose files. Every story has *Link to previous* and *Link to next* at the right: the two become one story (in Choose files, Stories and Auto-edit), and the page stays where they meet. A linked story gets *Split again* to undo it. Links are kept when the trip is analysed again or files are removed. A long story now has one header; its later parts say "continued".
- **0.26.1:** Choose files is one long list again, with no page buttons: more files appear by themselves as you scroll down. Ticking stays instant even far down a list of thousands, *Tick story* and *keep ⭐ only* no longer jump you back to the top, and a busy moment no longer makes the page reload itself.
- **0.26:** much better at finding doubles. Shots now count as look-alikes when their small details match even if the picture moved, turned or zoomed a little or someone moved, or when layout and colours match closely; within 10 minutes (not 3), and a burst stays together even with other shots in between. In a test with burst photos it kept 19 of 20 bursts together (before: 2 of 20) without mixing up different scenes. *Doubles are* (Step 1) chooses how strict: only near-identical shots, same people and setting (normal), or anything alike in the same place. *Keep only the ⭐ of every set* now only unticks doubles and leaves your other ticks alone. The first opening after updating compares all pictures once.
- **0.25.1:** in Choose files, as soon as anything is unticked a red bar stays at the top of the screen while you scroll: "N unticked · Remove them from the trip".
- **0.25:** *Step 1: sort out* at the top of Choose files, for big folders. It shows how many files (and GB) the trip has; *Show → Only doubles and triples* lists just the look-alike sets; ticking is instant; *Keep only the ⭐ of every set* unticks every double; *Remove unticked from the trip* takes them out of every tab for good (they are never analysed; the files on your Mac are not touched; *See removed* brings them back). Choose files shows 60 files a page and the Storyboard 60 clips at a time (*Show more*), and each tab is only built when you open it, so big trips stay quick. The local AI (Ollama) now looks at one smaller picture per clip and stays loaded, which is several times faster.
- **0.24.8:** the AI director shows its progress in its own card (which clip it is looking at, or that it is thinking about storylines), says when it's done, and tells you to wait if you click while it's still busy, instead of silently doing nothing.
- **0.24.7:** Music timeline: every pin shows its own time next to it (to the hundredth of a second when zoomed in); zoom buttons Whole song / 30 s / 10 s / 4 s replace the *Show* list; the line under it now says how many shots your pins make and up to when.
- **0.24.6:** the shot boxes under the Music timeline are no longer coloured orange or red for going past your clip limits, since your pins now always win; hover a box to see which pins it lies between.
- **0.24.5:** a finer Music timeline. It shows the song's waveform (the spikes are the hits). Scroll up or down on it, or pinch, to zoom in around the mouse, down to about a second across; scroll sideways to move along. Time marks get down to a tenth of a second. Click a pin and nudge it with ← → (10 ms, or 1 ms with Alt). *Show* also has 4 s and 1.5 s.
- **0.24.4:** your pins are the only cuts. No cut is added between two pins, however long the gap (your clip and photo limits don't apply there; a clip that is a bit short is slowed down to fill it), and pins can be as close as a tenth of a second. After *Make a first cut* the message says how many pins it followed, or that the song has no final pins yet.
- **0.24.3:** fixed 0.24.2 not starting (a typo in the screen code). Every update is now checked to start before it is published.
- **0.24.2:** People: faces are sorted into people straight after they are found, so a problem in a later analysis step can no longer leave the People tab empty. When it is empty, the tab now says why (not checked yet, found but not sorted, the face finder could not start, or really no faces) with the button to fix it, plus *Look for faces again*.
- **0.24.1:** taps are respected. Auto-edit never joins two of your taps into one shot any more (it did when there were more taps than shots) and never adds cuts between taps to fit more shots. *Snap to the music* now learns your usual tap delay (reaction time, Bluetooth headphones) from all your taps together and snaps onto the song's beats. Taps and the replay clicks are timed on the sound card's clock, so the clicks land exactly on the pins.
- **0.24:** the **AI director**, at the top of the Stories tab. *Describe clips* has an AI look at each chosen file and write what it shows (📝 in the Library and Storyboard) plus a 1 to 5 ★ "worth remembering" score that Auto-edit uses to prefer the best moments. *Suggest storylines* gives 2 to 4 ways to tell the trip (for example best moments first, day by day, the people), each with the story order and better story names; *Use this* puts the stories in that order and switches off the ones it leaves out. Run it free on your Mac with Ollama (ollama.com, then `ollama pull gemma3:4b`), or with Claude (paid, about a cent per clip, needs an API key from console.anthropic.com).
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
