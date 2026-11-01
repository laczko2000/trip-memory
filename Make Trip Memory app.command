#!/bin/bash
# Run this once: it puts a Trip Memory app (with its own icon) in your Applications folder.
# Click it to start Trip Memory: it always starts the newest copy you downloaded into Downloads.
cd "$(dirname "$0")" || exit 1
APP="$HOME/Applications/Trip Memory.app"
mkdir -p "$APP/Contents/MacOS" "$APP/Contents/Resources"

cat > "$APP/Contents/MacOS/Trip Memory" <<'LAUNCH'
#!/bin/bash
# Start the newest Trip Memory in Downloads, in a Terminal window (close that window to stop the app).
NEWEST="$(ls -td "$HOME"/Downloads/trip-memory-main*/ 2>/dev/null | head -1)"
if [ -z "$NEWEST" ]; then
  osascript -e 'display alert "Trip Memory" message "No Trip Memory found in Downloads. Download the ZIP from GitHub (Code, Download ZIP), then click this again."'
  exit 0
fi
NEWEST="${NEWEST%/}"
osascript - "$NEWEST" <<'OSA'
on run argv
  set folder to item 1 of argv
  tell application "Terminal"
    activate
    do script "cd " & quoted form of folder & " && bash 'Start Trip Memory.command'"
  end tell
end run
OSA
LAUNCH
chmod +x "$APP/Contents/MacOS/Trip Memory"

cat > "$APP/Contents/Info.plist" <<'PLIST'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>CFBundleName</key><string>Trip Memory</string>
  <key>CFBundleDisplayName</key><string>Trip Memory</string>
  <key>CFBundleIdentifier</key><string>local.tripmemory.launcher</string>
  <key>CFBundleExecutable</key><string>Trip Memory</string>
  <key>CFBundleIconFile</key><string>icon</string>
  <key>CFBundlePackageType</key><string>APPL</string>
  <key>CFBundleVersion</key><string>1</string>
</dict>
</plist>
PLIST

# The icon (made from assets/icon.png with the Mac's own tools).
if [ -f assets/icon.png ]; then
  SET="$(mktemp -d)/icon.iconset"
  mkdir -p "$SET"
  for n in 16 32 128 256 512; do
    sips -z $n $n assets/icon.png --out "$SET/icon_${n}x${n}.png" >/dev/null
    sips -z $((n * 2)) $((n * 2)) assets/icon.png --out "$SET/icon_${n}x${n}@2x.png" >/dev/null
  done
  iconutil -c icns "$SET" -o "$APP/Contents/Resources/icon.icns" 2>/dev/null
fi
touch "$APP"  # (so Finder shows the new icon)

echo
echo "Done: Trip Memory is in your Applications folder (the Finder window that just opened)."
echo "Drag it to the Dock at the bottom of the screen to start it with one click from now on."
echo "The first time, your Mac may ask to let Trip Memory control Terminal: click OK."
open -R "$APP"
