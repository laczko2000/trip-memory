#!/bin/bash
# Double-click this file to start Trip Memory.
cd "$(dirname "$0")" || exit 1
export PATH="$HOME/.local/bin:$PATH"

if ! command -v uv >/dev/null 2>&1; then
  echo "First start: installing the helper that runs the app (uv)..."
  curl -LsSf https://astral.sh/uv/install.sh | sh || { echo "Install failed. Check your internet connection."; read -r; exit 1; }
  export PATH="$HOME/.local/bin:$PATH"
fi

echo "Starting Trip Memory... The first start downloads what it needs and can take a few minutes."
uv run --python 3.12 trip-memory
echo "Trip Memory has stopped. You can close this window."
read -r
