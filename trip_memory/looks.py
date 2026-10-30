"""Looks: one colour mood for the whole video, applied as the clips are joined (so changing it is quick).

Each look is an FFmpeg filter. The previews in the app run the same filter on a still picture, so what you
see is what you get.
"""

from __future__ import annotations

import base64
import subprocess
from pathlib import Path

from .tools import ffmpeg_paths

# name -> (what it's called in the app, the filter; "" = the pictures as they are)
LOOKS: dict[str, tuple[str, str]] = {
    "natural": ("Natural", ""),
    "summer": ("Warm summer", "colorbalance=rs=0.08:gs=0.02:bs=-0.08:rm=0.05:bm=-0.05:rh=0.03:bh=-0.04,"
                              "eq=saturation=1.15:contrast=1.04"),
    "vivid": ("Vivid", "eq=saturation=1.35:contrast=1.08:gamma=0.97,unsharp=5:5:0.35"),
    "film": ("Film", "colorbalance=rs=0.04:bs=-0.03:bh=0.03,eq=saturation=0.82:contrast=1.06,"
                     "noise=alls=7:allf=t,vignette=angle=PI/5"),
    "cool": ("Cool blue hour", "colorbalance=rs=-0.05:bs=0.08:bm=0.04,eq=saturation=1.05:contrast=1.03"),
    "mono": ("Black and white", "hue=s=0,eq=contrast=1.15:brightness=0.01"),
}


def look_filter(name: str | None) -> str:
    """The filter for this look, ready to follow another filter (",..."), or "" for none."""
    graph = LOOKS.get(name or "natural", LOOKS["natural"])[1]
    return f",{graph}" if graph else ""


def preview(picture: str | Path, name: str, width: int = 240) -> str:
    """A small picture with this look, as a data URL for the app (empty if it could not be made)."""
    graph = f"scale={width}:-2{look_filter(name)}".replace(":allf=t", "")  # (grain still shows on a single frame)
    try:
        jpeg = subprocess.run([ffmpeg_paths()[0], "-loglevel", "error", "-i", str(picture), "-frames:v", "1",
                               "-vf", graph, "-f", "mjpeg", "-"], capture_output=True, check=True, timeout=20).stdout
    except (subprocess.SubprocessError, OSError):
        return ""
    return "data:image/jpeg;base64," + base64.b64encode(jpeg).decode()
