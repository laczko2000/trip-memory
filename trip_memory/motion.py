"""How much moves in a video, and whether it is the people or the camera.

Looks at small grey frames five times a second and measures how far every bit of the picture moves
(optical flow). The shared part of that movement is the camera; what is left over is the action.
No AI model is needed.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass, field

import cv2
import numpy as np

from .tools import ffmpeg_paths

WIDTH = 160
FPS = 5
MAX_SECONDS = 300

# Share of the picture (%) that changes between looks, once the camera's own movement is undone -> label.
ACTION_LEVELS = [(0.5, "Still"), (3, "Calm"), (12, "Lively")]
FAST = "Fast action"
CAMERA_MOVING = 1.2  # average camera movement
SHAKY = 1.0  # how much the camera movement jumps around from step to step
# How the camera itself moves.
CAMERA_TAGS = ["Static camera", "Pan left", "Pan right", "Tilt up", "Tilt down", "Zoom in", "Zoom out",
               "Whip pan", "Turn around", "Rolling", "Fast camera"]
MOVEMENT_TAGS = ["Still", "Calm", "Lively", FAST, "Camera moving", "Cinematic", "Shaky", *CAMERA_TAGS]
PAN = 1.0  # steady sideways movement (pixels of a 160-wide picture per look) that counts as a pan
ZOOM = 0.03  # growing or shrinking by 3% a second counts as a zoom
WHIP = 22  # a jump this big (pixels per look, ~14% of the picture in 0.2 s) is a whip pan


@dataclass
class Motion:
    action: float  # how much of the picture changes because people and things move (%)
    camera: float  # how much the camera itself moves (pans, walks)
    shake: float  # how jittery the camera is
    # Five times a second: how much changes in the scene (%) and how much the camera jumps.
    action_steps: list[float] = field(default_factory=list)
    shake_steps: list[float] = field(default_factory=list)
    # The camera's own move, look by look: sideways (x), up/down (y), zoom (1 = none) and turn (degrees).
    move_x: list[float] = field(default_factory=list)
    move_y: list[float] = field(default_factory=list)
    move_zoom: list[float] = field(default_factory=list)
    move_turn: list[float] = field(default_factory=list)
    # Picture quality, look by look: sharpness (higher = crisper) and brightness (0-255).
    sharp_steps: list[float] = field(default_factory=list)
    bright_steps: list[float] = field(default_factory=list)

    def camera_tags(self) -> list[str]:
        """How the camera moves: pans, tilts, zooms, whips, turning all the way round, rolling."""
        x, y = np.array(self.move_x), np.array(self.move_y)
        if len(x) < 2:
            return []
        tags = []
        steady = np.abs(x) < WHIP  # whips are counted on their own
        if self.camera < 0.3 and self.shake < 0.3:
            tags.append("Static camera")

        def mostly(v: np.ndarray, sign: int) -> bool:
            v = v[steady] if steady.any() else v
            return len(v) > 0 and float(np.median(v)) * sign > PAN and float(np.mean(np.sign(v) == sign)) > 0.6

        # The picture slides left when the camera turns right.
        if mostly(x, -1):
            tags.append("Pan right")
        elif mostly(x, 1):
            tags.append("Pan left")
        if mostly(y, 1):
            tags.append("Tilt up")
        elif mostly(y, -1):
            tags.append("Tilt down")
        zoom = np.log(np.clip(np.array(self.move_zoom), 0.5, 2.0)) * FPS  # per second
        if len(zoom) and abs(float(np.median(zoom))) > ZOOM and float(np.mean(np.sign(zoom) == np.sign(np.median(zoom)))) > 0.6:
            tags.append("Zoom in" if np.median(zoom) > 0 else "Zoom out")
        if (np.abs(x) >= WHIP).any():
            tags.append("Whip pan")
        if abs(float(x.sum())) > 3 * WIDTH:
            tags.append("Turn around")  # the view swept round by more than three pictures' widths
        turn = np.array(self.move_turn)
        if len(turn) and (abs(float(turn.sum())) > 45 or float(np.mean(np.abs(turn))) > 1.5):
            tags.append("Rolling")
        if float(np.mean(np.hypot(x, y))) > 8:  # a quarter of the picture a second or more
            tags.append("Fast camera")
        return tags

    def tags(self) -> list[str]:
        level = next((name for limit, name in ACTION_LEVELS if self.action < limit), FAST)
        tags = [level]
        if self.camera >= CAMERA_MOVING:
            tags = ["Camera moving"] if level == "Still" else [level, "Camera moving"]  # a pan over a still view
        if self.camera >= 1.0 and self.shake < 0.6 * SHAKY and self.action < 3:
            tags.append("Cinematic")  # a smooth, steady camera move over a calm scene
        if self.shake >= SHAKY:
            tags.append("Shaky")
        return tags


def measure(path: str) -> Motion | None:
    """Measure a video's movement (the first five minutes at most). Which way up it is doesn't matter here."""
    cmd = [
        ffmpeg_paths()[0], "-loglevel", "error", "-i", path, "-t", str(MAX_SECONDS),
        "-vf", f"fps={FPS},scale={WIDTH}:-2,format=gray", "-f", "rawvideo", "-",
    ]
    raw = subprocess.run(cmd, capture_output=True).stdout
    probe = subprocess.run(
        [ffmpeg_paths()[0], "-loglevel", "error", "-i", path, "-frames:v", "1",
         "-vf", f"scale={WIDTH}:-2", "-f", "rawvideo", "-pix_fmt", "gray", "-"],
        capture_output=True,
    ).stdout
    if not raw or not probe:
        return None
    height = len(probe) // WIDTH
    frames = np.frombuffer(raw, np.uint8)[: len(raw) // (WIDTH * height) * WIDTH * height].reshape(-1, height, WIDTH)
    return from_frames(frames)


def from_frames(frames: np.ndarray) -> Motion | None:
    if len(frames) < 2:
        return None
    h, w = frames.shape[1:]
    ys, xs = np.mgrid[0:h, 0:w]
    grid = np.stack([xs, ys], axis=2).reshape(-1, 2).astype(np.float32)
    rng = np.random.default_rng(0)
    action, camera, moves = [], [], []
    hann = cv2.createHanningWindow((w, h), cv2.CV_32F)
    for a, b in zip(frames[:-1], frames[1:]):
        flow = cv2.calcOpticalFlowFarneback(a, b, None, 0.5, 3, 15, 3, 5, 1.2, 0).reshape(-1, 2)
        # Only places with some detail show movement reliably (a blue sky or a white wall can't).
        edges = (np.abs(cv2.Sobel(a, cv2.CV_32F, 1, 0)) + np.abs(cv2.Sobel(a, cv2.CV_32F, 0, 1))).reshape(-1)
        detail = edges > 40
        if detail.sum() < 50:
            detail[:] = True
        pick = rng.choice(np.flatnonzero(detail), size=min(1500, int(detail.sum())), replace=False)
        # The camera = the one shift/zoom/turn that explains most of the picture's movement.
        matrix, _ = cv2.estimateAffinePartial2D(grid[pick], grid[pick] + flow[pick], method=cv2.RANSAC,
                                                ransacReprojThreshold=1.0)
        if matrix is None:
            cam = np.tile(np.median(flow[pick], axis=0), (len(grid), 1))
            moves.append((float(cam[0, 0]), float(cam[0, 1]), 1.0, 0.0))
        else:
            cam = grid @ matrix[:, :2].T + matrix[:, 2] - grid
            # Measured around the picture's centre: the shift, how much it grew, how much it turned.
            centre = np.float32([w / 2, h / 2])
            shift = centre @ matrix[:, :2].T + matrix[:, 2] - centre
            scale = float(np.hypot(matrix[0, 0], matrix[1, 0]))
            moves.append((float(shift[0]), float(shift[1]), scale, float(np.degrees(np.arctan2(matrix[1, 0], matrix[0, 0])))))
        # A very fast swing (a whip pan) is too big for the flow above; a whole-picture match still finds it.
        (px, py), response = cv2.phaseCorrelate(a.astype(np.float32), b.astype(np.float32), hann)
        if response > 0.1 and np.hypot(px, py) > max(8.0, np.hypot(*moves[-1][:2]) + 5):
            moves[-1] = (float(px), float(py), 1.0, 0.0)
        # What still changes once the camera's movement is undone is the action in the scene.
        if matrix is None:
            matrix = np.float32([[1, 0, cam[0, 0]], [0, 1, cam[0, 1]]])
        back = cv2.warpAffine(b, cv2.invertAffineTransform(matrix), (w, h), borderMode=cv2.BORDER_REPLICATE)
        changed = cv2.absdiff(cv2.GaussianBlur(a, (5, 5), 0), cv2.GaussianBlur(back, (5, 5), 0)) > 25
        margin = 6  # picture edges come into view as the camera moves; ignore them
        action.append(100 * float(changed[margin:-margin, margin:-margin].mean()))  # % of the picture
        camera.append(cam.mean(axis=0) if matrix is not None else cam[0])
        camera[-1] = np.append(camera[-1], np.linalg.norm(cam, axis=1).mean())
    camera = np.array(camera)
    jumps = np.linalg.norm(np.diff(camera[:, :2], axis=0), axis=1) if len(camera) > 1 else np.zeros(0)
    shake = float(jumps.mean()) if len(camera) > 2 else 0.0
    # Sharpness and brightness of each later look (so they line up with the steps above).
    sharp = [float(cv2.Laplacian(f, cv2.CV_32F).var()) for f in frames[1:]]
    bright = [float(f.mean()) for f in frames[1:]]
    return Motion(
        action=float(np.mean(action)), camera=float(camera[:, 2].mean()), shake=shake,
        action_steps=[round(a, 2) for a in action], shake_steps=[0.0] + [round(float(j), 2) for j in jumps],
        move_x=[round(m[0], 2) for m in moves], move_y=[round(m[1], 2) for m in moves],
        move_zoom=[round(m[2], 4) for m in moves], move_turn=[round(m[3], 2) for m in moves],
        sharp_steps=[round(v, 1) for v in sharp], bright_steps=[round(v, 1) for v in bright],
    )
