"""Movement tags: a still scene, a camera pan, a shaky camera and something moving in the picture."""

import numpy as np
import cv2

from trip_memory.motion import from_frames


def _scene(seed=1, size=(400, 700)):
    rng = np.random.default_rng(seed)
    img = np.zeros(size, np.uint8)
    for _ in range(300):
        cv2.circle(img, (int(rng.integers(0, size[1])), int(rng.integers(0, size[0]))), int(rng.integers(3, 20)),
                   int(rng.integers(0, 255)), -1)
    return cv2.GaussianBlur(img, (5, 5), 0)


def _clip(offsets, ball=None):
    big = _scene()
    frames = []
    for n, (x, y) in enumerate(offsets):
        frame = big[50 + y : 50 + y + 90, 100 + x : 100 + x + 160].copy()
        if ball:
            cv2.rectangle(frame, (10 + ball * n, 30), (40 + ball * n, 60), 255, -1)
        frames.append(frame)
    return np.array(frames)


def test_still():
    assert from_frames(_clip([(0, 0)] * 10)).tags() == ["Still"]


def test_pan_is_the_camera_not_action():
    assert from_frames(_clip([(6 * n, 0) for n in range(10)])).tags() == ["Camera moving", "Cinematic"]


def test_shaky_camera():
    rng = np.random.default_rng(2)
    tags = from_frames(_clip([(int(x), int(y)) for x, y in rng.integers(0, 12, (10, 2))])).tags()
    assert "Shaky" in tags


def test_something_moving():
    assert from_frames(_clip([(0, 0)] * 10, ball=8)).tags()[0] in ("Calm", "Lively")
