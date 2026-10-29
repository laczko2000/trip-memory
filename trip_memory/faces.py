"""Find faces in pictures and turn each face into a "fingerprint" so the same person can be grouped.

Uses the free InsightFace "buffalo_l" models (SCRFD face finder + ArcFace recogniser), run with
onnxruntime. The models are downloaded once (about 290 MB) and are licensed for non-commercial use.
"""

from __future__ import annotations

import io
import zipfile
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

MODELS_URL = "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip"
DETECTOR = "det_10g.onnx"
RECOGNISER = "w600k_r50.onnx"

# Where the five face landmarks (eyes, nose, mouth corners) sit in a 112x112 ArcFace crop.
ARCFACE_POINTS = np.array(
    [[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366], [41.5493, 92.3655], [70.7299, 92.2041]],
    dtype=np.float32,
)

MIN_FACE_PIXELS = 40  # smaller faces are too blurry to recognise reliably
MIN_SCORE = 0.6


@dataclass
class Face:
    box: tuple[float, float, float, float]  # x1, y1, x2, y2 in the original picture
    score: float
    embedding: np.ndarray  # 512 numbers, length 1


def download_models(model_dir: Path) -> None:
    """Fetch the two model files we need, once."""
    if (model_dir / DETECTOR).exists() and (model_dir / RECOGNISER).exists():
        return
    import requests

    model_dir.mkdir(parents=True, exist_ok=True)
    response = requests.get(MODELS_URL, timeout=600)
    response.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
        for name in archive.namelist():
            if Path(name).name in (DETECTOR, RECOGNISER):
                (model_dir / Path(name).name).write_bytes(archive.read(name))


def _session(path: Path) -> ort.InferenceSession:
    providers = [p for p in ("CoreMLExecutionProvider", "CPUExecutionProvider") if p in ort.get_available_providers()]
    try:
        return ort.InferenceSession(str(path), providers=providers)
    except Exception:  # CoreML can refuse some models; the CPU always works
        return ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])


class FaceFinder:
    def __init__(self, model_dir: Path):
        download_models(model_dir)
        self.detector = _session(model_dir / DETECTOR)
        self.recogniser = _session(model_dir / RECOGNISER)
        self.det_input = self.detector.get_inputs()[0].name
        self.rec_input = self.recogniser.get_inputs()[0].name

    def find(self, rgb: np.ndarray) -> list[Face]:
        """Return the faces in an RGB picture (height x width x 3, uint8)."""
        faces = []
        for box, score, points in self._detect(rgb):
            if min(box[2] - box[0], box[3] - box[1]) < MIN_FACE_PIXELS:
                continue
            faces.append(Face(box=tuple(float(v) for v in box), score=float(score), embedding=self._embed(rgb, points)))
        return faces

    def _detect(self, rgb: np.ndarray, size: int = 640, threshold: float = MIN_SCORE):
        h, w = rgb.shape[:2]
        scale = size / max(h, w)
        resized = cv2.resize(rgb, (round(w * scale), round(h * scale)))
        canvas = np.zeros((size, size, 3), dtype=np.uint8)
        canvas[: resized.shape[0], : resized.shape[1]] = resized
        blob = cv2.dnn.blobFromImage(canvas, 1 / 128, (size, size), (127.5, 127.5, 127.5), swapRB=False)
        outputs = [o[0] if o.ndim == 3 else o for o in self.detector.run(None, {self.det_input: blob})]

        boxes, scores, points = [], [], []
        for i, stride in enumerate((8, 16, 32)):
            cells = size // stride
            grid = np.stack(np.mgrid[:cells, :cells][::-1], axis=-1).reshape(-1, 2).astype(np.float32) * stride
            centres = np.repeat(grid, 2, axis=0)  # two anchors per cell
            s = outputs[i].reshape(-1)
            keep = s >= threshold
            if not keep.any():
                continue
            d = outputs[i + 3][keep] * stride
            k = outputs[i + 6][keep] * stride
            c = centres[keep]
            boxes.append(np.stack([c[:, 0] - d[:, 0], c[:, 1] - d[:, 1], c[:, 0] + d[:, 2], c[:, 1] + d[:, 3]], axis=1))
            points.append((np.repeat(c, 5, axis=0).reshape(-1, 5, 2) + k.reshape(-1, 5, 2)))
            scores.append(s[keep])
        if not boxes:
            return []
        boxes = np.concatenate(boxes) / scale
        points = np.concatenate(points) / scale
        scores = np.concatenate(scores)
        xywh = [[b[0], b[1], b[2] - b[0], b[3] - b[1]] for b in boxes]
        keep = cv2.dnn.NMSBoxes(xywh, scores.tolist(), threshold, 0.4)
        return [(boxes[i], scores[i], points[i]) for i in np.array(keep).reshape(-1)]

    def _embed(self, rgb: np.ndarray, points: np.ndarray) -> np.ndarray:
        matrix, _ = cv2.estimateAffinePartial2D(points.astype(np.float32), ARCFACE_POINTS, method=cv2.LMEDS)
        crop = cv2.warpAffine(rgb, matrix, (112, 112), borderValue=0)
        blob = cv2.dnn.blobFromImage(crop, 1 / 127.5, (112, 112), (127.5, 127.5, 127.5), swapRB=False)
        vector = self.recogniser.run(None, {self.rec_input: blob})[0][0]
        return (vector / np.linalg.norm(vector)).astype(np.float32)


def group_faces(
    embeddings: np.ndarray, known: dict[int, np.ndarray] | None = None, threshold: float = 0.42
) -> list[int | None]:
    """Put each face with the most similar known person, or start new groups.

    Returns, per face, a known person id or a negative number naming a new group (-1, -2, ...).
    """
    known = dict(known or {})
    centres = {pid: c / np.linalg.norm(c) for pid, c in known.items()}
    members: dict[int, list[np.ndarray]] = {pid: [c] for pid, c in centres.items()}
    next_new = -1
    result: list[int | None] = []
    for emb in embeddings:
        best, best_sim = None, threshold
        for pid, centre in centres.items():
            sim = float(emb @ centre)
            if sim > best_sim:
                best, best_sim = pid, sim
        if best is None:
            best = next_new
            next_new -= 1
            members[best] = []
        members[best].append(emb)
        mean = np.mean(members[best], axis=0)
        centres[best] = mean / np.linalg.norm(mean)
        result.append(best)
    return result
