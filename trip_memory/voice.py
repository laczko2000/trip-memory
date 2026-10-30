"""Speak your idea: what you say becomes text on this Mac (Whisper, free; the speech model is downloaded once)."""

from __future__ import annotations

import threading
from pathlib import Path

MODEL = "small"  # about 500 MB, good with long, rambling speech in many languages
_model = None
_lock = threading.Lock()


def transcribe(path: str, model_dir: Path) -> str:
    """The words in a recording, as one piece of text."""
    global _model
    with _lock:
        if _model is None:
            from faster_whisper import WhisperModel

            model_dir.mkdir(parents=True, exist_ok=True)
            _model = WhisperModel(MODEL, device="cpu", compute_type="int8", download_root=str(model_dir))
        segments, _info = _model.transcribe(path, vad_filter=True, beam_size=1)
        return " ".join(s.text.strip() for s in segments).strip()
