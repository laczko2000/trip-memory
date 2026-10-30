"""Speak your idea: a recording becomes text, and the AI makes a long idea short and clear."""

from types import SimpleNamespace

from trip_memory import director, voice


def test_a_recording_becomes_text(tmp_path, monkeypatch):
    made = []

    class FakeWhisper:
        def __init__(self, name, device, compute_type, download_root):
            made.append((name, download_root))

        def transcribe(self, path, vad_filter, beam_size):
            return iter([SimpleNamespace(text=" Two minutes of the beach,"), SimpleNamespace(text=" no, three. ")]), None

    import faster_whisper

    monkeypatch.setattr(faster_whisper, "WhisperModel", FakeWhisper)
    monkeypatch.setattr(voice, "_model", None)
    assert voice.transcribe("talk.webm", tmp_path / "model") == "Two minutes of the beach, no, three."
    voice.transcribe("again.webm", tmp_path / "model")
    assert made == [("small", str(tmp_path / "model"))]  # the model is loaded once


def test_a_long_idea_is_made_short_and_clear():
    prompt = director.brief_prompt("so um I want   like a video\nof the beach, two minutes, no actually three")
    assert '"so um I want like a video of the beach, two minutes, no actually three"' in prompt
    assert director.clean_brief({"brief": "  A three-minute video\n of the beach. "}) == "A three-minute video of the beach."
    assert director.clean_brief({}) == ""


def test_the_recording_reaches_the_app(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    from nicegui import app as web

    from trip_memory import app as trip_app

    monkeypatch.setattr(trip_app, "app_dir", lambda: tmp_path)
    answer = TestClient(web).post("/voice", content=b"sound", headers={"Content-Type": "audio/mp4"}).json()
    assert answer["name"].endswith(".m4a") and (tmp_path / "voice" / answer["name"]).read_bytes() == b"sound"
