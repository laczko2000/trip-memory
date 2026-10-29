"""The AI director: a short description of every clip, and a few storyline ideas for the whole trip.

Two ways to run it:
- Local, with Ollama (free, nothing leaves your Mac). Install it from ollama.com and download a model that can
  see pictures, for example: ollama pull gemma3:4b
- Claude (paid per use, usually a few cents to a dollar per trip). Needs an API key from console.anthropic.com.
"""

from __future__ import annotations

import base64
import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass

OLLAMA_URL = "http://localhost:11434"
OLLAMA_SUGGESTED = "gemma3:4b"
CLAUDE_MODEL = "claude-opus-5-5"
PROVIDERS = {"ollama": "On this Mac (Ollama, free)", "claude": "Claude (paid, needs an API key)"}

CAPTION_PROMPT = """These are pictures from a holiday video or photo. {what}
For each clip, write:
- caption: what it shows, in at most 12 plain words (who or what, doing what, where). No "This image shows".
- highlight: how much it deserves a place in a short holiday memory video, 1 (dull, blurry, accidental) to 5 (a moment you'd want to remember).
Answer in JSON."""

CAPTION_SCHEMA = {
    "type": "object",
    "properties": {
        "clips": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "clip": {"type": "integer"},
                    "caption": {"type": "string"},
                    "highlight": {"type": "integer"},
                },
                "required": ["clip", "caption", "highlight"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["clips"],
    "additionalProperties": False,
}

STORYLINE_PROMPT = """You are the editor of a short holiday memory video (a few minutes, set to music).
The trip is split into stories: groups of shots taken at the same time and place. Here they are in date order:

{stories}

Suggest 3 different storylines for the video. Each storyline:
- title: a short name for the idea (e.g. "Best moments first", "Day by day", "The people").
- idea: one sentence on why it works.
- stories: the story numbers to use, in the order the video should tell them. Leave out stories that are dull
  or repeat another; keep most of them.
Also suggest a better short name (2 to 5 words) for every story, from what its clips show and where it is.
Answer in JSON."""

STORYLINE_SCHEMA = {
    "type": "object",
    "properties": {
        "storylines": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "idea": {"type": "string"},
                    "stories": {"type": "array", "items": {"type": "integer"}},
                },
                "required": ["title", "idea", "stories"],
                "additionalProperties": False,
            },
        },
        "names": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"story": {"type": "integer"}, "name": {"type": "string"}},
                "required": ["story", "name"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["storylines", "names"],
    "additionalProperties": False,
}


class DirectorError(Exception):
    """Something the person can fix (no key, Ollama not running...), with a plain explanation."""


@dataclass
class Caption:
    text: str
    highlight: int  # 1..5


# ---------- Ollama (local) ----------


def _ollama(path: str, body: dict | None = None, timeout: float = 600) -> dict:
    data = json.dumps(body).encode() if body is not None else None
    request = urllib.request.Request(OLLAMA_URL + path, data=data, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read())
    except urllib.error.HTTPError as exc:
        raise DirectorError(f"Ollama said: {exc.read().decode(errors='replace')[:300]}") from exc
    except (urllib.error.URLError, OSError) as exc:
        raise DirectorError("Ollama is not running. Install it from ollama.com and open it, then try again.") from exc


def ollama_models() -> tuple[list[str], list[str]] | None:
    """(models that can see pictures, all models) installed in Ollama, or None if Ollama isn't running."""
    try:
        names = [m["name"] for m in _ollama("/api/tags", timeout=3).get("models", [])]
    except DirectorError:
        return None
    seeing = []
    for name in names:
        try:
            if "vision" in _ollama("/api/show", {"model": name}, timeout=5).get("capabilities", []):
                seeing.append(name)
        except DirectorError:
            pass
    return seeing, names


class Ollama:
    batch = 1  # small local models describe one clip at a time best

    def __init__(self, model: str):
        if not model:
            raise DirectorError(f"No picture-reading model in Ollama yet. In Terminal, run: ollama pull {OLLAMA_SUGGESTED}")
        self.model = model

    def _ask(self, prompt: str, schema: dict, images: list[bytes] | None = None) -> dict:
        message: dict = {"role": "user", "content": prompt}
        if images:
            message["images"] = [base64.b64encode(i).decode() for i in images]
        answer = _ollama("/api/chat", {"model": self.model, "messages": [message], "stream": False, "format": schema,
                                       "options": {"temperature": 0.2}})
        try:
            return json.loads(answer["message"]["content"])
        except (KeyError, ValueError) as exc:
            raise DirectorError("Ollama gave an answer the app could not read. Try again or try another model.") from exc

    def describe(self, clips: list[list[bytes]]) -> list[Caption | None]:
        return _captions(self._ask(_caption_prompt(clips), CAPTION_SCHEMA, [i for c in clips for i in c]), len(clips))

    def storylines(self, prompt: str) -> dict:
        return self._ask(prompt, STORYLINE_SCHEMA)


# ---------- Claude ----------


class Claude:
    batch = 6  # several clips per request: cheaper, and it can compare them

    def __init__(self, api_key: str = "", model: str = CLAUDE_MODEL):
        try:
            import anthropic
        except ImportError as exc:
            raise DirectorError("The Claude helper is missing. Start the app again with Start Trip Memory.command.") from exc
        key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        if not key:
            raise DirectorError("Paste your Claude API key first (from console.anthropic.com, under API keys).")
        self.anthropic = anthropic
        self.client = anthropic.Anthropic(api_key=key)
        self.model = model or CLAUDE_MODEL

    def _ask(self, content: list[dict], schema: dict) -> dict:
        a = self.anthropic
        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=16000,
                messages=[{"role": "user", "content": content}],
                output_config={"effort": "low", "format": {"type": "json_schema", "schema": schema}},
            )
        except a.AuthenticationError as exc:
            raise DirectorError("Claude did not accept the API key. Check it at console.anthropic.com.") from exc
        except a.PermissionDeniedError as exc:
            raise DirectorError("This API key may not use Claude yet. Check your account at console.anthropic.com.") from exc
        except a.NotFoundError as exc:
            raise DirectorError(f"Claude does not know the model \"{self.model}\". Check the model name.") from exc
        except a.RateLimitError as exc:
            raise DirectorError("Claude is busy or your account's limit is reached. Wait a minute and try again.") from exc
        except a.BadRequestError as exc:
            message = str(getattr(exc, "message", exc))
            if "credit" in message.lower():
                raise DirectorError("Your Claude account has no credit left. Add some at console.anthropic.com.") from exc
            raise DirectorError(f"Claude could not do this: {message[:300]}") from exc
        except a.APIStatusError as exc:
            raise DirectorError(f"Claude had a problem ({exc.status_code}). Try again in a minute.") from exc
        except a.APIConnectionError as exc:
            raise DirectorError("Could not reach Claude. Check your internet connection.") from exc
        if response.stop_reason == "refusal":
            raise DirectorError("Claude declined to describe these pictures.")
        text = next((b.text for b in response.content if b.type == "text"), "")
        try:
            return json.loads(text)
        except ValueError as exc:
            raise DirectorError("Claude's answer was cut off. Try again.") from exc

    def describe(self, clips: list[list[bytes]]) -> list[Caption | None]:
        content: list[dict] = []
        for n, images in enumerate(clips, 1):
            content.append({"type": "text", "text": f"Clip {n}:"})
            content.extend({"type": "image", "source": {"type": "base64", "media_type": "image/jpeg",
                                                         "data": base64.b64encode(i).decode()}} for i in images)
        content.append({"type": "text", "text": _caption_prompt(clips)})
        return _captions(self._ask(content, CAPTION_SCHEMA), len(clips))

    def storylines(self, prompt: str) -> dict:
        return self._ask([{"type": "text", "text": prompt}], STORYLINE_SCHEMA)


def make(settings: dict):
    """The director to use, from the app's settings ({"provider", "ollama_model", "claude_model", "api_key"})."""
    if settings.get("provider") == "claude":
        return Claude(settings.get("api_key", ""), settings.get("claude_model") or CLAUDE_MODEL)
    return Ollama(settings.get("ollama_model", ""))


# ---------- prompts and answers ----------


def _caption_prompt(clips: list[list[bytes]]) -> str:
    if len(clips) == 1:
        what = (f"They are {len(clips[0])} frames from one clip, in time order: call it clip 1."
                if len(clips[0]) > 1 else "It is one photo: call it clip 1.")
    else:
        what = f"There are {len(clips)} clips, each shown as one or more frames after its \"Clip n:\" label."
    return CAPTION_PROMPT.format(what=what)


def _captions(answer: dict, count: int) -> list[Caption | None]:
    result: list[Caption | None] = [None] * count
    for entry in answer.get("clips", []):
        try:
            n = int(entry["clip"]) - 1
            text = " ".join(str(entry["caption"]).split())[:120]
            highlight = max(1, min(5, int(entry["highlight"])))
        except (KeyError, TypeError, ValueError):
            continue
        if count == 1:
            n = 0  # a small model may number the only clip differently
        if 0 <= n < count and text and result[n] is None:
            result[n] = Caption(text, highlight)
    return result


def storyline_prompt(stories: list[dict]) -> str:
    """The trip, story by story, for the storyline question. Each story: name, when, where, its clips' captions."""
    lines = []
    for n, s in enumerate(stories, 1):
        captions = [c for c in s["captions"] if c][:10]
        lines.append(f"Story {n}: \"{s['name']}\" · {s['when']}" + (f" · {s['place']}" if s.get("place") else "")
                     + f" · {s['count']} clips")
        lines.extend(f"  - {c}" for c in captions)
    return STORYLINE_PROMPT.format(stories="\n".join(lines))


def clean_storylines(answer: dict, count: int) -> dict:
    """Keep only valid story numbers (1..count, each once) and drop empty storylines."""
    storylines = []
    for s in answer.get("storylines", []):
        order: list[int] = []
        for n in s.get("stories", []):
            if isinstance(n, int) and 1 <= n <= count and n not in order:
                order.append(n)
        if order and s.get("title"):
            storylines.append({"title": str(s["title"])[:60], "idea": str(s.get("idea", ""))[:300], "stories": order})
    names = {}
    for entry in answer.get("names", []):
        n, name = entry.get("story"), " ".join(str(entry.get("name", "")).split())[:40]
        if isinstance(n, int) and 1 <= n <= count and name:
            names[n] = name
    return {"storylines": storylines[:4], "names": names}
