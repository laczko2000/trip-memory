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
OLLAMA_FASTEST = "moondream"  # about 3x faster on a MacBook Air, simpler descriptions
CLAUDE_MODEL = "claude-haiku-5-5"  # describing many pictures: fast and costs very little
CLAUDE_STORY_MODEL = "claude-opus-5-5"  # thinking about the stories: few requests, so the best model
CLAUDE_MODELS = {"claude-haiku-5-5": "Fast and cheap (Haiku 5.5, a few cents a trip)",
                 "claude-opus-5-5": "Most detailed (Opus 5.5, about $2–5 a trip)"}
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


ROLES = {  # what a group of shots does in a story, and how Auto-edit cuts it
    "opening": "Opening: sets the scene",
    "build-up": "Build-up",
    "montage": "Quick montage on the beat",
    "highlight": "Highlight",
    "slow moment": "Slow, cinematic moment",
    "ending": "Ending",
}

PLAN_PROMPT = """You are the editor of a holiday memory video set to music. This is one story of the trip: shots
taken at the same time and place, in the order they were taken.

Story: "{name}" · {when}{place}
{shots}

Plan how this story is told in the video, like a short scene with a beginning, middle and end:
- theme: a short, evocative title for the story (2 to 5 words, e.g. "Lost in the forest").
- idea: one sentence on how the scene should feel and flow.
- parts: the scene in order. Each part has a role, the shot numbers it uses (in the order to show them), and a short
  note on why. Roles:
  opening (one or two wide, calm shots that set the scene), build-up, montage (several similar or related shots,
  especially photos, cut quickly one after another on the beat, e.g. five forest pictures), highlight (the best
  moment), slow moment (a calm, beautiful shot to let breathe), ending (a calm shot that closes the story).
  Use each shot at most once. Leave out dull, blurry or repeated shots. Not every role is needed.
Answer in JSON."""

PLAN_SCHEMA = {
    "type": "object",
    "properties": {
        "theme": {"type": "string"},
        "idea": {"type": "string"},
        "parts": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "role": {"type": "string", "enum": list(ROLES)},
                    "shots": {"type": "array", "items": {"type": "integer"}},
                    "note": {"type": "string"},
                },
                "required": ["role", "shots", "note"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["theme", "idea", "parts"],
    "additionalProperties": False,
}


WISH_PROMPT = """You are the editor of a holiday memory video set to music. The person typed the video they want:

"{wish}"

The trip is split into stories (shots taken at the same time and place), in date order:

{stories}

People they have named: {people}.
{song}
Turn their words into settings for the app:
- summary: one friendly sentence saying what video you will make.
- length_seconds: how long the video should be (0 if they did not say: it then lasts as long as the song).
- stories: the story numbers to use, in the order to tell them; leave out stories that don't fit the wish.
  An empty list keeps the stories as they are now.
- people: names (only from the list above) to show more of; empty if they named nobody.
- focus: a few short words for what to show more of (e.g. "beach", "kids", "boat"); empty if nothing special.
- pace: "calm", "medium" or "lively".
- teaser: true to open with a quick teaser of the most exciting shots.
- slow_motion: true for slow motion in calm moments.
- camera_effects: true for zooms, punch-ins and speed ramps.
- title: the opening title if they asked for one or the wish suggests a good one; else empty.
- look: the colour mood if the wish suggests one: "summer" (warm), "vivid", "film" (grain, vintage),
  "cool" (blue hour), "mono" (black and white), "natural"; empty to keep the current one.
Answer in JSON."""

WISH_SCHEMA = {
    "type": "object",
    "properties": {
        "summary": {"type": "string"},
        "length_seconds": {"type": "integer"},
        "stories": {"type": "array", "items": {"type": "integer"}},
        "people": {"type": "array", "items": {"type": "string"}},
        "focus": {"type": "array", "items": {"type": "string"}},
        "pace": {"type": "string", "enum": ["calm", "medium", "lively"]},
        "teaser": {"type": "boolean"},
        "slow_motion": {"type": "boolean"},
        "camera_effects": {"type": "boolean"},
        "title": {"type": "string"},
        "look": {"type": "string", "enum": ["", "natural", "summer", "vivid", "film", "cool", "mono"]},
    },
    "required": ["summary", "length_seconds", "stories", "people", "focus", "pace", "teaser", "slow_motion",
                 "camera_effects", "title", "look"],
    "additionalProperties": False,
}


class DirectorError(Exception):
    """Something the person can fix (no key, Ollama not running...), with a plain explanation."""


class Stopped(DirectorError):
    """You pressed Stop: what the AI finished before is kept, the rest is left as it was."""


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
    frames = 1  # one small picture per clip: a Mac without a big graphics card reads each picture slowly
    size = 384

    def __init__(self, model: str):
        if not model:
            raise DirectorError(f"No picture-reading model in Ollama yet. In Terminal, run: ollama pull {OLLAMA_SUGGESTED}")
        self.model = model

    def _ask(self, prompt: str, schema: dict, images: list[bytes] | None = None) -> dict:
        message: dict = {"role": "user", "content": prompt}
        if images:
            message["images"] = [base64.b64encode(i).decode() for i in images]
        answer = _ollama("/api/chat", {"model": self.model, "messages": [message], "stream": False, "format": schema,
                                       "keep_alive": "30m",  # stay loaded between clips (loading takes the longest)
                                       # a caption is a few dozen words: stopping early saves a lot of time
                                       "options": {"temperature": 0.2, "num_predict": 160 if images else 3000,
                                                   "num_ctx": 2048 if images else 8192}})
        try:
            return json.loads(answer["message"]["content"])
        except (KeyError, ValueError) as exc:
            raise DirectorError("Ollama gave an answer the app could not read. Try again or try another model.") from exc

    def describe(self, clips: list[list[bytes]]) -> list[Caption | None]:
        return _captions(self._ask(_caption_prompt(clips), CAPTION_SCHEMA, [i for c in clips for i in c]), len(clips))

    def storylines(self, prompt: str) -> dict:
        return self._ask(prompt, STORYLINE_SCHEMA)

    def plan(self, prompt: str) -> dict:
        return self._ask(prompt, PLAN_SCHEMA)

    def wish(self, prompt: str) -> dict:
        return self._ask(prompt, WISH_SCHEMA)


# ---------- Claude ----------


class Claude:
    batch = 6  # several clips per request: cheaper, and it can compare them
    frames = 3
    size = 512

    def __init__(self, api_key: str = "", model: str = CLAUDE_MODEL, story_model: str = CLAUDE_STORY_MODEL):
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
        self.story_model = story_model or CLAUDE_STORY_MODEL

    def _ask(self, content: list[dict], schema: dict, model: str = "") -> dict:
        a = self.anthropic
        model = model or self.model
        try:
            response = self.client.messages.create(
                model=model,
                max_tokens=16000,
                messages=[{"role": "user", "content": content}],
                output_config={"effort": "low", "format": {"type": "json_schema", "schema": schema}},
            )
        except a.AuthenticationError as exc:
            raise DirectorError("Claude did not accept the API key. Check it at console.anthropic.com.") from exc
        except a.PermissionDeniedError as exc:
            raise DirectorError("This API key may not use Claude yet. Check your account at console.anthropic.com.") from exc
        except a.NotFoundError as exc:
            raise DirectorError(f"Claude does not know the model \"{model}\". Check the model name.") from exc
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
        return self._ask([{"type": "text", "text": prompt}], STORYLINE_SCHEMA, self.story_model)

    def plan(self, prompt: str) -> dict:
        return self._ask([{"type": "text", "text": prompt}], PLAN_SCHEMA, self.story_model)

    def wish(self, prompt: str) -> dict:
        return self._ask([{"type": "text", "text": prompt}], WISH_SCHEMA, self.story_model)


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
        if s.get("idea"):
            lines.append(f"  Plan: {s['idea']}")
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


def plan_prompt(story: dict, shots: list[dict]) -> str:
    """The question for planning one story. shots: in time order, each with caption, highlight, kind, seconds, time."""
    lines = []
    for n, shot in enumerate(shots, 1):
        what = f"video {shot['seconds']:.0f} s" if shot["kind"] == "video" else "photo"
        score = f", highlight {shot['highlight']}/5" if shot.get("highlight") else ""
        lines.append(f"{n}. {shot['time']} · {what}{score}: {shot.get('caption') or '(not described)'}")
    return PLAN_PROMPT.format(name=story["name"], when=story["when"],
                              place=f" · {story['place']}" if story.get("place") else "", shots="\n".join(lines))


def clean_plan(answer: dict, count: int) -> dict:
    """Keep valid roles and shot numbers (1..count, each shot once); drop empty parts."""
    used: set[int] = set()
    parts = []
    for part in answer.get("parts", []):
        role = part.get("role")
        shots = []
        for n in part.get("shots", []):
            if isinstance(n, int) and 1 <= n <= count and n not in used:
                used.add(n)
                shots.append(n)
        if role in ROLES and shots:
            parts.append({"role": role, "shots": shots, "note": " ".join(str(part.get("note", "")).split())[:200]})
    theme = " ".join(str(answer.get("theme", "")).split())[:40]
    return {"theme": theme, "idea": " ".join(str(answer.get("idea", "")).split())[:300], "parts": parts}


def wish_prompt(wish: str, stories: list[dict], people: list[str], song_seconds: float | None) -> str:
    """The question for "type the video you want". stories: as for storyline_prompt, plus the people in each."""
    lines = []
    for n, s in enumerate(stories, 1):
        lines.append(f"Story {n}: \"{s['name']}\" · {s['when']}" + (f" · {s['place']}" if s.get("place") else "")
                     + f" · {s['count']} clips" + (f" · with {', '.join(s['people'])}" if s.get("people") else ""))
        lines.extend(f"  - {c}" for c in [c for c in s["captions"] if c][:6])
    song = (f"The chosen song lasts {song_seconds / 60:.1f} minutes.\n" if song_seconds
            else "No song is chosen yet.\n")
    return WISH_PROMPT.format(wish=" ".join(wish.split())[:1000], stories="\n".join(lines),
                              people=", ".join(people) or "nobody yet", song=song)


def clean_wish(answer: dict, count: int, people: list[str]) -> dict:
    """Keep valid story numbers and known names; keep the numbers sensible."""
    order: list[int] = []
    for n in answer.get("stories", []):
        if isinstance(n, int) and 1 <= n <= count and n not in order:
            order.append(n)
    known = {p.lower(): p for p in people}
    who = [known[str(p).strip().lower()] for p in answer.get("people", []) if str(p).strip().lower() in known]
    focus = [w for w in (" ".join(str(f).split()).lower()[:30] for f in answer.get("focus", [])) if w][:6]
    seconds = answer.get("length_seconds")
    seconds = int(min(max(seconds, 15), 1200)) if isinstance(seconds, (int, float)) and seconds > 0 else 0
    return {
        "summary": " ".join(str(answer.get("summary", "")).split())[:300],
        "length_seconds": seconds, "stories": order, "people": list(dict.fromkeys(who)), "focus": focus,
        "pace": answer.get("pace") if answer.get("pace") in ("calm", "medium", "lively") else "medium",
        "teaser": bool(answer.get("teaser")), "slow_motion": bool(answer.get("slow_motion", True)),
        "camera_effects": bool(answer.get("camera_effects", True)),
        "title": " ".join(str(answer.get("title", "")).split())[:60],
        "look": answer.get("look") if answer.get("look") in ("natural", "summer", "vivid", "film", "cool", "mono") else "",
    }
