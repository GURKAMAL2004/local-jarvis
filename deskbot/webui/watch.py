"""Clarify -> search -> pick loop behind `deskbot watch` (the distraction-free
YouTube kiosk). One WatchSession per kiosk window. Each user message advances
the loop by exactly one user-visible step: either a clarifying question
(pause and wait for the reply) or a final "playing" decision. In between,
the session may silently loop through several search rounds on its own —
never a question the user didn't ask for, never a video it didn't actually
find.

Grounding is structural, not a prompt request: `_pick()` can only return an
entry already present in `self.last_results`, and `last_results` is only ever
populated from `search_youtube()`'s real scrape. There is no code path that
plays a video the model merely described.
"""

from __future__ import annotations

import json
import logging
import uuid
from dataclasses import dataclass, field
from typing import Any

from deskbot.config import Config
from deskbot.llm import OllamaClient, OllamaConnectionError, OllamaModelError
from deskbot.tools.youtube import search_youtube

logger = logging.getLogger("deskbot.webui.watch")

_DECISION_SHAPES = (
    'Respond with ONLY a JSON object, no other text, in exactly one of these shapes:\n'
    '{"action": "ask", "question": "<one short clarifying question>"}\n'
    '{"action": "search", "query": "<a specific YouTube search query>"}\n'
    '{"action": "play", "index": <integer index into the numbered SEARCH_RESULTS above>}\n'
)

def _build_system_prompt(max_questions: int) -> str:
    # Built with an f-string rather than str.format() on a shared constant —
    # _DECISION_SHAPES contains literal JSON braces that str.format() would
    # misparse as format fields (e.g. {"action": ...} looks like a field
    # named "action").
    return (
        "You are a focused video-finding assistant inside a distraction-free YouTube "
        "kiosk. There is no search bar, homepage, or recommendation feed here — you "
        "are the only way to find a video. The user tells you what they want to "
        "watch. Ask short, specific clarifying questions ONLY when the answer would "
        "actually change which video is right (brand, skill level, exact topic) — "
        f"don't ask about things that don't matter, and ask at most {max_questions} "
        "questions total. Once you're confident, or once you've hit that limit, "
        "issue a search. When SEARCH_RESULTS are shown to you, pick the single best "
        "match by its index — never invent a video that isn't in that list. If none "
        "of the results fit, you may search again with a revised query instead of "
        "picking a bad match.\n\n" + _DECISION_SHAPES
    )


@dataclass
class WatchSession:
    config: Config
    client: OllamaClient
    history: list[dict[str, str]] = field(default_factory=list)
    turns: int = 0
    last_results: list[dict[str, str]] = field(default_factory=list)
    done: bool = False

    def __post_init__(self) -> None:
        max_questions = int(self.config.get("watch", "max_questions", default=3))
        self.history.append({"role": "system", "content": _build_system_prompt(max_questions)})

    @property
    def max_turns(self) -> int:
        return int(self.config.get("watch", "max_turns", default=6))

    @property
    def result_limit(self) -> int:
        return int(self.config.get("watch", "result_limit", default=8))

    @property
    def model(self) -> str:
        return self.config.get("watch", "model", default=None) or self.config.resolved_tier.text_model

    # --- public entry points -------------------------------------------------

    def start(self, request: str) -> dict[str, Any]:
        self.history.append({"role": "user", "content": request})
        return self._advance()

    def reply(self, message: str) -> dict[str, Any]:
        if self.done:
            return {"type": "error", "text": "This session already finished — start a new one."}
        self.history.append({"role": "user", "content": message})
        return self._advance()

    # --- internal loop ---------------------------------------------------------

    def _advance(self) -> dict[str, Any]:
        while self.turns < self.max_turns:
            self.turns += 1
            decision = self._decide()
            action = decision.get("action")

            if action == "ask":
                question = str(decision.get("question") or "What would you like to watch?").strip()
                self.history.append({"role": "assistant", "content": json.dumps(decision)})
                return {"type": "question", "text": question}

            if action == "search":
                query = str(decision.get("query") or "").strip()
                self.history.append({"role": "assistant", "content": json.dumps(decision)})
                if not query:
                    self.history.append({"role": "user", "content": "That query was empty — try again."})
                    continue
                self._run_search(query)
                continue

            if action == "play":
                video = self._pick(decision.get("index"))
                self.history.append({"role": "assistant", "content": json.dumps(decision)})
                if video is None:
                    self.history.append({
                        "role": "user",
                        "content": "That index wasn't valid — pick a real index from the numbered "
                        "SEARCH_RESULTS, or issue a new search.",
                    })
                    continue
                self.done = True
                return self._play_response(video, self.last_results)

            # Unrecognized/malformed decision (or an LLM call failure) — nudge
            # and retry, still bounded by max_turns so this always converges.
            self.history.append({
                "role": "user",
                "content": "Reply with only the JSON object described, in one of the three exact shapes.",
            })

        return self._fallback()

    def _run_search(self, query: str) -> None:
        result = search_youtube(query, limit=self.result_limit)
        if not result.get("ok"):
            self.last_results = []
            self.history.append({
                "role": "user",
                "content": f"Search failed: {result.get('error')}. Try a different query or ask a "
                "clarifying question instead.",
            })
            return
        self.last_results = result["results"]
        self.history.append({"role": "user", "content": self._format_results(self.last_results)})

    def _decide(self) -> dict[str, Any]:
        try:
            message = self.client.chat_once(self.model, self.history, temperature=0.3, format="json")
        except (OllamaConnectionError, OllamaModelError) as e:
            logger.warning("watch: LLM call failed: %s", e)
            return {}
        try:
            parsed = json.loads(message.content)
            return parsed if isinstance(parsed, dict) else {}
        except (json.JSONDecodeError, TypeError):
            return {}

    def _pick(self, index: Any) -> dict[str, str] | None:
        if not isinstance(index, int) or isinstance(index, bool) or not self.last_results:
            return None
        if 0 <= index < len(self.last_results):
            return self.last_results[index]
        return None

    def _fallback(self) -> dict[str, Any]:
        """Never leaves the user stuck with nothing playing: if the turn
        budget runs out, play the top hit of whatever was last searched, or
        as a last resort search the user's original request directly."""
        self.done = True
        if self.last_results:
            return self._play_response(self.last_results[0], self.last_results)

        original = next((m["content"] for m in self.history if m["role"] == "user"), "")
        result = search_youtube(original, limit=self.result_limit)
        if result.get("ok") and result["results"]:
            return self._play_response(result["results"][0], result["results"])
        return {"type": "error", "text": "Couldn't find a video for that — try rephrasing what you want."}

    @staticmethod
    def _format_results(results: list[dict[str, str]]) -> str:
        lines = [f"SEARCH_RESULTS ({len(results)}):"]
        for i, r in enumerate(results):
            meta = " — ".join(x for x in (r.get("channel"), r.get("duration")) if x)
            lines.append(f"{i}. {r['title']}" + (f" ({meta})" if meta else ""))
        return "\n".join(lines)

    @staticmethod
    def _play_response(video: dict[str, str], pool: list[dict[str, str]]) -> dict[str, Any]:
        """`candidates` orders the chosen video first, followed by the rest of
        the real, already-scraped results — so if playback of the chosen one
        fails client-side (e.g. the uploader disabled embedding, a common
        case for official music videos), the frontend can fall through to
        the next real match without a distracting error page or another
        round trip to the model."""
        chosen_id = video["video_id"]
        ordered = [video] + [v for v in pool if v["video_id"] != chosen_id]
        return {
            "type": "playing",
            "video_id": chosen_id,
            "title": video.get("title", ""),
            "channel": video.get("channel", ""),
            "candidates": [
                {"video_id": v["video_id"], "title": v.get("title", ""), "channel": v.get("channel", "")}
                for v in ordered
            ],
        }


# --- session registry (in-memory, one per kiosk window, mirrors the chess
# session pattern in webui/server.py) -----------------------------------------

_sessions: dict[str, WatchSession] = {}


def create_session(config: Config) -> str:
    session_id = uuid.uuid4().hex
    _sessions[session_id] = WatchSession(config=config, client=OllamaClient(host=config.ollama_host))
    return session_id


def get_session(session_id: str) -> WatchSession | None:
    return _sessions.get(session_id)


# --- command classifier -----------------------------------------------------
#
# The kiosk has no buttons, sliders, or keyboard shortcuts — once a video is
# playing, every typed message is either a new search or a request to
# control something already on screen (volume, mute, pause, close a pane,
# change the split layout, resize a pane's share of the screen). This is
# the router that decides which, and — since panes are addressed by
# position/title/"this" rather than an id the user can see — which pane it's
# about. It is deliberately stateless (unlike WatchSession): each call gets
# a fresh snapshot of every pane's current state and decides in one shot.

_COMMAND_SHAPES = (
    'Respond with ONLY a JSON array of one or more action objects, no other text, '
    'each in exactly one of these shapes:\n'
    '{"action": "volume", "pane": <index>, "direction": "up" or "down"}\n'
    '{"action": "mute", "pane": <index>}\n'
    '{"action": "unmute", "pane": <index>}\n'
    '{"action": "playback", "pane": <index>, "command": "play" or "pause"}\n'
    '{"action": "close", "pane": <index>}\n'
    '{"action": "layout", "screen_count": 1, 2, 3, or 4}\n'
    '{"action": "resize", "pane": <index>, "direction": "bigger" or "smaller"}\n'
    '{"action": "search", "pane": "new" or <index>}\n'
    '\n'
    "Examples:\n"
    '"turn it up" -> [{"action": "volume", "pane": 0, "direction": "up"}]\n'
    '"mute the recipe video" (pane 1 titled a recipe) -> [{"action": "mute", "pane": 1}]\n'
    '"split into 3" -> [{"action": "layout", "screen_count": 3}]\n'
    '"pause everything" (panes 0 and 1 both playing) -> '
    '[{"action": "playback", "pane": 0, "command": "pause"}, '
    '{"action": "playback", "pane": 1, "command": "pause"}]\n'
    '"play some lofi music" -> [{"action": "search", "pane": "new"}]\n'
)

_FALLBACK_ACTIONS: list[dict[str, Any]] = [{"action": "search", "pane": "new"}]


def _describe_pane(pane: dict[str, Any]) -> str:
    if not pane.get("has_video"):
        status = "empty"
    elif pane.get("playing"):
        status = "playing"
    else:
        status = "paused"
    audio = "muted" if pane.get("muted") else f"volume {pane.get('volume', 100)}"
    title = pane.get("title") or ""
    return f'  {pane.get("index")}: {status} — "{title}" ({audio})'


def _build_command_prompt(
    panes_state: list[dict[str, Any]], last_active_pane: int | None, layout: int
) -> str:
    lines = [
        "You are the control layer for a video-wall kiosk with no buttons, sliders, or "
        "keyboard shortcuts — every action happens through what the user types here. "
        "Decide what the user's message means and translate it into actions.",
        "",
        f"Current layout: {layout} pane(s) on screen.",
        f"Most recently active pane: {last_active_pane if last_active_pane is not None else 'none yet'}.",
        "Panes:",
    ]
    lines.extend(_describe_pane(p) for p in panes_state)
    lines.append(
        "\nUse \"search\" whenever the message describes what to WATCH rather than "
        "controlling something already on screen — that covers both a brand new video "
        "and replacing what's in a specific pane. Set its \"pane\" to a specific index "
        "only when the user clearly wants to replace that pane's video; otherwise \"new\". "
        "Resolve pane references from context: a position (\"the second one\"), a title "
        "match (\"the recipe video\"), or \"this\"/\"here\"/no reference at all meaning the "
        "most recently active pane. If the user means every pane at once (\"mute "
        "everything\", \"pause all\"), return one action per pane that currently has a video."
    )
    return "\n".join(lines) + "\n\n" + _COMMAND_SHAPES


def classify_command(
    text: str,
    panes_state: list[dict[str, Any]],
    last_active_pane: int | None,
    layout: int,
    config: Config,
) -> list[dict[str, Any]]:
    """Never raises, and never returns an empty list — an empty/unclear
    classification degrades to treating the message as a fresh search
    (`_FALLBACK_ACTIONS`) rather than silently doing nothing."""
    if not panes_state or not any(p.get("has_video") for p in panes_state):
        return _FALLBACK_ACTIONS

    client = OllamaClient(host=config.ollama_host)
    # Classification is a much smaller task than search/clarify reasoning —
    # picking one of ~8 shapes from a short, structured prompt — so it's a
    # good candidate for a separate, smaller/faster model. Falls back to the
    # same model search uses if no override is configured.
    model = (
        config.get("watch", "classify_model", default=None)
        or config.get("watch", "model", default=None)
        or config.resolved_tier.text_model
    )
    system = _build_command_prompt(panes_state, last_active_pane, layout)

    try:
        message = client.chat_once(
            model,
            [{"role": "system", "content": system}, {"role": "user", "content": text}],
            temperature=0.2,
            format="json",
        )
    except (OllamaConnectionError, OllamaModelError) as e:
        logger.warning("watch: command classify failed: %s", e)
        return _FALLBACK_ACTIONS

    try:
        parsed = json.loads(message.content)
    except (json.JSONDecodeError, TypeError):
        return _FALLBACK_ACTIONS

    if isinstance(parsed, dict):
        parsed = [parsed]
    if not isinstance(parsed, list):
        return _FALLBACK_ACTIONS

    actions = [a for a in parsed if isinstance(a, dict) and isinstance(a.get("action"), str)]
    return actions or _FALLBACK_ACTIONS
