"""Core agent loop shared by `deskbot`, `deskbot chat`, and `deskbot do`.

Persona chat (no tools) streams tokens directly. Tool-enabled turns (the bare
REPL and `deskbot do`) go through a non-streaming tool-calling loop: the model
either returns a final answer or a tool_calls batch, each tool runs with one
retry and a structured error on failure, and the loop repeats (capped) until
a final answer or the step cap is hit.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Callable

from rich.console import Console

from deskbot.config import Config
from deskbot.llm import OllamaClient, OllamaConnectionError, OllamaModelError
from deskbot.memory import Memory
from deskbot.persona import Persona, load_persona

console = Console()
logger = logging.getLogger("deskbot.agent")

MAX_TOOL_STEPS = 8
STUCK_REPEAT_THRESHOLD = 2


@dataclass
class ToolRegistry:
    """Empty by default (pure persona chat). Phase 2 populates this via
    deskbot.tools.build_tool_registry() for the tool-enabled REPL and `do`."""

    tools: dict[str, Callable[..., Any]] = field(default_factory=dict)
    schemas: list[dict[str, Any]] = field(default_factory=list)
    on_call: Callable[[str, dict[str, Any], dict[str, Any]], None] | None = None

    def register(self, name: str, fn: Callable[..., Any], schema: dict[str, Any]) -> None:
        self.tools[name] = fn
        self.schemas.append(schema)

    def as_ollama_tools(self) -> list[dict[str, Any]]:
        return self.schemas

    def invoke(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        fn = self.tools.get(name)
        if fn is None:
            return {"ok": False, "error": f"Unknown tool '{name}'"}
        last_error: Exception | None = None
        result: dict[str, Any] | None = None
        for attempt in range(2):  # one retry, per the Phase 2 quality bar
            try:
                result = fn(**args)
                break
            except Exception as e:  # noqa: BLE001 - surfaced to the LLM, not fatal
                last_error = e
                logger.warning("Tool '%s' failed on attempt %d: %s", name, attempt + 1, e)
        if result is None:
            result = {"ok": False, "error": f"Tool '{name}' failed after retry: {last_error}"}
        if self.on_call is not None:
            try:
                self.on_call(name, args, result)
            except Exception:  # noqa: BLE001 - a broken recorder must not break the tool call
                logger.exception("on_call hook raised for tool '%s'", name)
        return result


class Agent:
    def __init__(
        self,
        config: Config,
        memory: Memory | None = None,
        client: OllamaClient | None = None,
        tools: ToolRegistry | None = None,
    ):
        self.config = config
        self.memory = memory or Memory()
        self.client = client or OllamaClient(host=config.ollama_host)
        self.tools = tools or ToolRegistry()

    def _history_as_messages(self, persona: Persona, session_id: int) -> list[dict[str, str]]:
        messages = [{"role": "system", "content": persona.system_prompt()}]
        for m in self.memory.get_messages(session_id, limit=self.config.max_history_messages):
            messages.append({"role": m.role, "content": m.content})
        return messages

    def _stream_reply(self, model: str, messages: list[dict[str, str]]) -> str:
        full = ""
        try:
            for chunk in self.client.chat_stream(
                model, messages, temperature=self.config.temperature,
                tools=self.tools.as_ollama_tools() or None,
            ):
                if chunk.content:
                    console.print(chunk.content, end="")
                    full += chunk.content
            console.print()
        except (OllamaConnectionError, OllamaModelError) as e:
            console.print(f"\n[red]LLM error:[/red] {e}")
            raise
        return full

    def _run_agentic_turn(
        self, model: str, messages: list[dict[str, Any]], max_steps: int = MAX_TOOL_STEPS
    ) -> str:
        """Tool-calling loop: model proposes tool_calls or a final answer; each
        tool call runs (with retry) and its result is fed back, repeating until
        a final answer or max_steps is reached. Repeating the identical
        call is treated as stuck and reported back to the model as such."""
        call_counts: dict[str, int] = {}

        for _ in range(max_steps):
            try:
                message = self.client.chat_once(
                    model, messages, temperature=self.config.temperature,
                    tools=self.tools.as_ollama_tools() or None,
                )
            except (OllamaConnectionError, OllamaModelError) as e:
                console.print(f"\n[red]LLM error:[/red] {e}")
                raise

            if not message.tool_calls:
                if message.content:
                    console.print(message.content)
                return message.content

            messages.append({"role": "assistant", "content": message.content, "tool_calls": message.tool_calls})

            for call in message.tool_calls:
                fn = call.get("function", {})
                name = fn.get("name", "")
                args = fn.get("arguments") or {}
                signature = f"{name}:{json.dumps(args, sort_keys=True)}"
                call_counts[signature] = call_counts.get(signature, 0) + 1

                if call_counts[signature] > STUCK_REPEAT_THRESHOLD:
                    result: dict[str, Any] = {
                        "ok": False,
                        "error": (
                            f"'{name}' was already called with these exact arguments "
                            f"{call_counts[signature] - 1} time(s) with no progress. "
                            "Try a different tool, different arguments, or give the user "
                            "your best answer given what you've found so far."
                        ),
                    }
                else:
                    console.print(f"[dim]→ {name}({json.dumps(args)})[/dim]")
                    result = self.tools.invoke(name, args)

                messages.append({"role": "tool", "name": name, "content": json.dumps(result)})

        return "(stopped after the max number of tool steps without a final answer — try a narrower task)"

    # --- public entry points -------------------------------------------------

    def chat_repl(self, persona_name: str, tools_enabled: bool = False) -> None:
        persona = load_persona(persona_name)
        session_id = self.memory.get_or_create_session(persona_name, resume=True)
        model = self.config.resolved_tier.text_model
        use_tools = tools_enabled and bool(self.tools.tools)

        mode = "tools enabled" if use_tools else "pure persona chat, no tools"
        console.print(
            f"[dim]deskbot — persona '{persona_name}' ({mode}) — model {model}. "
            "Type 'exit' or Ctrl+C to quit.[/dim]"
        )

        while True:
            try:
                user_input = console.input("[bold cyan]you>[/bold cyan] ").strip()
            except (EOFError, KeyboardInterrupt):
                console.print("\n[dim]bye![/dim]")
                return

            if not user_input:
                continue
            if user_input.lower() in {"exit", "quit"}:
                console.print("[dim]bye![/dim]")
                return

            self.memory.add_message(session_id, "user", user_input)
            messages = self._history_as_messages(persona, session_id)

            console.print(f"[bold magenta]{persona_name}>[/bold magenta] ", end="" if not use_tools else "\n")
            try:
                reply = self._run_agentic_turn(model, messages) if use_tools else self._stream_reply(model, messages)
            except (OllamaConnectionError, OllamaModelError):
                continue
            self.memory.add_message(session_id, "assistant", reply)

    def one_shot(self, task: str, persona_name: str | None = None, max_steps: int = MAX_TOOL_STEPS) -> str:
        persona_name = persona_name or self.config.default_persona
        persona = load_persona(persona_name)
        session_id = self.memory.create_session(persona_name)
        model = self.config.resolved_tier.text_model
        use_tools = bool(self.tools.tools)

        self.memory.add_message(session_id, "user", task)
        messages = self._history_as_messages(persona, session_id)
        reply = (
            self._run_agentic_turn(model, messages, max_steps=max_steps)
            if use_tools
            else self._stream_reply(model, messages)
        )
        self.memory.add_message(session_id, "assistant", reply)
        return reply
