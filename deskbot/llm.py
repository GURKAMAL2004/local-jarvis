"""Thin client around the local Ollama HTTP API.

Kept deliberately dependency-light (raw `requests`, no ollama-python SDK) so
the tool layer added in later phases can pass `tools=[...]` through the same
chat() call without a client swap.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import requests


class OllamaConnectionError(RuntimeError):
    """Raised when Ollama isn't reachable at the configured host."""


class OllamaModelError(RuntimeError):
    """Raised when Ollama is reachable but the requested model isn't available."""


@dataclass
class ChatChunk:
    content: str
    done: bool
    tool_calls: list[dict[str, Any]] | None = None


@dataclass
class ChatMessage:
    content: str
    tool_calls: list[dict[str, Any]]


class OllamaClient:
    def __init__(self, host: str = "http://localhost:11434", timeout: float = 120.0):
        self.host = host.rstrip("/")
        self.timeout = timeout

    def is_up(self) -> bool:
        try:
            r = requests.get(f"{self.host}/api/tags", timeout=5)
            return r.ok
        except requests.RequestException:
            return False

    def list_models(self) -> list[str]:
        try:
            r = requests.get(f"{self.host}/api/tags", timeout=10)
            r.raise_for_status()
            return [m["name"] for m in r.json().get("models", [])]
        except requests.RequestException as e:
            raise OllamaConnectionError(f"Could not reach Ollama at {self.host}: {e}") from e

    def chat_stream(
        self,
        model: str,
        messages: list[dict[str, str]],
        temperature: float = 0.4,
        tools: list[dict[str, Any]] | None = None,
    ) -> Iterator[ChatChunk]:
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": True,
            "options": {"temperature": temperature},
        }
        if tools:
            payload["tools"] = tools

        try:
            with requests.post(
                f"{self.host}/api/chat", json=payload, stream=True, timeout=self.timeout
            ) as resp:
                if resp.status_code == 404:
                    raise OllamaModelError(
                        f"Model '{model}' not found on Ollama. Run: ollama pull {model}"
                    )
                resp.raise_for_status()
                for line in resp.iter_lines():
                    if not line:
                        continue
                    data = json.loads(line)
                    msg = data.get("message", {})
                    yield ChatChunk(
                        content=msg.get("content", ""),
                        done=data.get("done", False),
                        tool_calls=msg.get("tool_calls") or None,
                    )
        except requests.RequestException as e:
            raise OllamaConnectionError(f"Could not reach Ollama at {self.host}: {e}") from e

    def chat(
        self,
        model: str,
        messages: list[dict[str, str]],
        temperature: float = 0.4,
        tools: list[dict[str, Any]] | None = None,
    ) -> str:
        parts = [c.content for c in self.chat_stream(model, messages, temperature, tools)]
        return "".join(parts)

    def chat_once(
        self,
        model: str,
        messages: list[dict[str, str]],
        temperature: float = 0.4,
        tools: list[dict[str, Any]] | None = None,
    ) -> ChatMessage:
        """Single non-streaming round trip. Used for tool-calling turns, where the
        reply is a small structured tool_calls payload rather than prose worth
        streaming token-by-token."""
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": temperature},
        }
        if tools:
            payload["tools"] = tools

        try:
            resp = requests.post(f"{self.host}/api/chat", json=payload, timeout=self.timeout)
            if resp.status_code == 404:
                raise OllamaModelError(
                    f"Model '{model}' not found on Ollama. Run: ollama pull {model}"
                )
            resp.raise_for_status()
            data = resp.json()
            msg = data.get("message", {})
            return ChatMessage(content=msg.get("content", ""), tool_calls=msg.get("tool_calls") or [])
        except requests.RequestException as e:
            raise OllamaConnectionError(f"Could not reach Ollama at {self.host}: {e}") from e
