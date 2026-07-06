from __future__ import annotations

import sys

import pytest

from deskbot.agent import Agent, ToolRegistry
from deskbot.config import load_config
from deskbot.llm import ChatMessage, OllamaClient
from deskbot.memory import Memory
from deskbot.tools.apps import resolve_app
from deskbot.tools.browser import BrowserSession
from deskbot.tools.safety import SafetyTier, classify_command
from deskbot.tools.shell import make_run_shell


class _FakePage:
    def __init__(self):
        self.closed = False

    def set_default_timeout(self, *_a, **_k):
        pass

    def set_default_navigation_timeout(self, *_a, **_k):
        pass

    def close(self):
        self.closed = True


def test_classify_command_tiers():
    config = load_config()
    assert classify_command("Get-ChildItem", config) == SafetyTier.SAFE
    assert classify_command("New-Item foo.txt", config) == SafetyTier.CAUTION
    assert classify_command("Remove-Item -Recurse -Force C:\\temp", config) == SafetyTier.DESTRUCTIVE
    assert classify_command("shutdown /s", config) == SafetyTier.DESTRUCTIVE


@pytest.mark.skipif(sys.platform != "win32", reason="run_shell shells out to powershell.exe")
def test_run_shell_safe_command_executes():
    config = load_config()
    run_shell = make_run_shell(config)
    result = run_shell("Write-Output hello-deskbot")
    assert result["ok"] is True
    assert result["tier"] == "SAFE"
    assert "hello-deskbot" in result["stdout"]


@pytest.mark.skipif(sys.platform != "win32", reason="run_shell shells out to powershell.exe")
def test_run_shell_destructive_declined(monkeypatch):
    config = load_config()
    run_shell = make_run_shell(config)
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)  # force the interactive y/n path
    monkeypatch.setattr("rich.console.Console.input", lambda self, prompt="": "n")
    result = run_shell("Remove-Item -Recurse -Force C:\\definitely-not-real-xyz")
    assert result["ok"] is False
    assert result["tier"] == "DESTRUCTIVE"
    assert "declined" in result["error"]


def test_resolve_app_unknown_returns_none():
    assert resolve_app("definitely-not-a-real-installed-app-xyz-123") is None


@pytest.mark.skipif(sys.platform != "win32", reason="notepad.exe is a Windows-only sanity check")
def test_resolve_app_notepad_found():
    assert resolve_app("notepad") is not None


def test_tool_registry_invoke_retries_then_succeeds():
    registry = ToolRegistry()
    calls = {"count": 0}

    def flaky(x):
        calls["count"] += 1
        if calls["count"] == 1:
            raise RuntimeError("transient failure")
        return {"ok": True, "x": x}

    registry.register("flaky", flaky, {"type": "function", "function": {"name": "flaky", "parameters": {}}})
    result = registry.invoke("flaky", {"x": 1})
    assert result == {"ok": True, "x": 1}
    assert calls["count"] == 2


def test_tool_registry_invoke_unknown_tool():
    registry = ToolRegistry()
    result = registry.invoke("nope", {})
    assert result["ok"] is False
    assert "Unknown tool" in result["error"]


def test_agentic_turn_calls_tool_then_returns_final_answer(monkeypatch):
    load_config()
    registry = ToolRegistry()
    seen_args = []

    def fake_tool(query):
        seen_args.append(query)
        return {"ok": True, "result": f"results for {query}"}

    registry.register(
        "fake_search",
        fake_tool,
        {"type": "function", "function": {"name": "fake_search", "parameters": {}}},
    )

    responses = [
        ChatMessage(
            content="",
            tool_calls=[{"function": {"name": "fake_search", "arguments": {"query": "bottles"}}}],
        ),
        ChatMessage(content="Here is a summary of bottles.", tool_calls=[]),
    ]

    def fake_chat_once(self, model, messages, temperature=0.4, tools=None):
        return responses.pop(0)

    monkeypatch.setattr(OllamaClient, "chat_once", fake_chat_once)

    config = load_config()
    agent = Agent(config, memory=Memory(), tools=registry)
    reply = agent.one_shot("look up bottles", persona_name="friend")

    assert reply == "Here is a summary of bottles."
    assert seen_args == ["bottles"]


def test_agentic_turn_detects_stuck_repeated_calls(monkeypatch):
    load_config()
    registry = ToolRegistry()
    registry.register(
        "always_same",
        lambda: {"ok": True, "result": "same thing"},
        {"type": "function", "function": {"name": "always_same", "parameters": {}}},
    )

    def fake_chat_once(self, model, messages, temperature=0.4, tools=None):
        # The model stubbornly keeps calling the same tool with no args, forever.
        return ChatMessage(content="", tool_calls=[{"function": {"name": "always_same", "arguments": {}}}])

    monkeypatch.setattr(OllamaClient, "chat_once", fake_chat_once)

    config = load_config()
    agent = Agent(config, memory=Memory(), tools=registry)
    reply = agent.one_shot("do the same thing forever", persona_name="friend")

    # Never resolves to a real final answer, so it should hit the step cap message.
    assert "max number of tool steps" in reply


def test_browser_session_caps_open_windows():
    config = load_config()
    session = BrowserSession(config)
    session.max_open_windows = 2
    initial_page = _FakePage()
    session._open_pages = [initial_page]
    session._page = initial_page

    second_page = _FakePage()
    session._on_new_page(second_page)
    assert session._page is second_page
    assert len(session._open_pages) == 2
    assert second_page.closed is False

    third_page = _FakePage()
    session._on_new_page(third_page)
    assert third_page.closed is True  # over the cap - closed immediately
    assert session._page is second_page  # active page unchanged
    assert len(session._open_pages) == 2


def test_one_shot_respects_custom_max_steps(monkeypatch):
    load_config()
    registry = ToolRegistry()
    registry.register(
        "loop_tool",
        lambda: {"ok": True},
        {"type": "function", "function": {"name": "loop_tool", "parameters": {}}},
    )

    call_count = {"n": 0}

    def fake_chat_once(self, model, messages, temperature=0.4, tools=None):
        call_count["n"] += 1
        return ChatMessage(content="", tool_calls=[{"function": {"name": "loop_tool", "arguments": {}}}])

    monkeypatch.setattr(OllamaClient, "chat_once", fake_chat_once)

    config = load_config()
    agent = Agent(config, memory=Memory(), tools=registry)
    reply = agent.one_shot("loop forever", max_steps=3)

    assert call_count["n"] == 3
    assert "max number of tool steps" in reply
