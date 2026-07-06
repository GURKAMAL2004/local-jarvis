from __future__ import annotations

import sys

import pytest
from rich.console import Console

from deskbot.agent import Agent, ToolRegistry
from deskbot.config import load_config
from deskbot.llm import ChatMessage, OllamaClient
from deskbot.memory import Memory
from deskbot.routine_runner import run_routine
from deskbot.routines import Routine, RoutineStep, save_routine
from deskbot.scheduler import UnsupportedScheduleError, cron_to_schtasks_args
from deskbot.teach import teach_routine
from deskbot.tools.shell import make_run_shell


def test_routine_resolved_steps_substitutes_placeholders():
    routine = Routine(
        name="r",
        description="d",
        steps=[RoutineStep(tool="search", args={"query": "{term}", "engine": "google"})],
        placeholders={"term": "bottles"},
    )
    resolved = routine.resolved_steps({})
    assert resolved[0].args == {"query": "bottles", "engine": "google"}

    resolved_override = routine.resolved_steps({"term": "kettles"})
    assert resolved_override[0].args["query"] == "kettles"


def test_routine_resolved_steps_missing_param_raises():
    routine = Routine(name="r", description="d", steps=[RoutineStep(tool="t", args={"x": "{missing}"})])
    with pytest.raises(ValueError):
        routine.resolved_steps({})


def test_teach_records_steps_and_builds_placeholders(monkeypatch):
    load_config()
    registry = ToolRegistry()
    registry.register(
        "fake_search",
        lambda query: {"ok": True, "result": f"results for {query}"},
        {"type": "function", "function": {"name": "fake_search", "parameters": {}}},
    )

    responses = [
        ChatMessage(
            content="", tool_calls=[{"function": {"name": "fake_search", "arguments": {"query": "bottles"}}}]
        ),
        ChatMessage(content="Done searching.", tool_calls=[]),
    ]
    monkeypatch.setattr(OllamaClient, "chat_once", lambda self, *a, **k: responses.pop(0))

    answers = iter(["search for bottles", "y", "search_term"])
    monkeypatch.setattr(Console, "input", lambda self, prompt="": next(answers))

    config = load_config()
    agent = Agent(config, memory=Memory(), tools=registry)
    routine = teach_routine("test-routine", agent)

    assert routine is not None
    assert routine.steps[0].tool == "fake_search"
    assert routine.steps[0].args["query"] == "{search_term}"
    assert routine.placeholders["search_term"] == "bottles"


def test_teach_with_no_tool_calls_saves_nothing(monkeypatch):
    load_config()
    registry = ToolRegistry()
    registry.register(
        "fake_search",
        lambda query: {"ok": True, "result": query},
        {"type": "function", "function": {"name": "fake_search", "parameters": {}}},
    )
    monkeypatch.setattr(
        OllamaClient, "chat_once", lambda self, *a, **k: ChatMessage(content="just an answer", tool_calls=[])
    )
    monkeypatch.setattr(Console, "input", lambda self, prompt="": "do something with no tools")

    config = load_config()
    agent = Agent(config, memory=Memory(), tools=registry)
    routine = teach_routine("empty-routine", agent)
    assert routine is None


def test_run_routine_success(monkeypatch):
    load_config()
    registry = ToolRegistry()
    calls = []
    registry.register(
        "fake_search",
        lambda query: calls.append(query) or {"ok": True},
        {"type": "function", "function": {"name": "fake_search", "parameters": {}}},
    )
    routine = Routine(
        name="run-me",
        description="search",
        steps=[RoutineStep(tool="fake_search", args={"query": "{term}"})],
        placeholders={"term": "bottles"},
    )
    save_routine(routine)

    config = load_config()
    agent = Agent(config, memory=Memory(), tools=registry)
    ok = run_routine("run-me", {"term": "kettles"}, agent)

    assert ok is True
    assert calls == ["kettles"]


def test_run_routine_replans_once_then_succeeds(monkeypatch):
    load_config()
    registry = ToolRegistry()
    registry.register("bad_tool", lambda x: {"ok": False, "error": "bad arg"},
                       {"type": "function", "function": {"name": "bad_tool", "parameters": {}}})
    registry.register("good_tool", lambda y: {"ok": True, "y": y},
                       {"type": "function", "function": {"name": "good_tool", "parameters": {}}})

    routine = Routine(name="replan-ok", description="do thing",
                       steps=[RoutineStep(tool="bad_tool", args={"x": "1"})])
    save_routine(routine)

    monkeypatch.setattr(
        OllamaClient, "chat_once",
        lambda self, *a, **k: ChatMessage(content="", tool_calls=[{"function": {"name": "good_tool", "arguments": {"y": "2"}}}]),
    )

    config = load_config()
    agent = Agent(config, memory=Memory(), tools=registry)
    assert run_routine("replan-ok", {}, agent) is True


def test_run_routine_aborts_when_replan_fails(monkeypatch):
    load_config()
    registry = ToolRegistry()
    registry.register("bad_tool", lambda x: {"ok": False, "error": "bad arg"},
                       {"type": "function", "function": {"name": "bad_tool", "parameters": {}}})

    routine = Routine(name="replan-fail", description="do thing",
                       steps=[RoutineStep(tool="bad_tool", args={"x": "1"})])
    save_routine(routine)

    monkeypatch.setattr(
        OllamaClient, "chat_once",
        lambda self, *a, **k: ChatMessage(content="I can't fix this.", tool_calls=[]),
    )

    config = load_config()
    agent = Agent(config, memory=Memory(), tools=registry)
    assert run_routine("replan-fail", {}, agent) is False


def test_cron_to_schtasks_args_supported_patterns():
    assert cron_to_schtasks_args("30 9 * * *") == ["/sc", "daily", "/st", "09:30"]
    assert cron_to_schtasks_args("0 8 * * 1") == ["/sc", "weekly", "/d", "MON", "/st", "08:00"]
    assert cron_to_schtasks_args("*/15 * * * *") == ["/sc", "minute", "/mo", "15"]
    assert cron_to_schtasks_args("0 */2 * * *") == ["/sc", "hourly", "/mo", "2"]


def test_cron_to_schtasks_args_unsupported_raises():
    with pytest.raises(UnsupportedScheduleError):
        cron_to_schtasks_args("*/5 */3 1 * *")


@pytest.mark.skipif(sys.platform != "win32", reason="run_shell shells out to powershell.exe")
def test_run_shell_destructive_auto_declines_when_noninteractive(monkeypatch):
    config = load_config()
    run_shell = make_run_shell(config)
    monkeypatch.setattr(sys.stdin, "isatty", lambda: False)
    result = run_shell("Remove-Item -Recurse -Force C:\\definitely-not-real-xyz")
    assert result["ok"] is False
    assert "non-interactively" in result["error"]
