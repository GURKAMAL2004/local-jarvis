from __future__ import annotations

import pytest

from deskbot.agent import Agent, ToolRegistry
from deskbot.config import load_config
from deskbot.llm import OllamaClient
from deskbot.memory import Memory
from deskbot.resolver import resolve_quick_action
from deskbot.tools import apps as apps_module


# --- pure resolver behavior --------------------------------------------------


def test_bare_site_alias_resolves_to_browse():
    config = load_config()
    action = resolve_quick_action("youtube", config)
    assert action is not None
    assert action.tool == "browse"
    assert action.args == {"url": "https://youtube.com"}


@pytest.mark.parametrize(
    "text",
    [
        "open youtube",
        "  Open YouTube  ",
        "OPEN YOUTUBE!",
        "go to youtube",
        "navigate to youtube",
        "visit youtube",
    ],
)
def test_trigger_prefixed_site_alias_resolves(text):
    config = load_config()
    action = resolve_quick_action(text, config)
    assert action is not None
    assert action.args == {"url": "https://youtube.com"}


def test_bare_app_alias_resolves_without_filesystem_lookup(monkeypatch):
    config = load_config()

    def fail_if_called(name):
        raise AssertionError("resolve_app should not be needed for a known APP_ALIASES entry")

    monkeypatch.setattr(apps_module, "resolve_app", fail_if_called)
    # resolver.py imported resolve_app by name — patch the binding it actually uses.
    import deskbot.resolver as resolver_module

    monkeypatch.setattr(resolver_module, "resolve_app", fail_if_called)

    action = resolve_quick_action("notepad", config)
    assert action is not None
    assert action.tool == "open_app"
    assert action.args == {"name": "notepad"}


def test_trigger_prefixed_unknown_target_falls_through_when_unresolvable(monkeypatch):
    import deskbot.resolver as resolver_module

    monkeypatch.setattr(resolver_module, "resolve_app", lambda name: None)
    config = load_config()

    action = resolve_quick_action("open some nonexistent thing", config)
    assert action is None


def test_trigger_prefixed_real_app_resolves_via_filesystem_lookup(monkeypatch):
    import deskbot.resolver as resolver_module

    monkeypatch.setattr(resolver_module, "resolve_app", lambda name: r"C:\Fake\thing.exe")
    config = load_config()

    action = resolve_quick_action("launch some custom tool", config)
    assert action is not None
    assert action.tool == "open_app"
    assert action.args == {"name": "some custom tool"}


def test_ordinary_conversation_does_not_touch_the_filesystem(monkeypatch):
    import deskbot.resolver as resolver_module

    def fail_if_called(name):
        raise AssertionError("resolve_app must not run for plain conversational text")

    monkeypatch.setattr(resolver_module, "resolve_app", fail_if_called)
    config = load_config()

    assert resolve_quick_action("what's the weather like today", config) is None
    assert resolve_quick_action("hey, how are you doing", config) is None


def test_empty_input_resolves_to_none():
    config = load_config()
    assert resolve_quick_action("", config) is None
    assert resolve_quick_action("   ", config) is None


def test_custom_config_sites_override_defaults():
    config = load_config()
    config._raw["quick_open"] = {"sites": {"work portal": "https://intranet.example.com"}}

    action = resolve_quick_action("open work portal", config)
    assert action is not None
    assert action.args == {"url": "https://intranet.example.com"}

    # Defaults are gone once quick_open.sites is set explicitly, not merged.
    assert resolve_quick_action("youtube", config) is None


# --- integration: the point of this feature is skipping the LLM entirely ----


def test_one_shot_quick_action_never_calls_the_llm(monkeypatch):
    config = load_config()

    def fail_if_called(*args, **kwargs):
        raise AssertionError("a quick action must not trigger any Ollama call")

    monkeypatch.setattr(OllamaClient, "chat_once", fail_if_called)
    monkeypatch.setattr(OllamaClient, "chat_stream", fail_if_called)

    registry = ToolRegistry()
    calls: dict[str, str] = {}

    def fake_browse(url: str) -> dict:
        calls["url"] = url
        return {"ok": True, "url": url, "title": "YouTube"}

    registry.register(
        "browse",
        fake_browse,
        {
            "type": "function",
            "function": {
                "name": "browse",
                "description": "Navigate to a URL.",
                "parameters": {"type": "object", "properties": {"url": {"type": "string"}}, "required": ["url"]},
            },
        },
    )

    agent = Agent(config, memory=Memory(), tools=registry)
    reply = agent.one_shot("open youtube")

    assert calls["url"] == "https://youtube.com"
    assert "Opening youtube" in reply


def test_one_shot_falls_through_to_agentic_loop_when_unresolved(monkeypatch):
    config = load_config()

    from deskbot.llm import ChatMessage

    monkeypatch.setattr(
        OllamaClient,
        "chat_once",
        lambda self, model, messages, temperature=0.4, tools=None: ChatMessage(content="a normal reply", tool_calls=[]),
    )

    registry = ToolRegistry()
    registry.register("browse", lambda url: {"ok": True}, {"type": "function", "function": {"name": "browse", "description": "", "parameters": {"type": "object", "properties": {}, "required": []}}})

    agent = Agent(config, memory=Memory(), tools=registry)
    reply = agent.one_shot("tell me a joke")

    assert reply == "a normal reply"
