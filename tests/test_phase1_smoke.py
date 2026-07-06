from __future__ import annotations

import pytest

from deskbot import cli, paths
from deskbot.agent import Agent
from deskbot.config import load_config
from deskbot.llm import ChatChunk, OllamaClient
from deskbot.memory import Memory
from deskbot.persona import Persona, load_persona, save_persona


def test_config_seeds_defaults_and_loads():
    config = load_config()
    assert paths.CONFIG_PATH.exists()
    assert paths.PERSONAS_DIR.joinpath("friend.yaml").exists()
    assert config.default_persona == "friend"
    assert config.resolved_tier.text_model


def test_persona_roundtrip():
    load_config()  # seeds defaults, including personas dir
    persona = Persona(
        name="tester",
        role="A QA persona",
        tone="dry",
        greeting_style="Nods",
        sample_phrases=["works on my machine"],
        boundaries=["never lies about test results"],
        humor_level="low",
        language_mix="English",
    )
    save_persona(persona)
    loaded = load_persona("tester")
    assert loaded.name == "tester"
    assert loaded.role == "A QA persona"
    assert loaded.sample_phrases == ["works on my machine"]
    assert "never lies about test results" in loaded.system_prompt()


def test_memory_session_resume_and_ordering():
    mem = Memory()
    sid1 = mem.get_or_create_session("friend", resume=True)
    mem.add_message(sid1, "user", "hello")
    mem.add_message(sid1, "assistant", "hi there")

    sid2 = mem.get_or_create_session("friend", resume=True)
    assert sid1 == sid2

    messages = mem.get_messages(sid2, limit=10)
    assert [m.role for m in messages] == ["user", "assistant"]
    assert [m.content for m in messages] == ["hello", "hi there"]


def test_memory_contact_notes():
    mem = Memory()
    mem.add_contact_note("contact-1", "prefers email over calls")
    notes = mem.get_contact_notes("contact-1")
    assert notes == ["prefers email over calls"]


def test_agent_one_shot_streams_and_persists(monkeypatch):
    load_config()

    def fake_stream(self, model, messages, temperature=0.4, tools=None):
        yield ChatChunk(content="Hey ", done=False)
        yield ChatChunk(content="friend!", done=True)

    monkeypatch.setattr(OllamaClient, "chat_stream", fake_stream)

    config = load_config()
    agent = Agent(config, memory=Memory())
    reply = agent.one_shot("say hi", persona_name="friend")

    assert reply == "Hey friend!"


def test_cli_persona_list_and_create(monkeypatch, capsys):
    load_config()
    rc = cli.main(["persona", "list"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "friend" in out
