"""Shared test fixtures: every test gets an isolated ~/.deskbot so nothing
touches the real user home directory or a real Ollama server."""

from __future__ import annotations

import pytest

from deskbot import paths


@pytest.fixture(autouse=True)
def isolated_home(tmp_path, monkeypatch):
    home = tmp_path / ".deskbot"
    monkeypatch.setattr(paths, "HOME_DIR", home)
    monkeypatch.setattr(paths, "PERSONAS_DIR", home / "personas")
    monkeypatch.setattr(paths, "ROUTINES_DIR", home / "routines")
    monkeypatch.setattr(paths, "GAME_PROFILES_DIR", home / "game_profiles")
    monkeypatch.setattr(paths, "LOGS_DIR", home / "logs")
    monkeypatch.setattr(paths, "DB_PATH", home / "deskbot.db")
    monkeypatch.setattr(paths, "CONFIG_PATH", home / "config.yaml")
    yield home
