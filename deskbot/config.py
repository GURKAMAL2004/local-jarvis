"""Loads ~/.deskbot/config.yaml (seeding it from package defaults on first run),
and resolves the active RAM tier into concrete Ollama model names."""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import psutil
import yaml

from deskbot import paths


def _total_ram_gb() -> float:
    return psutil.virtual_memory().total / (1024**3)


def _tier_for_ram(ram_gb: float) -> str:
    if ram_gb >= 28:
        return "32gb"
    if ram_gb >= 14:
        return "16gb"
    return "8gb"


def seed_defaults_if_missing() -> None:
    """Copy packaged defaults into ~/.deskbot the first time deskbot runs."""
    paths.ensure_dirs()

    if not paths.CONFIG_PATH.exists():
        shutil.copyfile(paths.DEFAULT_CONFIG_PATH, paths.CONFIG_PATH)

    if paths.DEFAULT_PERSONAS_DIR.exists():
        for src in paths.DEFAULT_PERSONAS_DIR.glob("*.yaml"):
            dest = paths.PERSONAS_DIR / src.name
            if not dest.exists():
                shutil.copyfile(src, dest)


@dataclass
class ModelTier:
    tier: str
    text_model: str
    vision_model: str


class Config:
    def __init__(self, raw: dict[str, Any]):
        self._raw = raw

    @property
    def raw(self) -> dict[str, Any]:
        return self._raw

    def get(self, *keys: str, default: Any = None) -> Any:
        node: Any = self._raw
        for k in keys:
            if not isinstance(node, dict) or k not in node:
                return default
            node = node[k]
        return node

    @property
    def ollama_host(self) -> str:
        return self.get("models", "ollama_host", default="http://localhost:11434")

    @property
    def resolved_tier(self) -> ModelTier:
        selected = self.get("models", "selected_tier", default="auto")
        tier_name = _tier_for_ram(_total_ram_gb()) if selected in (None, "auto") else selected
        tiers = self.get("models", "ram_tiers", default={})
        tier_cfg = tiers.get(tier_name, {})
        return ModelTier(
            tier=tier_name,
            text_model=tier_cfg.get("text", "qwen2.5:7b-instruct-q4_K_M"),
            vision_model=tier_cfg.get("vision", "moondream"),
        )

    @property
    def default_persona(self) -> str:
        return self.get("agent", "default_persona", default="friend")

    @property
    def temperature(self) -> float:
        return float(self.get("agent", "temperature", default=0.4))

    @property
    def max_history_messages(self) -> int:
        return int(self.get("agent", "max_history_messages", default=40))

    @property
    def stream(self) -> bool:
        return bool(self.get("agent", "stream", default=True))

    @property
    def log_level(self) -> str:
        return str(self.get("logging", "level", default="INFO"))


def load_config(path: Path | None = None) -> Config:
    seed_defaults_if_missing()
    cfg_path = path or paths.CONFIG_PATH
    with open(cfg_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    return Config(raw)
