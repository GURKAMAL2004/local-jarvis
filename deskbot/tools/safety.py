"""Classifies a shell command into one of three safety tiers, driven entirely
by the pattern lists in config.yaml (safety.destructive_patterns /
safety.caution_patterns) — never hardcoded in code.

SAFE:        auto-runs, no prompt, no special logging
CAUTION:     auto-runs, but logged at WARNING so it's visible after the fact
DESTRUCTIVE: blocks on an explicit y/n confirmation in the terminal
"""

from __future__ import annotations

from enum import Enum

from deskbot.config import Config


class SafetyTier(str, Enum):
    SAFE = "SAFE"
    CAUTION = "CAUTION"
    DESTRUCTIVE = "DESTRUCTIVE"


def classify_command(cmd: str, config: Config) -> SafetyTier:
    lowered = cmd.lower()

    destructive_patterns = config.get("safety", "destructive_patterns", default=[])
    for pattern in destructive_patterns:
        if pattern.lower() in lowered:
            return SafetyTier.DESTRUCTIVE

    caution_patterns = config.get("safety", "caution_patterns", default=[])
    for pattern in caution_patterns:
        if pattern.lower() in lowered:
            return SafetyTier.CAUTION

    return SafetyTier.SAFE
