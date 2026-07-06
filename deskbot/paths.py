"""Central definition of every on-disk location deskbot uses.

deskbot is launched from arbitrary directories, so nothing here may depend on
the current working directory. All persistent state lives under the user's
home directory (~/.deskbot); the installed package only ships read-only
defaults that get copied out on first run.
"""

from __future__ import annotations

from pathlib import Path

HOME_DIR = Path.home() / ".deskbot"
PERSONAS_DIR = HOME_DIR / "personas"
ROUTINES_DIR = HOME_DIR / "routines"
GAME_PROFILES_DIR = HOME_DIR / "game_profiles"
LOGS_DIR = HOME_DIR / "logs"
DB_PATH = HOME_DIR / "deskbot.db"
CONFIG_PATH = HOME_DIR / "config.yaml"

PACKAGE_DIR = Path(__file__).resolve().parent
DEFAULTS_DIR = PACKAGE_DIR / "defaults"
DEFAULT_CONFIG_PATH = DEFAULTS_DIR / "config.yaml"
DEFAULT_PERSONAS_DIR = DEFAULTS_DIR / "personas"


def ensure_dirs() -> None:
    for d in (HOME_DIR, PERSONAS_DIR, ROUTINES_DIR, GAME_PROFILES_DIR, LOGS_DIR):
        d.mkdir(parents=True, exist_ok=True)
