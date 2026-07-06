"""open_app(name) — resolves an installed application by friendly name and
launches it. No hardcoded install paths: resolution goes through the Windows
App Paths registry, PATH, and a Start Menu shortcut search, in that order,
the same mechanisms Windows itself uses to resolve "chrome" -> the actual
install location regardless of where a given machine put it.
"""

from __future__ import annotations

import logging
import os
import shutil
import sys
from pathlib import Path
from typing import Any

logger = logging.getLogger("deskbot.tools.apps")

# Friendly-name -> canonical executable name. Not a path — just disambiguates
# common spoken names ("edge") from their actual exe name ("msedge.exe").
ALIASES = {
    "chrome": "chrome.exe",
    "google chrome": "chrome.exe",
    "edge": "msedge.exe",
    "microsoft edge": "msedge.exe",
    "notepad": "notepad.exe",
    "explorer": "explorer.exe",
    "file explorer": "explorer.exe",
    "calculator": "calc.exe",
    "calc": "calc.exe",
    "word": "winword.exe",
    "excel": "excel.exe",
    "powerpoint": "powerpnt.exe",
    "paint": "mspaint.exe",
    "terminal": "wt.exe",
    "cmd": "cmd.exe",
    "powershell": "powershell.exe",
    "task manager": "taskmgr.exe",
    "control panel": "control.exe",
}

_START_MENU_DIRS = [
    Path(os.environ.get("ProgramData", r"C:\ProgramData")) / "Microsoft/Windows/Start Menu/Programs",
    Path(os.environ.get("AppData", "")) / "Microsoft/Windows/Start Menu/Programs" if os.environ.get("AppData") else None,
]


def _app_paths_lookup(exe_name: str) -> str | None:
    if sys.platform != "win32":
        return None
    import winreg

    key_path = rf"Software\Microsoft\Windows\CurrentVersion\App Paths\{exe_name}"
    for hive in (winreg.HKEY_CURRENT_USER, winreg.HKEY_LOCAL_MACHINE):
        try:
            with winreg.OpenKey(hive, key_path) as key:
                value, _ = winreg.QueryValueEx(key, None)
                if value and Path(value).exists():
                    return value
        except (FileNotFoundError, OSError):
            continue
    return None


def _start_menu_lookup(name: str) -> str | None:
    needle = name.lower()
    for base in _START_MENU_DIRS:
        if base is None or not base.exists():
            continue
        try:
            for shortcut in base.rglob("*.lnk"):
                if needle in shortcut.stem.lower():
                    return str(shortcut)
        except OSError:
            continue
    return None


def resolve_app(name: str) -> str | None:
    """Best-effort resolution of a friendly app name to something launchable.
    Returns None if nothing matched."""
    candidate = ALIASES.get(name.strip().lower(), name.strip())

    exe_name = candidate if candidate.lower().endswith(".exe") else f"{candidate}.exe"
    found = _app_paths_lookup(exe_name)
    if found:
        return found

    on_path = shutil.which(candidate) or shutil.which(exe_name)
    if on_path:
        return on_path

    shortcut = _start_menu_lookup(candidate)
    if shortcut:
        return shortcut

    return None


def make_open_app():
    def open_app(name: str) -> dict[str, Any]:
        resolved = resolve_app(name)
        if not resolved:
            logger.warning("open_app: could not resolve '%s'", name)
            return {
                "ok": False,
                "error": f"Could not find an installed app matching '{name}'. "
                "Try the exact app name as it appears in the Start menu.",
            }
        try:
            os.startfile(resolved)  # noqa: S606 - intentional, this *is* the app launcher
            logger.info("open_app: launched '%s' -> %s", name, resolved)
            return {"ok": True, "resolved": resolved}
        except OSError as e:
            logger.warning("open_app: failed to launch '%s' (%s): %s", name, resolved, e)
            return {"ok": False, "error": f"Found '{resolved}' but failed to launch it: {e}"}

    return open_app


OPEN_APP_SCHEMA = {
    "type": "function",
    "function": {
        "name": "open_app",
        "description": (
            "Open/launch an installed desktop application by its common name "
            "(e.g. 'chrome', 'edge', 'notepad', 'explorer', 'calculator')."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Common name of the app to open"},
            },
            "required": ["name"],
        },
    },
}
