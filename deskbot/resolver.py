"""Instant open/launch resolver — a zero-LLM-call fast path for unambiguous
commands like "open youtube" or "launch notepad", checked before the full
agentic tool-loop runs. It only ever fires on an exact, unambiguous match
(a known site alias, or a real installed app after an explicit "open"/
"launch" trigger word); anything else returns None and falls through to the
normal LLM-driven path unchanged.

Deliberately does not handle compound commands ("search youtube for cooking
tutorials") — those need multiple steps (navigate, type, submit) chosen
based on what's actually on the page, which is exactly what the existing
agent tool-loop (browse + type + click) already does well. This module only
short-circuits the single-action case: go straight to a known destination.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from deskbot.config import Config
from deskbot.tools.apps import ALIASES as APP_ALIASES
from deskbot.tools.apps import resolve_app

_TRIGGER_RE = re.compile(r"^(?:open|launch|go to|navigate to|visit)\s+(.+)$")

# Fallback used only if quick_open.sites is absent from config.yaml.
_DEFAULT_SITES = {
    "youtube": "https://youtube.com",
    "gmail": "https://mail.google.com",
    "github": "https://github.com",
    "google": "https://google.com",
    "maps": "https://maps.google.com",
    "google maps": "https://maps.google.com",
    "twitter": "https://twitter.com",
    "x": "https://twitter.com",
    "reddit": "https://reddit.com",
    "amazon": "https://amazon.com",
    "netflix": "https://netflix.com",
    "spotify": "https://open.spotify.com",
    "outlook": "https://outlook.com",
    "wikipedia": "https://wikipedia.org",
    "whatsapp": "https://web.whatsapp.com",
}


@dataclass
class QuickAction:
    tool: str  # registered ToolRegistry name to invoke — "browse" or "open_app"
    args: dict[str, Any]
    label: str  # human-readable description, echoed to the user before running


def _site_aliases(config: Config) -> dict[str, str]:
    configured = config.get("quick_open", "sites", default=None)
    table = configured if configured else _DEFAULT_SITES
    return {str(k).strip().lower(): v for k, v in table.items()}


def resolve_quick_action(text: str, config: Config) -> QuickAction | None:
    """Returns a QuickAction if `text` is an unambiguous open/launch command,
    else None. Never raises — a resolution failure just means "not a quick
    action", not an error."""
    normalized = text.strip().lower().rstrip("?.! ")
    if not normalized:
        return None

    match = _TRIGGER_RE.match(normalized)
    if match:
        target = match.group(1).strip()
    elif normalized in _site_aliases(config) or normalized in APP_ALIASES:
        # Bare alias with no trigger word, e.g. the whole input is just
        # "youtube" or just "chrome" — still unambiguous, so it counts.
        target = normalized
    else:
        return None

    if not target:
        return None

    sites = _site_aliases(config)
    if target in sites:
        return QuickAction("browse", {"url": sites[target]}, f"Opening {target}")

    # Only worth the filesystem/registry lookup once there's an explicit
    # open/launch intent (trigger word matched, or a known app alias) —
    # not for every short phrase that happens to reach this point.
    if match is not None or target in APP_ALIASES:
        if target in APP_ALIASES or resolve_app(target) is not None:
            return QuickAction("open_app", {"name": target}, f"Launching {target}")

    return None
