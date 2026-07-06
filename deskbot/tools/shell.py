"""run_shell(cmd) — the general-purpose shell tool, gated by the SAFE /
CAUTION / DESTRUCTIVE tiers in safety.py. Runs everything through
`powershell -NoProfile -NonInteractive -Command <cmd>` so the LLM can use
normal PowerShell/cmd syntax without deskbot needing to parse it.
"""

from __future__ import annotations

import logging
import subprocess
import sys
from typing import Any, Callable

from rich.console import Console

from deskbot.config import Config
from deskbot.tools.safety import SafetyTier, classify_command

console = Console()
logger = logging.getLogger("deskbot.tools.shell")

OUTPUT_CAP = 4000
DEFAULT_TIMEOUT_SECONDS = 30


def make_run_shell(config: Config, timeout: float = DEFAULT_TIMEOUT_SECONDS) -> Callable[[str], dict[str, Any]]:
    def run_shell(cmd: str) -> dict[str, Any]:
        tier = classify_command(cmd, config)

        if tier == SafetyTier.DESTRUCTIVE:
            if not sys.stdin.isatty():
                # No one is at a terminal to approve this (e.g. a scheduled routine
                # run via Task Scheduler) — never block forever, never auto-run.
                logger.warning("DESTRUCTIVE command auto-declined (non-interactive session): %s", cmd)
                return {
                    "ok": False,
                    "tier": tier.value,
                    "error": "Auto-declined: destructive commands require interactive y/n approval, "
                    "and this is running non-interactively (no terminal attached).",
                }
            console.print(f"\n[bold red]DESTRUCTIVE command requested by the agent:[/bold red]\n  {cmd}")
            answer = console.input("[bold red]Run this? [y/N]:[/bold red] ").strip().lower()
            if answer != "y":
                logger.warning("DESTRUCTIVE command declined by user: %s", cmd)
                return {"ok": False, "tier": tier.value, "error": "User declined to run this destructive command."}
            logger.warning("DESTRUCTIVE command approved and executed: %s", cmd)
        elif tier == SafetyTier.CAUTION:
            logger.warning("CAUTION command executed: %s", cmd)
        else:
            logger.info("SAFE command executed: %s", cmd)

        proc = subprocess.run(
            ["powershell", "-NoProfile", "-NonInteractive", "-Command", cmd],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return {
            "ok": proc.returncode == 0,
            "tier": tier.value,
            "returncode": proc.returncode,
            "stdout": proc.stdout[-OUTPUT_CAP:],
            "stderr": proc.stderr[-OUTPUT_CAP:],
        }

    return run_shell


RUN_SHELL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "run_shell",
        "description": (
            "Run a PowerShell command on the user's machine. Commands are auto-classified "
            "SAFE/CAUTION/DESTRUCTIVE from config.yaml; DESTRUCTIVE commands pause for the "
            "user's y/n confirmation in the terminal before running."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "cmd": {"type": "string", "description": "The PowerShell command to execute"},
            },
            "required": ["cmd"],
        },
    },
}
