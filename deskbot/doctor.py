"""`deskbot doctor` — diagnoses the local environment: Ollama reachability and
the resolved text model, config/persona storage, and (Phase 2) that
Playwright is installed and a Chrome/Edge channel is resolvable. Later phases
append their own checks (node sidecar, game profiles, ...) here.
"""

from __future__ import annotations

import shutil
import sys
from dataclasses import dataclass

from rich.console import Console
from rich.table import Table

from deskbot import paths
from deskbot.config import Config, load_config
from deskbot.llm import OllamaClient

console = Console()


@dataclass
class CheckResult:
    name: str
    ok: bool
    detail: str


def _check_python() -> CheckResult:
    ok = sys.version_info >= (3, 11)
    return CheckResult("Python >= 3.11", ok, sys.version.split()[0])


def _check_ollama_binary() -> CheckResult:
    found = shutil.which("ollama") is not None
    return CheckResult("ollama CLI on PATH", found, "found" if found else "not found — run install.ps1")


def _check_ollama_server(config: Config) -> CheckResult:
    client = OllamaClient(host=config.ollama_host)
    up = client.is_up()
    return CheckResult(
        "Ollama server reachable", up, config.ollama_host if up else f"not reachable at {config.ollama_host}"
    )


def _check_model_pulled(config: Config) -> CheckResult:
    client = OllamaClient(host=config.ollama_host)
    tier = config.resolved_tier
    try:
        models = client.list_models()
    except Exception as e:  # noqa: BLE001 - surfaced to the user as a failed check
        return CheckResult(f"Text model '{tier.text_model}' pulled", False, str(e))
    ok = any(tier.text_model in m for m in models)
    return CheckResult(
        f"Text model '{tier.text_model}' pulled",
        ok,
        "present" if ok else f"run: ollama pull {tier.text_model}",
    )


def _check_storage() -> CheckResult:
    try:
        paths.ensure_dirs()
        probe = paths.HOME_DIR / ".write_probe"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
        return CheckResult("~/.deskbot writable", True, str(paths.HOME_DIR))
    except OSError as e:
        return CheckResult("~/.deskbot writable", False, str(e))


def _check_playwright_installed() -> CheckResult:
    try:
        import playwright  # noqa: F401

        return CheckResult("Playwright installed", True, "python package present")
    except ImportError:
        return CheckResult("Playwright installed", False, "run: pip install playwright")


def _check_browser_channel(config: Config) -> CheckResult:
    from deskbot.tools.apps import resolve_app

    engine = config.get("browser", "default_engine", default="edge")
    name = "edge" if str(engine).lower() == "edge" else "chrome"
    resolved = resolve_app(name)
    return CheckResult(
        f"Browser '{name}' found for the browser layer",
        resolved is not None,
        resolved or f"could not resolve '{name}' — install it or set browser.default_engine in config.yaml",
    )


def run_doctor() -> bool:
    config = load_config()
    checks = [
        _check_python(),
        _check_storage(),
        _check_ollama_binary(),
        _check_ollama_server(config),
        _check_model_pulled(config),
        _check_playwright_installed(),
        _check_browser_channel(config),
    ]

    table = Table(title="deskbot doctor")
    table.add_column("Check")
    table.add_column("Status")
    table.add_column("Detail")
    for c in checks:
        status = "[green]OK[/green]" if c.ok else "[red]FAIL[/red]"
        table.add_row(c.name, status, c.detail)
    console.print(table)

    all_ok = all(c.ok for c in checks)
    if not all_ok:
        console.print("[yellow]Some checks failed — see install.ps1 / README troubleshooting.[/yellow]")
    return all_ok
