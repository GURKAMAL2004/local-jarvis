"""Launches `deskbot watch`: a local FastAPI backend (the same app `deskbot
ui` serves, just pointed at /watch) plus a chromeless Chromium "app window"
aimed at it — no address bar, no tabs, no YouTube search box, nothing but
the video and the chat composer. Uses the same launch_persistent_context
pattern tools/browser.py already uses for the main agent's browser, just
with --app=<url> instead of a normal window.
"""

from __future__ import annotations

import threading
import time

from rich.console import Console

from deskbot import paths
from deskbot.config import Config

console = Console()

_START_TIMEOUT_SECONDS = 10.0


def run_watch_kiosk(config: Config, port: int) -> int:
    import uvicorn

    from deskbot.webui.server import create_app

    url = f"http://127.0.0.1:{port}/watch"
    server_config = uvicorn.Config(create_app(config), host="127.0.0.1", port=port, log_level="warning")
    server = uvicorn.Server(server_config)
    server_thread = threading.Thread(target=server.run, daemon=True)
    server_thread.start()

    deadline = time.time() + _START_TIMEOUT_SECONDS
    while not server.started and time.time() < deadline:
        time.sleep(0.05)
    if not server.started:
        console.print("[red]watch kiosk backend failed to start in time.[/red]")
        return 1

    console.print(f"[bold]deskbot watch[/bold] — {url}")
    console.print("[dim]Close the window to exit.[/dim]")

    try:
        _open_kiosk_window(config, url)
    finally:
        server.should_exit = True
        server_thread.join(timeout=5)

    return 0


def _open_kiosk_window(config: Config, url: str) -> None:
    from playwright.sync_api import sync_playwright

    width = int(config.get("watch", "window_width", default=1280))
    height = int(config.get("watch", "window_height", default=800))
    engine = config.get("browser", "default_engine", default="edge")
    channel = "msedge" if str(engine).lower() == "edge" else "chrome"

    profile_dir = paths.HOME_DIR / "watch_profile"
    profile_dir.mkdir(parents=True, exist_ok=True)

    with sync_playwright() as p:
        context = p.chromium.launch_persistent_context(
            user_data_dir=str(profile_dir),
            channel=channel,
            headless=False,
            # --app=<url> is what strips the address bar/tabs/bookmarks —
            # the closest thing to a dedicated kiosk app Chromium supports
            # without a native Electron/Tauri shell. --test-type suppresses
            # the "unsupported command-line flag" infobar some automated
            # launch environments trigger (e.g. a sandboxed/root shell that
            # forces --no-sandbox) — harmless if it wasn't going to show anyway.
            args=[f"--app={url}", f"--window-size={width},{height}", "--test-type"],
            ignore_default_args=["--enable-automation"],
        )
        closed = threading.Event()
        context.on("close", lambda: closed.set())
        try:
            closed.wait()
        except KeyboardInterrupt:
            pass
        finally:
            try:
                context.close()
            except Exception:  # noqa: BLE001 - browser process may already be gone
                pass
