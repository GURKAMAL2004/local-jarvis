"""Playwright-backed browser layer: persistent Chrome/Edge profile (or attach
to the user's real running Chrome via CDP), exposed as tool functions the LLM
drives one step at a time (browse -> extract_text/screenshot -> click -> ...).

The multi-step "see page, pick next action, repeat" loop lives in the agent's
general tool-calling loop (agent.py) — this module only needs to make each
individual action safe to call repeatedly and fail informatively so the LLM
can self-correct (e.g. retry with a different selector after a failed click).
"""

from __future__ import annotations

import datetime
import logging
import random
import time
from pathlib import Path
from typing import Any
from urllib.parse import quote_plus

from deskbot import paths
from deskbot.config import Config

logger = logging.getLogger("deskbot.tools.browser")

DEFAULT_NAV_TIMEOUT_MS = 15_000
DEFAULT_ACTION_TIMEOUT_MS = 8_000
DEFAULT_MIN_HUMAN_DELAY_SECONDS = 0.8
DEFAULT_MAX_HUMAN_DELAY_SECONDS = 2.2

SEARCH_ENGINE_URLS = {
    "google": "https://www.google.com/search?q={q}",
    "bing": "https://www.bing.com/search?q={q}",
    "duckduckgo": "https://duckduckgo.com/?q={q}",
}

# A plain Playwright-launched Chromium sets several tells search engines use
# to distinguish bots from real users: navigator.webdriver = true, the
# "--enable-automation" flag (which also pops the "Chrome is being
# controlled by automated test software" infobar), and a machine-gun request
# cadence no human produces. Google in particular acts on this — showing a
# CAPTCHA/"unusual traffic" interstitial instead of real results, which is
# what breaks deep research. These two mitigations remove the cheap, common
# tells without adding a stealth-plugin dependency; see _human_delay() for
# the request-cadence half of the fix.
_STEALTH_LAUNCH_ARGS = ["--disable-blink-features=AutomationControlled"]
_STEALTH_INIT_SCRIPT = """
Object.defineProperty(navigator, 'webdriver', { get: () => undefined });
Object.defineProperty(navigator, 'languages', { get: () => ['en-US', 'en'] });
Object.defineProperty(navigator, 'plugins', { get: () => [1, 2, 3, 4, 5] });
window.chrome = window.chrome || { runtime: {} };
"""


class BrowserSession:
    """Lazily launches on first use; stays open across tool calls within a
    single deskbot process so navigation state (logins, current page) persists
    the way a human's browser session would."""

    def __init__(self, config: Config):
        self.config = config
        self._playwright = None
        self._context = None
        self._browser = None
        self._page = None
        self._open_pages: list = []
        self.max_open_windows = int(config.get("browser", "max_open_windows", default=3))

    def _profile_dir(self, channel: str) -> Path:
        d = paths.HOME_DIR / "browser_profile" / channel
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _configure_page(self, page) -> None:
        page.set_default_timeout(DEFAULT_ACTION_TIMEOUT_MS)
        page.set_default_navigation_timeout(DEFAULT_NAV_TIMEOUT_MS)

    def _on_new_page(self, page) -> None:
        """Fires when a click (e.g. a target="_blank" link) opens a new tab/window.
        Caps total open pages at max_open_windows instead of letting research
        tasks spawn an unbounded pile of browser windows."""
        if len(self._open_pages) >= self.max_open_windows:
            logger.info("max_open_windows (%d) reached — closing extra tab", self.max_open_windows)
            try:
                page.close()
            except Exception:  # noqa: BLE001 - best-effort
                pass
            return
        self._configure_page(page)
        self._open_pages.append(page)
        self._page = page  # a new tab within budget becomes the active one

    def _is_page_dead(self) -> bool:
        if self._page is None:
            return True
        try:
            return self._page.is_closed()
        except Exception:  # noqa: BLE001 - underlying connection gone entirely
            return True

    def _reset_state(self) -> None:
        for closer in (
            lambda: self._context.close() if self._context else None,
            lambda: self._playwright.stop() if self._playwright else None,
        ):
            try:
                closer()
            except Exception:  # noqa: BLE001 - best-effort cleanup
                pass
        self._page = self._context = self._browser = self._playwright = None
        self._open_pages = []

    def _ensure_started(self) -> None:
        if not self._is_page_dead():
            return
        # Either never launched, or a previous browser/tab crashed mid-session —
        # relaunch fresh rather than handing back a permanently-dead reference.
        if self._playwright is not None:
            logger.warning("Browser session was closed/crashed — relaunching")
        self._reset_state()

        from playwright.sync_api import sync_playwright

        self._playwright = sync_playwright().start()
        cfg = self.config
        cdp_attach = cfg.get("browser", "cdp_attach", default=False)
        engine = cfg.get("browser", "default_engine", default="edge")
        channel = "msedge" if str(engine).lower() == "edge" else "chrome"
        headless = cfg.get("browser", "headless", default=False)
        cdp_port = cfg.get("browser", "cdp_port", default=9222)

        if cdp_attach:
            # Attaching to a real, already-running, already-logged-in browser
            # is the strongest anti-bot-detection option there is — it's
            # literally not an automation-launched process, so none of the
            # launch-time tells below even apply. See README's "Getting
            # blocked by Google?" section for how to set this up.
            self._browser = self._playwright.chromium.connect_over_cdp(f"http://localhost:{cdp_port}")
            self._context = self._browser.contexts[0] if self._browser.contexts else self._browser.new_context()
        else:
            self._context = self._playwright.chromium.launch_persistent_context(
                user_data_dir=str(self._profile_dir(channel)),
                channel=channel,
                headless=bool(headless),
                args=_STEALTH_LAUNCH_ARGS,
                ignore_default_args=["--enable-automation"],
            )
        try:
            self._context.add_init_script(_STEALTH_INIT_SCRIPT)
        except Exception:  # noqa: BLE001 - best-effort, never block a real session over this
            pass
        self._page = self._context.pages[0] if self._context.pages else self._context.new_page()
        self._configure_page(self._page)
        self._open_pages = [self._page]
        # Only pages opened AFTER this point (e.g. target="_blank" clicks) go
        # through the cap — the initial tab is always allowed.
        self._context.on("page", self._on_new_page)

    def _human_delay(self) -> None:
        """A brief randomized pause before each search — real users don't
        fire queries at machine-gun cadence, and that cadence is one of the
        strongest bot signals search engines act on. Set both bounds to 0
        in config to disable."""
        lo = float(self.config.get("browser", "min_human_delay_seconds", default=DEFAULT_MIN_HUMAN_DELAY_SECONDS))
        hi = float(self.config.get("browser", "max_human_delay_seconds", default=DEFAULT_MAX_HUMAN_DELAY_SECONDS))
        if hi <= 0:
            return
        time.sleep(random.uniform(min(lo, hi), hi))

    def close(self) -> None:
        self._reset_state()

    # --- actions, each returns a structured {"ok": ...} dict -------------

    def browse(self, url: str) -> dict[str, Any]:
        self._ensure_started()
        if not url.startswith(("http://", "https://")):
            url = "https://" + url
        try:
            self._page.goto(url, wait_until="domcontentloaded")
            return {"ok": True, "url": self._page.url, "title": self._page.title()}
        except Exception as e:  # noqa: BLE001 - fed back to the LLM, not re-raised
            return {"ok": False, "error": f"Could not open '{url}': {e}"}

    def search(self, query: str, engine: str = "google") -> dict[str, Any]:
        self._ensure_started()
        self._human_delay()
        template = SEARCH_ENGINE_URLS.get(engine.lower(), SEARCH_ENGINE_URLS["google"])
        url = template.format(q=quote_plus(query))
        try:
            self._page.goto(url, wait_until="domcontentloaded")
            return {"ok": True, "url": self._page.url, "title": self._page.title()}
        except Exception as e:  # noqa: BLE001
            return {"ok": False, "error": f"Search failed for '{query}' on {engine}: {e}"}

    def list_search_results(self, engine: str = "google", limit: int = 10) -> dict[str, Any]:
        """Deterministically scrapes real result URLs+titles off the current
        search-results page (call after search()). Used by the deep-research
        pipeline instead of asking the model to invent click targets, which
        small local models do unreliably (they'll hallucinate plausible-looking
        URLs rather than ground themselves in the actual page).

        Bing wraps organic links in click-tracking redirect URLs rather than
        direct hrefs; Google's are direct. Prefer engine="google" when the
        exact destination URL matters."""
        self._ensure_started()
        results: list[dict[str, str]] = []
        try:
            anchors = self._page.locator("li.b_algo h2 a") if engine.lower() == "bing" else self._page.locator("a:has(h3)")
            count = min(anchors.count(), limit)
            for i in range(count):
                a = anchors.nth(i)
                href = a.get_attribute("href") or ""
                if not href.startswith("http"):
                    continue
                try:
                    h3 = a.locator("h3").first
                    title = h3.inner_text() if h3.count() > 0 else a.inner_text()
                except Exception:  # noqa: BLE001
                    title = href
                results.append({"url": href, "title": title.strip() or href})
            return {"ok": True, "results": results}
        except Exception as e:  # noqa: BLE001
            return {"ok": False, "error": f"Could not list search results: {e}"}

    def click(self, target: str) -> dict[str, Any]:
        self._ensure_started()
        try:
            self._page.locator(target).first.click()
            return {"ok": True, "clicked": target, "matched_as": "selector"}
        except Exception:
            pass
        try:
            self._page.get_by_text(target, exact=False).first.click()
            return {"ok": True, "clicked": target, "matched_as": "visible text"}
        except Exception as e:  # noqa: BLE001
            return {
                "ok": False,
                "error": (
                    f"Could not click '{target}' as either a CSS selector or visible text: {e}. "
                    "Try extract_text() to see what's actually on the page, then click exact "
                    "visible text or a more specific selector."
                ),
            }

    def type_text(self, text: str, selector: str | None = None) -> dict[str, Any]:
        self._ensure_started()
        try:
            if selector:
                self._page.locator(selector).first.fill(text)
            else:
                self._page.keyboard.type(text)
            return {"ok": True}
        except Exception as e:  # noqa: BLE001
            return {"ok": False, "error": f"Could not type into '{selector or 'the focused element'}': {e}"}

    def extract_text(self, max_chars: int = 4000, url: str | None = None) -> dict[str, Any]:
        self._ensure_started()
        # Models sometimes pass a url here instead of calling browse() first —
        # honor that instead of erroring, since it's a reasonable thing to ask for.
        if url:
            nav = self.browse(url)
            if not nav.get("ok"):
                return nav
        try:
            text = self._page.inner_text("body")
            return {"ok": True, "url": self._page.url, "text": text[:max_chars]}
        except Exception as e:  # noqa: BLE001
            return {"ok": False, "error": f"Could not extract page text: {e}"}

    def screenshot(self) -> dict[str, Any]:
        self._ensure_started()
        out_dir = paths.HOME_DIR / "screenshots"
        out_dir.mkdir(parents=True, exist_ok=True)
        filename = out_dir / f"{datetime.datetime.now():%Y%m%d-%H%M%S}.png"
        try:
            self._page.screenshot(path=str(filename))
            return {"ok": True, "path": str(filename)}
        except Exception as e:  # noqa: BLE001
            return {"ok": False, "error": f"Screenshot failed: {e}"}

    def fill_form(self, fields: dict[str, str]) -> dict[str, Any]:
        self._ensure_started()
        results: dict[str, str] = {}
        for label, value in fields.items():
            try:
                self._page.get_by_label(label, exact=False).first.fill(value)
                results[label] = "ok (label)"
                continue
            except Exception:
                pass
            try:
                self._page.locator(label).first.fill(value)
                results[label] = "ok (selector)"
            except Exception as e:  # noqa: BLE001
                results[label] = f"failed: {e}"
        ok = all(v.startswith("ok") for v in results.values())
        return {"ok": ok, "results": results}

    def download(self, target: str) -> dict[str, Any]:
        self._ensure_started()
        out_dir = paths.HOME_DIR / "downloads"
        out_dir.mkdir(parents=True, exist_ok=True)
        try:
            with self._page.expect_download() as dl_info:
                if target.startswith(("http://", "https://")):
                    self._page.goto(target)
                else:
                    self.click(target)
            download = dl_info.value
            dest = out_dir / download.suggested_filename
            download.save_as(str(dest))
            return {"ok": True, "path": str(dest)}
        except Exception as e:  # noqa: BLE001
            return {"ok": False, "error": f"Download of '{target}' failed: {e}"}


def build_browser_tool_schemas() -> list[dict[str, Any]]:
    def fn(name: str, description: str, properties: dict[str, Any], required: list[str]) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": name,
                "description": description,
                "parameters": {"type": "object", "properties": properties, "required": required},
            },
        }

    return [
        fn(
            "browse",
            "Navigate the browser to a URL.",
            {"url": {"type": "string", "description": "URL to open (scheme optional)"}},
            ["url"],
        ),
        fn(
            "search",
            "Search the web via a search engine and open the results page.",
            {
                "query": {"type": "string"},
                "engine": {"type": "string", "description": "google | bing | duckduckgo", "default": "google"},
            },
            ["query"],
        ),
        fn(
            "list_search_results",
            "List the real result URLs and titles on the current search-results page "
            "(call after search()). Use this instead of guessing a click target — it "
            "returns the actual links so you can browse() to one directly.",
            {
                "engine": {"type": "string", "description": "google | bing", "default": "google"},
                "limit": {"type": "integer", "default": 10},
            },
            [],
        ),
        fn(
            "click",
            "Click an element on the current page, by CSS selector or visible text.",
            {"target": {"type": "string", "description": "CSS selector or visible text to click"}},
            ["target"],
        ),
        fn(
            "type",
            "Type text into an input on the current page.",
            {
                "text": {"type": "string"},
                "selector": {"type": "string", "description": "Optional CSS selector; omit to type into the focused element"},
            },
            ["text"],
        ),
        fn(
            "extract_text",
            "Extract the visible text of the current page, so you can decide what to do next.",
            {
                "max_chars": {"type": "integer", "default": 4000},
                "url": {"type": "string", "description": "Optional: navigate here first, then extract"},
            },
            [],
        ),
        fn("screenshot", "Take a screenshot of the current page and save it to disk.", {}, []),
        fn(
            "fill_form",
            "Fill multiple form fields at once. Keys are field labels or CSS selectors, values are the text to enter.",
            {"fields": {"type": "object", "description": "label/selector -> value"}},
            ["fields"],
        ),
        fn(
            "download",
            "Download a file, either directly from a URL or by clicking a download link/button (selector or visible text).",
            {"target": {"type": "string"}},
            ["target"],
        ),
    ]


def register_browser_tools(registry, session: BrowserSession) -> None:
    """Registers browse/search/click/type/extract_text/screenshot/fill_form/download
    on the given ToolRegistry, all backed by the same BrowserSession instance."""
    handlers = {
        "browse": lambda url: session.browse(url),
        "search": lambda query, engine="google": session.search(query, engine),
        "list_search_results": lambda engine="google", limit=10: session.list_search_results(engine, limit),
        "click": lambda target: session.click(target),
        "type": lambda text, selector=None: session.type_text(text, selector),
        "extract_text": lambda max_chars=4000, url=None: session.extract_text(max_chars, url),
        "screenshot": lambda: session.screenshot(),
        "fill_form": lambda fields: session.fill_form(fields),
        "download": lambda target: session.download(target),
    }
    schemas = {s["function"]["name"]: s for s in build_browser_tool_schemas()}
    for name, handler in handlers.items():
        registry.register(name, handler, schemas[name])
