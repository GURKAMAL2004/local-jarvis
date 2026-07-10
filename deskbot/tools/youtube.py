"""Headless, grounded YouTube search for the watch kiosk (deskbot.webui.watch).

Scrapes real search results off youtube.com/results — title, channel,
video_id, duration — the same "code searches, model picks from what's
actually there" pattern research.py and tools/browser.py's
list_search_results already use, so the LLM can never hand back a
hallucinated video that doesn't exist.

Runs its own short-lived headless Chromium instance rather than reusing
BrowserSession: this is a quick in-and-out scrape, not a persistent browsing
session, and it must not disturb whatever the main agent's browser tab is
currently showing.
"""

from __future__ import annotations

import logging
import re
from typing import Any
from urllib.parse import quote_plus

logger = logging.getLogger("deskbot.tools.youtube")

SEARCH_URL = "https://www.youtube.com/results?search_query={q}"
EMBED_URL = "https://www.youtube.com/embed/{video_id}?autoplay=1&fs=1&rel=0&modestbranding=1"

_VIDEO_ID_RE = re.compile(r"[?&]v=([A-Za-z0-9_-]{6,})")

# Common EU/UK consent-wall buttons — best-effort dismiss, never fatal if absent.
_CONSENT_SELECTORS = [
    "button:has-text('Accept all')",
    "button:has-text('Reject all')",
    "button:has-text('I agree')",
]


def embed_url(video_id: str) -> str:
    return EMBED_URL.format(video_id=video_id)


def search_youtube(query: str, limit: int = 8) -> dict[str, Any]:
    """Returns {"ok": True, "results": [{"video_id", "title", "channel",
    "duration"}, ...]} or {"ok": False, "error": ...}. Never raises."""
    query = query.strip()
    if not query:
        return {"ok": False, "error": "Empty search query"}

    try:
        from playwright.sync_api import sync_playwright
    except ImportError as e:  # pragma: no cover - Playwright is a hard dependency of this project
        return {"ok": False, "error": f"Playwright not available: {e}"}

    url = SEARCH_URL.format(q=quote_plus(query))
    try:
        with sync_playwright() as p:
            browser = _launch_headless(p)
            try:
                page = browser.new_page()
                page.set_default_timeout(15_000)
                page.goto(url, wait_until="domcontentloaded")
                _dismiss_consent(page)
                page.wait_for_selector("ytd-video-renderer", timeout=10_000)
                results = _scrape_results(page, limit)
            finally:
                browser.close()
    except Exception as e:  # noqa: BLE001 - fed back to the caller, not fatal
        logger.warning("YouTube search failed for %r: %s", query, e)
        return {"ok": False, "error": f"YouTube search failed for '{query}': {e}"}

    if not results:
        return {"ok": False, "error": f"No video results found for '{query}'"}
    return {"ok": True, "results": results}


def _launch_headless(p):
    """Prefers the system Edge/Chrome install (channel="msedge"/"chrome") —
    the same browser tools/browser.py already launches successfully — over
    Playwright's separately-downloaded headless Chromium, which isn't
    guaranteed to be installed (it needs its own `playwright install` step
    distinct from what a system-channel launch requires)."""
    for channel in ("msedge", "chrome"):
        try:
            return p.chromium.launch(headless=True, channel=channel)
        except Exception as e:  # noqa: BLE001 - try the next channel
            logger.debug("headless launch via channel=%r failed: %s", channel, e)
            continue
    return p.chromium.launch(headless=True)  # last resort: Playwright's bundled Chromium


def _dismiss_consent(page) -> None:
    for selector in _CONSENT_SELECTORS:
        try:
            page.locator(selector).first.click(timeout=1500)
            return
        except Exception:  # noqa: BLE001 - no consent wall present, or already dismissed
            continue


def _scrape_results(page, limit: int) -> list[dict[str, str]]:
    results: list[dict[str, str]] = []
    renderers = page.locator("ytd-video-renderer")
    # Over-fetch slightly since a few entries (ads, shorts shelves) won't
    # parse as a normal video and get skipped below.
    count = min(renderers.count(), max(limit * 2, limit + 4))

    for i in range(count):
        if len(results) >= limit:
            break
        node = renderers.nth(i)
        try:
            link = node.locator("a#video-title").first
            href = link.get_attribute("href") or ""
            video_id = _extract_video_id(href)
            if not video_id:
                continue

            title = (link.get_attribute("title") or link.inner_text() or "").strip()
            if not title:
                continue

            channel = _safe_text(node.locator("ytd-channel-name a").first)
            duration = _safe_text(node.locator("ytd-thumbnail-overlay-time-status-renderer span").first)

            results.append({
                "video_id": video_id,
                "title": title,
                "channel": channel,
                "duration": duration,
            })
        except Exception:  # noqa: BLE001 - skip one malformed result, don't fail the whole search
            continue

    return results


def _safe_text(locator) -> str:
    try:
        return locator.inner_text(timeout=1000).strip()
    except Exception:  # noqa: BLE001 - e.g. live streams have no fixed duration element
        return ""


def _extract_video_id(href: str) -> str | None:
    match = _VIDEO_ID_RE.search(href)
    return match.group(1) if match else None
