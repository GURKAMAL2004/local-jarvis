from __future__ import annotations

import json

import pytest

from deskbot.config import load_config
from deskbot.llm import ChatMessage, OllamaClient
from deskbot.tools import youtube as youtube_module
from deskbot.webui import watch as watch_module
from deskbot.webui.watch import WatchSession, classify_command


# --- deskbot/tools/youtube.py: pure parsing logic --------------------------


@pytest.mark.parametrize(
    "href,expected",
    [
        ("/watch?v=dQw4w9WgXcQ", "dQw4w9WgXcQ"),
        ("https://www.youtube.com/watch?v=abc123&list=PL1", "abc123"),
        ("/watch?other=1&v=xyz_-9", "xyz_-9"),
        ("/shorts/abc123", None),
        ("", None),
    ],
)
def test_extract_video_id(href, expected):
    assert youtube_module._extract_video_id(href) == expected


def test_embed_url_shape():
    url = youtube_module.embed_url("abc123")
    assert url.startswith("https://www.youtube.com/embed/abc123?")
    assert "autoplay=1" in url


def test_search_youtube_rejects_empty_query():
    result = youtube_module.search_youtube("   ")
    assert result == {"ok": False, "error": "Empty search query"}


# --- WatchSession: the clarify -> search -> pick state machine -------------


def _session(monkeypatch, decisions, search_results=None):
    """decisions: list of dicts, one per _decide() call, consumed in order.
    search_results: canned return value for search_youtube (defaults to two
    real-looking, grounded entries)."""
    config = load_config()
    session = WatchSession(config=config, client=OllamaClient(host=config.ollama_host))

    decisions_iter = iter(decisions)

    def fake_chat_once(self, model, messages, temperature=0.4, tools=None, format=None):
        try:
            decision = next(decisions_iter)
        except StopIteration:
            decision = {"action": "ask", "question": "ran out of scripted decisions"}
        return ChatMessage(content=json.dumps(decision), tool_calls=[])

    monkeypatch.setattr(OllamaClient, "chat_once", fake_chat_once)

    canned = search_results if search_results is not None else {
        "ok": True,
        "results": [
            {"video_id": "vid1", "title": "iPhone 12 screen repair tutorial", "channel": "RepairChan", "duration": "12:34"},
            {"video_id": "vid2", "title": "Unrelated cooking video", "channel": "CookChan", "duration": "5:00"},
        ],
    }
    monkeypatch.setattr(watch_module, "search_youtube", lambda query, limit=8: canned)

    return session


def test_ask_pauses_and_returns_a_question(monkeypatch):
    session = _session(monkeypatch, [{"action": "ask", "question": "What brand of phone?"}])
    result = session.start("mobile repair video")
    assert result == {"type": "question", "text": "What brand of phone?"}
    assert session.done is False


def test_search_then_play_returns_a_real_grounded_video(monkeypatch):
    session = _session(monkeypatch, [
        {"action": "search", "query": "iphone 12 screen repair"},
        {"action": "play", "index": 0},
    ])
    result = session.start("mobile repair video")
    assert result["type"] == "playing"
    assert result["video_id"] == "vid1"
    # candidates lists the chosen video first, then the rest of the real
    # results — the frontend's fallback chain if playback of vid1 fails
    # (e.g. embedding disabled).
    assert [c["video_id"] for c in result["candidates"]] == ["vid1", "vid2"]
    assert session.done is True


def test_reply_continues_an_existing_session(monkeypatch):
    session = _session(monkeypatch, [
        {"action": "ask", "question": "What brand?"},
        {"action": "search", "query": "iphone 12 screen repair"},
        {"action": "play", "index": 0},
    ])
    first = session.start("mobile repair video")
    assert first["type"] == "question"

    second = session.reply("iPhone 12")
    assert second["type"] == "playing"
    assert second["video_id"] == "vid1"


def test_reply_after_done_returns_error_without_calling_the_llm(monkeypatch):
    session = _session(monkeypatch, [{"action": "play", "index": 0}])
    # Force it into last_results/done via a search+play round.
    session.last_results = [{"video_id": "vid1", "title": "t", "channel": "c", "duration": ""}]
    session.done = True

    def fail_if_called(self, *a, **k):
        raise AssertionError("must not call the LLM once the session is done")

    monkeypatch.setattr(OllamaClient, "chat_once", fail_if_called)
    result = session.reply("something else")
    assert result["type"] == "error"


def test_play_with_out_of_range_index_retries_then_recovers(monkeypatch):
    session = _session(monkeypatch, [
        {"action": "search", "query": "iphone 12 screen repair"},
        {"action": "play", "index": 99},  # invalid — nudged to retry
        {"action": "play", "index": 1},
    ])
    result = session.start("mobile repair video")
    assert result["type"] == "playing"
    assert result["video_id"] == "vid2"


def test_model_can_never_play_a_video_outside_last_results(monkeypatch):
    """Structural grounding check: even if the model's JSON claims an index,
    _pick() only ever returns an entry that came from a real search_youtube()
    call — there is no path to inventing a video."""
    session = _session(monkeypatch, [
        {"action": "play", "index": 0},  # no search has run yet — last_results is empty
        {"action": "search", "query": "iphone 12 screen repair"},
        {"action": "play", "index": 0},
    ])
    result = session.start("mobile repair video")
    assert result["type"] == "playing"
    assert result["video_id"] in {"vid1", "vid2"}
    assert result["video_id"] in {r["video_id"] for r in session.last_results}


def test_malformed_decision_is_nudged_and_retried(monkeypatch):
    config = load_config()
    session = WatchSession(config=config, client=OllamaClient(host=config.ollama_host))

    responses = iter(["not json at all", json.dumps({"action": "play", "index": 0})])

    def fake_chat_once(self, model, messages, temperature=0.4, tools=None, format=None):
        return ChatMessage(content=next(responses), tool_calls=[])

    monkeypatch.setattr(OllamaClient, "chat_once", fake_chat_once)
    monkeypatch.setattr(watch_module, "search_youtube", lambda query, limit=8: {
        "ok": True, "results": [{"video_id": "vid1", "title": "t", "channel": "c", "duration": ""}],
    })
    session.last_results = [{"video_id": "vid1", "title": "t", "channel": "c", "duration": ""}]

    result = session.start("mobile repair video")
    assert result["type"] == "playing"
    assert result["video_id"] == "vid1"


def test_exhausting_turn_budget_falls_back_to_top_result(monkeypatch):
    """An "ask" decision always returns a question immediately, regardless of
    remaining budget — the cap only bites on the *next* call, once
    self.turns has already reached max_turns and the loop can't even start.
    max_turns=1 makes that a clean two-call scenario: call 1 asks (spending
    the only turn), call 2 finds no budget left and falls straight through
    to the top-result fallback without another LLM call."""
    config = load_config()
    config._raw["watch"] = {"max_turns": 1}
    session = WatchSession(config=config, client=OllamaClient(host=config.ollama_host))

    def fake_chat_once(self, model, messages, temperature=0.4, tools=None, format=None):
        return ChatMessage(content=json.dumps({"action": "ask", "question": "another question?"}), tool_calls=[])

    monkeypatch.setattr(OllamaClient, "chat_once", fake_chat_once)
    # No search ever ran, so the fallback must search the original request directly.
    monkeypatch.setattr(watch_module, "search_youtube", lambda query, limit=8: {
        "ok": True, "results": [{"video_id": "fallback-vid", "title": "t", "channel": "c", "duration": ""}],
    })

    result = session.start("mobile repair video")
    assert result["type"] == "question"

    result = session.reply("still vague")
    assert result["type"] == "playing"
    assert result["video_id"] == "fallback-vid"


def test_search_failure_is_reported_back_to_the_model_not_raised(monkeypatch):
    session = _session(
        monkeypatch,
        [
            {"action": "search", "query": "something obscure"},
            {"action": "ask", "question": "can you clarify?"},
        ],
        search_results={"ok": False, "error": "No video results found for 'something obscure'"},
    )
    result = session.start("something obscure")
    assert result == {"type": "question", "text": "can you clarify?"}
    assert session.last_results == []


# --- classify_command: the no-button control router -------------------------


_PLAYING_PANE = {"index": 0, "title": "iPhone 12 screen repair", "has_video": True, "playing": True, "muted": False, "volume": 100}
_EMPTY_PANE = {"index": 1, "title": "", "has_video": False, "playing": False, "muted": False, "volume": 100}


def _fake_classifier(monkeypatch, response_obj):
    def fake_chat_once(self, model, messages, temperature=0.4, tools=None, format=None):
        return ChatMessage(content=json.dumps(response_obj), tool_calls=[])

    monkeypatch.setattr(OllamaClient, "chat_once", fake_chat_once)


def test_classify_command_with_nothing_playing_always_searches_without_calling_the_llm(monkeypatch):
    def fail_if_called(self, *a, **k):
        raise AssertionError("no pane has a video yet — there's nothing to classify")

    monkeypatch.setattr(OllamaClient, "chat_once", fail_if_called)
    config = load_config()

    actions = classify_command("anything at all", [_EMPTY_PANE], None, 1, config)
    assert actions == [{"action": "search", "pane": "new"}]


def test_classify_command_volume_up(monkeypatch):
    _fake_classifier(monkeypatch, {"action": "volume", "pane": 0, "direction": "up"})
    config = load_config()
    actions = classify_command("turn it up", [_PLAYING_PANE], 0, 1, config)
    assert actions == [{"action": "volume", "pane": 0, "direction": "up"}]


def test_classify_command_prefers_classify_model_over_watch_model(monkeypatch):
    seen_models = []

    def fake_chat_once(self, model, messages, temperature=0.4, tools=None, format=None):
        seen_models.append(model)
        return ChatMessage(content=json.dumps({"action": "mute", "pane": 0}), tool_calls=[])

    monkeypatch.setattr(OllamaClient, "chat_once", fake_chat_once)
    config = load_config()
    config._raw["watch"] = {"model": "big-search-model", "classify_model": "small-fast-model"}

    classify_command("mute it", [_PLAYING_PANE], 0, 1, config)
    assert seen_models == ["small-fast-model"]


def test_classify_command_falls_back_to_watch_model_without_classify_model(monkeypatch):
    seen_models = []

    def fake_chat_once(self, model, messages, temperature=0.4, tools=None, format=None):
        seen_models.append(model)
        return ChatMessage(content=json.dumps({"action": "mute", "pane": 0}), tool_calls=[])

    monkeypatch.setattr(OllamaClient, "chat_once", fake_chat_once)
    config = load_config()
    config._raw["watch"] = {"model": "big-search-model"}

    classify_command("mute it", [_PLAYING_PANE], 0, 1, config)
    assert seen_models == ["big-search-model"]


def test_classify_command_accepts_a_bare_object_not_wrapped_in_an_array(monkeypatch):
    """Small local models don't always follow "always an array" — a bare
    object should still work rather than being discarded."""
    _fake_classifier(monkeypatch, {"action": "mute", "pane": 0})
    config = load_config()
    actions = classify_command("mute that", [_PLAYING_PANE], 0, 1, config)
    assert actions == [{"action": "mute", "pane": 0}]


def test_classify_command_supports_multiple_actions_for_mute_everything(monkeypatch):
    both_playing = [_PLAYING_PANE, {**_PLAYING_PANE, "index": 1, "title": "cooking video"}]
    _fake_classifier(monkeypatch, [{"action": "mute", "pane": 0}, {"action": "mute", "pane": 1}])
    config = load_config()
    actions = classify_command("mute everything", both_playing, 0, 2, config)
    assert actions == [{"action": "mute", "pane": 0}, {"action": "mute", "pane": 1}]


def test_classify_command_falls_back_to_search_on_malformed_json(monkeypatch):
    def fake_chat_once(self, model, messages, temperature=0.4, tools=None, format=None):
        return ChatMessage(content="not json", tool_calls=[])

    monkeypatch.setattr(OllamaClient, "chat_once", fake_chat_once)
    config = load_config()
    actions = classify_command("something ambiguous", [_PLAYING_PANE], 0, 1, config)
    assert actions == [{"action": "search", "pane": "new"}]


def test_classify_command_falls_back_to_search_on_llm_error(monkeypatch):
    from deskbot.llm import OllamaConnectionError

    def raise_error(self, *a, **k):
        raise OllamaConnectionError("down")

    monkeypatch.setattr(OllamaClient, "chat_once", raise_error)
    config = load_config()
    actions = classify_command("turn it up", [_PLAYING_PANE], 0, 1, config)
    assert actions == [{"action": "search", "pane": "new"}]


def test_classify_command_filters_out_entries_without_a_string_action(monkeypatch):
    _fake_classifier(monkeypatch, [{"action": "mute", "pane": 0}, {"pane": 1}, {"action": 5}])
    config = load_config()
    actions = classify_command("mute it", [_PLAYING_PANE], 0, 1, config)
    assert actions == [{"action": "mute", "pane": 0}]


# --- FastAPI routes ----------------------------------------------------------


@pytest.fixture
def client(monkeypatch, tmp_path):
    from deskbot import paths
    from deskbot.webui import server

    monkeypatch.setattr(paths, "HOME_DIR", tmp_path)

    from fastapi.testclient import TestClient

    app = server.create_app(load_config())
    return TestClient(app)


def test_watch_page_serves_html(client):
    r = client.get("/watch")
    assert r.status_code == 200
    assert "composer" in r.text


def test_watch_start_and_message_round_trip(client, monkeypatch):
    decisions = iter([
        {"action": "search", "query": "iphone 12 screen repair"},
        {"action": "play", "index": 0},
    ])

    def fake_chat_once(self, model, messages, temperature=0.4, tools=None, format=None):
        return ChatMessage(content=json.dumps(next(decisions)), tool_calls=[])

    monkeypatch.setattr(OllamaClient, "chat_once", fake_chat_once)
    monkeypatch.setattr(watch_module, "search_youtube", lambda query, limit=8: {
        "ok": True,
        "results": [{"video_id": "vid1", "title": "iPhone 12 screen repair", "channel": "RepairChan", "duration": "10:00"}],
    })

    r = client.post("/api/watch/start", json={"request": "mobile repair video"})
    assert r.status_code == 200
    data = r.json()
    assert data["type"] == "playing"
    assert data["video_id"] == "vid1"
    assert "session_id" in data


def test_watch_message_rejects_unknown_session(client):
    r = client.post("/api/watch/message", json={"session_id": "does-not-exist", "message": "hi"})
    assert r.status_code == 404


def test_watch_command_route_returns_classified_actions(client, monkeypatch):
    _fake_classifier(monkeypatch, {"action": "volume", "pane": 0, "direction": "up"})

    r = client.post("/api/watch/command", json={
        "text": "turn it up",
        "panes": [_PLAYING_PANE],
        "last_active_pane": 0,
        "layout": 1,
    })
    assert r.status_code == 200
    assert r.json() == {"actions": [{"action": "volume", "pane": 0, "direction": "up"}]}


def test_watch_command_route_with_no_video_playing_defaults_to_search(client):
    r = client.post("/api/watch/command", json={
        "text": "anything",
        "panes": [_EMPTY_PANE],
        "last_active_pane": None,
        "layout": 1,
    })
    assert r.status_code == 200
    assert r.json() == {"actions": [{"action": "search", "pane": "new"}]}
