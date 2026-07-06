from __future__ import annotations

from deskbot.agent import Agent, ToolRegistry
from deskbot.config import load_config
from deskbot.llm import ChatMessage, OllamaClient, OllamaConnectionError
from deskbot.memory import Memory
from deskbot.research import (
    RESEARCH_MODE_PRESETS,
    ResearchOptions,
    Source,
    _domain,
    _is_blocked_domain,
    _resolve_quick_model,
    _resolve_synthesis_model,
    _slugify,
    gather_sources,
    generate_additional_angles,
    generate_followup_questions,
    prompt_research_setup,
    run_deep_research,
    save_report,
    synthesize_report,
    verify_section,
)


class _FakeBrowserSession:
    def __init__(self, results_by_query=None, results=None, empty_engines=None):
        self._results_by_query = results_by_query
        self._results = results
        self._empty_engines = empty_engines or set()  # engines that always return zero results
        self.queries: list[str] = []
        self.search_calls: list[tuple[str, str]] = []  # (engine, query)
        self.extract_calls = []

    def search(self, query, engine="google"):
        self.queries.append(query)
        self.search_calls.append((engine, query))
        return {"ok": True, "url": f"https://www.google.com/search?q={query}"}

    def list_search_results(self, engine="google", limit=10):
        if engine in self._empty_engines:
            return {"ok": True, "results": []}
        query = self.queries[-1]
        if self._results_by_query is not None:
            results = self._results_by_query.get(query, [])
        else:
            results = self._results or []
        return {"ok": True, "results": results[:limit]}

    def extract_text(self, max_chars=4000, url=None):
        self.extract_calls.append(url)
        text = f"Extracted content from {url}. " * 50
        return {"ok": True, "url": url, "text": text[:max_chars]}


def _agent_with_fake_browser(results=None, results_by_query=None, empty_engines=None, no_verify=False):
    config = load_config()
    if no_verify:
        config._raw.setdefault("research", {})["verify_sections"] = False
    registry = ToolRegistry()
    registry.browser_session = _FakeBrowserSession(  # type: ignore[attr-defined]
        results_by_query=results_by_query, results=results, empty_engines=empty_engines
    )
    return Agent(config, memory=Memory(), tools=registry)


def test_domain_dedup_helper():
    assert _domain("https://www.example.com/page") == "example.com"
    assert _domain("https://sub.example.com/page") == "sub.example.com"
    assert _domain("not a url") == "not a url"


def test_slugify_produces_filesystem_safe_names():
    assert _slugify("Insulated Bottle Vacuum Physics!") == "insulated-bottle-vacuum-physics"
    assert _slugify("") == "research"


def test_gather_sources_dedupes_by_domain_and_respects_max_sources():
    results = [
        {"url": "https://a.com/1", "title": "A1"},
        {"url": "https://a.com/2", "title": "A2 (same domain as A1)"},
        {"url": "https://b.com/1", "title": "B1"},
        {"url": "https://c.com/1", "title": "C1"},
    ]
    agent = _agent_with_fake_browser(results)
    sources = gather_sources(agent, "topic", max_sources=2, per_source_chars=500, max_corpus_chars=80_000)

    assert len(sources) == 2
    assert {s.url for s in sources} == {"https://a.com/1", "https://b.com/1"}  # a.com/2 deduped


def test_gather_sources_respects_corpus_char_budget():
    results = [{"url": f"https://site{i}.com/", "title": f"Site {i}"} for i in range(10)]
    agent = _agent_with_fake_browser(results)
    sources = gather_sources(agent, "topic", max_sources=10, per_source_chars=1000, max_corpus_chars=2500)

    total_chars = sum(len(s.text) for s in sources)
    assert total_chars <= 2500
    assert len(sources) < 10  # budget ran out before reading everything


def test_resolve_synthesis_model_falls_back_when_deep_model_not_pulled(monkeypatch):
    config = load_config()
    config._raw.setdefault("research", {})["deep_model"] = "qwen2.5:14b-instruct-q4_K_M"
    monkeypatch.setattr(OllamaClient, "list_models", lambda self: ["qwen2.5:1.5b-instruct-q4_K_M"])

    agent = Agent(config, memory=Memory(), tools=ToolRegistry())
    model = _resolve_synthesis_model(agent)
    assert model == config.resolved_tier.text_model  # falls back, deep_model absent from list_models()


def test_resolve_synthesis_model_uses_deep_model_when_available(monkeypatch):
    config = load_config()
    config._raw.setdefault("research", {})["deep_model"] = "qwen2.5:14b-instruct-q4_K_M"
    monkeypatch.setattr(
        OllamaClient, "list_models", lambda self: ["qwen2.5:14b-instruct-q4_K_M:latest"]
    )

    agent = Agent(config, memory=Memory(), tools=ToolRegistry())
    model = _resolve_synthesis_model(agent)
    assert model == "qwen2.5:14b-instruct-q4_K_M"


def test_synthesize_report_cites_sources(monkeypatch):
    config = load_config()
    all_calls = []

    def fake_chat(self, model, messages, temperature=0.4, tools=None):
        all_calls.append(messages)
        return "This is a summary [Source 1]."

    monkeypatch.setattr(OllamaClient, "chat", fake_chat)

    config._raw.setdefault("research", {})["verify_sections"] = False
    agent = Agent(config, memory=Memory(), tools=ToolRegistry())
    sources = [Source(url="https://a.com", title="A", text="content about the topic")]
    report = synthesize_report(agent, "topic", sources)

    assert "This is a summary [Source 1]." in report
    assert any("https://a.com" in m["content"] for call in all_calls for m in call)


def test_synthesize_report_covers_every_round_independently(monkeypatch):
    """Regression test: a single giant synthesis call was observed (twice, in
    real runs) to let the last round crowd out earlier ones. Map-reduce
    synthesis must guarantee every round gets its own section regardless."""
    config = load_config()

    def fake_chat(self, model, messages, temperature=0.4, tools=None):
        angle_line = next((m for m in messages if m["role"] == "user"), None)["content"]
        if angle_line.startswith("Angle:"):
            angle = angle_line.split("\n", 1)[0].removeprefix("Angle:").strip()
            return f"Section content about {angle}."
        return "## Introduction\nIntro text.\n\n## Conclusion\nConclusion text."

    monkeypatch.setattr(OllamaClient, "chat", fake_chat)

    config._raw.setdefault("research", {})["verify_sections"] = False
    agent = Agent(config, memory=Memory(), tools=ToolRegistry())
    sources = [
        Source(url="https://a.com", title="A", text="round 1 content", round_label="round one"),
        Source(url="https://b.com", title="B", text="round 2 content", round_label="round two"),
        Source(url="https://c.com", title="C", text="round 3 content", round_label="round three"),
    ]
    report = synthesize_report(agent, "topic", sources)

    assert "Section content about round one." in report
    assert "Section content about round two." in report
    assert "Section content about round three." in report


def test_save_report_writes_markdown_with_sources(tmp_path, monkeypatch):
    monkeypatch.setattr("deskbot.research.Path.home", lambda: tmp_path)
    sources = [Source(url="https://a.com", title="Source A", text="...")]
    path = save_report("my topic", sources, "The report body.")

    assert path.exists()
    content = path.read_text(encoding="utf-8")
    assert "# Research: my topic" in content
    assert "The report body." in content
    assert "[Source A](https://a.com)" in content


def test_generate_followup_questions_parses_numbered_list(monkeypatch):
    config = load_config()
    monkeypatch.setattr(
        OllamaClient, "chat",
        lambda self, model, messages, temperature=0.4, tools=None: (
            "1. What affects real-world speed most?\n"
            "2. How do costs compare?\n"
            "3. What about future standards?\n"
        ),
    )
    agent = Agent(config, memory=Memory(), tools=ToolRegistry())
    sources = [Source(url="https://a.com", title="A", text="some findings")]
    questions = generate_followup_questions(agent, "topic", sources, n=3)

    assert questions == [
        "What affects real-world speed most?",
        "How do costs compare?",
        "What about future standards?",
    ]


def test_generate_followup_questions_returns_empty_when_disabled():
    config = load_config()
    agent = Agent(config, memory=Memory(), tools=ToolRegistry())
    assert generate_followup_questions(agent, "topic", [], n=3) == []
    assert generate_followup_questions(agent, "topic", [Source(url="https://a.com", title="A", text="x")], n=0) == []


def test_gather_sources_respects_exclude_domains():
    results = [
        {"url": "https://a.com/1", "title": "A1"},
        {"url": "https://b.com/1", "title": "B1"},
    ]
    agent = _agent_with_fake_browser(results=results)
    sources = gather_sources(
        agent, "topic", max_sources=5, per_source_chars=500, max_corpus_chars=80_000,
        exclude_domains={"a.com"},
    )
    assert {s.url for s in sources} == {"https://b.com/1"}


def test_is_blocked_domain_matches_domain_and_subdomains():
    blocked = {"pinterest.com"}
    assert _is_blocked_domain("pinterest.com", blocked)
    assert _is_blocked_domain("foo.pinterest.com", blocked)
    assert not _is_blocked_domain("notpinterest.com", blocked)
    assert not _is_blocked_domain("example.com", blocked)


def test_gather_sources_skips_blocked_domains():
    results = [
        {"url": "https://pinterest.com/pin/1", "title": "Some pin"},
        {"url": "https://real-source.com/article", "title": "Real article"},
    ]
    agent = _agent_with_fake_browser(results=results)
    agent.config._raw.setdefault("research", {})["blocked_domains"] = ["pinterest.com"]
    sources = gather_sources(agent, "topic", max_sources=5, per_source_chars=500, max_corpus_chars=80_000)
    assert {s.url for s in sources} == {"https://real-source.com/article"}


def test_gather_sources_falls_back_to_second_engine_when_first_is_empty():
    # Bing is tried first (Google's anti-bot wall is far more aggressive
    # against automated Chromium), so an empty Bing result triggers the
    # fallback to Google here.
    results = [{"url": "https://a.com/1", "title": "A1"}]
    agent = _agent_with_fake_browser(results=results, empty_engines={"bing"})
    sources = gather_sources(agent, "topic", max_sources=5, per_source_chars=500, max_corpus_chars=80_000)

    assert {s.url for s in sources} == {"https://a.com/1"}
    session = agent.tools.browser_session
    assert ("bing", "topic") in session.search_calls
    assert ("google", "topic") in session.search_calls


def test_gather_sources_returns_empty_when_all_engines_empty():
    agent = _agent_with_fake_browser(results=[], empty_engines={"google", "bing"})
    sources = gather_sources(agent, "topic", max_sources=5, per_source_chars=500, max_corpus_chars=80_000)
    assert sources == []


def test_run_deep_research_does_multiple_rounds(monkeypatch):
    initial_results = [{"url": "https://a.com/1", "title": "A1"}]
    followup_results = [{"url": "https://b.com/1", "title": "B1"}]

    agent = _agent_with_fake_browser(
        results_by_query={
            "topic": initial_results,
            "follow-up question 1": followup_results,
            "follow-up question 2": followup_results,
            "follow-up question 3": followup_results,
        },
        no_verify=True,
    )

    def fake_chat(self, model, messages, temperature=0.4, tools=None):
        system_content = next((m["content"] for m in messages if m["role"] == "system"), "")
        user_content = next((m["content"] for m in messages if m["role"] == "user"), "")
        if "still missing" in system_content.lower():
            return "NONE"  # nothing more to dig into — stop the adaptive loop immediately
        if "propose specific" in system_content.lower():
            return "1. follow-up question 1\n2. follow-up question 2\n3. follow-up question 3\n"
        if user_content.startswith("Angle:"):
            return "SECTION_CONTENT"
        return "## Introduction\nFRAMING_CONTENT\n\n## Conclusion\nFRAMING_CONTENT"

    monkeypatch.setattr(OllamaClient, "chat", fake_chat)

    result = run_deep_research(agent, "topic")

    assert result.followup_questions == ["follow-up question 1", "follow-up question 2", "follow-up question 3"]
    round_labels = {s.round_label for s in result.sources}
    assert "Overview: topic" in round_labels
    assert "follow-up question 1" in round_labels
    assert "SECTION_CONTENT" in result.report
    assert "FRAMING_CONTENT" in result.report


def test_run_deep_research_keeps_digging_when_model_reports_a_gap(monkeypatch):
    """The adaptive loop is the actual "human-like" behavior change: it
    shouldn't stop at the planned follow-up count if the model still thinks
    something important is missing."""
    initial_results = [{"url": "https://a.com/1", "title": "A1"}]
    gap_results = [{"url": "https://gap.com/1", "title": "Gap source"}]

    agent = _agent_with_fake_browser(
        results_by_query={"topic": initial_results, "a real gap": gap_results},
        no_verify=True,
    )
    agent.config._raw.setdefault("research", {})["followup_rounds"] = 0  # isolate the adaptive loop

    calls = {"additional_angles": 0}

    def fake_chat(self, model, messages, temperature=0.4, tools=None):
        system_content = next((m["content"] for m in messages if m["role"] == "system"), "")
        user_content = next((m["content"] for m in messages if m["role"] == "user"), "")
        if "still missing" in system_content.lower():
            calls["additional_angles"] += 1
            return "NONE" if calls["additional_angles"] > 1 else "1. a real gap\n"
        if user_content.startswith("Angle:"):
            return "SECTION_CONTENT"
        return "## Introduction\nFRAMING_CONTENT\n\n## Conclusion\nFRAMING_CONTENT"

    monkeypatch.setattr(OllamaClient, "chat", fake_chat)

    result = run_deep_research(agent, "topic")

    round_labels = {s.round_label for s in result.sources}
    assert "a real gap" in round_labels
    assert calls["additional_angles"] == 2  # asked again after digging, then stopped on NONE


def test_generate_additional_angles_stops_on_none(monkeypatch):
    config = load_config()
    monkeypatch.setattr(OllamaClient, "chat", lambda self, model, messages, temperature=0.4, tools=None: "NONE")
    agent = Agent(config, memory=Memory(), tools=ToolRegistry())
    sources = [Source(url="https://a.com", title="A", text="some findings")]
    assert generate_additional_angles(agent, "topic", sources, covered_angles=["Overview: topic"]) == []


def test_generate_additional_angles_parses_new_angles(monkeypatch):
    config = load_config()
    monkeypatch.setattr(
        OllamaClient, "chat",
        lambda self, model, messages, temperature=0.4, tools=None: "1. What about long-term costs?\n2. Any safety recalls?\n",
    )
    agent = Agent(config, memory=Memory(), tools=ToolRegistry())
    sources = [Source(url="https://a.com", title="A", text="some findings")]
    angles = generate_additional_angles(agent, "topic", sources, covered_angles=["Overview: topic"], n=2)
    assert angles == ["What about long-term costs?", "Any safety recalls?"]


def test_generate_additional_angles_returns_empty_without_sources():
    config = load_config()
    agent = Agent(config, memory=Memory(), tools=ToolRegistry())
    assert generate_additional_angles(agent, "topic", [], covered_angles=[]) == []


def test_verify_section_returns_corrected_text(monkeypatch):
    config = load_config()
    monkeypatch.setattr(
        OllamaClient, "chat",
        lambda self, model, messages, temperature=0.4, tools=None: "Corrected section text.",
    )
    agent = Agent(config, memory=Memory(), tools=ToolRegistry())
    sources = [Source(url="https://a.com", title="A", text="the real facts")]
    fixed = verify_section(agent, "some-model", "angle", sources, "Original draft with a made-up claim.")
    assert fixed == "Corrected section text."


def test_verify_section_falls_back_to_draft_on_model_error(monkeypatch):
    config = load_config()

    def raise_error(self, model, messages, temperature=0.4, tools=None):
        raise OllamaConnectionError("down")

    monkeypatch.setattr(OllamaClient, "chat", raise_error)
    agent = Agent(config, memory=Memory(), tools=ToolRegistry())
    sources = [Source(url="https://a.com", title="A", text="the real facts")]
    fixed = verify_section(agent, "some-model", "angle", sources, "Original draft.")
    assert fixed == "Original draft."


def test_resolve_quick_model_override_wins_over_config():
    config = load_config()
    agent = Agent(config, memory=Memory(), tools=ToolRegistry())
    assert _resolve_quick_model(agent, "my-custom-model") == "my-custom-model"
    assert _resolve_quick_model(agent) == agent.config.resolved_tier.text_model


def test_resolve_synthesis_model_override_used_when_pulled(monkeypatch):
    config = load_config()
    monkeypatch.setattr(OllamaClient, "list_models", lambda self: ["my-custom-model:latest"])
    agent = Agent(config, memory=Memory(), tools=ToolRegistry())
    assert _resolve_synthesis_model(agent, "my-custom-model") == "my-custom-model"


def test_resolve_synthesis_model_override_falls_back_when_not_pulled(monkeypatch):
    config = load_config()
    monkeypatch.setattr(OllamaClient, "list_models", lambda self: ["something-else"])
    agent = Agent(config, memory=Memory(), tools=ToolRegistry())
    assert _resolve_synthesis_model(agent, "my-custom-model") == agent.config.resolved_tier.text_model


def test_standard_preset_is_all_none_and_defers_to_config():
    assert RESEARCH_MODE_PRESETS["standard"] == ResearchOptions()


def test_quick_preset_disables_verification_and_adaptive_digging():
    quick = RESEARCH_MODE_PRESETS["quick"]
    assert quick.verify_sections is False
    assert quick.adaptive_digging is False
    assert quick.followup_rounds == 0
    assert quick.max_total_rounds == 1


def test_run_deep_research_quick_mode_does_a_single_round_even_if_model_suggests_more(monkeypatch):
    """The quick preset should short-circuit both the planned follow-ups
    (followup_rounds=0) and the adaptive loop (adaptive_digging=False) even
    if the model would happily propose more angles — the preset, not the
    model's opinion, decides how far a "quick" run goes."""
    agent = _agent_with_fake_browser(results_by_query={"topic": [{"url": "https://a.com/1", "title": "A1"}]})

    def fake_chat(self, model, messages, temperature=0.4, tools=None):
        # If the adaptive loop or follow-up planner ran, it would land here
        # and (wrongly, for "quick") propose more work.
        return "1. more work that should never be requested\n"

    monkeypatch.setattr(OllamaClient, "chat", fake_chat)

    result = run_deep_research(agent, "topic", options=RESEARCH_MODE_PRESETS["quick"])

    assert {s.round_label for s in result.sources} == {"Overview: topic"}
    assert result.followup_questions == []


def test_prompt_research_setup_quick_choice_with_default_models(monkeypatch):
    config = load_config()
    agent = Agent(config, memory=Memory(), tools=ToolRegistry())
    monkeypatch.setattr(OllamaClient, "list_models", lambda self: ["some-model"])

    answers = iter(["1", "", ""])  # method=Quick, planning model blank, writing model blank
    monkeypatch.setattr("builtins.input", lambda *_args: next(answers))

    options = prompt_research_setup(agent)

    assert options.verify_sections is False
    assert options.adaptive_digging is False
    assert options.quick_model is None
    assert options.synthesis_model is None


def test_prompt_research_setup_picks_models_and_supports_same_keyword(monkeypatch):
    config = load_config()
    agent = Agent(config, memory=Memory(), tools=ToolRegistry())
    monkeypatch.setattr(OllamaClient, "list_models", lambda self: ["model-a", "model-b"])

    answers = iter(["3", "model-a", "same"])  # method=Deep, planning=model-a, writing=same as planning
    monkeypatch.setattr("builtins.input", lambda *_args: next(answers))

    options = prompt_research_setup(agent)

    assert options.max_sources == RESEARCH_MODE_PRESETS["deep"].max_sources
    assert options.quick_model == "model-a"
    assert options.synthesis_model == "model-a"


def test_prompt_research_setup_custom_reads_every_field(monkeypatch):
    config = load_config()
    agent = Agent(config, memory=Memory(), tools=ToolRegistry())
    monkeypatch.setattr(OllamaClient, "list_models", lambda self: [])

    answers = iter([
        "4",     # method = Custom
        "5",     # max_sources
        "2000",  # per_source_chars
        "20000", # max_corpus_chars
        "1",     # followup_rounds
        "3",     # max_total_rounds
        "n",     # verify_sections
        "n",     # adaptive_digging
        "",      # planning model (blank)
        "",      # writing model (blank)
    ])
    monkeypatch.setattr("builtins.input", lambda *_args: next(answers))

    options = prompt_research_setup(agent)

    assert options == ResearchOptions(
        max_sources=5, per_source_chars=2000, max_corpus_chars=20000,
        followup_rounds=1, max_total_rounds=3, verify_sections=False, adaptive_digging=False,
    )
