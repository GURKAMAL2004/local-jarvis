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
    _score_credibility,
    _slugify,
    extract_factors,
    gather_sources,
    generate_additional_angles,
    generate_followup_questions,
    generate_hypothesis,
    pairwise_relationship_questions,
    prompt_research_setup,
    run_deep_research,
    save_report,
    scientific_relationship_questions,
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

    monkeypatch.setattr("deskbot.research._choose", lambda message, options, default_index=0: "quick")
    monkeypatch.setattr("deskbot.research._ask_model", lambda message, available: "")
    monkeypatch.setattr("deskbot.research._ask_text", lambda message, default="": "")

    options = prompt_research_setup(agent)

    assert options.verify_sections is False
    assert options.adaptive_digging is False
    assert options.quick_model is None
    assert options.synthesis_model is None


def test_prompt_research_setup_picks_models_and_supports_same_keyword(monkeypatch):
    config = load_config()
    agent = Agent(config, memory=Memory(), tools=ToolRegistry())
    monkeypatch.setattr(OllamaClient, "list_models", lambda self: ["model-a", "model-b"])

    monkeypatch.setattr("deskbot.research._choose", lambda message, options, default_index=0: "deep")
    monkeypatch.setattr("deskbot.research._ask_model", lambda message, available: "model-a")
    monkeypatch.setattr("deskbot.research._ask_text", lambda message, default="": "same")

    options = prompt_research_setup(agent)

    assert options.max_sources == RESEARCH_MODE_PRESETS["deep"].max_sources
    assert options.quick_model == "model-a"
    assert options.synthesis_model == "model-a"


def test_prompt_research_setup_custom_reads_every_field(monkeypatch):
    config = load_config()
    agent = Agent(config, memory=Memory(), tools=ToolRegistry())
    monkeypatch.setattr(OllamaClient, "list_models", lambda self: [])

    monkeypatch.setattr("deskbot.research._choose", lambda message, options, default_index=0: "custom")
    confirms = iter([False, False, False])  # factor_analysis, verify_sections, adaptive_digging
    numbers = iter([5, 2000, 20000, 1, 3])  # max_sources, per_source_chars, max_corpus_chars, followup_rounds, max_total_rounds
    monkeypatch.setattr("deskbot.research._ask_confirm", lambda message, default: next(confirms))
    monkeypatch.setattr("deskbot.research._ask_number", lambda message, default: next(numbers))
    monkeypatch.setattr("deskbot.research._ask_model", lambda message, available: "")
    monkeypatch.setattr("deskbot.research._ask_text", lambda message, default="": "")

    options = prompt_research_setup(agent)

    assert options == ResearchOptions(
        max_sources=5, per_source_chars=2000, max_corpus_chars=20000,
        followup_rounds=1, max_total_rounds=3, verify_sections=False, adaptive_digging=False,
        factor_analysis=False,
    )


def test_prompt_research_setup_relentless_choice(monkeypatch):
    config = load_config()
    agent = Agent(config, memory=Memory(), tools=ToolRegistry())
    monkeypatch.setattr(OllamaClient, "list_models", lambda self: [])

    monkeypatch.setattr("deskbot.research._choose", lambda message, options, default_index=0: "relentless")
    monkeypatch.setattr("deskbot.research._ask_model", lambda message, available: "")
    monkeypatch.setattr("deskbot.research._ask_text", lambda message, default="": "")

    options = prompt_research_setup(agent)

    assert options.factor_analysis is True
    assert options.max_total_rounds == RESEARCH_MODE_PRESETS["relentless"].max_total_rounds


def test_prompt_research_setup_custom_factor_analysis_uses_relentless_ceilings(monkeypatch):
    config = load_config()
    agent = Agent(config, memory=Memory(), tools=ToolRegistry())
    monkeypatch.setattr(OllamaClient, "list_models", lambda self: [])

    monkeypatch.setattr("deskbot.research._choose", lambda message, options, default_index=0: "custom")
    confirms = iter([True, False, False])  # factor_analysis, scientific_mode, verify_sections
    numbers = iter([3000, 2])  # per_source_chars, followup_rounds
    monkeypatch.setattr("deskbot.research._ask_confirm", lambda message, default: next(confirms))
    monkeypatch.setattr("deskbot.research._ask_number", lambda message, default: next(numbers))
    monkeypatch.setattr("deskbot.research._ask_model", lambda message, available: "")
    monkeypatch.setattr("deskbot.research._ask_text", lambda message, default="": "")

    options = prompt_research_setup(agent)

    assert options.factor_analysis is True
    assert options.scientific_mode is False
    assert options.max_total_rounds == RESEARCH_MODE_PRESETS["relentless"].max_total_rounds
    assert options.per_source_chars == 3000
    assert options.followup_rounds == 2
    assert options.verify_sections is False


def test_prompt_research_setup_custom_can_enable_scientific_mode(monkeypatch):
    config = load_config()
    agent = Agent(config, memory=Memory(), tools=ToolRegistry())
    monkeypatch.setattr(OllamaClient, "list_models", lambda self: [])

    monkeypatch.setattr("deskbot.research._choose", lambda message, options, default_index=0: "custom")
    confirms = iter([True, True, True])  # factor_analysis, scientific_mode, verify_sections
    numbers = iter([3000, 2])
    monkeypatch.setattr("deskbot.research._ask_confirm", lambda message, default: next(confirms))
    monkeypatch.setattr("deskbot.research._ask_number", lambda message, default: next(numbers))
    monkeypatch.setattr("deskbot.research._ask_model", lambda message, available: "")
    monkeypatch.setattr("deskbot.research._ask_text", lambda message, default="": "")

    options = prompt_research_setup(agent)

    assert options.factor_analysis is True
    assert options.scientific_mode is True


def test_prompt_research_setup_scientist_choice(monkeypatch):
    config = load_config()
    agent = Agent(config, memory=Memory(), tools=ToolRegistry())
    monkeypatch.setattr(OllamaClient, "list_models", lambda self: [])

    monkeypatch.setattr("deskbot.research._choose", lambda message, options, default_index=0: "scientist")
    monkeypatch.setattr("deskbot.research._ask_model", lambda message, available: "")
    monkeypatch.setattr("deskbot.research._ask_text", lambda message, default="": "")

    options = prompt_research_setup(agent)

    assert options.factor_analysis is True
    assert options.scientific_mode is True
    assert options.max_total_rounds == RESEARCH_MODE_PRESETS["scientist"].max_total_rounds


def test_extract_factors_stops_on_none(monkeypatch):
    config = load_config()
    monkeypatch.setattr(OllamaClient, "chat", lambda self, model, messages, temperature=0.4, tools=None: "NONE")
    agent = Agent(config, memory=Memory(), tools=ToolRegistry())
    sources = [Source(url="https://a.com", title="A", text="some findings")]
    assert extract_factors(agent, "topic", sources, covered_factors=[]) == []


def test_extract_factors_parses_and_excludes_covered(monkeypatch):
    config = load_config()
    monkeypatch.setattr(
        OllamaClient, "chat",
        lambda self, model, messages, temperature=0.4, tools=None: "1. Sleep quality\n2. Caffeine dose\n",
    )
    agent = Agent(config, memory=Memory(), tools=ToolRegistry())
    sources = [Source(url="https://a.com", title="A", text="some findings")]
    factors = extract_factors(agent, "topic", sources, covered_factors=["Caffeine dose"], n=2)
    assert factors == ["Sleep quality"]  # already-covered factor filtered out


def test_extract_factors_returns_empty_without_sources():
    config = load_config()
    agent = Agent(config, memory=Memory(), tools=ToolRegistry())
    assert extract_factors(agent, "topic", [], covered_factors=[]) == []


def test_extract_factors_samples_both_broad_and_recent_sources(monkeypatch):
    """Regression test: a real run on "does creatine improve cognitive
    performance" got extract_factors stuck returning nothing after several
    narrow follow-up rounds, because it only ever sampled the most recent
    (increasingly narrow) sources — the broad overview source with general
    framing (dosage, duration, population...) had aged out of the window.
    It must always include early sources too, however many rounds have run."""
    config = load_config()
    captured_user_content = {}

    def fake_chat(self, model, messages, temperature=0.4, tools=None):
        captured_user_content["text"] = next(m["content"] for m in messages if m["role"] == "user")
        return "NONE"

    monkeypatch.setattr(OllamaClient, "chat", fake_chat)
    agent = Agent(config, memory=Memory(), tools=ToolRegistry())

    sources = [Source(url=f"https://s{i}.com", title=f"Source {i}", text=f"text {i}") for i in range(10)]
    extract_factors(agent, "topic", sources, covered_factors=[])

    sample_text = captured_user_content["text"]
    assert "Source 0" in sample_text  # broad/overview source — must not age out
    assert "Source 9" in sample_text  # most recent/narrow source
    assert "Source 5" not in sample_text  # middle sources are deliberately excluded


def test_pairwise_relationship_questions_covers_every_unique_pair():
    factors = ["A", "B", "C"]
    pairs = pairwise_relationship_questions("topic", factors, already_asked=set())
    assert len(pairs) == 3  # 3 choose 2
    keys = {key for key, _ in pairs}
    assert frozenset({"A", "B"}) in keys
    assert frozenset({"A", "C"}) in keys
    assert frozenset({"B", "C"}) in keys
    assert all("topic" in question for _, question in pairs)


def test_pairwise_relationship_questions_skips_already_asked():
    factors = ["A", "B", "C"]
    already_asked = {frozenset({"A", "B"})}
    pairs = pairwise_relationship_questions("topic", factors, already_asked)
    keys = {key for key, _ in pairs}
    assert frozenset({"A", "B"}) not in keys
    assert len(pairs) == 2


def test_run_deep_research_factor_analysis_researches_pairs_and_stops_when_dry(monkeypatch):
    agent = _agent_with_fake_browser(
        results_by_query={
            "topic": [{"url": "https://a.com/1", "title": "A1"}],
            "How does Speed relate to or correlate with Cost, in the context of topic?": [
                {"url": "https://pair.com/1", "title": "Pair source"}
            ],
        },
        no_verify=True,
    )
    agent.config._raw.setdefault("research", {})["followup_rounds"] = 0

    calls = {"factors": 0}

    def fake_chat(self, model, messages, temperature=0.4, tools=None):
        system_content = next((m["content"] for m in messages if m["role"] == "system"), "")
        user_content = next((m["content"] for m in messages if m["role"] == "user"), "")
        if "concrete factors" in system_content.lower():
            calls["factors"] += 1
            return "1. Speed\n2. Cost\n" if calls["factors"] == 1 else "NONE"
        if user_content.startswith("Angle:"):
            return "SECTION_CONTENT"
        return "## Introduction\nFRAMING_CONTENT\n\n## Conclusion\nFRAMING_CONTENT"

    monkeypatch.setattr(OllamaClient, "chat", fake_chat)

    options = ResearchOptions(followup_rounds=0, adaptive_digging=False, factor_analysis=True, verify_sections=False)
    result = run_deep_research(agent, "topic", options=options)

    round_labels = {s.round_label for s in result.sources}
    assert "How does Speed relate to or correlate with Cost, in the context of topic?" in round_labels
    # One round finds factors, then two consecutive dry rounds confirm
    # exhaustion (a single empty response isn't enough to stop on).
    assert calls["factors"] == 3


def test_run_deep_research_stops_gracefully_on_keyboard_interrupt(monkeypatch):
    """A single Ctrl+C during digging should stop the digging loop but still
    let synthesis run normally afterward — it's a momentary interrupt, not a
    persistent failure state."""
    agent = _agent_with_fake_browser(
        results_by_query={"topic": [{"url": "https://a.com/1", "title": "A1"}]},
        no_verify=True,
    )
    agent.config._raw.setdefault("research", {})["followup_rounds"] = 0

    interrupted = {"done": False}

    def fake_chat(self, model, messages, temperature=0.4, tools=None):
        if not interrupted["done"]:
            interrupted["done"] = True
            raise KeyboardInterrupt  # simulates Ctrl+C during the adaptive-digging call
        return "REPORT_CONTENT"

    monkeypatch.setattr(OllamaClient, "chat", fake_chat)

    options = ResearchOptions(followup_rounds=0, adaptive_digging=True, verify_sections=False)
    result = run_deep_research(agent, "topic", options=options)

    assert result.sources  # the overview round's source survived
    assert "REPORT_CONTENT" in result.report  # synthesis still ran normally afterward


def test_choose_falls_back_to_numbered_prompt_without_questionary(monkeypatch):
    import deskbot.research as research_module

    monkeypatch.setattr(research_module, "questionary", None)
    monkeypatch.setattr("builtins.input", lambda *_args: "2")

    result = research_module._choose("Pick one:", [("a", "Option A"), ("b", "Option B")], default_index=0)
    assert result == "b"


def test_choose_falls_back_to_default_on_invalid_input_without_questionary(monkeypatch):
    import deskbot.research as research_module

    monkeypatch.setattr(research_module, "questionary", None)
    monkeypatch.setattr("builtins.input", lambda *_args: "not a number")

    result = research_module._choose("Pick one:", [("a", "Option A"), ("b", "Option B")], default_index=1)
    assert result == "b"


def test_ask_text_falls_back_to_input_without_questionary(monkeypatch):
    import deskbot.research as research_module

    monkeypatch.setattr(research_module, "questionary", None)
    monkeypatch.setattr("builtins.input", lambda *_args: "")

    assert research_module._ask_text("Anything?", default="fallback") == "fallback"


def test_ask_confirm_falls_back_to_input_without_questionary(monkeypatch):
    import deskbot.research as research_module

    monkeypatch.setattr(research_module, "questionary", None)
    monkeypatch.setattr("builtins.input", lambda *_args: "y")

    assert research_module._ask_confirm("Sure?", default=False) is True


def test_ask_number_parses_valid_and_falls_back_on_invalid(monkeypatch):
    import deskbot.research as research_module

    monkeypatch.setattr(research_module, "_ask_text", lambda message, default="": "42")
    assert research_module._ask_number("How many?", 7) == 42

    monkeypatch.setattr(research_module, "_ask_text", lambda message, default="": "not a number")
    assert research_module._ask_number("How many?", 7) == 7


def test_ask_model_uses_plain_text_when_no_models_available(monkeypatch):
    import deskbot.research as research_module

    monkeypatch.setattr(research_module, "_ask_text", lambda message, default="": "typed-model")
    assert research_module._ask_model("Which model?", available=[]) == "typed-model"


def test_score_credibility_gov_and_edu_are_always_high():
    assert _score_credibility("cdc.gov") == "high"
    assert _score_credibility("mit.edu") == "high"
    assert _score_credibility("sub.stanford.edu") == "high"


def test_score_credibility_known_journals_are_high():
    assert _score_credibility("nature.com") == "high"
    assert _score_credibility("pubmed.ncbi.nlm.nih.gov") == "high"


def test_score_credibility_known_news_sites_are_medium():
    assert _score_credibility("bbc.com") == "medium"
    assert _score_credibility("healthline.com") == "medium"


def test_score_credibility_unknown_domain_is_low():
    assert _score_credibility("some-random-blog.example") == "low"


def test_gather_sources_tags_credibility_on_each_source():
    results = [
        {"url": "https://cdc.gov/health-info", "title": "CDC info"},
        {"url": "https://randomblog.example/post", "title": "Blog post"},
    ]
    agent = _agent_with_fake_browser(results=results)
    sources = gather_sources(agent, "topic", max_sources=5, per_source_chars=500, max_corpus_chars=80_000)

    by_url = {s.url: s.credibility for s in sources}
    assert by_url["https://cdc.gov/health-info"] == "high"
    assert by_url["https://randomblog.example/post"] == "low"


def test_generate_hypothesis_returns_stripped_model_reply(monkeypatch):
    config = load_config()
    monkeypatch.setattr(
        OllamaClient, "chat",
        lambda self, model, messages, temperature=0.4, tools=None: "  Daily walking reduces cardiovascular risk.  ",
    )
    agent = Agent(config, memory=Memory(), tools=ToolRegistry())
    sources = [Source(url="https://a.com", title="A", text="some findings")]
    assert generate_hypothesis(agent, "topic", sources) == "Daily walking reduces cardiovascular risk."


def test_generate_hypothesis_returns_empty_without_sources():
    config = load_config()
    agent = Agent(config, memory=Memory(), tools=ToolRegistry())
    assert generate_hypothesis(agent, "topic", []) == ""


def test_generate_hypothesis_falls_back_to_empty_on_model_error(monkeypatch):
    config = load_config()

    def raise_error(self, model, messages, temperature=0.4, tools=None):
        raise OllamaConnectionError("down")

    monkeypatch.setattr(OllamaClient, "chat", raise_error)
    agent = Agent(config, memory=Memory(), tools=ToolRegistry())
    sources = [Source(url="https://a.com", title="A", text="some findings")]
    assert generate_hypothesis(agent, "topic", sources) == ""


def test_scientific_relationship_questions_generates_confirm_and_disconfirm_per_pair():
    factors = ["A", "B"]
    pairs = scientific_relationship_questions("topic", factors, already_asked=set())

    assert len(pairs) == 2  # one pair -> confirm + disconfirm
    keys = {key for key, _ in pairs}
    assert (frozenset({"A", "B"}), "confirm") in keys
    assert (frozenset({"A", "B"}), "disconfirm") in keys
    questions = {q for _, q in pairs}
    assert any("supports" in q.lower() for q in questions)
    assert any("no relationship" in q.lower() or "contradicts" in q.lower() for q in questions)


def test_scientific_relationship_questions_skips_already_asked():
    factors = ["A", "B"]
    already_asked = {(frozenset({"A", "B"}), "confirm")}
    pairs = scientific_relationship_questions("topic", factors, already_asked)

    assert len(pairs) == 1
    assert pairs[0][0] == (frozenset({"A", "B"}), "disconfirm")


def test_run_deep_research_scientific_mode_generates_hypothesis_and_evidence_rounds(monkeypatch):
    agent = _agent_with_fake_browser(
        results_by_query={
            "topic": [{"url": "https://a.com/1", "title": "A1"}],
            "What evidence supports a relationship between Speed and Cost, in the context of topic?": [
                {"url": "https://confirm.com/1", "title": "Confirm source"}
            ],
            "What evidence shows NO relationship or contradicts a link between Speed and Cost, in the context of topic?": [
                {"url": "https://disconfirm.com/1", "title": "Disconfirm source"}
            ],
        },
        no_verify=True,
    )
    agent.config._raw.setdefault("research", {})["followup_rounds"] = 0

    calls = {"factors": 0, "hypothesis": 0}

    def fake_chat(self, model, messages, temperature=0.4, tools=None):
        system_content = next((m["content"] for m in messages if m["role"] == "system"), "")
        sc = system_content.lower()
        if "formulating a hypothesis" in sc:
            calls["hypothesis"] += 1
            return "Speed and Cost are inversely related."
        if "concrete factors" in sc:
            calls["factors"] += 1
            return "1. Speed\n2. Cost\n" if calls["factors"] == 1 else "NONE"
        return "REPORT_CONTENT"

    monkeypatch.setattr(OllamaClient, "chat", fake_chat)

    options = ResearchOptions(
        followup_rounds=0, adaptive_digging=False, factor_analysis=True, scientific_mode=True, verify_sections=False,
    )
    result = run_deep_research(agent, "topic", options=options)

    assert result.hypothesis == "Speed and Cost are inversely related."
    round_labels = {s.round_label for s in result.sources}
    assert "What evidence supports a relationship between Speed and Cost, in the context of topic?" in round_labels
    assert (
        "What evidence shows NO relationship or contradicts a link between Speed and Cost, in the context of topic?"
        in round_labels
    )
    assert calls["hypothesis"] == 1
