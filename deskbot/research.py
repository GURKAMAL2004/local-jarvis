"""`deskbot research "<topic>"` — deep, multi-source research that's reliable
even on small local models.

Key design choice: navigation is CODE-DRIVEN, not left to the model's own
tool planning. Small local models call tools correctly one at a time but
don't reliably chain search -> read -> click -> read -> summarize on their
own (verified directly: they stall, or hallucinate plausible-looking click
targets instead of grounding themselves in the real page). So this module
does the searching and reading itself, deterministically, and only asks the
model to do the things it's reliably good at in small, well-scoped calls:
proposing angles, writing one section at a time, and checking its own work.

The pipeline, in order:
  1. Round 1: a broad overview search on the raw topic.
  2. Planned follow-up rounds (research.followup_rounds): the model reads a
     sample of round 1 and proposes specific angles to dig into.
  3. Adaptive "dig until satisfied" rounds: instead of stopping at a fixed
     count, the model is repeatedly asked what's still missing given
     everything found so far, and keeps researching new angles until it
     says nothing important is left (or a safety cap / character budget is
     hit). This is what makes it behave like a real researcher rather than
     a script that always does exactly N searches.
  4. Each round's search results are filtered against a low-quality-domain
     blocklist, and a round automatically retries on a second search engine
     if the first one returns nothing usable (e.g. blocked/empty).
  5. Section synthesis stays map-reduce (one call per angle — a single call
     covering every round was tried and failed twice in real testing, see
     synthesize_section), optionally followed by a self-fact-check pass
     that compares each section back against its own sources and corrects
     unsupported claims.
  6. A dedicated "Contradictions & Open Questions" pass reads across all the
     written sections and explicitly surfaces disagreements instead of
     letting synthesis quietly paper over them.
  7. The report opens with an introduction + bulleted key findings, and
     closes with a conclusion written last (after contradictions are known)
     so it can honestly reflect whatever uncertainty was found.

Two-model support: quick_model (config.research.quick_model, falls back to
the RAM-tier model) drives every planning/checking call; deep_model is
preferred for section synthesis if it's actually been pulled — long,
organized writing benefits from a bigger model, and that's the step in the
pipeline where it matters most.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from rich.console import Console

from deskbot import paths
from deskbot.agent import Agent
from deskbot.llm import OllamaConnectionError, OllamaModelError

console = Console()
logger = logging.getLogger("deskbot.research")

DEFAULT_MAX_SOURCES = 8
DEFAULT_PER_SOURCE_CHARS = 8_000
DEFAULT_MAX_CORPUS_CHARS = 80_000
DEFAULT_FOLLOWUP_ROUNDS = 3
DEFAULT_MAX_TOTAL_ROUNDS = 8
DEFAULT_VERIFY_SECTIONS = True
DEFAULT_ADAPTIVE_ANGLES_PER_ROUND = 2

# Low-signal domains a human researcher would skip even if they rank —
# content farms, redirect/share pages, and sites that block extraction
# behind logins/paywalls so extract_text() would just get boilerplate.
DEFAULT_BLOCKED_DOMAINS = {
    "pinterest.com",
    "quora.com",
    "answers.com",
    "scribd.com",
    "slideshare.net",
    "chegg.com",
}

# Engines list_search_results() knows how to scrape (see tools/browser.py).
# Bing goes first: Google's anti-bot system is far more aggressive about
# showing a CAPTCHA/"unusual traffic" wall to automated Chromium than Bing
# is, even with the stealth mitigations in tools/browser.py — so leading
# with Google just means eating that wall on nearly every research run.
# duckduckgo is deliberately excluded here: the default duckduckgo.com URL
# is a JS-rendered page list_search_results() has no selector for, so
# falling back to it would silently return zero results rather than help.
_SEARCH_ENGINE_FALLBACK_ORDER = ("bing", "google")

_STOPWORD_LEAD_RE = re.compile(r"^\d+[\.\)]\s*")


@dataclass
class ResearchOptions:
    """Per-run overrides for a `deskbot research` invocation — lets one run
    pick a depth preset (or fully custom values) and specific models without
    touching config.yaml. Any field left as None falls back to whatever
    config.yaml already resolves to, so ResearchOptions() behaves exactly
    like calling run_deep_research with no options at all."""

    max_sources: int | None = None
    per_source_chars: int | None = None
    max_corpus_chars: int | None = None
    followup_rounds: int | None = None
    max_total_rounds: int | None = None
    verify_sections: bool | None = None
    adaptive_digging: bool | None = None
    # quick_model drives the cheap planning calls (follow-up questions, "what's
    # still missing" gap-checking). synthesis_model drives every call that
    # writes report prose (sections, fact-check, key findings, contradictions,
    # conclusion) — see the module docstring's two-model split.
    quick_model: str | None = None
    synthesis_model: str | None = None


# Named depth presets offered by the research menu / `--mode` flag. "standard"
# is all-None on purpose: it just means "use whatever config.yaml says",
# which is the same behavior as running research had before presets existed.
RESEARCH_MODE_PRESETS: dict[str, ResearchOptions] = {
    "quick": ResearchOptions(
        max_sources=4,
        followup_rounds=0,
        max_total_rounds=1,
        verify_sections=False,
        adaptive_digging=False,
    ),
    "standard": ResearchOptions(),
    "deep": ResearchOptions(
        max_sources=14,
        per_source_chars=10_000,
        max_corpus_chars=140_000,
        followup_rounds=4,
        max_total_rounds=14,
        verify_sections=True,
        adaptive_digging=True,
    ),
}

RESEARCH_MODE_DESCRIPTIONS: dict[str, str] = {
    "quick": "Single broad search only, no follow-ups, no fact-check. Fastest.",
    "standard": "Broad search + follow-ups + adaptive digging + fact-check. (default)",
    "deep": "More sources/rounds, thorough fact-check. Slowest, most complete.",
}


@dataclass
class Source:
    url: str
    title: str
    text: str = ""
    round_label: str = "initial"  # which question/round this came from — shown in the report


@dataclass
class ResearchResult:
    topic: str
    sources: list[Source] = field(default_factory=list)
    followup_questions: list[str] = field(default_factory=list)
    report: str = ""
    saved_path: Path | None = None


def _domain(url: str) -> str:
    try:
        netloc = urlparse(url).netloc.lower()
    except ValueError:
        return url
    if not netloc:
        return url
    return netloc[4:] if netloc.startswith("www.") else netloc


def _blocked_domains(agent: Agent) -> set[str]:
    configured = agent.config.get("research", "blocked_domains", default=None)
    if not configured:
        return DEFAULT_BLOCKED_DOMAINS
    return {str(d).lower() for d in configured}


def _is_blocked_domain(domain: str, blocked: set[str]) -> bool:
    return any(domain == b or domain.endswith("." + b) for b in blocked)


def _search_and_list(session, query: str, limit: int) -> list[dict]:
    """Tries each engine in _SEARCH_ENGINE_FALLBACK_ORDER until one returns
    usable results — a query that's blocked or empty on Google (common for
    automated traffic) shouldn't sink the whole round when Bing would work."""
    for engine in _SEARCH_ENGINE_FALLBACK_ORDER:
        search_result = session.search(query, engine=engine)
        if not search_result.get("ok"):
            logger.warning("research search failed on %s: %s", engine, search_result.get("error"))
            continue
        listing = session.list_search_results(engine=engine, limit=limit)
        if not listing.get("ok"):
            logger.warning("research list_search_results failed on %s: %s", engine, listing.get("error"))
            continue
        if listing["results"]:
            return listing["results"]
        logger.info("no results on %s for %r — trying next engine", engine, query)
    return []


def gather_sources(
    agent: Agent,
    query: str,
    max_sources: int = DEFAULT_MAX_SOURCES,
    per_source_chars: int = DEFAULT_PER_SOURCE_CHARS,
    max_corpus_chars: int = DEFAULT_MAX_CORPUS_CHARS,
    round_label: str = "initial",
    exclude_domains: set[str] | None = None,
) -> list[Source]:
    session = agent.tools.browser_session  # type: ignore[attr-defined]
    blocked = _blocked_domains(agent)

    raw_results = _search_and_list(session, query, max_sources * 3)
    if not raw_results:
        logger.warning("research: no usable search results for %r on any engine", query)
        return []

    seen_domains: set[str] = set(exclude_domains or ())
    candidates = []
    for item in raw_results:
        domain = _domain(item["url"])
        if domain in seen_domains or _is_blocked_domain(domain, blocked):
            continue
        seen_domains.add(domain)
        candidates.append(item)

    sources: list[Source] = []
    total_chars = 0
    console.print(f"[dim]Found {len(candidates)} distinct source(s) for: {query!r}[/dim]")
    for cand in candidates:
        if len(sources) >= max_sources or total_chars >= max_corpus_chars:
            break
        remaining_budget = max_corpus_chars - total_chars
        extracted = session.extract_text(max_chars=min(per_source_chars, remaining_budget), url=cand["url"])
        if not extracted.get("ok"):
            logger.info("skipping source (extract failed): %s — %s", cand["url"], extracted.get("error"))
            continue
        text = extracted["text"]
        if not text.strip():
            logger.info("skipping source (empty page text): %s", cand["url"])
            continue
        console.print(f"  [dim]read:[/dim] {cand['title'][:70]} ({len(text)} chars)")
        sources.append(
            Source(url=extracted.get("url", cand["url"]), title=cand["title"], text=text, round_label=round_label)
        )
        total_chars += len(text)

    return sources


def _parse_numbered_list(text: str, n: int) -> list[str]:
    items = []
    for line in text.splitlines():
        cleaned = _STOPWORD_LEAD_RE.sub("", line.strip()).strip("-* ").strip()
        if cleaned:
            items.append(cleaned)
    return items[:n]


def _resolve_quick_model(agent: Agent, override: str | None = None) -> str:
    if override:
        return override
    return agent.config.get("research", "quick_model", default=None) or agent.config.resolved_tier.text_model


def generate_followup_questions(
    agent: Agent,
    topic: str,
    sources: list[Source],
    n: int = DEFAULT_FOLLOWUP_ROUNDS,
    quick_model: str | None = None,
) -> list[str]:
    """Reads a sample of round 1 and proposes specific follow-up angles worth
    digging into — this is what turns a single flat search into genuine
    multi-round research instead of "search once and summarize"."""
    if n <= 0 or not sources:
        return []

    model = _resolve_quick_model(agent, quick_model)
    sample = "\n\n".join(f"[{s.title}]\n{s.text[:1500]}" for s in sources[:4])
    messages = [
        {
            "role": "system",
            "content": (
                "You are planning a deep research investigation. Given the topic and a sample "
                "of what's already been found, propose specific, focused follow-up questions "
                "worth researching in more depth — the kind a thorough analyst would chase down "
                f"next, not questions already answered above. Reply with ONLY a numbered list of "
                f"exactly {n} questions, one per line. No other text, no preamble."
            ),
        },
        {"role": "user", "content": f"Topic: {topic}\n\nWhat's been found so far:\n{sample}"},
    ]
    try:
        text = agent.client.chat(model, messages, temperature=0.5)
    except (OllamaConnectionError, OllamaModelError) as e:
        logger.warning("generate_followup_questions failed: %s", e)
        return []
    return _parse_numbered_list(text, n)


def generate_additional_angles(
    agent: Agent,
    topic: str,
    sources: list[Source],
    covered_angles: list[str],
    n: int = DEFAULT_ADAPTIVE_ANGLES_PER_ROUND,
    quick_model: str | None = None,
) -> list[str]:
    """Loop-until-satisfied step: instead of stopping after a fixed number of
    rounds, asks whether a thorough human researcher would still consider
    the topic under-covered given everything found so far. Returns [] once
    the model reports nothing significant is missing — the actual signal
    that ends the adaptive digging loop in run_deep_research()."""
    if n <= 0 or not sources:
        return []

    model = _resolve_quick_model(agent, quick_model)

    sample_pool = sources[:2] + sources[-4:]
    seen_ids: set[int] = set()
    sample_sources = []
    for s in sample_pool:
        if id(s) not in seen_ids:
            seen_ids.add(id(s))
            sample_sources.append(s)
    sample = "\n\n".join(f"[{s.title}]\n{s.text[:1200]}" for s in sample_sources)
    covered = "\n".join(f"- {a}" for a in covered_angles) or "(none yet)"

    messages = [
        {
            "role": "system",
            "content": (
                "You are continuing a deep research investigation. Given the topic, the angles "
                "already researched, and a sample of what's been found, identify what's still "
                f"missing — up to {n} specific, IMPORTANT angles a thorough analyst would still "
                "dig into (real gaps, unclear points, likely disagreements). Do not repeat angles "
                "already covered, and do not propose vague or trivial angles just to fill the "
                "quota. If nothing significant is missing, reply with EXACTLY the single word: "
                f"NONE. Otherwise reply with ONLY a numbered list of up to {n} questions, one per "
                "line, no other text."
            ),
        },
        {
            "role": "user",
            "content": f"Topic: {topic}\n\nAngles already researched:\n{covered}\n\nFindings so far:\n{sample}",
        },
    ]
    try:
        text = agent.client.chat(model, messages, temperature=0.5)
    except (OllamaConnectionError, OllamaModelError) as e:
        logger.warning("generate_additional_angles failed: %s", e)
        return []

    if text.strip().upper().startswith("NONE"):
        return []
    return [q for q in _parse_numbered_list(text, n) if q.upper() != "NONE"]


def _resolve_synthesis_model(agent: Agent, override: str | None = None) -> str:
    cfg = agent.config
    quick_model = _resolve_quick_model(agent)
    deep_model = override or cfg.get("research", "deep_model", default=None)
    if not deep_model:
        return quick_model

    try:
        available = agent.client.list_models()
    except OllamaConnectionError:
        return quick_model

    if any(deep_model in m for m in available):
        return deep_model

    logger.info("research model '%s' not pulled yet — using '%s'", deep_model, quick_model)
    return quick_model


def _group_by_round(sources: list[Source]) -> list[tuple[str, list[Source]]]:
    order: list[str] = []
    by_round: dict[str, list[Source]] = {}
    for s in sources:
        if s.round_label not in by_round:
            by_round[s.round_label] = []
            order.append(s.round_label)
        by_round[s.round_label].append(s)
    return [(label, by_round[label]) for label in order]


def _build_corpus(sources: list[Source]) -> str:
    return "\n\n".join(f"[Source {i}: {s.title} — {s.url}]\n{s.text}" for i, s in enumerate(sources, 1))


def synthesize_section(agent: Agent, model: str, angle: str, sources: list[Source]) -> str:
    """One angle, synthesized on its own. A giant single call covering every
    round was tried first and failed twice in real testing — the model
    consistently let whichever angle appeared last in the context crowd out
    earlier ones (recency bias over long contexts). Synthesizing each round
    independently makes that structurally impossible: each call only ever
    sees one angle's material, so every angle reliably gets its own section."""
    corpus = _build_corpus(sources)
    messages = [
        {
            "role": "system",
            "content": (
                "You are a research analyst. Using ONLY the provided source excerpts, write a "
                "focused, well-organized section covering this specific angle. Cite claims by "
                "source number like [Source 2], and note if sources disagree. Do not invent "
                "facts that aren't present in the sources, and do not include unrelated "
                "boilerplate from the sources (contact details, ads, navigation text, etc.)."
            ),
        },
        {"role": "user", "content": f"Angle: {angle}\n\n{corpus}\n\nWrite this section now."},
    ]
    try:
        return agent.client.chat(model, messages, temperature=agent.config.temperature)
    except (OllamaConnectionError, OllamaModelError) as e:
        return f"(Section synthesis failed for '{angle}': {e})"


def verify_section(agent: Agent, model: str, angle: str, sources: list[Source], draft: str) -> str:
    """Self-fact-check pass: re-reads the drafted section against its own
    source excerpts and corrects anything unsupported. Small models
    sometimes drift from "summarize this" into adding a plausible-sounding
    but unstated detail — this catches that before it reaches the report."""
    corpus = _build_corpus(sources)
    messages = [
        {
            "role": "system",
            "content": (
                "You are a meticulous fact-checking editor reviewing a drafted research section "
                "against its source material. Compare every factual claim in the draft to the "
                "source excerpts below. If a claim is not supported by any source, remove it or "
                "rewrite it as clearly uncertain. Keep correct, well-supported claims and their "
                "[Source N] citations exactly as they are. Reply with ONLY the corrected section "
                "text — no preamble, no commentary about what you changed."
            ),
        },
        {"role": "user", "content": f"{corpus}\n\nDrafted section:\n{draft}\n\nReturn the corrected section now."},
    ]
    try:
        fixed = agent.client.chat(model, messages, temperature=0.2)
        return fixed.strip() or draft
    except (OllamaConnectionError, OllamaModelError) as e:
        logger.warning("verify_section failed for '%s': %s", angle, e)
        return draft


def synthesize_key_findings(agent: Agent, model: str, topic: str, combined_body: str) -> str:
    messages = [
        {
            "role": "system",
            "content": (
                "You are a research editor. You'll be given a topic and a set of already-written "
                "research sections. Write ONLY a short introduction (2-3 sentences framing the "
                "topic), then a '## Key Findings' section with 5-8 concise bullet points capturing "
                "the single most important, concrete takeaways a busy reader needs — specific "
                "numbers, names, and conclusions, not vague generalities. Do not rewrite or repeat "
                "the sections themselves. Label the findings list exactly '## Key Findings'."
            ),
        },
        {"role": "user", "content": f"Topic: {topic}\n\nSections already written:\n{combined_body}"},
    ]
    try:
        return agent.client.chat(model, messages, temperature=agent.config.temperature)
    except (OllamaConnectionError, OllamaModelError):
        return ""


def synthesize_contradictions(agent: Agent, model: str, topic: str, combined_body: str) -> str:
    """A dedicated pass whose only job is to look for disagreement — section
    synthesis is written per-angle and can't see across angles, so nothing
    upstream of this ever compares sections against each other."""
    messages = [
        {
            "role": "system",
            "content": (
                "You are a skeptical research editor. Given a topic and a set of already-written "
                "research sections (each cites sources like [Source 2]), identify any places "
                "where sources disagree, numbers conflict, or important questions remain "
                "open/unresolved. Be specific — name the conflicting claims and which "
                "sections/sources they came from. If you genuinely find nothing significant, say "
                "so in one sentence instead of inventing disagreement. Label your output exactly "
                "'## Contradictions & Open Questions'."
            ),
        },
        {"role": "user", "content": f"Topic: {topic}\n\nSections already written:\n{combined_body}"},
    ]
    try:
        return agent.client.chat(model, messages, temperature=agent.config.temperature)
    except (OllamaConnectionError, OllamaModelError):
        return ""


def synthesize_conclusion(agent: Agent, model: str, topic: str, combined_body: str, contradictions: str) -> str:
    """Written last, after contradictions are known, so it can honestly
    reflect whatever uncertainty was found instead of a falsely tidy wrap-up."""
    messages = [
        {
            "role": "system",
            "content": (
                "You are a research editor writing the closing section of a report. Given the "
                "topic, the written sections, and a summary of any contradictions/open questions, "
                "write ONLY a short conclusion (3-5 sentences) synthesizing across the sections, "
                "honestly reflecting any uncertainty noted. Label it exactly '## Conclusion'."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Topic: {topic}\n\nSections already written:\n{combined_body}\n\n"
                f"Contradictions/open questions noted:\n{contradictions}"
            ),
        },
    ]
    try:
        return agent.client.chat(model, messages, temperature=agent.config.temperature)
    except (OllamaConnectionError, OllamaModelError):
        return ""


def synthesize_report(
    agent: Agent,
    topic: str,
    sources: list[Source],
    followup_questions: list[str] | None = None,
    options: ResearchOptions | None = None,
) -> str:
    options = options or ResearchOptions()
    model = _resolve_synthesis_model(agent, options.synthesis_model)
    verify = (
        options.verify_sections
        if options.verify_sections is not None
        else bool(agent.config.get("research", "verify_sections", default=DEFAULT_VERIFY_SECTIONS))
    )
    grouped = _group_by_round(sources)
    console.print(f"[dim]Writing {len(grouped)} section(s) with '{model}'...[/dim]")

    sections: list[tuple[str, str]] = []
    for angle, angle_sources in grouped:
        console.print(f"  [dim]section:[/dim] {angle[:70]}")
        draft = synthesize_section(agent, model, angle, angle_sources)
        if verify:
            draft = verify_section(agent, model, angle, angle_sources, draft)
        sections.append((angle, draft))

    combined_body = "\n\n".join(f"## {angle}\n\n{text}" for angle, text in sections)

    console.print("[dim]Writing introduction and key findings...[/dim]")
    opening = synthesize_key_findings(agent, model, topic, combined_body)

    console.print("[dim]Checking for contradictions and open questions...[/dim]")
    contradictions = synthesize_contradictions(agent, model, topic, combined_body)

    console.print("[dim]Writing conclusion...[/dim]")
    conclusion = synthesize_conclusion(agent, model, topic, combined_body, contradictions)

    return "\n\n".join(p for p in (opening, combined_body, contradictions, conclusion) if p)


def _slugify(topic: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", topic.lower()).strip("-")
    return slug[:60] or "research"


def save_report(topic: str, sources: list[Source], report: str) -> Path:
    filename = f"{_slugify(topic)}.md"
    lines = [f"# Research: {topic}", "", report, "", "## Sources", ""]
    lines.extend(f"{i}. [{s.title}]({s.url}) — *{s.round_label}*" for i, s in enumerate(sources, 1))
    content = "\n".join(lines)

    fallback_dir = paths.HOME_DIR / "research_reports"
    fallback_dir.mkdir(parents=True, exist_ok=True)
    (fallback_dir / filename).write_text(content, encoding="utf-8")

    desktop_dir = Path.home() / "Desktop" / "deskbot-research"
    try:
        desktop_dir.mkdir(parents=True, exist_ok=True)
        desktop_path = desktop_dir / filename
        desktop_path.write_text(content, encoding="utf-8")
        return desktop_path
    except OSError:
        return fallback_dir / filename


def run_deep_research(agent: Agent, topic: str, options: ResearchOptions | None = None) -> ResearchResult:
    """Multi-round deep research: an initial broad search, then planned
    follow-up rounds, then adaptive "dig until satisfied" rounds that keep
    going until the model reports nothing important is left — a fixed round
    count reliably misses angles a real analyst would chase down next, and
    also keeps going past the point of diminishing returns on easy topics.

    `options` overrides config.yaml's research.* defaults for this one run
    (e.g. a --mode preset or menu-picked models); leave it None to just use
    config.yaml as-is."""
    options = options or ResearchOptions()
    cfg = agent.config
    max_sources = options.max_sources if options.max_sources is not None else int(
        cfg.get("research", "max_sources", default=DEFAULT_MAX_SOURCES)
    )
    per_source_chars = options.per_source_chars if options.per_source_chars is not None else int(
        cfg.get("research", "per_source_chars", default=DEFAULT_PER_SOURCE_CHARS)
    )
    max_corpus_chars = options.max_corpus_chars if options.max_corpus_chars is not None else int(
        cfg.get("research", "max_corpus_chars", default=DEFAULT_MAX_CORPUS_CHARS)
    )
    followup_rounds = options.followup_rounds if options.followup_rounds is not None else int(
        cfg.get("research", "followup_rounds", default=DEFAULT_FOLLOWUP_ROUNDS)
    )
    max_total_rounds = options.max_total_rounds if options.max_total_rounds is not None else int(
        cfg.get("research", "max_total_rounds", default=DEFAULT_MAX_TOTAL_ROUNDS)
    )
    adaptive_digging = options.adaptive_digging if options.adaptive_digging is not None else True

    console.print(f"[bold]Researching:[/bold] {topic}")
    console.print("[bold]Round 1 (broad):[/bold]")
    num_rounds_planned = followup_rounds + 1
    initial_budget = max_corpus_chars // num_rounds_planned
    initial_sources_cap = max(2, max_sources // num_rounds_planned)

    all_sources = gather_sources(
        agent, topic, initial_sources_cap, per_source_chars, initial_budget, round_label=f"Overview: {topic}"
    )
    if not all_sources:
        console.print(
            "[red]Could not gather any sources — check your internet connection and that "
            "the browser layer works (`deskbot doctor`).[/red]"
        )
        return ResearchResult(topic=topic, sources=[], report="")

    total_chars = sum(len(s.text) for s in all_sources)
    seen_domains = {_domain(s.url) for s in all_sources}
    covered_angles = [f"Overview: {topic}"]
    round_count = 1

    def _research_round(question: str) -> None:
        nonlocal total_chars, round_count
        remaining_budget = max_corpus_chars - total_chars
        round_cap = max(2, max_sources // num_rounds_planned)
        extra = gather_sources(
            agent, question, round_cap, per_source_chars, remaining_budget,
            round_label=question, exclude_domains=seen_domains,
        )
        all_sources.extend(extra)
        seen_domains.update(_domain(s.url) for s in extra)
        total_chars += sum(len(s.text) for s in extra)
        covered_angles.append(question)
        round_count += 1

    followups: list[str] = []
    if followup_rounds > 0:
        console.print("[dim]Identifying follow-up angles to dig into...[/dim]")
        followups = generate_followup_questions(
            agent, topic, all_sources, n=followup_rounds, quick_model=options.quick_model
        )
        for question in followups:
            if total_chars >= max_corpus_chars or round_count >= max_total_rounds:
                console.print("[dim]Budget/round cap reached — stopping planned follow-ups early.[/dim]")
                break
            console.print(f"[bold]Round {round_count + 1}:[/bold] {question}")
            _research_round(question)

    # Adaptive "dig until satisfied" pass — keeps asking what's still
    # missing and researching it, instead of stopping at a fixed round
    # count, until the model says nothing important is left or the
    # safety cap / character budget is hit. Skippable (options.adaptive_digging
    # = False, e.g. the "quick" preset) for a fast, fixed-round-count run.
    while adaptive_digging and total_chars < max_corpus_chars and round_count < max_total_rounds:
        console.print("[dim]Checking whether anything important is still missing...[/dim]")
        extra_angles = generate_additional_angles(
            agent, topic, all_sources, covered_angles, quick_model=options.quick_model
        )
        if not extra_angles:
            console.print("[dim]Nothing significant left to dig into — stopping.[/dim]")
            break
        for question in extra_angles:
            if total_chars >= max_corpus_chars or round_count >= max_total_rounds:
                break
            console.print(f"[bold]Round {round_count + 1} (adaptive):[/bold] {question}")
            _research_round(question)

    console.print(
        f"[dim]Read {len(all_sources)} source(s) across {round_count} round(s), "
        f"{total_chars} characters total.[/dim]"
    )

    report = synthesize_report(agent, topic, all_sources, followups, options=options)
    saved_path = save_report(topic, all_sources, report)

    console.print(f"\n{report}\n")
    console.print(f"[green]Report saved to:[/green] {saved_path}")
    return ResearchResult(
        topic=topic, sources=all_sources, followup_questions=followups, report=report, saved_path=saved_path
    )


# --- interactive setup menu, used by `deskbot research` when run from a real
# terminal with no --mode/--model flags — plain input()/print() to match the
# style of the persona-create wizard, not the rich-formatted progress output
# above (that's for a run already in flight; this is picked before it starts).

def _list_available_models(agent: Agent) -> list[str]:
    try:
        return agent.client.list_models()
    except OllamaConnectionError:
        return []


def _ask_int(prompt: str, default: int) -> int:
    raw = input(f"{prompt} [{default}]: ").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        print(f"Not a number, using default ({default}).")
        return default


def _ask_bool(prompt: str, default: bool) -> bool:
    raw = input(f"{prompt} (y/n) [{'y' if default else 'n'}]: ").strip().lower()
    if not raw:
        return default
    return raw.startswith("y")


def _prompt_custom_options() -> ResearchOptions:
    print("\n-- custom research settings --")
    return ResearchOptions(
        max_sources=_ask_int("Max sources to read (across all rounds)", DEFAULT_MAX_SOURCES),
        per_source_chars=_ask_int("Max characters read per source", DEFAULT_PER_SOURCE_CHARS),
        max_corpus_chars=_ask_int("Total character budget for the whole run", DEFAULT_MAX_CORPUS_CHARS),
        followup_rounds=_ask_int("Planned follow-up rounds after the overview search", DEFAULT_FOLLOWUP_ROUNDS),
        max_total_rounds=_ask_int("Max total rounds, safety cap (overview + follow-ups + adaptive)", DEFAULT_MAX_TOTAL_ROUNDS),
        verify_sections=_ask_bool("Fact-check each section against its own sources", DEFAULT_VERIFY_SECTIONS),
        adaptive_digging=_ask_bool("Keep digging adaptively until nothing important is missing", True),
    )


def prompt_research_setup(agent: Agent) -> ResearchOptions:
    """Interactive menu: pick a depth preset (or go custom), then optionally
    pick specific models for this one run. Returns a ResearchOptions ready to
    hand to run_deep_research(); any field the user leaves blank falls back
    to config.yaml as usual."""
    print("\n=== deskbot deep research ===")
    print("Pick a research method:")
    print(f"  1) Quick    - {RESEARCH_MODE_DESCRIPTIONS['quick']}")
    print(f"  2) Standard - {RESEARCH_MODE_DESCRIPTIONS['standard']}")
    print(f"  3) Deep     - {RESEARCH_MODE_DESCRIPTIONS['deep']}")
    print("  4) Custom   - configure every setting yourself.")
    choice = input("> ").strip()

    if choice == "1":
        options = replace(RESEARCH_MODE_PRESETS["quick"])
    elif choice == "3":
        options = replace(RESEARCH_MODE_PRESETS["deep"])
    elif choice == "4":
        options = _prompt_custom_options()
    else:
        options = replace(RESEARCH_MODE_PRESETS["standard"])

    available = _list_available_models(agent)
    if available:
        print(f"\nModels available locally: {', '.join(available)}")
    else:
        print("\n(Could not reach Ollama to list models — you can still type a name manually.)")
    print("Pick models for this run, or leave blank to use config.yaml's defaults.")
    planning = input("  Planning model (proposes follow-up/gap questions) [default]: ").strip()
    writing = input("  Writing model (drafts sections, fact-checks, writes the report) [same/default]: ").strip()

    if planning:
        options.quick_model = planning
    if writing:
        options.synthesis_model = planning if writing.lower() == "same" else writing

    return options
