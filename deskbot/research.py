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
     unsupported claims. The "map" half of that (every angle's draft+verify)
     runs concurrently across a small thread pool (research.synthesis_workers,
     default DEFAULT_SYNTHESIS_WORKERS) instead of one angle at a time — each
     angle's synthesis is independent of every other angle's, so there's
     nothing to lose by overlapping the HTTP round trips to Ollama, and
     nothing to gain by waiting on them serially. The three report-level
     passes that don't depend on each other (key findings, contradictions,
     factor/scientific analysis) run concurrently too; the conclusion still
     runs last since it specifically reads the contradictions output. Network
     fetching (gather_sources) deliberately stays sequential through the one
     shared browser page — that serialization plus human_delay() is what
     keeps research from looking like bot traffic (see tools/browser.py);
     parallelizing it would undo that.
  6. A dedicated "Contradictions & Open Questions" pass reads across all the
     written sections and explicitly surfaces disagreements instead of
     letting synthesis quietly paper over them.
  7. The report opens with an introduction + bulleted key findings, and
     closes with a conclusion written last (after contradictions are known)
     so it can honestly reflect whatever uncertainty was found.
  8. Optional factor/correlation digging (ResearchOptions.factor_analysis,
     on for the "relentless" preset): once the above finishes, deskbot
     extracts the concrete factors/variables the topic depends on
     (extract_factors) and researches every pairwise relationship between
     them (pairwise_relationship_questions — pure combinatorics, not left to
     the model), pulling in more factors as it goes. This phase only ends on
     genuine exhaustion (two rounds in a row with nothing new) or the user
     pressing Ctrl+C — the "relentless" preset's round/budget ceilings are
     set so high they're not the practical limit. A "Factors & Correlations"
     section in the report summarizes what was actually found.
  9. Optional scientific mode (ResearchOptions.scientific_mode, implies
     factor_analysis, on for the "scientist" preset): before digging,
     generate_hypothesis states one clear, falsifiable hypothesis. Each
     factor pair then gets TWO research questions instead of one
     (scientific_relationship_questions) — one hunting for evidence that
     SUPPORTS the relationship, one hunting for evidence that CONTRADICTS
     it, because a real scientist tries to falsify their own hypothesis
     rather than only confirming it. Every source is tagged with a
     code-computed credibility tier (_score_credibility — .gov/.edu/known
     journals = high, reputable news = medium, everything else = low,
     never model-judged), and synthesize_scientific_assessment rates each
     relationship's evidence strength (Strong/Moderate/Weak/Conflicting/
     Insufficient) weighted by that credibility instead of source count alone.
  10. Optional academic/"authority" mode (ResearchOptions.academic_mode, on
      for the "authority" preset): generate_expert_persona invents one
      plausible, field-appropriate domain-expert persona for the topic (a
      name, credentials, institutions, decades of experience — the same
      trick used everywhere else in this module: a small, narrowly-scoped
      call the model is actually reliable at, instead of asking it to just
      "sound like an expert" inline). Every section, the key findings, the
      contradictions pass, and the conclusion are then written in that
      persona's voice with LaTeX equations for quantitative relationships
      and (Author, Year, Journal, Vol:Pages)-style citations (uncertain ones
      explicitly flagged as `[citation needed — proposed: ...]` rather than
      invented with false confidence). Finally, generate_research_taxonomy
      builds a large hierarchy of specific, testable research
      question/hypothesis/method leaves — domains, then subtopics per
      domain, then leaves per subtopic — using the exact same deterministic
      pagination pattern as factor/correlation digging (ask for a numbered
      list, parse it, recurse) rather than one impossible mega-call, with
      every independent branch fanned out concurrently on the same thread
      pool synthesize_report's section synthesis uses.

Two-model support: quick_model (config.research.quick_model, falls back to
the RAM-tier model) drives every planning/checking call; deep_model is
preferred for section synthesis if it's actually been pulled — long,
organized writing benefits from a bigger model, and that's the step in the
pipeline where it matters most.
"""

from __future__ import annotations

import logging
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from rich.console import Console
from rich.panel import Panel

try:
    import questionary
except ImportError:  # pragma: no cover - exercised only when the optional dep is missing
    questionary = None

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
DEFAULT_FACTORS_PER_EXTRACTION = 8
# Academic/"authority" mode taxonomy shape: domains x subtopics x leaves. Fixed
# counts rather than a range so the total is a deterministic floor (assuming
# the model returns everything asked for) — 10*6*10 = 600, comfortably over
# "at least 569". Real local-model output usually falls a bit short of the
# ask on some branches; run_deep_research reports the actual count achieved
# rather than silently claiming the full total (see the module docstring).
DEFAULT_TAXONOMY_DOMAINS = 10
DEFAULT_TAXONOMY_SUBTOPICS_PER_DOMAIN = 6
DEFAULT_TAXONOMY_LEAVES_PER_SUBTOPIC = 10
# How many angle-sections (draft + verify) synthesize_report runs concurrently.
# Each angle's synthesis only reads its own sources, so this is purely
# overlapping independent HTTP round trips to Ollama — safe regardless of
# whether the local Ollama server itself processes them one at a time or in
# parallel (OLLAMA_NUM_PARALLEL): worst case it's no slower than the old
# strictly-sequential loop, best case it's several times faster.
DEFAULT_SYNTHESIS_WORKERS = 4

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
    # When True, after the normal digging finishes deskbot extracts the key
    # factors/variables underlying the topic and researches the relationship
    # between every pair of them, pulling in more factors as it goes — see
    # the module docstring section on factor/correlation digging. Combined
    # with the "relentless" preset's huge round/budget ceilings, the only
    # practical way this run ends is genuine exhaustion or Ctrl+C.
    factor_analysis: bool | None = None
    # When True (implies factor_analysis), deskbot behaves like a scientist
    # instead of a generalist: states an explicit falsifiable hypothesis up
    # front, actively searches for evidence that would DISPROVE each factor
    # relationship (not just evidence that supports it), and weighs sources
    # by a code-computed credibility tier when rating how strong the
    # evidence actually is — see generate_hypothesis,
    # scientific_relationship_questions, and synthesize_scientific_assessment.
    scientific_mode: bool | None = None
    # When True, deskbot writes like a named domain-expert authority instead
    # of a generic analyst: it invents a plausible, field-appropriate expert
    # persona for the topic (generate_expert_persona), writes sections in
    # that formal, quantitative voice with LaTeX equations and
    # (Author, Year, Journal, Vol:Pages)-style citations where appropriate,
    # and appends a large hierarchical taxonomy of specific, testable
    # research questions (generate_research_taxonomy) — the equivalent of a
    # review article's "future work" section taken to its logical extreme.
    # See the "authority" preset and the module docstring's academic-mode
    # section.
    academic_mode: bool | None = None
    # quick_model drives the cheap planning calls (follow-up questions, "what's
    # still missing" gap-checking). synthesis_model drives every call that
    # writes report prose (sections, fact-check, key findings, contradictions,
    # conclusion) — see the module docstring's two-model split.
    quick_model: str | None = None
    synthesis_model: str | None = None
    # How many angle-sections synthesize_report drafts/verifies concurrently —
    # see DEFAULT_SYNTHESIS_WORKERS. None falls back to config.yaml.
    synthesis_workers: int | None = None


# Effectively-unbounded ceilings for the "relentless" preset — large finite
# numbers rather than infinity so the existing int math (division, budget
# subtraction) keeps working unchanged. In practice a run this size only
# ever ends via genuine exhaustion (extract_factors/generate_additional_angles
# both come back empty) or the user hitting Ctrl+C.
_RELENTLESS_MAX_SOURCES = 100_000
_RELENTLESS_MAX_CORPUS_CHARS = 100_000_000
_RELENTLESS_MAX_TOTAL_ROUNDS = 100_000

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
    "relentless": ResearchOptions(
        max_sources=_RELENTLESS_MAX_SOURCES,
        max_corpus_chars=_RELENTLESS_MAX_CORPUS_CHARS,
        followup_rounds=3,
        max_total_rounds=_RELENTLESS_MAX_TOTAL_ROUNDS,
        verify_sections=True,
        adaptive_digging=True,
        factor_analysis=True,
    ),
    "scientist": ResearchOptions(
        max_sources=_RELENTLESS_MAX_SOURCES,
        max_corpus_chars=_RELENTLESS_MAX_CORPUS_CHARS,
        followup_rounds=3,
        max_total_rounds=_RELENTLESS_MAX_TOTAL_ROUNDS,
        verify_sections=True,
        adaptive_digging=True,
        factor_analysis=True,
        scientific_mode=True,
    ),
    "authority": ResearchOptions(
        max_sources=14,
        per_source_chars=10_000,
        max_corpus_chars=140_000,
        followup_rounds=4,
        max_total_rounds=14,
        verify_sections=True,
        adaptive_digging=True,
        academic_mode=True,
    ),
}

RESEARCH_MODE_DESCRIPTIONS: dict[str, str] = {
    "quick": "Single broad search only, no follow-ups, no fact-check. Fastest.",
    "standard": "Broad search + follow-ups + adaptive digging + fact-check. (default)",
    "deep": "More sources/rounds, thorough fact-check. Slowest, most complete.",
    "relentless": (
        "Extracts key factors, digs into every relationship/correlation between "
        "them, and keeps going with no real cap until YOU stop it (Ctrl+C)."
    ),
    "scientist": (
        "Forms a falsifiable hypothesis, actively hunts for evidence that would "
        "DISPROVE each factor relationship (not just confirm it), and rates "
        "confidence by source credibility. No real cap — Ctrl+C when satisfied."
    ),
    "authority": (
        "Writes as an invented domain-expert authority — formal, quantitative, LaTeX "
        "equations and (Author, Year, Journal, Vol:Pages) citations — then appends a "
        "large taxonomy of hundreds of specific, testable research questions."
    ),
}


@dataclass
class Source:
    url: str
    title: str
    text: str = ""
    round_label: str = "initial"  # which question/round this came from — shown in the report
    credibility: str = "unknown"  # "high" | "medium" | "low" — see _score_credibility()


@dataclass
class ResearchResult:
    topic: str
    sources: list[Source] = field(default_factory=list)
    followup_questions: list[str] = field(default_factory=list)
    report: str = ""
    saved_path: Path | None = None
    hypothesis: str = ""  # set only in scientific_mode — see generate_hypothesis()
    persona: str = ""  # set only in academic_mode — see generate_expert_persona()


# Credibility is scored from the domain alone — code-driven, not model-judged,
# for the same reason blocked-domain filtering is: a small model asked "is
# this source credible?" is unreliable, but "is this domain .gov/.edu or on
# a known list" is deterministic. Any .gov/.edu domain is always "high"
# regardless of these lists. Everything unmatched defaults to "low" — the
# safe assumption for an unknown blog/forum is not to over-trust it.
HIGH_CREDIBILITY_DOMAINS = {
    "pubmed.ncbi.nlm.nih.gov", "ncbi.nlm.nih.gov", "nih.gov", "cdc.gov", "who.int",
    "nature.com", "sciencedirect.com", "thelancet.com", "nejm.org", "jamanetwork.com",
    "bmj.com", "cochranelibrary.com", "springer.com", "wiley.com", "frontiersin.org",
    "plos.org", "arxiv.org", "pnas.org", "cell.com", "mayoclinic.org",
}
MEDIUM_CREDIBILITY_DOMAINS = {
    "reuters.com", "bbc.com", "nytimes.com", "healthline.com", "webmd.com",
    "medicalnewstoday.com", "statnews.com", "apnews.com", "npr.org", "theguardian.com",
    "scientificamerican.com", "wikipedia.org",
}


def _score_credibility(domain: str) -> str:
    if domain.endswith(".gov") or domain.endswith(".edu"):
        return "high"
    if any(domain == d or domain.endswith("." + d) for d in HIGH_CREDIBILITY_DOMAINS):
        return "high"
    if any(domain == d or domain.endswith("." + d) for d in MEDIUM_CREDIBILITY_DOMAINS):
        return "medium"
    return "low"


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
        source_url = extracted.get("url", cand["url"])
        sources.append(
            Source(
                url=source_url, title=cand["title"], text=text, round_label=round_label,
                credibility=_score_credibility(_domain(source_url)),
            )
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


def _mixed_sample(sources: list[Source], first_n: int = 2, last_n: int = 4) -> list[Source]:
    """A small sample mixing early sources (broad overview framing) with the
    most recent ones (latest depth) — used for planning calls that need
    context without feeding the whole corpus. Recency-only sampling was
    observed to starve factor/angle extraction of the broad framing once
    several rounds of narrow follow-ups had run (e.g. after digging into
    specific biochemical mechanisms, "what factors does this depend on?"
    got no signal about dosage/duration/population from the overview
    source). Dedupes by identity in case the source lists overlap."""
    pool = sources[:first_n] + sources[-last_n:]
    seen_ids: set[int] = set()
    out = []
    for s in pool:
        if id(s) not in seen_ids:
            seen_ids.add(id(s))
            out.append(s)
    return out


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

    sample_sources = _mixed_sample(sources)
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


def extract_factors(
    agent: Agent,
    topic: str,
    sources: list[Source],
    covered_factors: list[str],
    n: int = DEFAULT_FACTORS_PER_EXTRACTION,
    quick_model: str | None = None,
) -> list[str]:
    """Factor/correlation digging, step 1: identifies concrete
    factors/variables the topic actually depends on (e.g. for "coffee and
    heart health": caffeine dose, genetics, existing conditions, timing,
    tolerance...), grounding the correlation-chasing in real named things
    from the sources instead of the model free-associating relationships out
    of nothing. Returns [] once the model can't name any more that aren't
    already covered — the natural end of this phase (see
    pairwise_relationship_questions for step 2)."""
    if n <= 0 or not sources:
        return []

    model = _resolve_quick_model(agent, quick_model)
    sample_sources = _mixed_sample(sources)
    sample = "\n\n".join(f"[{s.title}]\n{s.text[:1500]}" for s in sample_sources)
    covered = "\n".join(f"- {f}" for f in covered_factors) or "(none yet)"
    messages = [
        {
            "role": "system",
            "content": (
                "You are analyzing a topic to identify the concrete factors/variables it "
                f"actually depends on. Given the topic and what's been found so far, name up to "
                f"{n} specific factors — real, named variables a thorough analyst would examine "
                "(not vague themes) — that are NOT already in the covered list. Examples of the "
                "right level of specificity: dosage/amount, duration of exposure, population/age "
                "group, timing, individual variation (e.g. genetics), and method of measurement — "
                "not the topic restated. If you genuinely can't name any more, reply with EXACTLY "
                f"the single word: NONE. Otherwise reply with ONLY a numbered list of up to {n} "
                "short factor names, one per line, no other text."
            ),
        },
        {
            "role": "user",
            "content": f"Topic: {topic}\n\nFactors already covered:\n{covered}\n\nFindings so far:\n{sample}",
        },
    ]
    try:
        text = agent.client.chat(model, messages, temperature=0.6)
    except (OllamaConnectionError, OllamaModelError) as e:
        logger.warning("extract_factors failed: %s", e)
        return []

    if text.strip().upper().startswith("NONE"):
        logger.info("extract_factors: model reported no more factors")
        return []
    covered_lower = {f.lower() for f in covered_factors}
    factors = [f for f in _parse_numbered_list(text, n) if f.upper() != "NONE" and f.lower() not in covered_lower]
    if not factors:
        logger.info("extract_factors: got a reply but parsed no usable factors: %r", text[:200])
    return factors


def pairwise_relationship_questions(
    topic: str, factors: list[str], already_asked: set[frozenset[str]]
) -> list[tuple[frozenset[str], str]]:
    """Factor/correlation digging, step 2 — pure code, no model call: every
    unordered pair of known factors becomes its own research question. This
    is what actually drives "relation, correlation between each thing" —
    deterministic combinatorics rather than hoping the model thinks to
    compare things on its own, and it naturally produces a lot of rounds as
    the factor list grows (K factors -> K*(K-1)/2 pairs)."""
    from itertools import combinations

    pairs = []
    for a, b in combinations(factors, 2):
        key = frozenset((a, b))
        if key in already_asked:
            continue
        pairs.append((key, f"How does {a} relate to or correlate with {b}, in the context of {topic}?"))
    return pairs


def generate_hypothesis(agent: Agent, topic: str, sources: list[Source], quick_model: str | None = None) -> str:
    """Scientific mode, step 1: the first thing a real scientist does is
    state a clear, falsifiable hypothesis before digging further — not just
    "research the topic" but "here's a specific claim, now let's see if the
    evidence actually holds up." Returns "" (mode degrades gracefully to
    plain factor analysis) if the model call fails."""
    if not sources:
        return ""

    model = _resolve_quick_model(agent, quick_model)
    sample = "\n\n".join(f"[{s.title}]\n{s.text[:1500]}" for s in sources[:4])
    messages = [
        {
            "role": "system",
            "content": (
                "You are a scientist formulating a hypothesis. Given the topic and initial "
                "findings, state ONE clear, specific, falsifiable hypothesis worth testing against "
                "the evidence — not a vague theme. Reply with ONLY the hypothesis statement, one or "
                "two sentences, no preamble, no labels."
            ),
        },
        {"role": "user", "content": f"Topic: {topic}\n\nInitial findings:\n{sample}"},
    ]
    try:
        return agent.client.chat(model, messages, temperature=0.5).strip()
    except (OllamaConnectionError, OllamaModelError) as e:
        logger.warning("generate_hypothesis failed: %s", e)
        return ""


def scientific_relationship_questions(
    topic: str, factors: list[str], already_asked: set[tuple[frozenset[str], str]]
) -> list[tuple[tuple[frozenset[str], str], str]]:
    """Scientific mode, step 2 — like pairwise_relationship_questions, but
    actively hunts for BOTH confirming AND disconfirming evidence per pair:
    a real scientist tries to falsify their own hypothesis rather than only
    looking for support. Still pure code, no model call — deterministic
    combinatorics, just two questions per pair instead of one."""
    from itertools import combinations

    out = []
    for a, b in combinations(factors, 2):
        pair = frozenset((a, b))
        confirm_key = (pair, "confirm")
        disconfirm_key = (pair, "disconfirm")
        if confirm_key not in already_asked:
            out.append((confirm_key, f"What evidence supports a relationship between {a} and {b}, in the context of {topic}?"))
        if disconfirm_key not in already_asked:
            out.append(
                (disconfirm_key, f"What evidence shows NO relationship or contradicts a link between {a} and {b}, in the context of {topic}?")
            )
    return out


def generate_expert_persona(agent: Agent, topic: str, quick_model: str | None = None) -> str:
    """Academic mode, step 1: invents one plausible, field-appropriate expert
    persona for the topic (name, credentials, institutions, decades of
    experience) — a narrow, well-scoped call the model is actually reliable
    at, rather than asking every downstream writing call to "sound like an
    expert" with no concrete voice to anchor to. Returns "" (mode degrades to
    the normal generic-analyst voice) if the call fails."""
    model = _resolve_quick_model(agent, quick_model)
    messages = [
        {
            "role": "system",
            "content": (
                "You invent plausible expert personas for scientific/technical writing. Given a "
                "topic, invent ONE fictional-but-realistic domain expert: a name and title, the "
                "single most relevant credential or fellowship for this field, 2-3 real-sounding "
                "institutions or companies where they worked, a rough number of decades of "
                "experience, and an approximate publication count in journals plausible for this "
                "field. Reply with ONLY a 2-3 sentence persona description written in second "
                "person, e.g. 'You are Dr. ..., a ... with N years of ... experience at ..., ... "
                "and .... You have published over N papers in ..., ..., and ....' No preamble, no "
                "other text."
            ),
        },
        {"role": "user", "content": f"Topic: {topic}"},
    ]
    try:
        return agent.client.chat(model, messages, temperature=0.7).strip()
    except (OllamaConnectionError, OllamaModelError) as e:
        logger.warning("generate_expert_persona failed: %s", e)
        return ""


def _academic_style_instructions(persona: str) -> str:
    """Appended to a synthesis call's system prompt when academic_mode is on
    — the shared voice/formatting contract every writing call in the report
    follows, so the persona, LaTeX equations, and citation format stay
    consistent across sections instead of each call inventing its own style."""
    return (
        f"\n\n{persona}\n\nWrite in that voice: precise technical language, a formal but engaged "
        "expository style suitable for a top-tier peer-reviewed review journal. Where a "
        "quantitative relationship is central to the point being made, express it as a LaTeX "
        "equation (inline like $F = Bli$, or a display equation on its own line for anything "
        "derived, e.g. $$\\nabla^2 p = \\frac{1}{c^2}\\frac{\\partial^2 p}{\\partial t^2}$$). When "
        "you reference a specific established finding, cite it inline as "
        "(Author, Year, Journal, Volume:Pages); if you aren't certain of the exact citation, still "
        "give your best real candidate but mark it '[citation needed — proposed: Author, Year, "
        "Journal]' rather than presenting a fabricated citation with false confidence."
    )


_LEAF_LINE_RE = re.compile(r"^\s*(Q|H|M)\s*:\s*(.+)$", re.IGNORECASE | re.MULTILINE)
_LEAF_KEY_BY_TAG = {"Q": "question", "H": "hypothesis", "M": "method"}


def _parse_leaf_items(text: str, n: int) -> list[dict[str, str]]:
    """Parses the Q:/H:/M: leaf format generate_taxonomy_leaves asks for.
    Tolerant of models that drop the blank lines between items or occasionally
    skip a field: a leaf is considered complete (and appended) the moment all
    three tags have been seen since the last complete leaf, or immediately if
    a tag repeats before the previous leaf was completed (the model started a
    new item without finishing the last one) — a partial leaf missing a field
    is dropped rather than emitted with a blank field."""
    leaves: list[dict[str, str]] = []
    current: dict[str, str] = {}
    for tag, content in _LEAF_LINE_RE.findall(text):
        key = _LEAF_KEY_BY_TAG[tag.upper()]
        if key in current:
            current = {}
        current[key] = content.strip()
        if len(current) == 3:
            leaves.append(current)
            current = {}
    return leaves[:n]


def generate_taxonomy_domains(
    agent: Agent, topic: str, n: int = DEFAULT_TAXONOMY_DOMAINS, quick_model: str | None = None
) -> list[str]:
    """Academic mode taxonomy, level 1: the broad primary research domains a
    review article on this topic would organize its "future work" agenda
    around — pure list generation, the same parse_numbered_list pattern used
    for follow-up questions and factors elsewhere in this module."""
    model = _resolve_quick_model(agent, quick_model)
    messages = [
        {
            "role": "system",
            "content": (
                "You are cataloguing a field of research for a comprehensive review article. Given "
                f"a topic, name exactly {n} distinct, non-overlapping PRIMARY research domains "
                "within it — broad areas the review would organize its research agenda around "
                "(not the topic restated, not vague themes). Reply with ONLY a numbered list of "
                f"exactly {n} short domain names, one per line, no other text."
            ),
        },
        {"role": "user", "content": f"Topic: {topic}"},
    ]
    try:
        text = agent.client.chat(model, messages, temperature=0.6)
    except (OllamaConnectionError, OllamaModelError) as e:
        logger.warning("generate_taxonomy_domains failed: %s", e)
        return []
    return _parse_numbered_list(text, n)


def generate_taxonomy_subtopics(
    agent: Agent,
    topic: str,
    domain: str,
    n: int = DEFAULT_TAXONOMY_SUBTOPICS_PER_DOMAIN,
    quick_model: str | None = None,
) -> list[str]:
    """Academic mode taxonomy, level 2: narrower secondary topics within one
    primary domain, each broad enough to still contain several individual
    research questions."""
    model = _resolve_quick_model(agent, quick_model)
    messages = [
        {
            "role": "system",
            "content": (
                "You are cataloguing a field of research. Given an overall topic and one primary "
                f"research domain within it, name exactly {n} distinct SECONDARY topics within "
                "that domain specifically — narrower than the domain but still broad enough to "
                "contain several individual research questions. Reply with ONLY a numbered list of "
                f"exactly {n} short secondary-topic names, one per line, no other text."
            ),
        },
        {"role": "user", "content": f"Overall topic: {topic}\nPrimary domain: {domain}"},
    ]
    try:
        text = agent.client.chat(model, messages, temperature=0.6)
    except (OllamaConnectionError, OllamaModelError) as e:
        logger.warning("generate_taxonomy_subtopics failed for domain %r: %s", domain, e)
        return []
    return _parse_numbered_list(text, n)


def generate_taxonomy_leaves(
    agent: Agent,
    topic: str,
    domain: str,
    subtopic: str,
    n: int = DEFAULT_TAXONOMY_LEAVES_PER_SUBTOPIC,
    quick_model: str | None = None,
) -> list[dict[str, str]]:
    """Academic mode taxonomy, level 3 (the leaves): specific, testable,
    thesis-chapter-sized research questions within one secondary topic, each
    paired with a falsifiable hypothesis and a concrete method — grounded,
    answerable research units rather than the topic restated at increasing
    specificity."""
    model = _resolve_quick_model(agent, quick_model)
    messages = [
        {
            "role": "system",
            "content": (
                "You are proposing original research questions for a scientific review's future-work "
                "agenda. Given an overall topic, a primary research domain, and one secondary topic "
                f"within it, propose exactly {n} specific, testable, thesis-chapter-sized research "
                "questions — real, narrow, answerable questions, not restatements of the topic. For "
                "each one give: a specific research QUESTION, a falsifiable HYPOTHESIS it would test, "
                "and a concrete experimental or theoretical METHOD to answer it. Reply with ONLY the "
                "leaf items in EXACTLY this format, one after another with a blank line between "
                "items, no numbering, no other text:\nQ: <question>\nH: <hypothesis>\nM: <method>"
            ),
        },
        {"role": "user", "content": f"Overall topic: {topic}\nPrimary domain: {domain}\nSecondary topic: {subtopic}"},
    ]
    try:
        text = agent.client.chat(model, messages, temperature=0.7)
    except (OllamaConnectionError, OllamaModelError) as e:
        logger.warning("generate_taxonomy_leaves failed for %r / %r: %s", domain, subtopic, e)
        return []
    leaves = _parse_leaf_items(text, n)
    for leaf in leaves:
        leaf["domain"] = domain
        leaf["subtopic"] = subtopic
    return leaves


def generate_research_taxonomy(
    agent: Agent,
    topic: str,
    quick_model: str | None = None,
    n_domains: int = DEFAULT_TAXONOMY_DOMAINS,
    subtopics_per_domain: int = DEFAULT_TAXONOMY_SUBTOPICS_PER_DOMAIN,
    leaves_per_subtopic: int = DEFAULT_TAXONOMY_LEAVES_PER_SUBTOPIC,
    workers: int = DEFAULT_SYNTHESIS_WORKERS,
) -> list[dict[str, str]]:
    """Builds the full 3-level taxonomy the same way the rest of this module
    handles anything too big for one call: deterministic pagination (ask for
    a numbered list, parse it, recurse into each item) rather than asking a
    small local model for "hundreds of rigorous research questions" in a
    single shot, which it cannot do reliably. Every independent branch at
    each level is fanned out concurrently on a thread pool — level 2 across
    domains, level 3 across every (domain, subtopic) pair — the same pattern
    synthesize_report uses for section synthesis, since these calls are just
    as independent of each other as separate report sections are."""
    domains = generate_taxonomy_domains(agent, topic, n_domains, quick_model)
    if not domains:
        return []

    with ThreadPoolExecutor(max_workers=max(1, min(workers, len(domains)))) as pool:
        subtopic_futures = {
            pool.submit(generate_taxonomy_subtopics, agent, topic, domain, subtopics_per_domain, quick_model): domain
            for domain in domains
        }
        subtopics_by_domain = {subtopic_futures[f]: f.result() for f in subtopic_futures}

    pairs = [(domain, subtopic) for domain in domains for subtopic in subtopics_by_domain.get(domain, [])]
    if not pairs:
        return []

    with ThreadPoolExecutor(max_workers=max(1, min(workers, len(pairs)))) as pool:
        leaf_futures = [
            pool.submit(generate_taxonomy_leaves, agent, topic, domain, subtopic, leaves_per_subtopic, quick_model)
            for domain, subtopic in pairs
        ]
        return [leaf for future in leaf_futures for leaf in future.result()]


def render_taxonomy_markdown(leaves: list[dict[str, str]]) -> str:
    """Regroups the flat leaf list back into domain -> subtopic Markdown
    headings, in first-seen order, with each leaf numbered sequentially."""
    if not leaves:
        return ""

    domain_order: list[str] = []
    by_domain: dict[str, list[dict[str, str]]] = {}
    for leaf in leaves:
        d = leaf["domain"]
        if d not in by_domain:
            by_domain[d] = []
            domain_order.append(d)
        by_domain[d].append(leaf)

    lines = [
        "## Research Taxonomy",
        "",
        f"*{len(leaves)} distinct, testable research questions across {len(domain_order)} primary domain(s).*",
        "",
    ]
    n = 0
    for domain in domain_order:
        lines.append(f"### {domain}")
        lines.append("")
        subtopic_order: list[str] = []
        by_subtopic: dict[str, list[dict[str, str]]] = {}
        for leaf in by_domain[domain]:
            s = leaf["subtopic"]
            if s not in by_subtopic:
                by_subtopic[s] = []
                subtopic_order.append(s)
            by_subtopic[s].append(leaf)
        for subtopic in subtopic_order:
            lines.append(f"#### {subtopic}")
            lines.append("")
            for leaf in by_subtopic[subtopic]:
                n += 1
                lines.append(f"{n}. **Q:** {leaf['question']}")
                lines.append(f"   **Hypothesis:** {leaf['hypothesis']}")
                lines.append(f"   **Method:** {leaf['method']}")
                lines.append("")
    return "\n".join(lines)


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


def synthesize_section(agent: Agent, model: str, angle: str, sources: list[Source], persona: str = "") -> str:
    """One angle, synthesized on its own. A giant single call covering every
    round was tried first and failed twice in real testing — the model
    consistently let whichever angle appeared last in the context crowd out
    earlier ones (recency bias over long contexts). Synthesizing each round
    independently makes that structurally impossible: each call only ever
    sees one angle's material, so every angle reliably gets its own section.

    persona (academic mode only — see generate_expert_persona) switches the
    voice to that invented domain expert and requires LaTeX equations /
    formal citations where appropriate; empty string keeps the plain
    generic-analyst voice."""
    corpus = _build_corpus(sources)
    system = (
        "You are a research analyst. Using ONLY the provided source excerpts, write a "
        "focused, well-organized section covering this specific angle. Cite claims by "
        "source number like [Source 2], and note if sources disagree. Do not invent "
        "facts that aren't present in the sources, and do not include unrelated "
        "boilerplate from the sources (contact details, ads, navigation text, etc.)."
    )
    if persona:
        system += _academic_style_instructions(persona)
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": f"Angle: {angle}\n\n{corpus}\n\nWrite this section now."},
    ]
    try:
        return agent.client.chat(model, messages, temperature=agent.config.temperature)
    except (OllamaConnectionError, OllamaModelError) as e:
        return f"(Section synthesis failed for '{angle}': {e})"


def verify_section(agent: Agent, model: str, angle: str, sources: list[Source], draft: str, academic: bool = False) -> str:
    """Self-fact-check pass: re-reads the drafted section against its own
    source excerpts and corrects anything unsupported. Small models
    sometimes drift from "summarize this" into adding a plausible-sounding
    but unstated detail — this catches that before it reaches the report.

    academic=True (mirrors ResearchOptions.academic_mode) adds an explicit
    carve-out for LaTeX equations and `[citation needed — proposed: ...]`
    markers: those are standard field/derivation content and explicit
    uncertainty flags, not sourced claims, so the fact-checker shouldn't
    strip them just because the scraped web sources don't restate them
    verbatim."""
    corpus = _build_corpus(sources)
    system = (
        "You are a meticulous fact-checking editor reviewing a drafted research section "
        "against its source material. Compare every factual claim in the draft to the "
        "source excerpts below. If a claim is not supported by any source, remove it or "
        "rewrite it as clearly uncertain. Keep correct, well-supported claims and their "
        "[Source N] citations exactly as they are. Reply with ONLY the corrected section "
        "text — no preamble, no commentary about what you changed."
    )
    if academic:
        system += (
            " This section may contain LaTeX equations and citations already marked "
            "'[citation needed — proposed: ...]' — these are standard field/derivation "
            "content and explicit uncertainty flags, not claims sourced from the excerpts "
            "above. Preserve them exactly as written unless they directly contradict a source."
        )
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": f"{corpus}\n\nDrafted section:\n{draft}\n\nReturn the corrected section now."},
    ]
    try:
        fixed = agent.client.chat(model, messages, temperature=0.2)
        return fixed.strip() or draft
    except (OllamaConnectionError, OllamaModelError) as e:
        logger.warning("verify_section failed for '%s': %s", angle, e)
        return draft


def synthesize_key_findings(agent: Agent, model: str, topic: str, combined_body: str, persona: str = "") -> str:
    system = (
        "You are a research editor. You'll be given a topic and a set of already-written "
        "research sections. Write ONLY a short introduction (2-3 sentences framing the "
        "topic), then a '## Key Findings' section with 5-8 concise bullet points capturing "
        "the single most important, concrete takeaways a busy reader needs — specific "
        "numbers, names, and conclusions, not vague generalities. Do not rewrite or repeat "
        "the sections themselves. Label the findings list exactly '## Key Findings'."
    )
    if persona:
        system += _academic_style_instructions(persona)
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": f"Topic: {topic}\n\nSections already written:\n{combined_body}"},
    ]
    try:
        return agent.client.chat(model, messages, temperature=agent.config.temperature)
    except (OllamaConnectionError, OllamaModelError):
        return ""


def synthesize_contradictions(agent: Agent, model: str, topic: str, combined_body: str, persona: str = "") -> str:
    """A dedicated pass whose only job is to look for disagreement — section
    synthesis is written per-angle and can't see across angles, so nothing
    upstream of this ever compares sections against each other."""
    system = (
        "You are a skeptical research editor. Given a topic and a set of already-written "
        "research sections (each cites sources like [Source 2]), identify any places "
        "where sources disagree, numbers conflict, or important questions remain "
        "open/unresolved. Be specific — name the conflicting claims and which "
        "sections/sources they came from. If you genuinely find nothing significant, say "
        "so in one sentence instead of inventing disagreement. Label your output exactly "
        "'## Contradictions & Open Questions'."
    )
    if persona:
        system += _academic_style_instructions(persona)
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": f"Topic: {topic}\n\nSections already written:\n{combined_body}"},
    ]
    try:
        return agent.client.chat(model, messages, temperature=agent.config.temperature)
    except (OllamaConnectionError, OllamaModelError):
        return ""


def synthesize_factor_relationships(agent: Agent, model: str, topic: str, factors: list[str], combined_body: str) -> str:
    """Factor/correlation digging, step 3: maps out the factors identified
    during research and summarizes what the sections actually found about
    how each pair relates — the analytical payoff of the pairwise-question
    rounds pairwise_relationship_questions() generated."""
    factor_list = "\n".join(f"- {f}" for f in factors)
    messages = [
        {
            "role": "system",
            "content": (
                "You are a research analyst summarizing a factor/correlation analysis. You will "
                "be given a topic, a list of factors that were investigated, and the written "
                "research sections (which include sections specifically about how pairs of these "
                "factors relate to each other). Write a '## Factors & Correlations' section that: "
                "(1) briefly lists the factors examined, and (2) for each meaningful relationship "
                "found between factors, states it plainly — e.g. 'X increases with Y' or "
                "'X and Z showed no clear relationship in the sources' — citing sources like "
                "[Source 2]. Only state relationships the sections actually support; say so "
                "explicitly when the evidence is thin or mixed. Label your output exactly "
                "'## Factors & Correlations'."
            ),
        },
        {
            "role": "user",
            "content": f"Topic: {topic}\n\nFactors examined:\n{factor_list}\n\nSections already written:\n{combined_body}",
        },
    ]
    try:
        return agent.client.chat(model, messages, temperature=agent.config.temperature)
    except (OllamaConnectionError, OllamaModelError):
        return ""


def synthesize_scientific_assessment(
    agent: Agent, model: str, topic: str, hypothesis: str, factors: list[str], sources: list[Source]
) -> str:
    """Scientific mode's payoff step: rates evidence strength per factor
    relationship the way a systematic review would, weighting sources by
    their code-computed credibility tier (_score_credibility) rather than
    treating a blog post and a peer-reviewed study as equally trustworthy —
    the difference between "a report" and "a scientist's assessment"."""
    factor_list = "\n".join(f"- {f}" for f in factors)
    sample = "\n\n".join(f"[{s.title} — credibility: {s.credibility}]\n{s.text[:1000]}" for s in sources[:40])
    messages = [
        {
            "role": "system",
            "content": (
                "You are a scientist evaluating evidence for a hypothesis, the way a systematic "
                "review would. You will be given a hypothesis, the factors examined, and source "
                "excerpts each tagged with a credibility tier (high = peer-reviewed/.gov/.edu "
                "sources, medium = reputable news/health sites, low = blogs/forums/unknown). For "
                "each meaningful factor relationship: state the relationship, rate the strength of "
                "evidence (Strong / Moderate / Weak / Conflicting / Insufficient) weighting "
                "high-credibility sources more heavily, and explicitly note if the evidence shows "
                "correlation without established causation. Then list confounding variables or "
                "limitations a careful scientist would flag, and state whether the evidence "
                "supports, contradicts, or is inconclusive about the hypothesis. Do not overstate "
                "certainty. Label your output exactly '## Scientific Assessment'."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Topic: {topic}\n\nHypothesis: {hypothesis}\n\nFactors examined:\n{factor_list}\n\n"
                f"Source excerpts:\n{sample}"
            ),
        },
    ]
    try:
        return agent.client.chat(model, messages, temperature=agent.config.temperature)
    except (OllamaConnectionError, OllamaModelError):
        return ""


def synthesize_conclusion(
    agent: Agent, model: str, topic: str, combined_body: str, contradictions: str, persona: str = ""
) -> str:
    """Written last, after contradictions are known, so it can honestly
    reflect whatever uncertainty was found instead of a falsely tidy wrap-up."""
    system = (
        "You are a research editor writing the closing section of a report. Given the "
        "topic, the written sections, and a summary of any contradictions/open questions, "
        "write ONLY a short conclusion (3-5 sentences) synthesizing across the sections, "
        "honestly reflecting any uncertainty noted. Label it exactly '## Conclusion'."
    )
    if persona:
        system += _academic_style_instructions(persona)
    messages = [
        {"role": "system", "content": system},
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
    factors: list[str] | None = None,
    hypothesis: str = "",
    persona: str = "",
) -> str:
    options = options or ResearchOptions()
    model = _resolve_synthesis_model(agent, options.synthesis_model)
    verify = (
        options.verify_sections
        if options.verify_sections is not None
        else bool(agent.config.get("research", "verify_sections", default=DEFAULT_VERIFY_SECTIONS))
    )
    grouped = _group_by_round(sources)
    workers = options.synthesis_workers if options.synthesis_workers is not None else int(
        agent.config.get("research", "synthesis_workers", default=DEFAULT_SYNTHESIS_WORKERS)
    )
    workers = max(1, min(workers, len(grouped) or 1))
    console.print(f"[dim]Writing {len(grouped)} section(s) with '{model}' ({workers} at a time)...[/dim]")

    def _draft_and_verify(angle: str, angle_sources: list[Source]) -> str:
        draft = synthesize_section(agent, model, angle, angle_sources, persona=persona)
        return verify_section(agent, model, angle, angle_sources, draft, academic=bool(persona)) if verify else draft

    sections: list[tuple[str, str] | None] = [None] * len(grouped)
    if grouped:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(_draft_and_verify, angle, angle_sources): i
                for i, (angle, angle_sources) in enumerate(grouped)
            }
            for future in futures:
                i = futures[future]
                angle = grouped[i][0]
                sections[i] = (angle, future.result())
                console.print(f"  [dim]section done:[/dim] {angle[:70]}")

    combined_body = "\n\n".join(f"## {angle}\n\n{text}" for angle, text in sections)

    hypothesis_block = f"## Hypothesis\n\n{hypothesis}" if hypothesis else ""

    # Key findings, contradictions, and factor/scientific analysis each only
    # read combined_body/factors/sources — none depends on another's output —
    # so they run concurrently too. The conclusion is the one pass that
    # genuinely has to wait: it reads the contradictions text.
    console.print("[dim]Writing introduction, contradictions, and factor analysis (in parallel)...[/dim]")
    with ThreadPoolExecutor(max_workers=3) as pool:
        opening_future = pool.submit(synthesize_key_findings, agent, model, topic, combined_body, persona)
        contradictions_future = pool.submit(synthesize_contradictions, agent, model, topic, combined_body, persona)
        factor_future = None
        if factors:
            if options.scientific_mode:
                factor_future = pool.submit(
                    synthesize_scientific_assessment, agent, model, topic, hypothesis, factors, sources
                )
            else:
                factor_future = pool.submit(synthesize_factor_relationships, agent, model, topic, factors, combined_body)
        opening = opening_future.result()
        contradictions = contradictions_future.result()
        factor_relationships = factor_future.result() if factor_future else ""

    console.print("[dim]Writing conclusion...[/dim]")
    conclusion = synthesize_conclusion(agent, model, topic, combined_body, contradictions, persona)

    taxonomy_block = ""
    if options.academic_mode:
        console.print(
            "[dim]Building research taxonomy (many small, narrowly-scoped model calls, "
            "fanned out concurrently)...[/dim]"
        )
        leaves = generate_research_taxonomy(agent, topic, quick_model=options.quick_model, workers=workers)
        if leaves:
            taxonomy_block = render_taxonomy_markdown(leaves)
            console.print(f"[dim]Research taxonomy: {len(leaves)} leaf question(s) generated.[/dim]")
        else:
            console.print("[yellow]Research taxonomy generation produced no usable leaves.[/yellow]")

    return "\n\n".join(
        p for p in (
            opening, hypothesis_block, factor_relationships, combined_body, contradictions, conclusion,
            taxonomy_block,
        ) if p
    )


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
    factors: list[str] = []
    hypothesis = ""
    persona = ""
    factor_analysis = bool(options.factor_analysis)
    scientific_mode = bool(options.scientific_mode)
    academic_mode = bool(options.academic_mode)

    # Everything that digs for more material is interruptible: Ctrl+C at any
    # point here drops straight to synthesis with whatever's been gathered
    # so far, instead of crashing — the actual "stop when I say stop"
    # mechanism for a run with no real round/budget ceiling (factor_analysis).
    try:
        if scientific_mode:
            hypothesis = generate_hypothesis(agent, topic, all_sources, quick_model=options.quick_model)
            if hypothesis:
                console.print(f"[bold]Hypothesis:[/bold] {hypothesis}")

        if academic_mode:
            persona = generate_expert_persona(agent, topic, quick_model=options.quick_model)
            if persona:
                console.print(f"[bold]Writing as:[/bold] {persona}")

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

        # Factor/correlation digging — the "read article, pull out the
        # factors it depends on, then dig into every relationship between
        # them" mode. Keeps extracting more factors and researching every
        # new pair they produce until either extraction genuinely runs dry
        # (two rounds in a row with nothing new) or the round/budget cap
        # (effectively unbounded for the "relentless" preset) is hit — so in
        # practice this only ends via real exhaustion or Ctrl+C.
        if factor_analysis:
            asked_pairs = set()
            dry_rounds = 0
            while dry_rounds < 2 and total_chars < max_corpus_chars and round_count < max_total_rounds:
                console.print(f"[dim]Extracting factors ({len(factors)} found so far)...[/dim]")
                new_factors = extract_factors(agent, topic, all_sources, factors, quick_model=options.quick_model)
                factors.extend(new_factors)
                pairs = (
                    scientific_relationship_questions(topic, factors, asked_pairs)
                    if scientific_mode
                    else pairwise_relationship_questions(topic, factors, asked_pairs)
                )
                if not new_factors and not pairs:
                    dry_rounds += 1
                    continue
                dry_rounds = 0
                for key, question in pairs:
                    if total_chars >= max_corpus_chars or round_count >= max_total_rounds:
                        break
                    asked_pairs.add(key)
                    label = "(evidence)" if scientific_mode else "(correlation)"
                    console.print(f"[bold]Round {round_count + 1} {label}:[/bold] {question}")
                    _research_round(question)
            if factors:
                console.print(f"[dim]Factor analysis complete — {len(factors)} factor(s), {len(asked_pairs)} question(s) researched.[/dim]")
    except KeyboardInterrupt:
        console.print(
            "\n[yellow]Stopped by user — writing up findings from what's been gathered so far...[/yellow]"
        )

    console.print(
        f"[dim]Read {len(all_sources)} source(s) across {round_count} round(s), "
        f"{total_chars} characters total.[/dim]"
    )

    report = synthesize_report(
        agent, topic, all_sources, followups, options=options, factors=factors,
        hypothesis=hypothesis, persona=persona,
    )
    saved_path = save_report(topic, all_sources, report)

    console.print(f"\n{report}\n")
    console.print(f"[green]Report saved to:[/green] {saved_path}")
    return ResearchResult(
        topic=topic, sources=all_sources, followup_questions=followups, report=report,
        saved_path=saved_path, hypothesis=hypothesis, persona=persona,
    )


# --- interactive setup menu, used by `deskbot research` when run from a real
# terminal with no --mode/--model flags. Arrow-key selection via questionary
# when it's installed (the normal case — it's a hard dependency), with a
# plain numbered-prompt fallback so a missing/broken optional import can
# never take the whole command down. Every actual prompt goes through the
# small _choose/_ask_* wrappers below rather than input()/questionary calls
# inline, so tests can patch just the wrapper and assert on the menu logic
# (preset selection, option assembly) without needing a real terminal.

def _list_available_models(agent: Agent) -> list[str]:
    try:
        return agent.client.list_models()
    except OllamaConnectionError:
        return []


def _choose(message: str, options: list[tuple[str, str]], default_index: int = 0) -> str:
    """options is a list of (value, label) pairs; returns the chosen value."""
    if questionary is not None:
        choices = [questionary.Choice(title=label, value=value) for value, label in options]
        answer = questionary.select(message, choices=choices, default=choices[default_index]).ask()
        return answer if answer is not None else options[default_index][0]
    print(f"\n{message}")
    for i, (_value, label) in enumerate(options, 1):
        print(f"  {i}) {label}")
    raw = input("> ").strip()
    if raw.isdigit() and 1 <= int(raw) <= len(options):
        return options[int(raw) - 1][0]
    return options[default_index][0]


def _ask_text(message: str, default: str = "") -> str:
    if questionary is not None:
        answer = questionary.text(message, default=default).ask()
        return (answer if answer is not None else default).strip()
    raw = input(f"{message} [{default}]: ").strip()
    return raw or default


def _ask_confirm(message: str, default: bool) -> bool:
    if questionary is not None:
        answer = questionary.confirm(message, default=default).ask()
        return default if answer is None else answer
    raw = input(f"{message} (y/n) [{'y' if default else 'n'}]: ").strip().lower()
    return default if not raw else raw.startswith("y")


def _ask_number(message: str, default: int) -> int:
    raw = _ask_text(message, str(default))
    try:
        return int(raw)
    except ValueError:
        console.print(f"[dim]Not a number, using default ({default}).[/dim]")
        return default


def _ask_model(message: str, available: list[str]) -> str:
    """Tab-completion over locally pulled models when questionary supports
    it; plain free-text entry otherwise (still lets you type any name)."""
    if questionary is not None and available:
        answer = questionary.autocomplete(message, choices=available, default="").ask()
        return (answer or "").strip()
    return _ask_text(message, "")


_MODE_MENU_OPTIONS: list[tuple[str, str]] = [
    ("quick", f"Quick      — {RESEARCH_MODE_DESCRIPTIONS['quick']}"),
    ("standard", f"Standard   — {RESEARCH_MODE_DESCRIPTIONS['standard']}"),
    ("deep", f"Deep       — {RESEARCH_MODE_DESCRIPTIONS['deep']}"),
    ("relentless", f"Relentless — {RESEARCH_MODE_DESCRIPTIONS['relentless']}"),
    ("scientist", f"Scientist  — {RESEARCH_MODE_DESCRIPTIONS['scientist']}"),
    ("authority", f"Authority  — {RESEARCH_MODE_DESCRIPTIONS['authority']}"),
    ("custom", "Custom     — configure every setting yourself."),
]

_ACADEMIC_MODE_PROMPT = (
    "Write as an invented domain-expert authority (LaTeX equations, formal citations) and "
    "append a large taxonomy of specific research questions?"
)


def _prompt_custom_options() -> ResearchOptions:
    console.print("\n[bold]-- custom research settings --[/bold]")
    factor_analysis = _ask_confirm(
        "Dig into factor relationships/correlations until YOU stop it (Ctrl+C) instead of a round cap?", False
    )
    if factor_analysis:
        console.print(
            "[dim](factor analysis on: max sources/rounds/budget below are effectively ignored — "
            "Ctrl+C is the stop.)[/dim]"
        )
        scientific_mode = _ask_confirm(
            "Also use the scientific method — falsifiable hypothesis, actively hunt for "
            "disconfirming evidence too, rate confidence by source credibility?", False
        )
        academic_mode = _ask_confirm(_ACADEMIC_MODE_PROMPT, False)
        return ResearchOptions(
            max_sources=_RELENTLESS_MAX_SOURCES,
            per_source_chars=_ask_number("Max characters read per source", DEFAULT_PER_SOURCE_CHARS),
            max_corpus_chars=_RELENTLESS_MAX_CORPUS_CHARS,
            followup_rounds=_ask_number("Planned follow-up rounds after the overview search", DEFAULT_FOLLOWUP_ROUNDS),
            max_total_rounds=_RELENTLESS_MAX_TOTAL_ROUNDS,
            verify_sections=_ask_confirm("Fact-check each section against its own sources?", DEFAULT_VERIFY_SECTIONS),
            adaptive_digging=True,
            factor_analysis=True,
            scientific_mode=scientific_mode,
            academic_mode=academic_mode,
        )
    academic_mode = _ask_confirm(_ACADEMIC_MODE_PROMPT, False)
    return ResearchOptions(
        max_sources=_ask_number("Max sources to read (across all rounds)", DEFAULT_MAX_SOURCES),
        per_source_chars=_ask_number("Max characters read per source", DEFAULT_PER_SOURCE_CHARS),
        max_corpus_chars=_ask_number("Total character budget for the whole run", DEFAULT_MAX_CORPUS_CHARS),
        followup_rounds=_ask_number("Planned follow-up rounds after the overview search", DEFAULT_FOLLOWUP_ROUNDS),
        max_total_rounds=_ask_number(
            "Max total rounds, safety cap (overview + follow-ups + adaptive)", DEFAULT_MAX_TOTAL_ROUNDS
        ),
        verify_sections=_ask_confirm("Fact-check each section against its own sources?", DEFAULT_VERIFY_SECTIONS),
        adaptive_digging=_ask_confirm("Keep digging adaptively until nothing important is missing?", True),
        factor_analysis=False,
        academic_mode=academic_mode,
    )


def prompt_research_setup(agent: Agent) -> ResearchOptions:
    """Interactive menu: pick a depth preset (or go custom) with arrow keys,
    then optionally pick specific models for this one run. Returns a
    ResearchOptions ready to hand to run_deep_research(); any field the user
    leaves blank falls back to config.yaml as usual."""
    console.print(Panel("[bold]deskbot deep research[/bold]", expand=False, border_style="cyan"))
    mode = _choose("Pick a research method:", _MODE_MENU_OPTIONS, default_index=1)

    options = _prompt_custom_options() if mode == "custom" else replace(RESEARCH_MODE_PRESETS[mode])

    available = _list_available_models(agent)
    if available:
        console.print(f"\n[dim]Models available locally: {', '.join(available)}[/dim]")
    else:
        console.print("\n[dim](Could not reach Ollama to list models — you can still type a name manually.)[/dim]")
    console.print("[dim]Pick models for this run, or leave blank to use config.yaml's defaults.[/dim]")
    planning = _ask_model("Planning model (proposes follow-up/gap questions):", available)
    writing = _ask_text("Writing model (drafts sections, fact-checks, writes the report) ['same' to reuse planning]:")

    if planning:
        options.quick_model = planning
    if writing:
        options.synthesis_model = planning if writing.lower() == "same" else writing

    return options
