import asyncio
import os
from pathlib import Path
from time import perf_counter
from typing import Awaitable

import httpx
from agentica import spawn

from .errors import AgentExecutionError, AgenticaConnectionError
from .ingestion import fetch_paper
from .models import PaperContent
from .prompts import (
    CROSSPOLLINATOR_PREMISE,
    DECOMPOSER_PREMISE,
    DEFAULT_MODEL,
    DESTROYER_PREMISE,
    INFRA_INVERSION_PREMISE,
    PAIN_SCANNER_PREMISE,
    SYNTHESIZER_PREMISE,
    TEMPORAL_PREMISE,
)
from .reporting import build_report
from .research import SearchTrace, make_disabled_web_search_tool, make_web_search_tool


PRIORITY_SECTION_KEYS = [
    "abstract",
    "preamble",
    "introduction",
    "method",
    "approach",
    "experiments",
    "results",
    "conclusion",
    "discussion",
]
SPAWN_TIMEOUT_SECONDS = 30.0
FULL_SECTION_CHARS = 8_000
FULL_CONTEXT_CHARS = 80_000
COMPACT_SECTION_CHARS = 2_500
COMPACT_CONTEXT_CHARS = 18_000
FULL_FIGURE_COUNT = 20
FULL_TABLE_COUNT = 10
FULL_REFERENCE_COUNT = 30
COMPACT_FIGURE_COUNT = 6
COMPACT_TABLE_COUNT = 4
COMPACT_REFERENCE_COUNT = 10
PRIMITIVE_SUMMARY_CHARS = 4_500
PAIN_SUMMARY_CHARS = 3_000
IDEA_SUMMARY_CHARS = 2_500


def _get_speed_profile() -> str:
    profile = os.getenv("PIPELINE_SPEED_PROFILE", "balanced").strip().lower()
    return profile if profile in {"balanced", "exhaustive"} else "balanced"


def _get_phase_timeout_seconds() -> float:
    default = 180.0 if _get_speed_profile() == "balanced" else 300.0
    raw_value = os.getenv("AGENT_PHASE_TIMEOUT_SECONDS", str(default))
    try:
        value = float(raw_value)
    except ValueError:
        return default
    return max(30.0, value)


def _redteam_search_enabled() -> bool:
    return os.getenv("ENABLE_REDTEAM_SEARCH", "0").strip().lower() in {"1", "true", "yes"}


def _agent_logs_enabled() -> bool:
    return os.getenv("ENABLE_AGENT_LOGS", "0").strip().lower() in {"1", "true", "yes"}


def _truncate_text(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + "\n\n[...truncated...]"


def _phase_started(label: str) -> float:
    print(label)
    return perf_counter()


def _phase_finished(label: str, started_at: float, details: str = "") -> None:
    elapsed = perf_counter() - started_at
    suffix = f" {details}" if details else ""
    print(f"  ✅ {label} complete in {elapsed:.1f}s{suffix}")


def _agentica_connection_help() -> str:
    base_url = os.getenv("AGENTICA_BASE_URL", "https://api.platform.symbolica.ai")
    session_manager_url = os.getenv("S_M_BASE_URL")
    target = session_manager_url or base_url
    return (
        "Timed out while connecting to the Agentica backend. "
        f"Current target: {target}. "
        "Check outbound network access, verify the backend URL, or set "
        "S_M_BASE_URL to a reachable local session manager."
    )


async def spawn_agent(**kwargs):
    if "listener" not in kwargs and not _agent_logs_enabled():
        kwargs["listener"] = None
    try:
        return await asyncio.wait_for(
            spawn(**kwargs),
            timeout=SPAWN_TIMEOUT_SECONDS,
        )
    except asyncio.TimeoutError as exc:
        raise AgenticaConnectionError(
            f"Timed out after {SPAWN_TIMEOUT_SECONDS}s waiting for Agentica "
            f"to create an agent. {_agentica_connection_help()}"
        ) from exc
    except httpx.TimeoutException as exc:
        raise AgenticaConnectionError(_agentica_connection_help()) from exc
    except httpx.HTTPError as exc:
        raise AgenticaConnectionError(
            f"Agentica request failed while creating an agent: {exc}"
        ) from exc


def _format_agent_error(phase: str, exc: BaseException) -> str:
    if isinstance(exc, asyncio.TimeoutError):
        return (
            f"{phase} timed out inside Agentica while finalizing the response. "
            "This is usually a transient Agentica invocation timeout."
        )
    return f"{phase} failed with {exc.__class__.__name__}: {exc}"


async def call_agent_text(
    agent,
    prompt: str,
    *,
    phase: str,
) -> str:
    try:
        return await asyncio.wait_for(
            agent.call(str, prompt),
            timeout=_get_phase_timeout_seconds(),
        )
    except BaseException as exc:
        raise AgentExecutionError(_format_agent_error(phase, exc)) from exc
    finally:
        # Attempt graceful teardown so Agentica finalizers don't outlive the
        # pipeline.  If the agent has no .close(), silently skip.
        close = getattr(agent, "close", None)
        if close is not None:
            try:
                await asyncio.wait_for(asyncio.shield(close()), timeout=5.0)
            except Exception:
                pass  # best-effort; don't mask the real error


async def gather_agent_calls(calls: dict[str, Awaitable[str]]) -> dict[str, str]:
    names = list(calls)
    results = await asyncio.gather(*(calls[name] for name in names), return_exceptions=True)

    failures: list[str] = []
    outputs: dict[str, str] = {}
    for name, result in zip(names, results):
        if isinstance(result, BaseException):
            failures.append(_format_agent_error(name, result))
            continue
        outputs[name] = result

    if failures:
        raise AgentExecutionError(" | ".join(failures))

    return outputs


def _collect_key_sections(
    paper: PaperContent,
    *,
    section_char_limit: int,
) -> dict[str, str]:
    key_sections: dict[str, str] = {}
    for key in PRIORITY_SECTION_KEYS:
        for section_name, content in paper.sections.items():
            if key in section_name.lower():
                key_sections[section_name] = content[:section_char_limit]
    return key_sections


def _build_paper_context(
    paper: PaperContent,
    *,
    section_char_limit: int,
    context_char_limit: int,
    figure_count: int,
    table_count: int,
    reference_count: int,
    primitives_summary: str = "",
) -> str:
    key_sections = _collect_key_sections(
        paper,
        section_char_limit=section_char_limit,
    )
    context = (
        f"TITLE: {paper.title}\n"
        f"AUTHORS: {', '.join(paper.authors[:10])}\n"
        f"ABSTRACT: {paper.abstract}\n\n"
        f"KEY SECTIONS:\n"
        + "\n\n".join(f"=== {name} ===\n{content}" for name, content in key_sections.items())
        + "\n\nFIGURE CAPTIONS:\n"
        + "\n".join(paper.figures_captions[:figure_count])
        + "\n\nTABLE SUMMARIES:\n"
        + "\n".join(paper.tables_text[:table_count])
        + "\n\nREFERENCED WORKS:\n"
        + "\n".join(paper.references_titles[:reference_count])
    )
    if primitives_summary:
        context += "\n\nTECHNICAL PRIMITIVES SUMMARY:\n" + primitives_summary
    if len(context) > context_char_limit:
        return context[:context_char_limit] + "\n\n[...truncated...]"
    return context


def build_full_paper_context(paper: PaperContent) -> str:
    return _build_paper_context(
        paper,
        section_char_limit=FULL_SECTION_CHARS,
        context_char_limit=FULL_CONTEXT_CHARS,
        figure_count=FULL_FIGURE_COUNT,
        table_count=FULL_TABLE_COUNT,
        reference_count=FULL_REFERENCE_COUNT,
    )


def build_compact_paper_context(
    paper: PaperContent,
    *,
    primitives_summary: str,
) -> str:
    return _build_paper_context(
        paper,
        section_char_limit=COMPACT_SECTION_CHARS,
        context_char_limit=COMPACT_CONTEXT_CHARS,
        figure_count=COMPACT_FIGURE_COUNT,
        table_count=COMPACT_TABLE_COUNT,
        reference_count=COMPACT_REFERENCE_COUNT,
        primitives_summary=primitives_summary,
    )


async def run_pipeline(arxiv_id_or_url: str, model: str = DEFAULT_MODEL) -> str:
    """Run the multi-agent paper analysis pipeline and write the markdown report."""
    speed_profile = _get_speed_profile()
    print(f"📄 Fetching paper: {arxiv_id_or_url}")
    paper = await fetch_paper(arxiv_id_or_url)
    print(f"✅ Loaded: {paper.title} ({len(paper.full_text)} chars)")
    print(f"⚙️ Speed profile: {speed_profile}")

    full_context = build_full_paper_context(paper)
    print(f"🧠 Phase 1 context: {len(full_context)} chars")

    phase_started_at = _phase_started("🔬 Phase 1: Extracting technical primitives...")
    decomposer = await spawn_agent(premise=DECOMPOSER_PREMISE, model=model)
    primitives_raw = await call_agent_text(
        decomposer,
        f"Analyze this paper and extract all atomic technical primitives:\n\n{full_context}",
        phase="technical primitive extraction",
    )
    _phase_finished("Phase 1", phase_started_at)
    primitives_summary = _truncate_text(primitives_raw, PRIMITIVE_SUMMARY_CHARS)
    compact_context = build_compact_paper_context(
        paper,
        primitives_summary=primitives_summary,
    )
    print(f"🧠 Downstream context: {len(compact_context)} chars")

    phase_started_at = _phase_started("🚀 Phase 2A: Running pain scanner and infrastructure inversion...")
    pain_trace = SearchTrace(section_name="Market Pain Mapping")
    temporal_trace = SearchTrace(section_name="Temporal Arbitrage")

    pain_agent = await spawn_agent(
        premise=PAIN_SCANNER_PREMISE,
        model=model,
        scope={
            "web_search": make_web_search_tool(
                default_intent="fast",
                trace=pain_trace,
            )
        },
    )
    infra_agent = await spawn_agent(premise=INFRA_INVERSION_PREMISE, model=model)

    pain_task = call_agent_text(
        pain_agent,
        f"Technical primitives:\n\n{primitives_summary}\n\n"
        f"Paper context:\n{compact_context}\n\n"
        "Search the web to find real, current market pain mapping to these primitives. "
        "Go FAR beyond the paper's own domain.",
        phase="pain scanner",
    )
    infra_task = call_agent_text(
        infra_agent,
        f"Paper context:\n{compact_context}\n\n"
        f"Technical primitives:\n{primitives_summary}\n\n"
        "What NEW problems does widespread adoption of this technique CREATE? "
        "What products solve those second-order problems?",
        phase="infrastructure inversion",
    )

    phase_two_results = await gather_agent_calls(
        {
            "pain scanner": pain_task,
            "infrastructure inversion": infra_task,
        }
    )
    pain_raw = phase_two_results["pain scanner"]
    infra_raw = phase_two_results["infrastructure inversion"]
    _phase_finished(
        "Phase 2A",
        phase_started_at,
        details=(
            f"(pain web calls={pain_trace.calls_used}"
            + (", budget hit" if pain_trace.budget_exhausted else "")
            + ")"
        ),
    )

    phase_started_at = _phase_started("🕒 Phase 2B: Running temporal arbitrage...")
    temporal_agent = await spawn_agent(
        premise=TEMPORAL_PREMISE,
        model=model,
        scope={
            "web_search": make_web_search_tool(
                default_intent="fresh",
                trace=temporal_trace,
            )
        },
    )
    temporal_raw = await call_agent_text(
        temporal_agent,
        f"Paper context:\n{compact_context}\n\n"
        f"Technical primitives:\n{primitives_summary}\n\n"
        "Identify temporal arbitrage windows. What can be built RIGHT NOW that "
        "won't be obvious for 12-24 months? Search the web for recent related "
        "papers and industry trends.",
        phase="temporal arbitrage",
    )
    _phase_finished(
        "Phase 2B",
        phase_started_at,
        details=(
            f"(temporal web calls={temporal_trace.calls_used}"
            + (", budget hit" if temporal_trace.budget_exhausted else "")
            + ")"
        ),
    )

    phase_started_at = _phase_started("🧬 Phase 3: Cross-pollination...")
    crosspoll_agent = await spawn_agent(
        premise=CROSSPOLLINATOR_PREMISE,
        model=model,
    )
    crosspoll_raw = await call_agent_text(
        crosspoll_agent,
        f"Technical primitives:\n{primitives_summary}\n\n"
        f"Market pain points found:\n{_truncate_text(pain_raw, PAIN_SUMMARY_CHARS)}\n\n"
        "Force non-obvious cross-pollination. Skip direct/obvious matches.",
        phase="cross-pollination",
    )
    _phase_finished("Phase 3", phase_started_at)

    phase_started_at = _phase_started("💀 Phase 4: Red team destruction...")
    all_ideas = (
        f"=== IDEAS FROM PAIN MAPPING ===\n{_truncate_text(pain_raw, IDEA_SUMMARY_CHARS)}\n\n"
        f"=== IDEAS FROM CROSS-POLLINATION ===\n{_truncate_text(crosspoll_raw, IDEA_SUMMARY_CHARS)}\n\n"
        f"=== IDEAS FROM INFRASTRUCTURE INVERSION ===\n{_truncate_text(infra_raw, IDEA_SUMMARY_CHARS)}\n\n"
        f"=== IDEAS FROM TEMPORAL ARBITRAGE ===\n{_truncate_text(temporal_raw, IDEA_SUMMARY_CHARS)}\n\n"
    )
    destroyer_scope = (
        {"web_search": make_web_search_tool(default_intent="fast")}
        if _redteam_search_enabled()
        else {"web_search": make_disabled_web_search_tool()}
    )
    destroyer = await spawn_agent(
        premise=DESTROYER_PREMISE,
        model=model,
        scope=destroyer_scope,
    )
    redteam_raw = await call_agent_text(
        destroyer,
        "Here are product ideas from a research paper. Destroy every one.\n\n"
        f"Paper: {paper.title}\n\n{all_ideas}",
        phase="red team destruction",
    )
    _phase_finished(
        "Phase 4",
        phase_started_at,
        details="(red-team search disabled)" if not _redteam_search_enabled() else "",
    )

    phase_started_at = _phase_started("🎯 Phase 5: Final synthesis...")
    synthesizer = await spawn_agent(premise=SYNTHESIZER_PREMISE, model=model)
    final_raw = await call_agent_text(
        synthesizer,
        f"PAPER: {paper.title}\nABSTRACT: {paper.abstract}\n\n"
        f"=== TECHNICAL PRIMITIVES ===\n{primitives_summary}\n\n"
        f"=== MARKET PAIN MAPPING ===\n{_truncate_text(pain_raw, IDEA_SUMMARY_CHARS)}\n\n"
        f"=== CROSS-POLLINATED IDEAS ===\n{_truncate_text(crosspoll_raw, IDEA_SUMMARY_CHARS)}\n\n"
        f"=== INFRASTRUCTURE INVERSION ===\n{_truncate_text(infra_raw, IDEA_SUMMARY_CHARS)}\n\n"
        f"=== TEMPORAL ARBITRAGE ===\n{_truncate_text(temporal_raw, IDEA_SUMMARY_CHARS)}\n\n"
        f"=== RED TEAM DESTRUCTION RESULTS ===\n{_truncate_text(redteam_raw, IDEA_SUMMARY_CHARS)}\n\n"
        "Synthesize all of the above into a final ranked list of the BEST ideas. "
        "Only include ideas that survived red-teaming or were strengthened by it.",
        phase="final synthesis",
    )
    _phase_finished("Phase 5", phase_started_at)

    report = build_report(
        paper=paper,
        primitives=primitives_raw,
        pain=pain_raw,
        crosspoll=crosspoll_raw,
        infra=infra_raw,
        temporal=temporal_raw,
        pain_sources=pain_trace.render_markdown(),
        temporal_sources=temporal_trace.render_markdown(),
        redteam=redteam_raw,
        redteam_sources="",
        final=final_raw,
    )

    safe_id = paper.arxiv_id.replace("/", "_").replace(".", "_")
    output_path = Path(f"products_{safe_id}.md")
    output_path.write_text(report, encoding="utf-8")

    print(f"\n✅ Done! Report saved to: {output_path}")
    print(f"   {len(report)} chars, ~{len(report.splitlines())} lines")
    return str(output_path)
