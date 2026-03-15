import asyncio
import os
from pathlib import Path
from time import perf_counter
from typing import Awaitable

import httpx
from agno.agent import Agent

from .backend import (
    OPENAI_COMPATIBLE_BACKEND,
    get_agno_model,
    get_execution_backend_name,
)
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
    QUERY_PLANNER_PREMISE,
    SYNTHESIZER_PREMISE,
    TEMPORAL_PREMISE,
)
from .paper_search import is_topic_query, run_paper_search, _paper_search_enabled
from .paper_search import is_topic_query, run_paper_search, _paper_search_enabled
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
FULL_SECTION_CHARS = 5_000
FULL_CONTEXT_CHARS = 25_000
COMPACT_SECTION_CHARS = 2_500
COMPACT_CONTEXT_CHARS = 10_000
FULL_FIGURE_COUNT = 15
FULL_TABLE_COUNT = 6
FULL_REFERENCE_COUNT = 30
COMPACT_FIGURE_COUNT = 6
COMPACT_TABLE_COUNT = 4
COMPACT_REFERENCE_COUNT = 10
PRIMITIVE_SUMMARY_CHARS = 4_500
PAIN_SUMMARY_CHARS = 3_000
IDEA_SUMMARY_CHARS = 2_500
QUERY_MAX_TOKENS = 120
PHASE_MAX_TOKENS = {
    "technical primitive extraction": 2200,
    "pain scanner": 1600,
    "infrastructure inversion": 1400,
    "temporal arbitrage": 1400,
    "cross-pollination": 1600,
    "red team destruction": 1600,
    "final synthesis": 1800,
}


def _get_speed_profile() -> str:
    profile = os.getenv("PIPELINE_SPEED_PROFILE", "balanced").strip().lower()
    return profile if profile in {"balanced", "exhaustive"} else "balanced"


def _get_phase_timeout_seconds() -> float:
    default = 360.0 if _get_speed_profile() == "balanced" else 480.0
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


def _phase_max_tokens(phase: str) -> int | None:
    return PHASE_MAX_TOKENS.get(phase)


def _agentica_connection_help() -> str:
    return "Failed to run agent (timeout or connect error)."


async def spawn_agent(premise: str, model: str, scope: dict | None = None) -> Agent:
    tools = []
    if scope:
        for v in scope.values():
            tools.append(v)

    return Agent(
        model=get_agno_model(model),
        description=premise,
        tools=tools,
    )


def _format_agent_error(phase: str, exc: BaseException) -> str:
    if isinstance(exc, asyncio.TimeoutError):
        return (
            f"{phase} timed out inside Agentica while finalizing the response. "
            "This is usually a transient Agentica invocation timeout."
        )
    return f"{phase} failed with {exc.__class__.__name__}: {exc}"


def _format_direct_error(phase: str, exc: BaseException) -> str:
    if isinstance(exc, asyncio.TimeoutError):
        return f"{phase} timed out while waiting for the direct execution backend."
    return f"{phase} failed with {exc.__class__.__name__}: {exc}"


async def call_agent_text(
    agent: Agent,
    prompt: str,
    *,
    phase: str,
) -> str:
    try:
        # Agno provides synchronous and asynchronous run methods
        response = await asyncio.to_thread(agent.run, prompt)
        return response.content
    except BaseException as exc:
        raise AgentExecutionError(_format_agent_error(phase, exc)) from exc


# Removing direct openai_backend fallback.
# In Agno, we just use the agno model abstraction everywhere!


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


def _parse_search_queries(text: str) -> list[str]:
    queries: list[str] = []
    seen: set[str] = set()
    for raw_line in text.splitlines():
        line = raw_line.strip().lstrip("-*0123456789. ").strip()
        if len(line) < 8:
            continue
        if line in seen:
            continue
        seen.add(line)
        queries.append(line)
    return queries[:2]


def _fallback_queries(
    *,
    phase: str,
    paper: PaperContent,
) -> list[str]:
    if phase == "pain scanner":
        return [
            f"{paper.title} enterprise pain budget current market",
            f"{paper.title} customer pain companies spending current",
        ]
    return [
        f"{paper.title} related paper 2025 2026",
        f"{paper.title} industry trend startup adoption 2025 2026",
    ]


async def build_search_packet(
    *,
    agent: Agent,
    paper: PaperContent,
    primitives_summary: str,
    trace: SearchTrace,
    phase: str,
    default_intent: str,
    model: str,
) -> str:
    planner_prompt = (
        f"Paper title: {paper.title}\n"
        f"Abstract: {paper.abstract}\n\n"
        f"Technical primitives summary:\n{primitives_summary}\n\n"
        f"Generate the best web search queries for the {phase} phase."
    )
    
    planner = await spawn_agent(
        premise=QUERY_PLANNER_PREMISE,
        model=model,
    )
    try:
        planned = await call_agent_text(
            planner,
            planner_prompt,
            phase=f"{phase} search planning",
        )
        queries = _parse_search_queries(planned)
    except AgentExecutionError:
        queries = []

    if not queries:
        queries = _fallback_queries(phase=phase, paper=paper)

    search_tool = make_web_search_tool(
        default_intent=default_intent,
        trace=trace,
    )
    packets: list[str] = []
    for query in queries[:2]:
        packets.append(f"QUERY: {query}\n{await search_tool(query)}")
    return "\n\n".join(packets)

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


async def _run_pipeline(arxiv_id_or_url: str, model: str = DEFAULT_MODEL) -> str:
    """Run the pipeline using Agno."""
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

    phase_started_at = _phase_started("🚀 Phase 2: Running parallel analysis agents...")
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
    temporal_task = call_agent_text(
        temporal_agent,
        f"Paper context:\n{compact_context}\n\n"
        f"Technical primitives:\n{primitives_summary}\n\n"
        "Identify temporal arbitrage windows. What can be built RIGHT NOW that "
        "won't be obvious for 12-24 months? Search the web for recent related "
        "papers and industry trends.",
        phase="temporal arbitrage",
    )

    phase_two_results = await gather_agent_calls(
        {
            "pain scanner": pain_task,
            "infrastructure inversion": infra_task,
            "temporal arbitrage": temporal_task,
        }
    )
    pain_raw = phase_two_results["pain scanner"]
    infra_raw = phase_two_results["infrastructure inversion"]
    temporal_raw = phase_two_results["temporal arbitrage"]
    _phase_finished(
        "Phase 2",
        phase_started_at,
        details=(
            f"(pain web calls={pain_trace.calls_used}, temporal web calls={temporal_trace.calls_used})"
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


async def run_pipeline(arxiv_id_or_url: str, model: str = DEFAULT_MODEL) -> str:
    """Run the paper-to-product pipeline using the configured execution backend."""
    # Phase 0 (optional): PASA-style paper search for topic queries
    if _paper_search_enabled() and is_topic_query(arxiv_id_or_url):
        results = await run_paper_search(arxiv_id_or_url, model=model)
        if not results:
            raise AgentExecutionError(
                f"Paper search found no relevant papers for topic: {arxiv_id_or_url}"
            )
        top_paper = results[0]
        print(f"📄 Selected top paper: [{top_paper.arxiv_id}] {top_paper.title}")
        arxiv_id_or_url = top_paper.arxiv_id

    return await _run_pipeline(arxiv_id_or_url, model)
