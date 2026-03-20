import asyncio
import os
from pathlib import Path
from time import perf_counter
from typing import Awaitable, Optional

import httpx
from agentica import spawn
from agentica.logging import set_default_agent_listener

from .backend import (
    AGENTICA_BACKEND,
    OPENAI_COMPATIBLE_BACKEND,
    OpenAICompatibleBackend,
    build_openai_compatible_backend,
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
from .paper_search import is_topic_query, run_paper_search
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
    return os.getenv("ENABLE_REDTEAM_SEARCH", "0").strip().lower() in {
        "1",
        "true",
        "yes",
    }


def _agent_logs_enabled() -> bool:
    return os.getenv("ENABLE_AGENT_LOGS", "0").strip().lower() in {"1", "true", "yes"}


from rich.console import Console
console = Console()

def _truncate_text(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + "\n\n[...truncated...]"


def _phase_started(label: str) -> float:
    console.print(label)
    return perf_counter()


def _phase_finished(label: str, started_at: float, details: str = "") -> None:
    elapsed = perf_counter() - started_at
    suffix = f" {details}" if details else ""
    console.print(f"  ✅ {label} complete in {elapsed:.1f}s{suffix}")


def _phase_max_tokens(phase: str) -> int | None:
    return PHASE_MAX_TOKENS.get(phase)


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


def _format_direct_error(phase: str, exc: BaseException) -> str:
    if isinstance(exc, asyncio.TimeoutError):
        return f"{phase} timed out while waiting for the direct execution backend."
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


async def call_direct_text(
    backend: OpenAICompatibleBackend,
    *,
    system_prompt: str,
    user_prompt: str,
    phase: str,
    model: str,
    max_tokens: int | None = None,
) -> str:
    try:
        return await asyncio.wait_for(
            backend.generate_text(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                model=model,
                phase=phase,
                max_tokens=max_tokens,
            ),
            timeout=_get_phase_timeout_seconds(),
        )
    except BaseException as exc:
        raise AgentExecutionError(_format_direct_error(phase, exc)) from exc


async def gather_agent_calls(calls: dict[str, Awaitable[str]]) -> dict[str, str]:
    names = list(calls)
    results = await asyncio.gather(
        *(calls[name] for name in names), return_exceptions=True
    )

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
    backend: OpenAICompatibleBackend,
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
    try:
        planned = await call_direct_text(
            backend,
            system_prompt=QUERY_PLANNER_PREMISE,
            user_prompt=planner_prompt,
            phase=f"{phase} search planning",
            model=model,
            max_tokens=QUERY_MAX_TOKENS,
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
        + "\n\n".join(
            f"=== {name} ===\n{content}" for name, content in key_sections.items()
        )
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


async def _run_pipeline_with_agentica(
    arxiv_ids: list[str], model: str = DEFAULT_MODEL, user_idea: str = "", github_mapping: dict[str, str] | None = None
) -> str:
    """Run the pipeline using Agentica as the execution backend."""
    if not _agent_logs_enabled():
        set_default_agent_listener(None)
    speed_profile = _get_speed_profile()
    github_mapping = github_mapping or {}
    
    console.print(f"📄 Fetching {len(arxiv_ids)} paper(s): {', '.join(arxiv_ids)}")
    papers = await asyncio.gather(*(fetch_paper(aid, github_mapping.get(aid, "")) for aid in arxiv_ids))
    titles = [p.title for p in papers]
    console.print(f"✅ Loaded: {titles}")
    console.print(f"⚙️ Speed profile: {speed_profile}")

    anchor_text = f"\n\nUSER'S CORE IDEA TO VALIDATE/AUGMENT:\n{user_idea}\nFocus all analysis, synthesis, and simulation around structurally building and stress-testing this specific idea." if user_idea else ""

    full_context = "\n\n---\n\n".join(
        [
            f"PAPER {i+1}:\n{build_full_paper_context(p)}"
            + (f"\nGITHUB REPOSITORY: {p.github_url}" if p.github_url else "")
            for i, p in enumerate(papers)
        ]
    )
    console.print(f"🧠 Phase 1 context: {len(full_context)} chars")

    phase_started_at = _phase_started("🔬 Phase 1: Extracting technical primitives...")
    # Add web search tool to decomposer for code skimming
    decomposer_trace = SearchTrace(section_name="Code Skimming")
    decomposer = await spawn_agent(
        premise=DECOMPOSER_PREMISE,
        model=model,
        scope={
            "web_search": make_web_search_tool(
                default_intent="fast",
                trace=decomposer_trace,
            )
        },
    )
    primitives_raw = await call_agent_text(
        decomposer,
        f"Analyze these research papers and extract all atomic technical primitives. "
        f"For papers with GitHub URLs, skim the repository (README and core code) to ensure practical orientation. "
        f"Think about interaction hooks between elements of DIFFERENT papers.\n\n{full_context}",
        phase="technical primitive extraction",
    )
    _phase_finished("Phase 1", phase_started_at, details=f"(code skimming calls={decomposer_trace.calls_used})")
    primitives_summary = _truncate_text(primitives_raw, PRIMITIVE_SUMMARY_CHARS)
    
    compact_context = "\n\n---\n\n".join(
        [f"PAPER {i+1} SUMMARY:\n{p.title}\n{p.abstract}" for i, p in enumerate(papers)]
    )
    if primitives_summary:
        compact_context += "\n\nTECHNICAL PRIMITIVES SUMMARY:\n" + primitives_summary

    console.print(f"🧠 Downstream context: {len(compact_context)} chars")

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
        f"Consolidated context:\n{compact_context}\n\n"
        "Search the web to find real, current market pain mapping to these primitives. "
        f"Go FAR beyond the papers' own domain.{anchor_text}",
        phase="pain scanner",
    )
    infra_task = call_agent_text(
        infra_agent,
        f"Consolidated context:\n{compact_context}\n\n"
        f"Technical primitives:\n{primitives_summary}\n\n"
        "What NEW problems does widespread adoption of these techniques CREATE? "
        f"What products solve those second-order problems?{anchor_text}",
        phase="infrastructure inversion",
    )
    temporal_task = call_agent_text(
        temporal_agent,
        f"Consolidated context:\n{compact_context}\n\n"
        f"Technical primitives:\n{primitives_summary}\n\n"
        "Identify temporal arbitrage windows. What can be built RIGHT NOW that "
        "won't be obvious for 12-24 months? Search the web for recent related "
        f"papers and industry trends.{anchor_text}",
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

    phase_started_at = _phase_started("🧬 Phase 3: Compound Synthesis...")
    crosspoll_agent = await spawn_agent(
        premise=CROSSPOLLINATOR_PREMISE,
        model=model,
    )
    crosspoll_raw = await call_agent_text(
        crosspoll_agent,
        f"Technical primitives (Elements):\n{primitives_summary}\n\n"
        f"Market pain points found:\n{_truncate_text(pain_raw, PAIN_SUMMARY_CHARS)}\n\n"
        "Synthesize multiple primitives into 'Compound Opportunities'. "
        f"Think about architectural hints for how these elements bond.{anchor_text}",
        phase="compound synthesis",
    )
    _phase_finished("Phase 3", phase_started_at)

    phase_started_at = _phase_started("🛡️  Phase 4: Structural Simulation (Red Team)...")
    destroyer_trace = SearchTrace(section_name="Structural Simulation")
    destroyer_agent = await spawn_agent(
        premise=DESTROYER_PREMISE,
        model=model,
        scope={
            "web_search": make_web_search_tool(
                default_intent="fresh",
                trace=destroyer_trace,
            )
        }
        if _redteam_search_enabled()
        else {"web_search": make_disabled_web_search_tool()},
    )
    redteam_raw = await call_agent_text(
        destroyer_agent,
        f"Consolidated context:\n{compact_context}\n\n"
        f"Technical primitives:\n{primitives_summary}\n\n"
        f"Candidate compound opportunities:\n{crosspoll_raw}\n\n"
        f"Simulate failure modes. Be brutal on mechanical logic, fair on potential.{anchor_text}",
        phase="structural simulation",
    )
    _phase_finished("Phase 4", phase_started_at)

    phase_started_at = _phase_started("🎯 Phase 5: Final synthesis...")
    synthesizer_agent = await spawn_agent(premise=SYNTHESIZER_PREMISE, model=model)
    final_raw = await call_agent_text(
        synthesizer_agent,
        f"Consolidated primitives:\n{primitives_summary}\n\n"
        f"Market analysis:\n{_truncate_text(pain_raw, PAIN_SUMMARY_CHARS)}\n\n"
        f"Compound opportunities & hints:\n{_truncate_text(crosspoll_raw, IDEA_SUMMARY_CHARS)}\n\n"
        f"Simulation failure modes:\n{_truncate_text(redteam_raw, IDEA_SUMMARY_CHARS)}\n\n"
        f"Synthesize these into a final ranked set of actionable Product Compounds.{anchor_text}",
        phase="final synthesis",
    )
    _phase_finished("Phase 5", phase_started_at)

    # Use the primary paper for reporting metadata
    primary_paper = papers[0]
    report = build_report(
        paper=primary_paper,
        primitives=primitives_raw,
        pain=pain_raw,
        pain_sources=pain_trace.render_markdown(),
        crosspoll=crosspoll_raw,
        infra=infra_raw,
        temporal=temporal_raw,
        temporal_sources=temporal_trace.render_markdown(),
        redteam=redteam_raw,
        redteam_sources=destroyer_trace.render_markdown()
        if _redteam_search_enabled()
        else "",
        final=final_raw,
    )

    safe_id = primary_paper.arxiv_id.replace("/", "_").replace(".", "_")
    if len(papers) > 1:
        safe_id += f"_plus_{len(papers)-1}_others"
    output_path = Path(f"products_{safe_id}.md")
    output_path.write_text(report, encoding="utf-8")

    console.print(f"\n✅ Done! Report saved to: {output_path}")
    console.print(f"   {len(report)} chars, ~{len(report.splitlines())} lines")
    return str(output_path)


async def _run_pipeline_with_openai_compatible(
    arxiv_ids: list[str],
    model: str,
    backend: OpenAICompatibleBackend,
    user_idea: str = "",
    github_mapping: dict[str, str] | None = None,
) -> str:
    github_mapping = github_mapping or {}
    console.print(f"📄 Fetching {len(arxiv_ids)} paper(s): {', '.join(arxiv_ids)}")
    papers = await asyncio.gather(*(fetch_paper(aid, github_mapping.get(aid, "")) for aid in arxiv_ids))
    titles = [p.title for p in papers]
    console.print(f"✅ Loaded: {titles}")
    console.print("⚙️ Execution backend: openai_compatible")
    console.print(f"⚙️ Speed profile: {_get_speed_profile()}")

    anchor_text = f"\n\nUSER'S CORE IDEA TO VALIDATE/AUGMENT:\n{user_idea}\nFocus all analysis, synthesis, and simulation around structurally building and stress-testing this specific idea." if user_idea else ""

    full_context = "\n\n---\n\n".join(
        [
            f"PAPER {i+1}:\n{build_full_paper_context(p)}"
            + (f"\nGITHUB REPOSITORY: {p.github_url}" if p.github_url else "")
            for i, p in enumerate(papers)
        ]
    )
    console.print(f"🧠 Phase 1 context: {len(full_context)} chars")

    phase_started_at = _phase_started("🔬 Phase 1: Extracting technical primitives...")
    # For openai_compatible, we'll try to build a search packet for the github repos if they exist
    skimming_packet = ""
    for p in papers:
        if p.github_url:
            skimming_packet += f"\n\nGITHUB SKIM FOR {p.title} ({p.github_url}):\n"
            # Build a trace just for the packet collection
            skim_trace = SearchTrace(section_name=f"GitHub Skim: {p.title}")
            skimming_packet += await build_search_packet(
                backend=backend,
                paper=p,
                primitives_summary="",
                trace=skim_trace,
                phase="code skimming",
                default_intent="fast",
                model=model,
            )

    primitives_raw = await call_direct_text(
        backend,
        system_prompt=DECOMPOSER_PREMISE,
        user_prompt=(
            "Analyze these research papers and extract all atomic technical primitives. "
            "Focus on implementation-level primitives. "
            f"Think about interaction hooks between elements of DIFFERENT papers.\n\n{full_context}"
            + (f"\n\nEXTERNAL IMPLEMENTATION EVIDENCE:\n{skimming_packet}" if skimming_packet else "")
        ),
        phase="technical primitive extraction",
        model=model,
        max_tokens=_phase_max_tokens("technical primitive extraction"),
    )
    _phase_finished("Phase 1", phase_started_at)

    primitives_summary = _truncate_text(primitives_raw, PRIMITIVE_SUMMARY_CHARS)
    compact_context = "\n\n---\n\n".join(
        [f"PAPER {i+1} SUMMARY:\n{p.title}\n{p.abstract}" for i, p in enumerate(papers)]
    )
    if primitives_summary:
        compact_context += "\n\nTECHNICAL PRIMITIVES SUMMARY:\n" + primitives_summary

    console.print(f"🧠 Downstream context: {len(compact_context)} chars")

    pain_trace = SearchTrace(section_name="Market Pain Mapping")
    temporal_trace = SearchTrace(section_name="Temporal Arbitrage")

    phase_started_at = _phase_started(
        "🚀 Phase 2: Running parallel analysis backend calls..."
    )

    # Note: build_search_packet uses primary paper metadata for simplicity
    primary_paper = papers[0]

    async def get_pain_raw():
        pain_search_packet = await build_search_packet(
            backend=backend,
            paper=primary_paper,
            primitives_summary=primitives_summary,
            trace=pain_trace,
            phase="pain scanner",
            default_intent="fast",
            model=model,
        )
        return await call_direct_text(
            backend,
            system_prompt=PAIN_SCANNER_PREMISE,
            user_prompt=(
                f"Technical primitives:\n{primitives_summary}\n\n"
                f"Consolidated context:\n{compact_context}\n\n"
                f"External market evidence:\n{pain_search_packet}\n\n"
                f"Find the strongest current market pain points linked to these primitives.{anchor_text}"
            ),
            phase="pain scanner",
            model=model,
            max_tokens=_phase_max_tokens("pain scanner"),
        )

    async def get_infra_raw():
        return await call_direct_text(
            backend,
            system_prompt=INFRA_INVERSION_PREMISE,
            user_prompt=(
                f"Consolidated context:\n{compact_context}\n\n"
                f"Technical primitives:\n{primitives_summary}\n\n"
                f"What new problems does widespread adoption of these techniques create?{anchor_text}"
            ),
            phase="infrastructure inversion",
            model=model,
            max_tokens=_phase_max_tokens("infrastructure inversion"),
        )

    async def get_temporal_raw():
        temporal_search_packet = await build_search_packet(
            backend=backend,
            paper=primary_paper,
            primitives_summary=primitives_summary,
            trace=temporal_trace,
            phase="temporal arbitrage",
            default_intent="fresh",
            model=model,
        )
        return await call_direct_text(
            backend,
            system_prompt=TEMPORAL_PREMISE,
            user_prompt=(
                f"Consolidated context:\n{compact_context}\n\n"
                f"Technical primitives:\n{primitives_summary}\n\n"
                f"External evidence:\n{temporal_search_packet}\n\n"
                f"Identify temporal arbitrage windows for the consolidated primitives.{anchor_text}"
            ),
            phase="temporal arbitrage",
            model=model,
            max_tokens=_phase_max_tokens("temporal arbitrage"),
        )

    phase_two_results = await gather_agent_calls(
        {
            "pain scanner": get_pain_raw(),
            "infrastructure inversion": get_infra_raw(),
            "temporal arbitrage": get_temporal_raw(),
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

    phase_started_at = _phase_started("🧬 Phase 3: Compound Synthesis...")
    crosspoll_raw = await call_direct_text(
        backend,
        system_prompt=CROSSPOLLINATOR_PREMISE,
        user_prompt=(
            f"Technical primitives (Elements):\n{primitives_summary}\n\n"
            f"Market pain points found:\n{_truncate_text(pain_raw, PAIN_SUMMARY_CHARS)}\n\n"
            "Synthesize multiple primitives into 'Compound Opportunities'. "
            f"Think about architectural hints for how these elements bond.{anchor_text}"
        ),
        phase="compound synthesis",
        model=model,
        max_tokens=_phase_max_tokens("cross-pollination"),
    )
    _phase_finished("Phase 3", phase_started_at)

    phase_started_at = _phase_started("🛡️  Phase 4: Structural Simulation (Red Team)...")
    all_ideas = (
        f"=== COMPOUNDS FROM PAIN MAPPING ===\n{_truncate_text(pain_raw, IDEA_SUMMARY_CHARS)}\n\n"
        f"=== COMPOUNDS FROM SYNTHESIS ===\n{_truncate_text(crosspoll_raw, IDEA_SUMMARY_CHARS)}\n\n"
        f"=== COMPOUNDS FROM INFRASTRUCTURE INVERSION ===\n{_truncate_text(infra_raw, IDEA_SUMMARY_CHARS)}\n\n"
        f"=== COMPOUNDS FROM TEMPORAL ARBITRAGE ===\n{_truncate_text(temporal_raw, IDEA_SUMMARY_CHARS)}\n\n"
    )
    redteam_raw = await call_direct_text(
        backend,
        system_prompt=DESTROYER_PREMISE,
        user_prompt=(
            "Here are product compounds from research. Simulate failure modes.\n\n"
            f"Papers: {', '.join(titles)}\n\n{all_ideas}\n{anchor_text}"
        ),
        phase="structural simulation",
        model=model,
        max_tokens=_phase_max_tokens("red team destruction"),
    )
    _phase_finished(
        "Phase 4", phase_started_at, details="(direct backend, no live red-team search)"
    )

    phase_started_at = _phase_started("🎯 Phase 5: Final synthesis...")
    final_raw = await call_direct_text(
        backend,
        system_prompt=SYNTHESIZER_PREMISE,
        user_prompt=(
            f"Consolidated primitives:\n{primitives_summary}\n\n"
            f"Market analysis:\n{_truncate_text(pain_raw, PAIN_SUMMARY_CHARS)}\n\n"
            f"Compound opportunities & hints:\n{_truncate_text(crosspoll_raw, IDEA_SUMMARY_CHARS)}\n\n"
            f"Simulation failure modes:\n{_truncate_text(redteam_raw, IDEA_SUMMARY_CHARS)}\n\n"
            f"Synthesize these into a final ranked set of actionable Product Compounds.{anchor_text}"
        ),
        phase="final synthesis",
        model=model,
        max_tokens=_phase_max_tokens("final synthesis"),
    )
    _phase_finished("Phase 5", phase_started_at)

    report = build_report(
        paper=primary_paper,
        primitives=primitives_raw,
        pain=pain_raw,
        pain_sources=pain_trace.render_markdown(),
        crosspoll=crosspoll_raw,
        infra=infra_raw,
        temporal=temporal_raw,
        temporal_sources=temporal_trace.render_markdown(),
        redteam=redteam_raw,
        redteam_sources="",
        final=final_raw,
    )

    safe_id = primary_paper.arxiv_id.replace("/", "_").replace(".", "_")
    if len(papers) > 1:
        safe_id += f"_plus_{len(papers)-1}_others"
    output_path = Path(f"products_{safe_id}.md")
    output_path.write_text(report, encoding="utf-8")

    console.print(f"\n✅ Done! Report saved to: {output_path}")
    console.print(f"   {len(report)} chars, ~{len(report.splitlines())} lines")
    return str(output_path)


async def run_pipeline(
    arxiv_id_or_url: str | list[str],
    model: str = DEFAULT_MODEL,
    save: bool = True,
    output_path: Optional[str] = None,
    display: bool = False,
    quiet: bool = False,
    search_papers: bool = False,
    user_idea: str = "",
) -> str:
    """Run the paper-to-product pipeline using the configured execution backend."""
    arxiv_ids: list[str] = []
    github_mapping: dict[str, str] = {}

    # Phase 0 (optional): PASA-style paper search for topic queries
    if search_papers and isinstance(arxiv_id_or_url, str) and is_topic_query(arxiv_id_or_url):
        results = await run_paper_search(arxiv_id_or_url, model=model)
        if not results:
            raise AgentExecutionError(
                f"Paper search found no relevant papers for topic: {arxiv_id_or_url}"
            )
        # Pick top 5 papers if user provided an idea to anchor against, else top 2
        num_papers = 5 if user_idea else 2
        top_papers = results[:num_papers]
        for p in top_papers:
            msg = f"📄 Selected paper: [{p.arxiv_id}] {p.title}"
            if p.github_url:
                msg += f" [GitHub: {p.github_url}]"
                github_mapping[p.arxiv_id] = p.github_url
            console.print(msg)
            arxiv_ids.append(p.arxiv_id)
    else:
        if isinstance(arxiv_id_or_url, str):
            arxiv_ids = [arxiv_id_or_url]
        else:
            arxiv_ids = arxiv_id_or_url

    backend_name = get_execution_backend_name()
    if backend_name == OPENAI_COMPATIBLE_BACKEND:
        backend = build_openai_compatible_backend()
        return await _run_pipeline_with_openai_compatible(
            arxiv_ids, model, backend, user_idea, github_mapping
        )
    return await _run_pipeline_with_agentica(arxiv_ids, model, user_idea, github_mapping)
