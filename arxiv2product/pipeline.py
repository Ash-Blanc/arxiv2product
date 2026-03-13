import asyncio
import os
from pathlib import Path

import httpx
from agentica import spawn

from .errors import AgenticaConnectionError
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
from .research import web_search


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
MAX_SECTION_CHARS = 8_000
MAX_CONTEXT_CHARS = 80_000


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
        return await spawn(**kwargs)
    except httpx.TimeoutException as exc:
        raise AgenticaConnectionError(_agentica_connection_help()) from exc
    except httpx.HTTPError as exc:
        raise AgenticaConnectionError(
            f"Agentica request failed while creating an agent: {exc}"
        ) from exc


def build_paper_context(paper: PaperContent) -> str:
    key_sections: dict[str, str] = {}
    for key in PRIORITY_SECTION_KEYS:
        for section_name, content in paper.sections.items():
            if key in section_name.lower():
                key_sections[section_name] = content[:MAX_SECTION_CHARS]

    context = (
        f"TITLE: {paper.title}\n"
        f"AUTHORS: {', '.join(paper.authors[:10])}\n"
        f"ABSTRACT: {paper.abstract}\n\n"
        f"KEY SECTIONS:\n"
        + "\n\n".join(f"=== {name} ===\n{content}" for name, content in key_sections.items())
        + "\n\nFIGURE CAPTIONS:\n"
        + "\n".join(paper.figures_captions[:20])
        + "\n\nTABLE SUMMARIES:\n"
        + "\n".join(paper.tables_text[:10])
        + "\n\nREFERENCED WORKS:\n"
        + "\n".join(paper.references_titles[:30])
    )
    if len(context) > MAX_CONTEXT_CHARS:
        return context[:MAX_CONTEXT_CHARS] + "\n\n[...truncated...]"
    return context


async def run_pipeline(arxiv_id_or_url: str, model: str = DEFAULT_MODEL) -> str:
    """Run the multi-agent paper analysis pipeline and write the markdown report."""
    print(f"📄 Fetching paper: {arxiv_id_or_url}")
    paper = await fetch_paper(arxiv_id_or_url)
    print(f"✅ Loaded: {paper.title} ({len(paper.full_text)} chars)")

    paper_context = build_paper_context(paper)
    print(f"🧠 Paper context: {len(paper_context)} chars")

    print("🔬 Phase 1: Extracting technical primitives...")
    decomposer = await spawn_agent(premise=DECOMPOSER_PREMISE, model=model)
    primitives_raw: str = await decomposer.call(
        str,
        f"Analyze this paper and extract all atomic technical primitives:\n\n{paper_context}",
    )
    print("  ✅ Primitives extracted")

    print("🚀 Phase 2: Spawning parallel analysis agents...")
    pain_agent = await spawn_agent(
        premise=PAIN_SCANNER_PREMISE,
        model=model,
        scope={"web_search": web_search},
    )
    infra_agent = await spawn_agent(premise=INFRA_INVERSION_PREMISE, model=model)
    temporal_agent = await spawn_agent(
        premise=TEMPORAL_PREMISE,
        model=model,
        scope={"web_search": web_search},
    )

    pain_task = pain_agent.call(
        str,
        f"Technical primitives:\n\n{primitives_raw}\n\n"
        f"Paper context:\n{paper_context}\n\n"
        "Search the web to find real, current market pain mapping to these primitives. "
        "Go FAR beyond the paper's own domain.",
    )
    infra_task = infra_agent.call(
        str,
        f"Paper context:\n{paper_context}\n\n"
        f"Technical primitives:\n{primitives_raw}\n\n"
        "What NEW problems does widespread adoption of this technique CREATE? "
        "What products solve those second-order problems?",
    )
    temporal_task = temporal_agent.call(
        str,
        f"Paper context:\n{paper_context}\n\n"
        f"Technical primitives:\n{primitives_raw}\n\n"
        "Identify temporal arbitrage windows. What can be built RIGHT NOW that "
        "won't be obvious for 12-24 months? Search the web for recent related "
        "papers and industry trends.",
    )

    print("  ⏳ Running pain scanner, infra inverter, temporal agent in parallel...")
    pain_raw, infra_raw, temporal_raw = await asyncio.gather(
        pain_task,
        infra_task,
        temporal_task,
    )
    print("  ✅ Phase 2 complete")

    print("🧬 Phase 3: Cross-pollination...")
    crosspoll_agent = await spawn_agent(
        premise=CROSSPOLLINATOR_PREMISE,
        model=model,
    )
    crosspoll_raw: str = await crosspoll_agent.call(
        str,
        f"Technical primitives:\n{primitives_raw}\n\n"
        f"Market pain points found:\n{pain_raw}\n\n"
        "Force non-obvious cross-pollination. Skip direct/obvious matches.",
    )
    print("  ✅ Cross-pollination complete")

    print("💀 Phase 4: Red team destruction...")
    all_ideas = (
        f"=== IDEAS FROM PAIN MAPPING ===\n{pain_raw}\n\n"
        f"=== IDEAS FROM CROSS-POLLINATION ===\n{crosspoll_raw}\n\n"
        f"=== IDEAS FROM INFRASTRUCTURE INVERSION ===\n{infra_raw}\n\n"
        f"=== IDEAS FROM TEMPORAL ARBITRAGE ===\n{temporal_raw}\n\n"
    )
    destroyer = await spawn_agent(
        premise=DESTROYER_PREMISE,
        model=model,
        scope={"web_search": web_search},
    )
    redteam_raw: str = await destroyer.call(
        str,
        "Here are product ideas from a research paper. Destroy every one.\n\n"
        f"Paper: {paper.title}\n\n{all_ideas}",
    )
    print("  ✅ Red team complete")

    print("🎯 Phase 5: Final synthesis...")
    synthesizer = await spawn_agent(premise=SYNTHESIZER_PREMISE, model=model)
    final_raw: str = await synthesizer.call(
        str,
        f"PAPER: {paper.title}\nABSTRACT: {paper.abstract}\n\n"
        f"=== TECHNICAL PRIMITIVES ===\n{primitives_raw}\n\n"
        f"=== MARKET PAIN MAPPING ===\n{pain_raw}\n\n"
        f"=== CROSS-POLLINATED IDEAS ===\n{crosspoll_raw}\n\n"
        f"=== INFRASTRUCTURE INVERSION ===\n{infra_raw}\n\n"
        f"=== TEMPORAL ARBITRAGE ===\n{temporal_raw}\n\n"
        f"=== RED TEAM DESTRUCTION RESULTS ===\n{redteam_raw}\n\n"
        "Synthesize all of the above into a final ranked list of the BEST ideas. "
        "Only include ideas that survived red-teaming or were strengthened by it.",
    )
    print("  ✅ Synthesis complete")

    report = build_report(
        paper=paper,
        primitives=primitives_raw,
        pain=pain_raw,
        crosspoll=crosspoll_raw,
        infra=infra_raw,
        temporal=temporal_raw,
        redteam=redteam_raw,
        final=final_raw,
    )

    safe_id = paper.arxiv_id.replace("/", "_").replace(".", "_")
    output_path = Path(f"products_{safe_id}.md")
    output_path.write_text(report, encoding="utf-8")

    print(f"\n✅ Done! Report saved to: {output_path}")
    print(f"   {len(report)} chars, ~{len(report.splitlines())} lines")
    return str(output_path)
