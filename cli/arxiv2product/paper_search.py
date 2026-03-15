"""PASA-inspired paper search agent — Crawler + Selector for topic-based discovery."""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from time import perf_counter

import arxiv

from .backend import (
    OPENAI_COMPATIBLE_BACKEND,
    get_execution_backend_name,
)
from .errors import AgentExecutionError
from .prompts import (
    DEFAULT_MODEL,
    PAPER_CRAWLER_PREMISE,
    PAPER_SELECTOR_PREMISE,
)
from .research import SearchTrace, make_web_search_tool

ARXIV_ID_PATTERN = re.compile(r"^\d{4}\.\d{4,5}(v\d+)?$")
ARXIV_URL_PATTERN = re.compile(
    r"^https?://(www\.)?(arxiv\.org|alphaxiv\.org)/(abs|pdf)/"
)
PAPER_SEARCH_MAX_RESULTS = 10
PAPER_SEARCH_TIMEOUT_SECONDS = 60.0


@dataclass
class PaperSearchResult:
    arxiv_id: str
    title: str
    abstract: str
    score: float
    reason: str


def is_topic_query(input_str: str) -> bool:
    """Return True if input is a free-text topic, not an arXiv ID or URL."""
    stripped = input_str.strip()
    if ARXIV_ID_PATTERN.match(stripped):
        return False
    if ARXIV_URL_PATTERN.match(stripped):
        return False
    return True


def _paper_search_enabled() -> bool:
    return os.getenv("ENABLE_PAPER_SEARCH", "0").strip().lower() in {"1", "true", "yes"}


async def _arxiv_search(query: str, max_results: int = PAPER_SEARCH_MAX_RESULTS) -> str:
    """Search arXiv and return formatted results with title, abstract, and ID."""
    client = arxiv.Client()
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance,
    )
    results: list[str] = []
    for paper in client.results(search):
        arxiv_id = paper.entry_id.split("/abs/")[-1].split("v")[0]
        abstract_short = paper.summary.replace("\n", " ")[:300]
        results.append(
            f"- [{arxiv_id}] {paper.title}\n  Abstract: {abstract_short}"
        )
    if not results:
        return f"[arxiv_search] No results for: {query}"
    return f"[arxiv_search results={len(results)}]\n" + "\n".join(results)


def _make_arxiv_search_tool():
    """Create an arxiv search tool callable for agent scope."""
    async def arxiv_search_tool(query: str) -> str:
        """Search the arXiv database for research papers.
        
        Args:
            query: The search string to query arXiv.
        """
        return await _arxiv_search(query)
    return arxiv_search_tool


def _parse_selector_output(text: str) -> list[PaperSearchResult]:
    """Parse the Selector agent's JSON output into PaperSearchResult list."""
    # Try to extract JSON array from the response
    text = text.strip()
    # Handle markdown code blocks
    if "```" in text:
        match = re.search(r"```(?:json)?\s*(\[.*?\])\s*```", text, re.DOTALL)
        if match:
            text = match.group(1)
    # Find the JSON array
    start = text.find("[")
    end = text.rfind("]")
    if start == -1 or end == -1:
        return []
    try:
        items = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return []

    results: list[PaperSearchResult] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        arxiv_id = item.get("arxiv_id", "")
        if not arxiv_id:
            continue
        results.append(
            PaperSearchResult(
                arxiv_id=str(arxiv_id),
                title=str(item.get("title", "")),
                abstract=str(item.get("abstract", "")),
                score=float(item.get("score", 0.0)),
                reason=str(item.get("reason", "")),
            )
        )
    results.sort(key=lambda r: r.score, reverse=True)
    return results


async def _run_paper_search(
    topic: str,
    model: str,
) -> list[PaperSearchResult]:
    """Run paper search using Agno backend."""
    import asyncio

    from .pipeline import call_agent_text, spawn_agent

    search_trace = SearchTrace(section_name="Paper Discovery")

    # Phase 0a: Crawler
    crawler = await spawn_agent(
        premise=PAPER_CRAWLER_PREMISE,
        model=model,
        scope={
            "web_search": make_web_search_tool(
                default_intent="fresh",
                trace=search_trace,
            ),
            "arxiv_search": _make_arxiv_search_tool(),
        },
    )
    crawler_output = await call_agent_text(
        crawler,
        f"Find the most relevant research papers for this topic:\n\n{topic}\n\n"
        "Use arxiv_search and web_search to build a comprehensive candidate list. "
        "Follow key citations when you find highly relevant papers.",
        phase="paper search: crawling",
    )

    # Phase 0b: Selector — score and rank
    selector = await spawn_agent(
        premise=PAPER_SELECTOR_PREMISE,
        model=model,
    )

    # Enrich crawler output with abstracts from arXiv
    enrichment = await _enrich_candidates(crawler_output)

    selector_output = await call_agent_text(
        selector,
        f"Topic: {topic}\n\n"
        f"Candidate papers from crawler:\n{crawler_output}\n\n"
        f"Additional abstracts:\n{enrichment}\n\n"
        "Score and rank these papers. Return the top 3-5 as a JSON array.",
        phase="paper search: selection",
    )
    return _parse_selector_output(selector_output)

async def _enrich_candidates(crawler_output: str) -> str:
    """Extract arXiv IDs from crawler output and fetch their abstracts."""
    ids = re.findall(r"\b(\d{4}\.\d{4,5})\b", crawler_output)
    if not ids:
        return ""
    unique_ids = list(dict.fromkeys(ids))[:10]

    client = arxiv.Client()
    search = arxiv.Search(id_list=unique_ids)
    lines: list[str] = []
    for paper in client.results(search):
        arxiv_id = paper.entry_id.split("/abs/")[-1].split("v")[0]
        abstract_short = paper.summary.replace("\n", " ")[:400]
        lines.append(
            f"### [{arxiv_id}] {paper.title}\n{abstract_short}"
        )
    return "\n\n".join(lines)


async def run_paper_search(
    topic: str,
    model: str = DEFAULT_MODEL,
) -> list[PaperSearchResult]:
    """Run the PASA-style paper search pipeline. Returns ranked papers."""
    started_at = perf_counter()
    print(f"🔍 Paper search: discovering papers for topic: {topic}")

    results = await _run_paper_search(topic, model)
    elapsed = perf_counter() - started_at
    print(f"  ✅ Paper search complete in {elapsed:.1f}s — found {len(results)} papers")
    for i, r in enumerate(results, 1):
        print(f"     {i}. [{r.arxiv_id}] {r.title} (score={r.score:.2f})")

    return results
