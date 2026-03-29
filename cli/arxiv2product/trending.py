"""Functions for fetching trending papers from Hugging Face and AlphaXiv."""

from __future__ import annotations

import re
from typing import List, Optional
from urllib.parse import urljoin

import httpx

from .models import PaperContent
from .ingestion import fetch_paper


HF_PAPERS_API_BASE = "https://huggingface.co/api/papers"
ALPHAXIV_API_BASE = "https://alphaxiv.org/api"


async def _fetch_hf_trending_papers(
    period: str = "daily",
    limit: int = 10,
) -> List[dict]:
    """Fetch trending papers from Hugging Face Papers API."""
    if period not in {"daily", "weekly", "monthly"}:
        raise ValueError(
            f"Invalid period: {period}. Must be 'daily', 'weekly', or 'monthly'"
        )

    url = f"{HF_PAPERS_API_BASE}?sort={period}&limit={limit}"

    async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
        try:
            response = await client.get(url)
            response.raise_for_status()
            return response.json()
        except httpx.HTTPError as exc:
            raise RuntimeError(
                f"Failed to fetch Hugging Face trending papers: {exc}"
            ) from exc


async def _fetch_alphaxiv_trending_papers(
    period: str = "daily",
    limit: int = 10,
) -> List[dict]:
    """Fetch trending papers from AlphaXiv API."""
    if period not in {"daily", "weekly", "monthly"}:
        raise ValueError(
            f"Invalid period: {period}. Must be 'daily', 'weekly', or 'monthly'"
        )

    url = urljoin(ALPHAXIV_API_BASE + "/", f"papers/trending/{period}")

    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            response = await client.get(url, params={"limit": limit})
            response.raise_for_status()
            return response.json()
        except httpx.HTTPError as exc:
            raise RuntimeError(
                f"Failed to fetch AlphaXiv trending papers: {exc}"
            ) from exc


async def _convert_hf_paper_to_content(paper_data: dict) -> Optional[PaperContent]:
    """Convert Hugging Face paper data to PaperContent object."""
    try:
        arxiv_id = paper_data.get("id", "")
        if not arxiv_id:
            return None
        return await fetch_paper(arxiv_id)
    except Exception:
        return None


async def _convert_alphaxiv_paper_to_content(
    paper_data: dict,
) -> Optional[PaperContent]:
    """Convert AlphaXiv paper data to PaperContent object."""
    try:
        arxiv_id = paper_data.get("arxivId", "")
        if not arxiv_id:
            url = paper_data.get("url", "")
            arxiv_match = re.search(r"arxiv\.org/(?:abs|pdf)/([^/]+)", url)
            if arxiv_match:
                arxiv_id = arxiv_match.group(1)

        if not arxiv_id:
            return None
        return await fetch_paper(arxiv_id)
    except Exception:
        return None


async def get_trending_papers(
    source: str = "huggingface",
    period: str = "daily",
    limit: int = 10,
    topics: Optional[List[str]] = None,
) -> List[PaperContent]:
    """Get trending papers from specified source.

    Args:
        source: Source of trending papers - "huggingface" or "alphaxiv"
        period: Time period - "daily", "weekly", or "monthly"
        limit: Maximum number of papers to return
        topics: Optional list of topics to filter by (not implemented)

    Returns:
        List of PaperContent objects for trending papers
    """
    if source not in {"huggingface", "alphaxiv"}:
        raise ValueError(
            f"Invalid source: {source}. Must be 'huggingface' or 'alphaxiv'"
        )

    if source == "huggingface":
        papers_data = await _fetch_hf_trending_papers(period=period, limit=limit)
        converter = _convert_hf_paper_to_content
    else:
        papers_data = await _fetch_alphaxiv_trending_papers(period=period, limit=limit)
        converter = _convert_alphaxiv_paper_to_content

    papers: List[PaperContent] = []
    for paper_data in papers_data:
        try:
            paper_content = await converter(paper_data)
            if paper_content:
                papers.append(paper_content)
        except Exception:
            continue

    return papers[:limit]
