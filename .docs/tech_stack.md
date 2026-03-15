# Technical Stack

`arxiv2product` is built with a modern, async-first Python stack designed for high-performance agentic workflows.

## Core Language & Runtime
- **Python 3.13+**: Leveraging the latest async features and type hinting.
- **uv**: Package management and project orchestration. Ultra-fast and reliable.

## Agentic Framework
- **Agentica (`symbolica-agentica`)**: Our primary agent orchestration library. 
    - Uses a `spawn` + `agent.call` pattern.
    - Supports `agentic` decorators for seamless AI integration into functions.
    - Handles persistence, state management, and tool-calling (MCP support).

## Models (via OpenRouter)
- **Anthropic Claude 3.5/4.5 (Sonnet/Haiku)**: Primary models for complex reasoning.
- **OpenAI GPT-4o/o1**: Used for specific tasks where appropriate.
- *Any OpenRouter model slug is supported.*

## Web & API
- **FastAPI**: Provides the local API service (`arxiv2product-api`).
- **Uvicorn**: High-performance ASGI server for the FastAPI app.

## Research & Search Tools
- **Serper API**: High-speed Google Search results for market research.
- **Exa API**: Neural search specialized for finding high-quality web content.
- **Parallel.ai**: Broad web research for competitor intelligence.
- **Tinyfish**: Headless browser automation for deep-crawling competitor sites.

## PDF & Data Processing
- **arxiv**: Library for fetching paper metadata/PDFs from arXiv.
- **pdfplumber**: Precision PDF text extraction for research papers.
- **Pydantic**: Robust data validation and settings management.

## Project Structure
- `cli/`: All source code and project configuration.
- `cli/arxiv2product/`: Core package logic.
- `cli/tests/`: Unit and integration test suite.
