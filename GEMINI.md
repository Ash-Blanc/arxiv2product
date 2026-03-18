# GEMINI.md

## Project Overview
**arxiv2product** is a multi-agent AI pipeline designed to transform arXiv research papers into comprehensive company and product opportunity reports. It employs a sophisticated, 5-phase adversarial process to analyze technical primitives and map them to market needs, infrastructure gaps, and temporal opportunities.

### Core Architecture (5-Phase Pipeline)
1.  **Decomposer:** Extracts atomic technical primitives and building blocks from the paper.
2.  **Pain Scanner:** Maps primitives to real-world market pain points using web research.
3.  **Infrastructure Inversion:** Identifies second-order problems created by the adoption of the technology.
4.  **Temporal Arbitrage:** Finds time-limited opportunities for building.
5.  **Red Team Destroyer:** Critically evaluates every idea, ensuring only the most robust survive.
*Final Synthesis:* A ranked list of 4-6 company ideas with market analysis and execution plans.

### Key Technologies
- **Language:** Python (>= 3.13)
- **Dependency Management:** `uv`
- **Frameworks:** `symbolica-agentica` (Agent orchestration), `FastAPI` (Service), `Click` (CLI)
- **APIs:** arXiv (PDF ingestion), Serper/Exa (Search), Parallel/Tinyfish (Competitor intel)
- **Utilities:** `pdfplumber`, `pydantic`, `python-dotenv`

---

## Building and Running

### Setup
Ensure you have `uv` installed, then run:
```bash
cd cli
uv sync
cp .env.example .env # Configure your API keys (AGENTICA_API_KEY, etc.)
```

### CLI Commands
- **Generate Report:** `uv run arxiv2product <arxiv_id_or_url>`
- **Topic Discovery:** `uv run arxiv2product "research topic" --search-papers` (Requires `ENABLE_PAPER_SEARCH=1` in `.env`)
- **Competitor Intelligence:** `uv run arxiv2product-compete <report_path>.md`
- **Start API Service:** `uv run arxiv2product-api` (Default port: 8010)

### Testing
Run the test suite using `unittest`:
```bash
cd cli
uv run python -m unittest discover -s tests
```

---

## Development Conventions

### Coding Style
- **Indentation:** 4-space indentation.
- **Type Hints:** Required for all public-facing functions and methods.
- **Naming:** `snake_case` for variables and functions; `UPPER_SNAKE_CASE` for prompt constants and environment variables.
- **Orchestration:** Use `asyncio` for all agentic and network-bound orchestration.

### Directory Structure
- `cli/arxiv2product/`: Core package logic.
- `cli/tests/`: Unit and integration tests.
- `cli/main.py`: Main CLI entry point wrapper.
- `cli/agentica-docs.md`: Reference for the Agentica framework used in this project.
- `AGENTS.md`: Specific behavioral and structural mandates for AI assistants working in this repo.

### Agent Configuration
- **Prompts:** Centralized in `cli/arxiv2product/prompts.py` as `UPPER_SNAKE_CASE` constants.
- **Execution:** Uses the `spawn` + `agent.call` pattern from `agentica`.
- **Backends:** Supports both `agentica` and `openai_compatible` (e.g., OpenRouter) via `backend.py`.

### Environment Configuration
The project looks for `.env` files in multiple locations, prioritizing the current working directory. Key variables include:
- `AGENTICA_API_KEY`: Required for the default backend.
- `ENABLE_PAPER_SEARCH`: Toggle for topic-to-paper discovery mode.
- `PIPELINE_SPEED_PROFILE`: `balanced` or `exhaustive` tuning.
