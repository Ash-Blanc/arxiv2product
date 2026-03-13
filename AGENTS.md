# Repository Guidelines

## Project Structure & Module Organization
This repository is currently a small Python application centered on [`main.py`](/home/ab916/src/arxiv2product/main.py). That file contains the full pipeline: arXiv ingestion, PDF parsing, agent prompts, report synthesis, and the CLI entrypoint. Project metadata lives in [`pyproject.toml`](/home/ab916/src/arxiv2product/pyproject.toml), and the `uv` lockfile is [`uv.lock`](/home/ab916/src/arxiv2product/uv.lock). Generated reports should stay out of version control unless they are part of a reviewed example.

## Build, Test, and Development Commands
Use `uv` for local setup and execution:

- `uv sync`: create/update the local environment from `pyproject.toml` and `uv.lock`.
- `uv run python main.py 2603.09229`: run the pipeline against an arXiv ID.
- `uv run python main.py https://alphaxiv.org/abs/2603.09229 anthropic/claude-sonnet-4`: run with an explicit model override.

Before running, export required secrets:

- `export AGENTICA_API_KEY=...`
- `export SERPER_API_KEY=...` for live web search instead of the stub.

## Coding Style & Naming Conventions
Follow existing Python style in `main.py`: 4-space indentation, type hints on public functions, `snake_case` for functions and variables, `UPPER_SNAKE_CASE` for prompt constants, and small dataclasses for structured payloads. Keep orchestration logic asynchronous (`async`/`await`) and prefer focused helper functions over adding more inline logic to the CLI path.

## Testing Guidelines
There is no `tests/` directory yet. Add tests under `tests/` using `test_<feature>.py` naming, and prefer `pytest` for new coverage. Prioritize unit tests for pure helpers such as PDF parsing, arXiv ID normalization, reference extraction, and markdown report building. For agent-driven flows, mock network and model calls rather than hitting external services.

## Commit & Pull Request Guidelines
This repository has no commit history yet, so use clear imperative commit subjects such as `Add pytest coverage for report builder`. Keep commits scoped to one change. Pull requests should include: purpose, key implementation notes, local verification commands, and sample output or screenshots if the markdown report format changes.

## Configuration Notes
`main.py` imports `agentica`, `arxiv`, `httpx`, and `pdfplumber`, but they are not currently declared in [`pyproject.toml`](/home/ab916/src/arxiv2product/pyproject.toml). Update dependencies and refresh `uv.lock` whenever runtime imports change.
