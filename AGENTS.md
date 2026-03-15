# Repository Guidelines

## Project Structure
Single Python app in `cli/`, with package code under `cli/arxiv2product/` and tests under `cli/tests/`. Generated reports, logs, and local SQLite data stay out of version control.

## Build, Test, and Development Commands

- `cd cli && uv sync`: install/update the Python environment.
- `cd cli && uv run arxiv2product 2603.09229`: generate a report from an arXiv ID.
- `cd cli && uv run arxiv2product-api`: start the local Python API service.
- `cd cli && python -m unittest discover -s tests`: run the Python test suite.

## Coding Style & Naming Conventions
Follow existing Python conventions in `cli/arxiv2product/`: 4-space indentation, type hints on public functions, `snake_case` for functions/variables, `UPPER_SNAKE_CASE` for prompt constants. Keep orchestration async.

## Agentica Framework
Reference docs: `cli/agentica-docs.md`. Uses `spawn` + `agent.call(str, prompt)` pattern. Model names must be valid OpenRouter slugs.

## Testing Guidelines
Python tests live in `cli/tests/` and use `unittest`. Add new test files as `test_<feature>.py`. Prefer mocked network/model calls for pipeline and service coverage.

## Commit & Pull Request Guidelines
Use short imperative commit subjects. Keep commits scoped to one concern.

## Configuration
Environment examples live in `cli/.env.example`. Do not add Python package-manager metadata at the repo root.
