# arxiv2product

Transforms arXiv research papers into company/product opportunity reports using a multi-agent AI pipeline.

## Setup

```bash
cd cli
uv sync
cp .env.example .env   # fill in your keys
```

## Usage

```bash
cd cli

# Generate a report
uv run arxiv2product 2603.09229
uv run arxiv2product https://arxiv.org/abs/2603.09229

# Start the API service
uv run arxiv2product-api
```

## API Endpoints

- `GET  /health`
- `GET  /dashboard/{userId}`
- `POST /reports`
- `GET  /reports/{jobId}`
- `POST /feedback/score`

## Tests

```bash
cd cli
python -m unittest discover -s tests
```

## Repository Layout

- `cli/` — Python package, pipeline, API service, and tests
- `cli/agentica-docs.md` — Agentica framework reference
- `cli/.env.example` — environment variable reference

## Configuration

Copy `cli/.env.example` to `cli/.env` and set:

- `EXECUTION_BACKEND` — `agentica` (default) or `openai_compatible`
- `AGENTICA_API_KEY` or `OPENROUTER_API_KEY` depending on backend
- `SERPER_API_KEY` / `EXA_API_KEY` for web search
