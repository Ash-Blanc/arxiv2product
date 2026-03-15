import asyncio
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

from .errors import AgentExecutionError
from .prompts import DEFAULT_MODEL

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = PACKAGE_ROOT.parent

load_dotenv(PACKAGE_ROOT / ".env")
load_dotenv(WORKSPACE_ROOT / ".env")


USAGE = """\
Usage: uv run arxiv2product <arxiv_id_or_url> [model]
   or: uv run python main.py <arxiv_id_or_url> [model]
   or: python -m arxiv2product <arxiv_id_or_url> [model]

Examples:
  uv run arxiv2product 2603.09229
  uv run python main.py 2603.09229
  python -m arxiv2product 2603.09229
  uv run arxiv2product https://alphaxiv.org/abs/2603.09229 openrouter:google/gemini-2.5-pro
"""


async def main() -> None:
    if len(sys.argv) < 2:
        print(USAGE)
        raise SystemExit(1)

    from .pipeline import run_pipeline

    paper_id = sys.argv[1]
    model = sys.argv[2] if len(sys.argv) > 2 else os.getenv("ARXIV2PRODUCT_MODEL", DEFAULT_MODEL)
    try:
        await run_pipeline(paper_id, model=model)
    except AgentExecutionError as exc:
        print(f"Agent execution error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


def run() -> None:
    asyncio.run(main())
