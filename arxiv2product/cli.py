import asyncio
import sys

from dotenv import load_dotenv

from .errors import AgenticaConnectionError
from .prompts import DEFAULT_MODEL

load_dotenv()


USAGE = """\
Usage: python main.py <arxiv_id_or_url> [model]
   or: python -m arxiv2product <arxiv_id_or_url> [model]
   or: arxiv2product <arxiv_id_or_url> [model]

Examples:
  python main.py 2603.09229
  python -m arxiv2product 2603.09229
  python main.py https://alphaxiv.org/abs/2603.09229
  python main.py 2603.09229 openrouter:google/gemini-2.5-pro
"""


async def main() -> None:
    if len(sys.argv) < 2:
        print(USAGE)
        raise SystemExit(1)

    from .pipeline import run_pipeline

    paper_id = sys.argv[1]
    model = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_MODEL
    try:
        await run_pipeline(paper_id, model=model)
    except AgenticaConnectionError as exc:
        print(f"Agentica connection error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


def run() -> None:
    asyncio.run(main())
