import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from click import Group, command, option, argument, echo

from .errors import AgentExecutionError, AgenticaConnectionError
from .prompts import DEFAULT_MODEL

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = PACKAGE_ROOT.parent

load_dotenv(PACKAGE_ROOT / ".env")
load_dotenv(WORKSPACE_ROOT / ".env")


@command()
@argument("arxiv_id_or_url")
@option("--model", default=DEFAULT_MODEL, help="LLM model to use")
@option("--save", is_flag=True, help="Save report to file")
@option("--output", type=click.Path(), help="Output file path")
@option("--display", is_flag=True, help="Display report in terminal")
@option("--quiet", is_flag=True, help="Suppress progress output")
async def cli(
    arxiv_id_or_url: str,
    model: str = DEFAULT_MODEL,
    save: bool = False,
    output: Optional[str] = None,
    display: bool = False,
    quiet: bool = False,
):
    """Generate product ideas from arXiv papers with a multi-agent pipeline."""
    if not quiet:
        echo(f"📄 Processing: {arxiv_id_or_url}")
        echo(f"⚙️ Model: {model}")

    from .pipeline import run_pipeline

    try:
        report_path = await run_pipeline(
            arxiv_id_or_url,
            model=model,
            save=save,
            output_path=output,
            display=display,
            quiet=quiet,
        )

        if not quiet:
            if save or output:
                echo(f"✅ Report saved to: {report_path}")
            else:
                echo(f"✅ Processing complete: {arxiv_id_or_url}")
    except AgenticaConnectionError as exc:
        echo(f"❌ Agentica connection error: {exc}", err=True)
        raise SystemExit(1) from exc
    except AgentExecutionError as exc:
        echo(f"❌ Agent execution error: {exc}", err=True)
        raise SystemExit(1) from exc
