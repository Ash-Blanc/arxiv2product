import asyncio
import os
import sys
from pathlib import Path
from typing import Optional

import cli2
import pyfiglet
from dotenv import load_dotenv, set_key
from rich.console import Console
from rich.markdown import Markdown

from .errors import AgentExecutionError, AgenticaConnectionError
from .prompts import DEFAULT_MODEL

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = PACKAGE_ROOT.parent

console = Console()


def load_environment():
    checked = []
    for env_path in [
        Path.cwd() / ".env",
        Path.cwd() / "cli" / ".env",
        WORKSPACE_ROOT / ".env",
        WORKSPACE_ROOT / "cli" / ".env",
        PACKAGE_ROOT / ".env",
        Path.home() / ".arxiv2product" / ".env",
    ]:
        checked.append(str(env_path))
        if env_path.exists():
            load_dotenv(env_path, override=True)
            return


def check_agentica_key():
    if not os.getenv("AGENTICA_API_KEY") and not os.getenv("OPENROUTER_API_KEY"):
        console.print(
            "[yellow]WARN: AGENTICA_API_KEY or OPENROUTER_API_KEY not found. You might want to run `arxiv2product init` first.[/yellow]"
        )


def print_banner():
    banner = pyfiglet.figlet_format("arxiv2product", font="double_blocky")
    console.print(f"[bold cyan]{banner}[/bold cyan]")


async def _run_analyze(
    arxiv_id_or_url: str,
    model: str,
    save: bool,
    output: Optional[str],
    display: bool,
    quiet: bool,
    search_papers: bool,
    open_report: bool,
):
    load_environment()
    if not quiet:
        print_banner()
        check_agentica_key()
        console.print(f"📄 [bold]Processing:[/bold] {arxiv_id_or_url}")
        console.print(f"⚙️  [bold]Model:[/bold] {model}")

    from .pipeline import run_pipeline

    try:
        report_path = await run_pipeline(
            arxiv_id_or_url,
            model=model,
            save=save,
            output_path=output,
            display=display,
            quiet=quiet,
            search_papers=search_papers,
        )

        if not quiet:
            if save or output:
                console.print(
                    f"✅ [bold green]Report saved to:[/bold green] {report_path}"
                )
            else:
                console.print(
                    f"✅ [bold green]Processing complete:[/bold green] {arxiv_id_or_url}"
                )

        if open_report and report_path and Path(report_path).exists():
            console.print(f"📖 [bold]Opening report:[/bold] {report_path}...")
            if sys.platform == "win32":
                os.startfile(report_path)
            elif sys.platform == "darwin":
                import subprocess

                subprocess.call(["open", report_path])
            else:
                import subprocess

                subprocess.call(["xdg-open", report_path])

    except AgenticaConnectionError as exc:
        console.print(f"❌ [bold red]Agentica connection error:[/bold red] {exc}")
        sys.exit(1)
    except AgentExecutionError as exc:
        console.print(f"❌ [bold red]Agent execution error:[/bold red] {exc}")
        sys.exit(1)


def analyze(
    arxiv_id_or_url: str,
    model: str = DEFAULT_MODEL,
    save: bool = False,
    output: str = "",
    display: bool = False,
    quiet: bool = False,
    search_papers: bool = False,
    open: bool = False,
):
    """Generate product ideas from arXiv papers with a multi-agent pipeline.
    
    Usage:
        arxiv2product analyze 2603.09229
        arxiv2product analyze 2603.09229 --open
        arxiv2product analyze "research topic" --search-papers
    """
    out_opt = output if output else None
    asyncio.run(
        _run_analyze(
            arxiv_id_or_url, model, save, out_opt, display, quiet, search_papers, open
        )
    )


def compete(
    report: str,
    ideas: str = "",
    idea: str = "",
    model: str = DEFAULT_MODEL,
):
    """Run competitor intelligence on arxiv2product report ideas.
    
    Usage:
        arxiv2product compete report.md
        arxiv2product compete report.md --ideas 1,2
        arxiv2product compete report.md --idea "Idea Name"
    """
    load_environment()
    from .compete import run_compete

    idea_indices = None
    if ideas:
        idea_indices = [int(x.strip()) for x in ideas.split(",")]

    idea_name = idea if idea else None

    if not Path(report).exists():
        console.print(f"[bold red]Error:[/bold red] Report file not found: {report}")
        sys.exit(1)

    try:
        asyncio.run(
            run_compete(
                report_path=report,
                idea_indices=idea_indices,
                idea_name=idea_name,
                model=model,
            )
        )
    except Exception as exc:
        console.print(f"❌ [bold red]Error:[/bold red] {exc}")
        sys.exit(1)


def serve():
    """Start the arxiv2product API service.
    
    Usage:
        arxiv2product serve
    """
    load_environment()
    from .service import run

    console.print("🚀 [bold green]Starting arxiv2product API service...[/bold green]")
    run()


def init():
    """Interactive setup to configure API keys.
    
    Usage:
        arxiv2product init
    """
    console.print("[bold cyan]Welcome to arxiv2product setup![/bold cyan]")

    env_path = Path.home() / ".arxiv2product" / ".env"
    if not env_path.parent.exists():
        env_path.parent.mkdir(parents=True, exist_ok=True)

    if not env_path.exists():
        env_path.touch()

    load_dotenv(env_path)

    keys_to_prompt = {
        "AGENTICA_API_KEY": "Agentica API Key (Required for default backend)",
        "OPENROUTER_API_KEY": "OpenRouter API Key (Optional, for OpenAI-compatible backend)",
        "SERPER_API_KEY": "Serper API Key (Optional, for Google search)",
        "EXA_API_KEY": "Exa API Key (Optional, for deep web search)",
        "PARALLEL_API_KEY": "Parallel.ai API Key (Optional, for competitor intel)",
        "TINYFISH_API_KEY": "Tinyfish API Key (Optional, for competitor site crawl)",
    }

    for key, desc in keys_to_prompt.items():
        current_val = os.getenv(key, "")
        prompt_text = f"Enter {desc}"
        if current_val:
            masked = (
                f"...{current_val[-4:]}" if len(current_val) > 4 else "***"
            )
            prompt_text += f" [Current: {masked}]"

        try:
            val = input(f"{prompt_text}: ").strip()
            if val:
                set_key(str(env_path), key, val)
                os.environ[key] = val
        except (KeyboardInterrupt, EOFError):
            console.print("\n[yellow]Setup cancelled.[/yellow]")
            return

    console.print(
        f"\n✅ [bold green]Setup complete![/bold green] Configuration saved to {env_path}"
    )


def main():
    group = cli2.Group("arxiv2product", "arxiv2product CLI")
    group.cmd(analyze)
    group.cmd(compete)
    group.cmd(serve)
    group.cmd(init)
    group.entry_point()
