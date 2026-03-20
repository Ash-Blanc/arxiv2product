import re
import tempfile

import arxiv
import pdfplumber

from .models import PaperContent


SECTION_HEADER_PATTERN = re.compile(r"^\d+\.?\s+[A-Z]")
NAMED_SECTION_PATTERN = re.compile(
    r"^(Abstract|Introduction|Related Work|Method|Approach|"
    r"Experiments?|Results?|Discussion|Conclusion|References|"
    r"Appendix)",
    re.I,
)
FIGURE_PATTERN = re.compile(r"^(Figure|Fig\.?)\s+\d+", re.I)
TABLE_PATTERN = re.compile(r"^Table\s+\d+", re.I)
REFERENCE_TITLE_PATTERN = re.compile(r'"(.+?)"')
ARXIV_PREFIX_PATTERN = re.compile(
    r"^https?://(www\.)?(arxiv\.org/abs/|alphaxiv\.org/abs/)"
)


def normalize_arxiv_id(arxiv_id_or_url: str) -> str:
    return (
        ARXIV_PREFIX_PATTERN.sub("", arxiv_id_or_url).strip("/").split("v")[0]
    )


async def fetch_paper(arxiv_id_or_url: str, github_url: str = "") -> PaperContent:
    """Fetch paper metadata via arxiv API and parse the full PDF."""
    arxiv_id = normalize_arxiv_id(arxiv_id_or_url)
    client = arxiv.Client()
    search = arxiv.Search(id_list=[arxiv_id])
    result = next(client.results(search))

    with tempfile.TemporaryDirectory() as tmpdir:
        pdf_path = result.download_pdf(dirpath=tmpdir)
        full_text, sections, fig_captions, tables = parse_pdf(pdf_path)

    ref_section = sections.get("references", sections.get("bibliography", ""))

    if not github_url:
        github_match = re.search(r"https?://github\.com/[a-zA-Z0-9\-_./]+", result.summary)
        if github_match:
            github_url = github_match.group(0)

    return PaperContent(
        arxiv_id=arxiv_id,
        title=result.title,
        authors=[author.name for author in result.authors],
        abstract=result.summary,
        full_text=full_text,
        sections=sections,
        figures_captions=fig_captions,
        tables_text=tables,
        references_titles=extract_reference_titles(ref_section),
        github_url=github_url,
    )


def parse_pdf(path: str) -> tuple[str, dict[str, str], list[str], list[str]]:
    full_text_parts: list[str] = []
    current_section = "preamble"
    sections: dict[str, list[str]] = {current_section: []}
    figure_captions: list[str] = []
    tables: list[str] = []

    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            text = page.extract_text() or ""
            full_text_parts.append(text)

            for line in text.splitlines():
                stripped = line.strip()
                if SECTION_HEADER_PATTERN.match(stripped) and len(stripped) < 120:
                    current_section = stripped.lower()
                    sections.setdefault(current_section, [])
                elif NAMED_SECTION_PATTERN.match(stripped):
                    current_section = stripped.lower()
                    sections.setdefault(current_section, [])

                sections.setdefault(current_section, [])
                sections[current_section].append(stripped)

                if FIGURE_PATTERN.match(stripped):
                    figure_captions.append(stripped)
                if TABLE_PATTERN.match(stripped):
                    tables.append(stripped)

            for table in page.extract_tables() or []:
                tables.append(str(table))

    joined_sections = {name: "\n".join(lines) for name, lines in sections.items()}
    return "\n".join(full_text_parts), joined_sections, figure_captions, tables


def extract_reference_titles(reference_text: str) -> list[str]:
    titles: list[str] = []
    for line in reference_text.splitlines():
        match = REFERENCE_TITLE_PATTERN.search(line)
        if match:
            titles.append(match.group(1))
    return titles[:50]
