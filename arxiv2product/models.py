from dataclasses import dataclass


@dataclass
class PaperContent:
    arxiv_id: str
    title: str
    authors: list[str]
    abstract: str
    full_text: str
    sections: dict[str, str]
    figures_captions: list[str]
    tables_text: list[str]
    references_titles: list[str]
