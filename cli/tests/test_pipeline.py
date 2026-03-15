import asyncio
import os
import unittest
from unittest.mock import AsyncMock, patch

from arxiv2product.errors import AgentExecutionError
from arxiv2product.models import PaperContent
from arxiv2product.pipeline import (
    build_compact_paper_context,
    build_full_paper_context,
)


class PipelineAsyncTests(unittest.IsolatedAsyncioTestCase):
    def test_compact_context_is_smaller_than_full_context(self):
        paper = PaperContent(
            arxiv_id="2603.09229",
            title="Example",
            authors=["Alice", "Bob"],
            abstract="Abstract",
            full_text="Full text",
            sections={
                "introduction": "intro " * 5000,
                "method": "method " * 5000,
                "results": "results " * 5000,
            },
            figures_captions=["Figure 1"] * 20,
            tables_text=["Table 1"] * 10,
            references_titles=["Reference"] * 30,
        )
        full_context = build_full_paper_context(paper)
        compact_context = build_compact_paper_context(
            paper,
            primitives_summary="primitive summary " * 400,
        )
        self.assertLess(len(compact_context), len(full_context))
        self.assertIn("TECHNICAL PRIMITIVES SUMMARY", compact_context)

    async def test_run_pipeline_uses_agno_and_bypasses_agentica(self):
        with (
            patch("arxiv2product.pipeline._run_pipeline", new_callable=AsyncMock, return_value="products_2603_09229.md") as run_agno,
            patch("arxiv2product.pipeline._get_speed_profile", return_value="slow"),
        ):
            from arxiv2product.pipeline import run_pipeline

            output = await run_pipeline("2603.09229", model="anthropic/claude-sonnet-4")

        self.assertEqual(output, "products_2603_09229.md")
        run_agno.assert_awaited_once_with("2603.09229", "anthropic/claude-sonnet-4")


if __name__ == "__main__":
    unittest.main()
