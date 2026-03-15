import os
import unittest
from unittest.mock import patch

from arxiv2product.backend import (
    OPENAI_COMPATIBLE_BACKEND,
    get_agno_model,
    get_execution_backend_name,
    normalize_model_name,
)

class BackendSelectionTests(unittest.TestCase):
    def test_normalize_model_name_removes_openrouter_prefix(self):
        self.assertEqual(
            normalize_model_name("openrouter:google/gemini-2.5-pro"),
            "google/gemini-2.5-pro",
        )

    def test_execution_backend_defaults_to_agentica_without_keys(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(get_execution_backend_name(), "agentica")

    def test_execution_backend_prefers_openai_compatible_with_direct_key(self):
        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "key"}, clear=True):
            self.assertEqual(get_execution_backend_name(), OPENAI_COMPATIBLE_BACKEND)

class AgnoModelTests(unittest.TestCase):
    def test_get_agno_model_returns_openai_chat_when_only_openai_key_present(self):
        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-123"}, clear=True):
            from agno.models.openai import OpenAIChat
            model = get_agno_model("gpt-4o")
            self.assertIsInstance(model, OpenAIChat)
            self.assertEqual(model.id, "gpt-4o")

    def test_get_agno_model_returns_openrouter_when_openrouter_key_present(self):
        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "sk-or-123", "OPENAI_API_KEY": "sk-123"}, clear=True):
            from agno.models.openrouter import OpenRouter
            model = get_agno_model("anthropic/claude-3-5-sonnet")
            self.assertIsInstance(model, OpenRouter)
            self.assertEqual(model.id, "anthropic/claude-3-5-sonnet")

if __name__ == "__main__":
    unittest.main()
