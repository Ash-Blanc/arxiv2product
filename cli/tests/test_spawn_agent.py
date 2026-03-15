import asyncio
import unittest
from unittest.mock import AsyncMock, patch

from arxiv2product.errors import AgentExecutionError


class SpawnAgentTests(unittest.IsolatedAsyncioTestCase):
    """Tests for spawn_agent."""

    async def test_spawn_agent_creates_agno_agent(self):
        """Verify Agno agent creation."""
        from arxiv2product.pipeline import spawn_agent
        from agno.agent import Agent

        agent = await spawn_agent(premise="test", model="test-model")
        self.assertIsInstance(agent, Agent)
        self.assertEqual(agent.description, "test")

    async def test_call_agent_text_runs_agent(self):
        """Verify call_agent_text uses Agno's run method."""
        from arxiv2product.pipeline import call_agent_text
        from agno.agent import Agent
        from agno.models.message import Message

        agent = AsyncMock(spec=Agent)
        agent.run.return_value = AsyncMock(content="result")

        result = await call_agent_text(agent, "prompt", phase="test")
        self.assertEqual(result, "result")
        agent.run.assert_called_once_with("prompt")


if __name__ == "__main__":
    unittest.main()
