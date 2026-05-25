import unittest
from unittest.mock import AsyncMock, patch

from app.schemas.agents_schema import CVAgentChatResult, CVAgentResponse
from app.schemas.tools_schema import CVAgentToolInput
from app.tools.tools import CVAgentTool


class CVAgentToolTests(unittest.IsolatedAsyncioTestCase):
    async def test_cv_agent_tool_uses_cv_agent_class(self) -> None:
        fake_agent = AsyncMock()
        with patch(
            "app.ai.cv_agent.CVAgent.client_from_settings",
            return_value=fake_agent,
        ) as factory:
            tool = CVAgentTool()

            resolved = tool._get_agent()

        self.assertIs(resolved, fake_agent)
        factory.assert_called_once_with()

    async def test_cv_agent_tool_calls_cv_agent_with_session_id_none(self) -> None:
        fake_agent = AsyncMock()
        fake_agent.chat = AsyncMock(
            return_value=CVAgentChatResult(
                session_id=None,
                response=CVAgentResponse(
                    answer="Azure projects found.", sources=["Projects"]
                ),
            )
        )

        with patch(
            "app.ai.cv_agent.CVAgent.client_from_settings",
            return_value=fake_agent,
        ):
            tool = CVAgentTool()
            await tool.ask_cv_agent(
                CVAgentToolInput(query="What Azure projects are in my CV?")
            )

        fake_agent.chat.assert_awaited_once_with(
            user_message="What Azure projects are in my CV?",
            session_id=None,
        )

    async def test_cv_agent_tool_returns_answer_and_sources(self) -> None:
        fake_agent = AsyncMock()
        fake_agent.chat = AsyncMock(
            return_value=CVAgentChatResult(
                session_id=None,
                response=CVAgentResponse(
                    answer="The CV lists Azure AI Search.", sources=["Skills"]
                ),
            )
        )

        with patch(
            "app.ai.cv_agent.CVAgent.client_from_settings",
            return_value=fake_agent,
        ):
            tool = CVAgentTool()
            result = await tool.ask_cv_agent(
                CVAgentToolInput(query="Which Azure services are listed?")
            )

        self.assertEqual(
            result,
            {"answer": "The CV lists Azure AI Search.", "sources": ["Skills"]},
        )

    async def test_cv_agent_tool_handles_string_response(self) -> None:
        fake_agent = AsyncMock()
        fake_agent.chat = AsyncMock(
            return_value=CVAgentChatResult(
                session_id=None, response="Fallback string answer"
            )
        )

        with patch(
            "app.ai.cv_agent.CVAgent.client_from_settings",
            return_value=fake_agent,
        ):
            tool = CVAgentTool()
            result = await tool.ask_cv_agent(
                CVAgentToolInput(query="Tell me about my skills")
            )

        self.assertEqual(
            result,
            {"answer": "Fallback string answer", "sources": ["none"]},
        )
