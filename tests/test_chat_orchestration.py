import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from app.ai.chat_main_orchestration import MultiAgentCVChat
from app.schemas.agents_schema import (
    AgentTurn,
    ChatIntent,
    MultiAgentChatResult,
    RouterResponse,
)


class FakeHistoryStore:
    def __init__(self, messages: list[dict] | None = None):
        self.messages = messages or []
        self.append_calls: list[dict[str, str | None]] = []
        self.closed = False

    async def get_messages(self, session_id: str, max_messages: int | None = None):
        return list(self.messages)

    async def append_message(self, session_id: str | None, role: str, content: str):
        self.append_calls.append(
            {"session_id": session_id, "role": role, "content": content}
        )
        return {"session_id": session_id or "session-generated"}

    async def close(self) -> None:
        self.closed = True


def build_chat(history_store: FakeHistoryStore | None = None) -> MultiAgentCVChat:
    fake_cv_agent = SimpleNamespace(
        agent="cv-agent", model="gpt-5", run_agent=AsyncMock()
    )
    fake_skill_agent = SimpleNamespace(
        agent="skills-agent", model="gpt-5", run_agent=AsyncMock()
    )
    return MultiAgentCVChat(
        history_store=history_store or FakeHistoryStore(),
        cv_agent=fake_cv_agent,
        skill_agent=fake_skill_agent,
        router_agent=object(),
    )


class ChatOrchestrationTests(unittest.IsolatedAsyncioTestCase):
    def test_get_agents_excludes_router_agent_from_collaboration(self) -> None:
        chat = build_chat()

        agents = chat.get_agents()

        self.assertEqual(agents, ["cv-agent", "skills-agent"])

    async def test_chat_returns_direct_response_without_execution(self) -> None:
        chat = build_chat()
        chat.route_message = AsyncMock(
            return_value=RouterResponse(
                intent=None,
                reasoning="Greeting detected.",
                direct_answer=True,
                direct_response="Hello there",
            )
        )
        chat.execute_route = AsyncMock()

        result = await chat.chat("hello")

        self.assertEqual(result.session_id, "direct")
        self.assertIsNone(result.intent)
        self.assertEqual(result.response, "Hello there")
        chat.execute_route.assert_not_awaited()

    async def test_execute_route_uses_cv_handler(self) -> None:
        chat = build_chat()
        expected = MultiAgentChatResult(
            session_id="session-1",
            intent=ChatIntent.CV_FACTS,
            response="CV answer",
        )
        chat.run_cv_only = AsyncMock(return_value=expected)
        chat.run_skill_only = AsyncMock()
        chat.run_collaboration = AsyncMock()

        result = await chat.execute_route(
            "What projects do I have?",
            "session-1",
            RouterResponse(
                intent=ChatIntent.CV_FACTS,
                reasoning="Fact lookup",
                direct_answer=False,
                direct_response=None,
            ),
        )

        chat.run_cv_only.assert_awaited_once()
        chat.run_skill_only.assert_not_awaited()
        chat.run_collaboration.assert_not_awaited()
        self.assertEqual(result.routing_reasoning, "Fact lookup")

    async def test_execute_route_uses_skill_handler(self) -> None:
        chat = build_chat()
        chat.run_cv_only = AsyncMock()
        chat.run_skill_only = AsyncMock(
            return_value=MultiAgentChatResult(
                session_id="session-2",
                intent=ChatIntent.SKILL_SUGGESTION,
                response="Skills answer",
            )
        )
        chat.run_collaboration = AsyncMock()

        await chat.execute_route(
            "How can I improve this CV?",
            "session-2",
            RouterResponse(
                intent=ChatIntent.SKILL_SUGGESTION,
                reasoning="Recommendation request",
                direct_answer=False,
                direct_response=None,
            ),
        )

        chat.run_skill_only.assert_awaited_once()
        chat.run_cv_only.assert_not_awaited()
        chat.run_collaboration.assert_not_awaited()

    async def test_execute_route_uses_collaboration_handler(self) -> None:
        chat = build_chat()
        chat.run_cv_only = AsyncMock()
        chat.run_skill_only = AsyncMock()
        chat.run_collaboration = AsyncMock(
            return_value=MultiAgentChatResult(
                session_id="session-3",
                intent=ChatIntent.MIXED,
                response="Mixed answer",
            )
        )

        await chat.execute_route(
            "How does my background fit this AI role?",
            "session-3",
            RouterResponse(
                intent=ChatIntent.MIXED,
                reasoning="Needs collaboration",
                direct_answer=False,
                direct_response=None,
            ),
        )

        chat.run_collaboration.assert_awaited_once()
        chat.run_cv_only.assert_not_awaited()
        chat.run_skill_only.assert_not_awaited()

    async def test_run_single_agent_persists_history(self) -> None:
        history_store = FakeHistoryStore(
            messages=[{"role": "user", "content": "Earlier"}]
        )
        chat = build_chat(history_store=history_store)
        agent_runner = AsyncMock(return_value=SimpleNamespace(answer="Grounded answer"))

        with patch(
            "app.ai.chat_main_orchestration.build_agent_messages",
            return_value=["message"],
        ) as build_messages:
            result = await chat.run_single_agent(
                agent_runner=agent_runner,
                model="gpt-5",
                user_message="What is in my CV?",
                session_id=None,
                intent=ChatIntent.CV_FACTS,
            )

        self.assertEqual(result.session_id, "session-generated")
        self.assertEqual(result.response, "Grounded answer")
        self.assertEqual(
            history_store.append_calls,
            [
                {
                    "session_id": None,
                    "role": "user",
                    "content": "What is in my CV?",
                },
                {
                    "session_id": "session-generated",
                    "role": "assistant",
                    "content": "Grounded answer",
                },
            ],
        )
        build_messages.assert_called_once()
        agent_runner.assert_awaited_once_with(["message"])

    async def test_route_message_falls_back_on_invalid_router_output(self) -> None:
        history_store = FakeHistoryStore(
            messages=[{"role": "assistant", "content": "Earlier"}]
        )
        chat = build_chat(history_store=history_store)

        with (
            patch(
                "app.ai.chat_main_orchestration.build_agent_messages",
                return_value=["message"],
            ),
            patch(
                "app.ai.chat_main_orchestration.run_agent",
                new=AsyncMock(return_value="invalid"),
            ),
        ):
            result = await chat.route_message(
                "Tell me about my CV", session_id="session-1"
            )

        self.assertEqual(result.intent, ChatIntent.CV_FACTS)
        self.assertFalse(result.direct_answer)
        self.assertIn("defaulted to cv_facts", result.reasoning)

    def test_agent_response_callback_records_turn(self) -> None:
        chat = build_chat()

        chat.agent_response_callback(
            SimpleNamespace(name="cv-agent", role="assistant", content="Evidence")
        )

        self.assertEqual(
            chat.agent_turns,
            [AgentTurn(agent="cv-agent", role="assistant", content="Evidence")],
        )
