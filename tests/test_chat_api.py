import unittest

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.routers.chat import router as chat_router
from app.api.routers.system import router as system_router
from app.schemas.agents_schema import AgentTurn, ChatIntent, MultiAgentChatResult


class FakeChatService:
    def __init__(self, *, result=None, error: Exception | None = None):
        self._result = result
        self._error = error

    async def chat(self, user_message: str, session_id: str | None = None):
        if self._error is not None:
            raise self._error
        return self._result


def build_test_client(chat_service: FakeChatService) -> TestClient:
    app = FastAPI()
    app.include_router(system_router)
    app.include_router(chat_router)
    app.state.chat_service = chat_service
    return TestClient(app)


class ChatApiTests(unittest.TestCase):
    def test_health_endpoint_returns_ok(self) -> None:
        client = build_test_client(FakeChatService())

        response = client.get("/health")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "ok"})

    def test_chat_endpoint_returns_chat_response(self) -> None:
        client = build_test_client(
            FakeChatService(
                result=MultiAgentChatResult(
                    session_id="session-1",
                    intent=ChatIntent.CV_FACTS,
                    response="The CV mentions Azure AI Search.",
                    routing_reasoning="Question is factual.",
                )
            )
        )

        response = client.post(
            "/chat",
            json={"message": "What Azure services are mentioned?", "session_id": None},
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json(),
            {
                "session_id": "session-1",
                "intent": "cv_facts",
                "response": "The CV mentions Azure AI Search.",
                "routing_reasoning": "Question is factual.",
                "agent_turns": [],
            },
        )

    def test_chat_endpoint_maps_value_error_to_400(self) -> None:
        client = build_test_client(
            FakeChatService(error=ValueError("message cannot be empty"))
        )

        response = client.post("/chat", json={"message": "Hello"})

        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json(), {"detail": "message cannot be empty"})

    def test_chat_endpoint_maps_unexpected_error_to_500(self) -> None:
        client = build_test_client(FakeChatService(error=RuntimeError("boom")))

        response = client.post("/chat", json={"message": "Hello"})

        self.assertEqual(response.status_code, 500)
        self.assertEqual(response.json(), {"detail": "Chat request failed"})

    def test_chat_endpoint_returns_agent_turns_for_mixed_flow(self) -> None:
        client = build_test_client(
            FakeChatService(
                result=MultiAgentChatResult(
                    session_id="session-2",
                    intent=ChatIntent.MIXED,
                    response="Grounded collaboration answer.",
                    routing_reasoning="Needs CV facts and recommendations.",
                    agent_turns=[
                        AgentTurn(
                            agent="cv-agent",
                            role="assistant",
                            content="The candidate has Azure and RAG experience.",
                        )
                    ],
                )
            )
        )

        response = client.post(
            "/chat",
            json={"message": "How does my CV fit an AI engineer role?"},
        )

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["intent"], "mixed")
        self.assertEqual(len(body["agent_turns"]), 1)
        self.assertEqual(body["agent_turns"][0]["agent"], "cv-agent")
