import unittest

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.routers.chat import router as chat_router
from app.schemas.agents_schema import ChatIntent, MultiAgentChatResult

QUESTIONS = [
    "What programming languages are mentioned in the CV?",
    "What Azure services are listed in the CV?",
    "What AI projects are mentioned?",
    "What experience with RAG is described?",
    "What certifications are included?",
    "What education background is listed?",
    "What machine learning experience is included?",
    "What backend technologies are mentioned?",
    "What cloud platforms appear in the CV?",
    "What AI development tools and frameworks are listed?",
]


class ScenarioChatService:
    async def chat(self, user_message: str, session_id: str | None = None):
        return MultiAgentChatResult(
            session_id=session_id or "scenario-session",
            intent=ChatIntent.CV_FACTS,
            response=f"Indexed answer only for: {user_message}",
            routing_reasoning="Scenario mock",
        )


class CVQuestionScenarioTests(unittest.TestCase):
    def test_ten_cv_related_questions_return_answers(self) -> None:
        app = FastAPI()
        app.include_router(chat_router)
        app.state.chat_service = ScenarioChatService()
        client = TestClient(app)

        for question in QUESTIONS:
            with self.subTest(question=question):
                response = client.post("/chat", json={"message": question})

                self.assertEqual(response.status_code, 200)
                body = response.json()
                self.assertEqual(body["intent"], "cv_facts")
                self.assertEqual(body["session_id"], "scenario-session")
                self.assertIn(question, body["response"])
