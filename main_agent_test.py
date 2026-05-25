import asyncio
import sys
from dataclasses import dataclass
from typing import Final

from app.ai.chat_main_orchestration import MultiAgentCVChat


@dataclass(frozen=True)
class TestScenario:
    category: str
    question: str


QUESTIONS: Final[list[TestScenario]] = [
    TestScenario(
        category="direct_answer",
        question="Hi",
    ),
    TestScenario(
        category="cv_facts",
        question="Give me a short summary of the CV.",
    ),
    TestScenario(
        category="cv_facts",
        question="What Azure services are mentioned in the CV?",
    ),
    TestScenario(
        category="cv_facts",
        question="What AI or GenAI projects are listed?",
    ),
    TestScenario(
        category="cv_facts",
        question="What certifications are listed in the CV?",
    ),
    TestScenario(
        category="skill_suggestion",
        question=(
            "Based only on my CV, what skills should I strengthen for an AI engineer role?"
        ),
    ),
    TestScenario(
        category="skill_suggestion",
        question=(
            "Suggest improvements I should add to my CV to be stronger for GenAI and RAG roles."
        ),
    ),
    TestScenario(
        category="mixed",
        question=(
            "Using the facts in my CV, tell me which AI engineer strengths I already have and which missing skills I should add."
        ),
    ),
    TestScenario(
        category="mixed",
        question=(
            "Compare my current CV against a typical Azure AI engineer profile and give me a grounded gap analysis."
        ),
    ),
    TestScenario(
        category="mixed",
        question=(
            "Give me a final assessment of how well this CV fits an AI engineer position, using evidence from the CV first and then recommendations."
        ),
    ),
]


async def run_main_agent_test(save_history: bool = False) -> None:
    chat = MultiAgentCVChat()
    session_id: str | None = "controlled-test-session"

    try:
        for index, scenario in enumerate(QUESTIONS, start=1):
            print(f"\n{'=' * 100}")
            print(f"Question {index} [{scenario.category}]: {scenario.question}")
            print(f"{'-' * 100}")

            result = await chat.chat(scenario.question, session_id=session_id)
            print("\n\n", "#" * 15, "RESULT", "#" * 15, "\n")
            session_id = result.session_id

            print(f"Session ID: {result.session_id}")
            print(f"Intent: {result.intent}")
            if result.routing_reasoning:
                print(f"Routing: {result.routing_reasoning}")
            print("Response:")
            print(result.response)

            if result.agent_turns:
                print(f"{'-' * 100}")
                print("Agent Turns:")
                for turn in result.agent_turns:
                    print(f"[{turn.agent}] {turn.content}")
            print("\n", "#" * 11, "RESULT END", "#" * 11)
    finally:
        if session_id and not save_history:
            print(f"\nDeleting test session: {session_id}")
            await chat.history_store.delete_session(session_id)
        elif session_id and save_history:
            print(f"\nPreserving test session: {session_id}")
        await chat.close()


if __name__ == "__main__":
    asyncio.run(run_main_agent_test(save_history="--save" in sys.argv[1:]))
