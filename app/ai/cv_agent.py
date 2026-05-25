import asyncio

from semantic_kernel.agents import AzureResponsesAgent

from ..config.settings import settings
from .agent_factory import create_registered_agent
from .base_agent import BaseChatAgent
from .agents_management import build_agent_messages
from ..schemas.agents_schema import CVAgentResponse, CVAgentChatResult


class CVAgent(BaseChatAgent[CVAgentResponse]):

    def __init__(
        self,
        agent: AzureResponsesAgent,
        model: str,
        max_history_tokens: int,
    ):
        super().__init__(
            agent=agent,
            response_model=CVAgentResponse,
            model=model,
        )
        self.max_history_tokens = max_history_tokens

    @classmethod
    def client_from_settings(cls):
        agent = create_registered_agent("cv")

        return cls(
            agent=agent,
            model=settings.AZURE_OPENAI_CHAT_DEPLOYMENT_NAME,
            max_history_tokens=settings.MAX_HISTORY_TOKENS,
        )

    async def chat(
        self, user_message: str, session_id: str | None = None
    ) -> CVAgentChatResult:
        messages = build_agent_messages(
            user_message=user_message,
            chat_history=[],
            model=self.model,
            max_history_tokens=self.max_history_tokens,
        )
        response = await self.run_agent(messages)
        return CVAgentChatResult(session_id=session_id, response=response)


async def main():
    cv_agent = CVAgent.client_from_settings()
    session_id = None

    try:
        while True:
            print("*" * 80)
            user_input = input("You: ")
            print("*" * 80)
            if user_input.lower() in {"exit", "quit"}:
                print("Exiting CV Agent. Goodbye!")
                break

            result = await cv_agent.chat(user_input, session_id=session_id)
            session_id = result.session_id
            print("*" * 80)
            print(f"Session: {session_id}")
            print("AI:\n", result.response)
            print("*" * 80)
    finally:
        await cv_agent.close()


if __name__ == "__main__":
    asyncio.run(main())
