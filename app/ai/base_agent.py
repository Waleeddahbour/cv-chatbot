from typing import Generic, TypeVar

from pydantic import BaseModel
from semantic_kernel.agents import AzureResponsesAgent
from semantic_kernel.contents.chat_message_content import ChatMessageContent

from .agents_management import run_agent

ResponseModelT = TypeVar("ResponseModelT", bound=BaseModel)


class BaseChatAgent(Generic[ResponseModelT]):
    def __init__(
        self,
        agent: AzureResponsesAgent,
        response_model: type[ResponseModelT],
        model: str,
    ):
        self.agent = agent
        self._response_model = response_model
        self.model = model

    async def close(self) -> None:
        return None

    async def run_agent(
        self, messages: list[ChatMessageContent]
    ) -> ResponseModelT | str:
        return await run_agent(
            agent=self.agent,
            messages=messages,
            response_model=self._response_model,
        )

    @staticmethod
    def extract_assistant_message(response: ResponseModelT | str) -> str:
        if isinstance(response, str):
            return response

        answer = getattr(response, "answer", None)
        if isinstance(answer, str):
            return answer

        return str(response)
