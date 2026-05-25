import asyncio
from typing import Any, Awaitable, Callable
from semantic_kernel.agents import (
    Agent,
    GroupChatOrchestration,
    RoundRobinGroupChatManager,
)
from semantic_kernel.agents.runtime import InProcessRuntime
from semantic_kernel.contents import ChatMessageContent

from ..config.settings import settings
from .agents_management import build_agent_messages, run_agent
from .cv_agent import CVAgent
from .skills_agent import SkillSuggester
from ..cosmos.azure_cosmos_service import CosmosChatHistory
from ..schemas.agents_schema import (
    AgentTurn,
    RouterResponse,
    MultiAgentChatResult,
    ChatIntent,
    MixedAgentResponse,
)
from .router_agent import create_router_agent
from semantic_kernel.utils.logging import setup_logging
import logging

setup_logging()
logging.getLogger("semantic_kernel").setLevel(logging.INFO)
logger = logging.getLogger(__name__)


class MultiAgentCVChat:
    def __init__(
        self,
        history_store: CosmosChatHistory | None = None,
        cv_agent: CVAgent | None = None,
        skill_agent: SkillSuggester | None = None,
        router_agent=None,
    ):
        self.history_store = history_store or CosmosChatHistory()
        self.cv_agent = cv_agent or CVAgent.client_from_settings()
        self.skill_agent = skill_agent or SkillSuggester.client_from_settings()
        self.router_agent = router_agent or create_router_agent()
        self.model = settings.AZURE_OPENAI_CHAT_DEPLOYMENT_NAME
        self.max_history_tokens = settings.MAX_HISTORY_TOKENS
        self.max_chat_history_messages = settings.MAX_CHAT_HISTORY_MESSAGES
        self.agent_turns: list[AgentTurn] = []

    async def close(self) -> None:
        await self.history_store.close()

    def get_agents(self) -> list[Agent]:
        return [self.cv_agent.agent, self.skill_agent.agent]

    async def chat(
        self, user_message: str, session_id: str | None = None
    ) -> MultiAgentChatResult:
        logger.info("Chat request received. session_id=%s", session_id)
        route = await self.route_message(user_message, session_id=session_id)
        logger.info(
            "Chat route resolved. session_id=%s intent=%s direct_answer=%s",
            session_id,
            route.intent,
            route.direct_answer,
        )
        if route.direct_answer and route.direct_response:
            return MultiAgentChatResult(
                session_id=session_id or "direct",
                intent=None,
                response=route.direct_response,
                routing_reasoning=route.reasoning,
            )
        return await self.execute_route(user_message, session_id, route)

    async def route_message(
        self, user_message: str, session_id: str | None = None
    ) -> RouterResponse:
        chat_history = (
            await self.history_store.get_messages(
                session_id,
                max_messages=self.max_chat_history_messages,
            )
            if session_id
            else []
        )
        messages = build_agent_messages(
            user_message=user_message,
            chat_history=chat_history,
            model=self.model,
            max_history_tokens=self.max_history_tokens,
        )
        response = await run_agent(
            agent=self.router_agent,
            messages=messages,
            response_model=RouterResponse,
        )

        if isinstance(response, RouterResponse):
            return response

        return RouterResponse(
            intent=ChatIntent.CV_FACTS,
            reasoning=(
                "Router did not return valid structured output; defaulted to cv_facts."
            ),
            direct_answer=False,
            direct_response=None,
        )

    async def execute_route(
        self,
        user_message: str,
        session_id: str | None,
        route: RouterResponse,
    ) -> MultiAgentChatResult:
        if route.intent is None:
            return MultiAgentChatResult(
                session_id=session_id or "direct",
                intent=None,
                response=route.direct_response or "",
                routing_reasoning=route.reasoning,
            )

        handlers: dict[
            ChatIntent,
            Callable[[str, str | None, ChatIntent], Awaitable[MultiAgentChatResult]],
        ] = {
            ChatIntent.CV_FACTS: self.run_cv_only,
            ChatIntent.SKILL_SUGGESTION: self.run_skill_only,
            ChatIntent.MIXED: self.run_collaboration,
        }

        handler = handlers.get(route.intent, self.run_collaboration)
        result = await handler(user_message, session_id, route.intent)

        result.routing_reasoning = route.reasoning
        return result

    @staticmethod
    def extract_response_text(response: Any) -> str:
        return response.answer if hasattr(response, "answer") else str(response)

    @staticmethod
    def normalize_mixed_response(value: Any) -> MixedAgentResponse:
        if isinstance(value, MixedAgentResponse):
            return value

        if isinstance(value, ChatMessageContent):
            content = value.content or ""
            try:
                return MixedAgentResponse.model_validate_json(content)
            except ValueError:
                return MixedAgentResponse(answer=content, sources=[])

        if isinstance(value, list):
            messages = [
                message.content or ""
                for message in value
                if isinstance(message, ChatMessageContent)
                and (message.content or "").strip()
            ]
            content = messages[-1] if messages else str(value)
            try:
                return MixedAgentResponse.model_validate_json(content)
            except ValueError:
                return MixedAgentResponse(answer=content, sources=[])

        if isinstance(value, str):
            try:
                return MixedAgentResponse.model_validate_json(value)
            except ValueError:
                return MixedAgentResponse(answer=value, sources=[])

        content = str(value)
        try:
            return MixedAgentResponse.model_validate_json(content)
        except ValueError:
            return MixedAgentResponse(answer=content, sources=[])

    async def run_single_agent(
        self,
        agent_runner: Callable[[list[ChatMessageContent]], Awaitable[Any]],
        model: str,
        user_message: str,
        session_id: str | None,
        intent: ChatIntent,
    ) -> MultiAgentChatResult:
        logger.info(
            "Running single-agent flow. intent=%s session_id=%s", intent, session_id
        )
        chat_history = (
            await self.history_store.get_messages(
                session_id,
                max_messages=self.max_chat_history_messages,
            )
            if session_id
            else []
        )
        session = await self.history_store.append_message(
            session_id=session_id,
            role="user",
            content=user_message,
        )
        resolved_session_id = session["session_id"]
        messages = build_agent_messages(
            user_message=user_message,
            chat_history=chat_history,
            model=model,
            max_history_tokens=self.max_history_tokens,
        )
        response = await agent_runner(messages)
        response_text = self.extract_response_text(response)
        await self.history_store.append_message(
            session_id=resolved_session_id,
            role="assistant",
            content=response_text,
        )

        return MultiAgentChatResult(
            session_id=resolved_session_id,
            intent=intent,
            response=response_text,
        )

    async def run_cv_only(
        self,
        user_message: str,
        session_id: str | None,
        intent: ChatIntent,
    ) -> MultiAgentChatResult:
        return await self.run_single_agent(
            agent_runner=self.cv_agent.run_agent,
            model=self.cv_agent.model,
            user_message=user_message,
            session_id=session_id,
            intent=intent,
        )

    async def run_skill_only(
        self,
        user_message: str,
        session_id: str | None,
        intent: ChatIntent,
    ) -> MultiAgentChatResult:
        return await self.run_single_agent(
            agent_runner=self.skill_agent.run_agent,
            model=self.skill_agent.model,
            user_message=user_message,
            session_id=session_id,
            intent=intent,
        )

    async def run_collaboration(
        self,
        user_message: str,
        session_id: str | None,
        intent: ChatIntent,
    ) -> MultiAgentChatResult:
        logger.info("Running collaboration flow. session_id=%s", session_id)
        self.agent_turns = []
        session = await self.history_store.append_message(
            session_id=session_id,
            role="user",
            content=user_message,
        )
        resolved_session_id = session["session_id"]

        value = await self.run_group_chat(user_message)
        mixed_response = self.normalize_mixed_response(value)
        final_answer = mixed_response.answer

        await self.history_store.append_message(
            session_id=resolved_session_id,
            role="assistant",
            content=final_answer,
        )

        return MultiAgentChatResult(
            session_id=resolved_session_id,
            intent=intent,
            response=final_answer,
            agent_turns=self.agent_turns,
        )

    async def run_group_chat(self, user_message: str) -> str:
        logger.info("Starting group chat orchestration.")
        group_chat_orchestration = GroupChatOrchestration(
            members=self.get_agents(),
            manager=RoundRobinGroupChatManager(max_rounds=3),
            agent_response_callback=self.agent_response_callback,
        )

        runtime = InProcessRuntime()
        runtime.start()
        try:
            orchestration_result = await group_chat_orchestration.invoke(
                task=self.build_collaboration_task(user_message),
                runtime=runtime,
            )
            result = await orchestration_result.get()
            logger.info("Group chat orchestration completed.")
            if isinstance(result, ChatMessageContent):
                return result.content or ""

            if isinstance(result, list):
                return "\n".join(
                    message.content or ""
                    for message in result
                    if isinstance(message, ChatMessageContent)
                )

            return str(result)
        finally:
            await runtime.stop_when_idle()

    def build_collaboration_task(self, user_message: str) -> str:
        return (
            "Collaborate on a mixed CV chatbot request.\n"
            "Turn 1: cv-agent should retrieve and state the factual CV evidence needed to answer the request.\n"
            "Turn 2: skill-suggester should use those facts to provide any requested recommendations or gap analysis.\n"
            "Turn 3: cv-agent should verify the final answer remains grounded in the CV and remove unsupported claims.\n"
            "Return one concise final answer for the user.\n\n"
            f"User message:\n{user_message}"
        )

    def agent_response_callback(self, message: ChatMessageContent) -> None:
        logger.debug(
            "Agent turn recorded. agent=%s role=%s",
            message.name or "agent",
            message.role,
        )
        self.agent_turns.append(
            AgentTurn(
                agent=message.name or "agent",
                role=str(message.role),
                content=message.content or "",
            )
        )


async def main() -> None:
    chat = MultiAgentCVChat()
    session_id = None
    try:
        while True:
            user_input = input("You: ")
            if user_input.lower() in {"exit", "quit"}:
                break

            result = await chat.chat(user_input, session_id=session_id)
            session_id = result.session_id
            print(f"Intent: {result.intent}")
            print(f"AI:\n{result.response}")
    finally:
        await chat.close()


if __name__ == "__main__":
    asyncio.run(main())
