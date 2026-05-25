from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel
from semantic_kernel.connectors.ai.function_choice_behavior import (
    FunctionChoiceBehavior,
)

from ..schemas.agents_schema import CVAgentResponse, RouterResponse, SkillsAgentResponse
from ..tools.tools import CVAgentTool, CVSearchPlugin, SkillsPlugin
from .prompts import CV_AGENT_PROMPT, ROUTER_AGENT_PROMPT, SKILL_SUGGESTER_PROMPT


@dataclass(frozen=True)
class AgentDefinition:
    name: str
    description: str
    instructions: str
    response_model: type[BaseModel]
    schema_name: str
    strict: bool
    plugins: list[Any] | None = None
    function_choice_behavior: Any | None = None
    store_enabled: bool = False


AGENT_REGISTRY: dict[str, AgentDefinition] = {
    "cv": AgentDefinition(
        name="cv-agent",
        description="Answers factual questions about the user's CV using indexed CV data.",
        instructions=CV_AGENT_PROMPT,
        response_model=CVAgentResponse,
        schema_name="cv_agent_response",
        strict=False,
        plugins=[CVSearchPlugin()],
        function_choice_behavior=FunctionChoiceBehavior.Auto(auto_invoke=True),
        store_enabled=False,
    ),
    "skills": AgentDefinition(
        name="skill-suggester",
        description="Suggests skills and CV improvements based on CV facts and job requirements.",
        instructions=SKILL_SUGGESTER_PROMPT,
        response_model=SkillsAgentResponse,
        schema_name="skills_agent_response",
        strict=True,
        plugins=[CVAgentTool(), SkillsPlugin()],
        function_choice_behavior=FunctionChoiceBehavior.Auto(auto_invoke=True),
        store_enabled=False,
    ),
    "router": AgentDefinition(
        name="chat-router",
        description="Routes CV chatbot requests to cv-agent, skill-suggester, or both.",
        instructions=ROUTER_AGENT_PROMPT,
        response_model=RouterResponse,
        schema_name="chat_router_response",
        strict=True,
        plugins=None,
        function_choice_behavior=None,
        store_enabled=False,
    ),
}
