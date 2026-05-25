from pydantic import BaseModel, ConfigDict, Field
from enum import Enum

######################
# CV AGENT SCHEMAS
######################


class CVAgentResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    answer: str
    sources: list[str]


class CVAgentChatResult(BaseModel):
    session_id: str | None
    response: CVAgentResponse | str


#########################
# SKILLS AGENT SCHEMAS
##########################


class SkillsAgentResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    answer: str = Field(
        ...,
        description="Your conversational answer to the user's question based on the CV",
    )
    sources: list[str] = Field(
        ...,
        description="Sections, file names, and/or web sources used to answer the user's question, or 'none' if not found in CV",
    )


class SkillsAgentChatResult(BaseModel):
    session_id: str | None
    response: SkillsAgentResponse | str


class MixedAgentResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    answer: str = Field(
        ...,
        description="Final user-facing mixed answer combining grounded CV evidence and recommendations.",
    )
    sources: list[str] = Field(
        ...,
        description="Sections, file names, and/or web sources used to answer the user's question, or 'none' if not available.",
    )


#######################
# ROUTER AGENT SCHEMAS
########################


class ChatIntent(str, Enum):
    CV_FACTS = "cv_facts"
    SKILL_SUGGESTION = "skill_suggestion"
    MIXED = "mixed"


class RouterResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    intent: ChatIntent | None = Field(
        ...,
        description="Routing intent. Must be null when direct_answer is true.",
    )
    reasoning: str
    direct_answer: bool = Field(
        ...,
        description="Set to true when the users inquiry is simple greetings, thanks, or other non factual introductory messages.",
    )
    direct_response: str | None = Field(
        ...,
        description="Direct response when direct_answer is True",
    )


#####################################
# MULTI AGENT ORCHESTRATION SCHEMAS
#####################################


class AgentTurn(BaseModel):
    agent: str
    role: str
    content: str
    sources: list[str] = Field(default_factory=list)


class MultiAgentChatResult(BaseModel):
    session_id: str
    intent: ChatIntent | None
    response: str
    agent_turns: list[AgentTurn] = Field(default_factory=list)
    routing_reasoning: str | None = None
