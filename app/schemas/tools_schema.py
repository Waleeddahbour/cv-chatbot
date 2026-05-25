from pydantic import BaseModel, Field
from typing import Optional
from enum import Enum


class SectionFilterEnum(str, Enum):
    Summary = "Summary"
    Skills = "Skills"
    Experience = "Experience"
    Education = "Education"
    Certifications = "Certifications"
    Languages = "Languages"
    Strengths = "Strengths"
    Projects = "Projects"


class SearchKB(BaseModel):
    query: str = Field(
        ...,
        description="The search query string. Elaborate for better retrieval.",
    )
    top_k: Optional[int] = Field(
        3,
        description="Number of top results to return. Maximum is 3.",
    )
    filter: Optional[list[SectionFilterEnum]] = Field(
        None,
        description="Optional section filters. Must be a list of the section filters to apply to the search.",
    )


class SkillsToolInput(BaseModel):
    query: str = Field(
        ...,
        description="Based on the users cv and job description, write a natural search query under 400 characters to help the user improve CV skills based on the target role or market context.",
    )


class CVAgentToolInput(BaseModel):
    query: str = Field(
        ...,
        description="A factual question for the CV agent about the user's CV.",
    )
