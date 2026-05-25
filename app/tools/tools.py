import asyncio
from typing import Annotated, Awaitable, Callable
import logging
import json

from ..RAG.search.azure_search_retrieval import AzureSearchRetrieval
from ..config.settings import settings
from ..schemas.tools_schema import (
    CVAgentToolInput,
    SearchKB,
    SkillsToolInput,
)

from semantic_kernel.functions import kernel_function
from semantic_kernel.filters import FunctionInvocationContext
from tavily import AsyncTavilyClient

logger = logging.getLogger(__name__)

MAX_TAVILY_QUERY_LENGTH = 400


def _truncate_search_query(query: str, max_length: int = MAX_TAVILY_QUERY_LENGTH) -> str:
    normalized_query = " ".join(query.split())
    if len(normalized_query) <= max_length:
        return normalized_query

    truncated_query = normalized_query[:max_length].rstrip()
    last_space_index = truncated_query.rfind(" ")
    if last_space_index > max_length // 2:
        truncated_query = truncated_query[:last_space_index].rstrip()

    return truncated_query


async def logger_filter(
    context: FunctionInvocationContext,
    next: Callable[FunctionInvocationContext, Awaitable[None]],  # type: ignore
):
    logger.info(
        "Function invocation: %s.%s",
        context.function.plugin_name,
        context.function.name,
    )
    await next(context)

    logger.info(
        "Function completed: %s.%s",
        context.function.plugin_name,
        context.function.name,
    )


class CVSearchPlugin:
    @kernel_function(
        name="search_kb",
        description="Search the user's CV knowledge base for relevant facts and sources.",
    )
    async def search_kb(
        self, search_parameters: Annotated[SearchKB, "Expected tool input schema"]
    ):
        logger.info(
            "CV search requested. query=%r top_k=%s filter=%s",
            search_parameters.query,
            search_parameters.top_k,
            search_parameters.filter,
        )
        async with AzureSearchRetrieval() as search_service:
            context = await search_service.retrieve_relevant_chunks(
                **search_parameters.model_dump(exclude_none=True)
            )

        # Return only the fields the model needs to answer and cite the CV.
        results = [
            {
                "text": item.get("text", ""),
                "section": item.get("section", ""),
                "source_file": item.get("source_file", ""),
            }
            for item in context
        ]
        logger.debug("CV search results: %s", json.dumps(results, ensure_ascii=False))
        return results


class CVAgentTool:
    def __init__(self):
        self._agent = None

    def _get_agent(self):
        if self._agent is None:
            from ..ai.cv_agent import CVAgent

            self._agent = CVAgent.client_from_settings()
        return self._agent

    @kernel_function(
        name="ask_cv_agent",
        description="Ask the CV agent factual questions about the user's CV. Use this to gather grounded CV evidence before making skill recommendations.",
    )
    async def ask_cv_agent(
        self,
        cv_agent_input: Annotated[CVAgentToolInput, "Expected tool input schema"],
    ):
        logger.info(
            "Delegating factual CV follow-up to CVAgent. query=%r", cv_agent_input.query
        )
        response = await self._get_agent().chat(
            user_message=cv_agent_input.query,
            session_id=None,
        )
        result = response.response
        if isinstance(result, str):
            return {
                "answer": result,
                "sources": ["none"],
            }
        return result.model_dump()


class SkillsPlugin:
    @kernel_function(
        name="suggest_skills",
        description="Suggest relevant skills based on CV and job description. Write a natural query to search for context to help the user improve his cv skills based on the job description.",
    )
    async def suggest_skills(
        self, websearch_input: Annotated[SkillsToolInput, "Tool input schema"]
    ):
        settings.require("tavily")
        search_query = _truncate_search_query(websearch_input.query)
        logger.info(
            "Skills suggestion requested. original_query_length=%s effective_query_length=%s",
            len(websearch_input.query),
            len(search_query),
        )
        logger.debug("Skills suggestion query: %s", search_query)

        tavily_client = AsyncTavilyClient(api_key=settings.TAVILY_API_KEY)
        try:
            response = await tavily_client.search(search_query)
        finally:
            await tavily_client.close()

        logger.debug(
            "Skills suggestion response: %s", json.dumps(response, ensure_ascii=False)
        )
        return [
            {
                "title": results.get("title", "No title"),
                "url": results.get("url", "No URL"),
                "content": results.get("content", "No content"),
            }
            for results in response["results"]
        ]


if __name__ == "__main__":
    plugin = SkillsPlugin()
    query = "What skills should I add to my CV for an AI engineering role?"
    result = asyncio.run(plugin.suggest_skills(SkillsToolInput(query=query)))
    logger.info("Suggested skills result: %s", result)
