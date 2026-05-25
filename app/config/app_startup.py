import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI

from ..RAG.ingestion.run_ingestion import ensure_ingestion_resources
from ..ai.chat_main_orchestration import MultiAgentCVChat
from .settings import settings

REQUIRED_CAPABILITIES = (
    "azure_openai",
    "azure_search",
    "azure_cosmos",
    "azure_ai_services",
    "azure_blob",
    "tavily",
)


async def initialize_application_services() -> MultiAgentCVChat:
    settings.require(*REQUIRED_CAPABILITIES)

    await asyncio.to_thread(ensure_ingestion_resources)

    chat_service = MultiAgentCVChat()
    await chat_service.history_store.ensure_db_and_container()
    return chat_service


@asynccontextmanager
async def lifespan(app: FastAPI):
    chat_service = await initialize_application_services()
    app.state.chat_service = chat_service
    try:
        yield
    finally:
        await chat_service.close()
