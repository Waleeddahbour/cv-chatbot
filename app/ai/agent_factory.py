from typing import Any

from pydantic import BaseModel
from semantic_kernel.agents import AzureResponsesAgent

from ..config.settings import settings
from ..config.service_bootstrap import get_azure_openai_client_kwargs
from .agent_registry import AGENT_REGISTRY


def create_azure_responses_client() -> Any:
    return AzureResponsesAgent.create_client(**get_azure_openai_client_kwargs())


def create_structured_agent(
    *,
    name: str,
    description: str,
    instructions: str,
    response_model: type[BaseModel],
    schema_name: str,
    strict: bool,
    plugins: list[Any] | None = None,
    function_choice_behavior: Any | None = None,
    store_enabled: bool = False,
) -> AzureResponsesAgent:
    client = create_azure_responses_client()

    agent_kwargs: dict[str, Any] = {
        "ai_model_id": settings.AZURE_OPENAI_CHAT_DEPLOYMENT_NAME,
        "client": client,
        "name": name,
        "description": description,
        "instructions": instructions,
        "store_enabled": store_enabled,
        "text": {
            "format": {
                "type": "json_schema",
                "name": schema_name,
                "schema": response_model.model_json_schema(),
                "strict": strict,
            }
        },
    }

    if plugins is not None:
        agent_kwargs["plugins"] = plugins

    if function_choice_behavior is not None:
        agent_kwargs["function_choice_behavior"] = function_choice_behavior

    return AzureResponsesAgent(**agent_kwargs)


def create_registered_agent(agent_key: str) -> AzureResponsesAgent:
    definition = AGENT_REGISTRY.get(agent_key)
    if definition is None:
        available = ", ".join(sorted(AGENT_REGISTRY))
        raise ValueError(
            f"Unknown agent key '{agent_key}'. Available keys: {available}"
        )

    return create_structured_agent(
        name=definition.name,
        description=definition.description,
        instructions=definition.instructions,
        response_model=definition.response_model,
        schema_name=definition.schema_name,
        strict=definition.strict,
        plugins=definition.plugins,
        function_choice_behavior=definition.function_choice_behavior,
        store_enabled=definition.store_enabled,
    )