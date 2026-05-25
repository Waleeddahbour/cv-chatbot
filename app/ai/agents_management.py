from typing import TypeVar
import logging

import tiktoken
from pydantic import BaseModel
from semantic_kernel.contents.chat_message_content import ChatMessageContent
from semantic_kernel.contents.utils.author_role import AuthorRole

AgentResponseModel = TypeVar("AgentResponseModel", bound=BaseModel)
logger = logging.getLogger(__name__)


def num_tokens_from_messages(messages: list[dict], model: str = "5.4") -> int:
    """Return the number of tokens used by a list of chat messages."""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        logger.warning("Model %s not found. Using o200k_base encoding.", model)
        encoding = tiktoken.get_encoding("o200k_base")

    if model in {
        "gpt-4o",
        "gpt-4o-mini",
        "gpt-5",
        "gpt-4.1",
        "o1",
        "o1-mini",
        "o3",
        "o3-mini",
        "o4-mini",
    }:
        tokens_per_message = 3
        tokens_per_name = 1
    elif any(
        model.startswith(prefix)
        for prefix in [
            "gpt-4o-",
            "gpt-5",
            "gpt-4.1-",
            "o1-",
            "o3-",
            "o4-mini-",
        ]
    ):
        tokens_per_message = 3
        tokens_per_name = 1
    else:
        raise NotImplementedError(
            f"num_tokens_from_messages() is not implemented for model {model}."
        )

    num_tokens = 0
    for message in messages:
        num_tokens += tokens_per_message
        for key, value in message.items():
            num_tokens += len(encoding.encode(value))
            if key == "name":
                num_tokens += tokens_per_name
    num_tokens += 3
    return num_tokens


def build_agent_messages(
    user_message: str,
    model: str,
    max_history_tokens: int,
    chat_history: list[dict] | None = None,
) -> list[ChatMessageContent]:
    token_messages = []
    token_messages.extend(chat_history or [])
    token_messages.append({"role": "user", "content": user_message})

    # Drop the oldest conversation turns first; keep the latest user message in
    # the token budget.
    messages_tokens = num_tokens_from_messages(token_messages, model=model)
    while len(token_messages) > 1 and messages_tokens > max_history_tokens:
        del token_messages[0]
        messages_tokens = num_tokens_from_messages(token_messages, model=model)

    role_map = {
        "user": AuthorRole.USER,
        "assistant": AuthorRole.ASSISTANT,
    }
    return [
        ChatMessageContent(role=role_map[message["role"]], content=message["content"])
        for message in token_messages
        if message["role"] in role_map
    ]


async def run_agent(
    agent,
    messages: list[ChatMessageContent],
    response_model: type[AgentResponseModel],
) -> AgentResponseModel | str:
    response = await agent.get_response(messages=messages)
    content = response.message.content or ""
    try:
        return response_model.model_validate_json(content)
    except ValueError:
        return content
