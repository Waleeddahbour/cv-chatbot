from fastapi import Request

from ..ai.chat_main_orchestration import MultiAgentCVChat


def get_chat_service(request: Request) -> MultiAgentCVChat:
    return request.app.state.chat_service
