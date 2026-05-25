from fastapi import APIRouter, Depends, HTTPException

from ..dependencies import get_chat_service
from ..schemas import ChatRequest, ChatResponse
from ...ai.chat_main_orchestration import MultiAgentCVChat

router = APIRouter(tags=["chat"])


@router.post("/chat", response_model=ChatResponse)
async def chat(
    payload: ChatRequest,
    chat_service: MultiAgentCVChat = Depends(get_chat_service),
) -> ChatResponse:
    try:
        result = await chat_service.chat(
            user_message=payload.message,
            session_id=payload.session_id,
        )
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except Exception as error:
        raise HTTPException(status_code=500, detail="Chat request failed") from error

    return ChatResponse(
        session_id=result.session_id,
        intent=result.intent.value if result.intent else None,
        response=result.response,
        routing_reasoning=result.routing_reasoning,
        agent_turns=[turn.model_dump() for turn in result.agent_turns],
    )
