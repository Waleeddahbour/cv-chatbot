from fastapi import APIRouter

from ..schemas import HealthResponse

router = APIRouter(tags=["system"])


@router.get("/")
async def root() -> dict[str, str]:
    return {"message": "CV Bot API is running."}


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(status="ok")
