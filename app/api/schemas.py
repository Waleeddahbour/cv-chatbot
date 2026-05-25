from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1)
    session_id: str | None = None


class ChatResponse(BaseModel):
    session_id: str
    intent: str | None
    response: str
    routing_reasoning: str | None = None
    agent_turns: list[dict] = Field(default_factory=list)


class HealthResponse(BaseModel):
    status: str


class CVIngestionResponse(BaseModel):
    status: str
    filename: str
    chunk_count: int
    local_pdf_path: str
    raw_output_path: str
    processed_output_path: str
    blob_name: str
    indexer_name: str
