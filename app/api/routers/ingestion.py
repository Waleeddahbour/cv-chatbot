import asyncio
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile, status

from ..schemas import CVIngestionResponse
from ...config.settings import settings
from ...RAG.ingestion.run_ingestion import ingest_pdf

router = APIRouter(prefix="/ingestion", tags=["ingestion"])


@router.post(
    "/cv", response_model=CVIngestionResponse, status_code=status.HTTP_202_ACCEPTED
)
async def upload_cv(file: UploadFile = File(...)) -> CVIngestionResponse:
    try:
        local_pdf_path = await save_upload(file)
        result = await asyncio.to_thread(ingest_pdf, local_pdf_path)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except FileNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except Exception as error:
        raise HTTPException(status_code=500, detail="CV ingestion failed") from error

    return CVIngestionResponse(
        status="accepted",
        filename=result.filename,
        chunk_count=result.chunk_count,
        local_pdf_path=str(result.local_pdf_path),
        raw_output_path=str(result.raw_output_path),
        processed_output_path=str(result.processed_output_path),
        blob_name=result.blob_name,
        indexer_name=result.indexer_name,
    )


async def save_upload(file: UploadFile) -> Path:
    filename = normalize_filename(file.filename)
    local_path = settings.DATA_DIR / filename
    local_path.parent.mkdir(parents=True, exist_ok=True)
    content = await file.read()
    local_path.write_bytes(content)
    await file.close()
    return local_path


def normalize_filename(filename: str | None) -> str:
    candidate = Path(filename or "uploaded-cv.pdf").name
    if not candidate.lower().endswith(".pdf"):
        raise ValueError("Only PDF files are supported.")
    return candidate
