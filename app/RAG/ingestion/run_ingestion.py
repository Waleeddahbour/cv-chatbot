from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
import logging

from ...config.settings import settings
from .azure_blob import AzureBlobService
from .azure_di import AzureDocumentIntelligence
from .azure_search_ingestion import AzureSearchIngestion
from .cv_chunker import create_cv_chunks
from .di_cleaner import clean_di_markdown
from .processed_writer import write_jsonl

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class IngestionResult:
    filename: str
    chunk_count: int
    local_pdf_path: Path
    raw_output_path: Path
    processed_output_path: Path
    blob_name: str
    indexer_name: str


def ensure_ingestion_resources() -> str:
    blob_service = AzureBlobService(create_container_if_missing=True)
    search_ingestion = AzureSearchIngestion()
    search_ingestion.create_index()
    search_ingestion.create_data_source()
    search_ingestion.create_skills()
    search_ingestion.create_indexer()
    logger.info(
        "Ingestion resources ensured. container=%s indexer=%s",
        blob_service.container_name,
        search_ingestion.indexer_name,
    )
    return blob_service.container_name


def ingest_pdf(pdf_path: str | Path) -> IngestionResult:
    source_path = Path(pdf_path)
    if not source_path.exists():
        raise FileNotFoundError(f"PDF not found: {source_path}")
    if source_path.suffix.lower() != ".pdf":
        raise ValueError("Only PDF files are supported.")

    processed_dir = settings.DATA_DIR / "processed"
    raw_dir = processed_dir / "raw"
    output_path = processed_dir / f"{source_path.stem}_chunks.jsonl"

    di_client = AzureDocumentIntelligence()
    blob_service = AzureBlobService()
    search_ingestion = AzureSearchIngestion()

    # DI runs locally in Python so we can keep the full AnalyzeResult for page
    # metadata and debugging before Search indexes the processed JSONL.
    logger.info("Starting CV ingestion. source=%s", source_path)
    raw_output_path = raw_dir / f"{source_path.stem}-di.json"
    di_result = di_client.analyze_and_save(source_path, raw_output_path)
    cleaned_markdown = clean_di_markdown(di_result)
    chunks = create_cv_chunks(cleaned_markdown, di_result, source_path)

    write_jsonl(chunks, output_path)
    blob_name = f"processed/{output_path.name}"
    upload_result = blob_service.upload_blob(output_path, blob_name)
    search_ingestion.run_indexer()

    logger.info(
        "CV ingestion completed. source=%s chunks=%s output=%s blob=%s indexer=%s",
        source_path,
        len(chunks),
        output_path,
        upload_result["blob_name"],
        search_ingestion.indexer_name,
    )
    return IngestionResult(
        filename=source_path.name,
        chunk_count=len(chunks),
        local_pdf_path=source_path,
        raw_output_path=raw_output_path,
        processed_output_path=output_path,
        blob_name=upload_result["blob_name"],
        indexer_name=search_ingestion.indexer_name,
    )


def run_ingestion(
    pdf_paths: Iterable[str | Path] | None = None,
) -> list[IngestionResult]:
    resolved_paths = (
        list(pdf_paths) if pdf_paths is not None else list(settings.DATA_FILES)
    )
    if not resolved_paths:
        raise FileNotFoundError(f"No PDF files found in {settings.DATA_DIR}")

    ensure_ingestion_resources()
    return [ingest_pdf(pdf_path) for pdf_path in resolved_paths]


if __name__ == "__main__":
    run_ingestion()
