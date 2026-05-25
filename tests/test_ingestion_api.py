import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.RAG.ingestion.run_ingestion import IngestionResult
from app.api.routers.ingestion import router as ingestion_router


def build_client() -> TestClient:
    app = FastAPI()
    app.include_router(ingestion_router)
    return TestClient(app)


class IngestionApiTests(unittest.TestCase):
    def test_upload_cv_returns_202_on_success(self) -> None:
        client = build_client()
        result = IngestionResult(
            filename="resume.pdf",
            chunk_count=4,
            local_pdf_path=Path("resume.pdf"),
            raw_output_path=Path("processed/raw/resume-di.json"),
            processed_output_path=Path("processed/resume_chunks.jsonl"),
            blob_name="processed/resume_chunks.jsonl",
            indexer_name="cv-indexer",
        )

        with (
            patch(
                "app.api.routers.ingestion.save_upload",
                new=AsyncMock(return_value=Path("resume.pdf")),
            ),
            patch(
                "app.api.routers.ingestion.asyncio.to_thread",
                new=AsyncMock(return_value=result),
            ),
        ):
            response = client.post(
                "/ingestion/cv",
                files={"file": ("resume.pdf", b"pdf-bytes", "application/pdf")},
            )

        self.assertEqual(response.status_code, 202)
        self.assertEqual(response.json()["filename"], "resume.pdf")
        self.assertEqual(response.json()["chunk_count"], 4)

    def test_upload_cv_rejects_non_pdf(self) -> None:
        client = build_client()

        with patch(
            "app.api.routers.ingestion.save_upload",
            new=AsyncMock(side_effect=ValueError("Only PDF files are supported.")),
        ):
            response = client.post(
                "/ingestion/cv",
                files={"file": ("resume.txt", b"text", "text/plain")},
            )

        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json(), {"detail": "Only PDF files are supported."})

    def test_upload_cv_maps_missing_file_error(self) -> None:
        client = build_client()

        with (
            patch(
                "app.api.routers.ingestion.save_upload",
                new=AsyncMock(return_value=Path("resume.pdf")),
            ),
            patch(
                "app.api.routers.ingestion.asyncio.to_thread",
                new=AsyncMock(side_effect=FileNotFoundError("PDF not found")),
            ),
        ):
            response = client.post(
                "/ingestion/cv",
                files={"file": ("resume.pdf", b"pdf-bytes", "application/pdf")},
            )

        self.assertEqual(response.status_code, 404)
        self.assertEqual(response.json(), {"detail": "PDF not found"})

    def test_upload_cv_maps_internal_failure_to_500(self) -> None:
        client = build_client()

        with (
            patch(
                "app.api.routers.ingestion.save_upload",
                new=AsyncMock(return_value=Path("resume.pdf")),
            ),
            patch(
                "app.api.routers.ingestion.asyncio.to_thread",
                new=AsyncMock(side_effect=RuntimeError("boom")),
            ),
        ):
            response = client.post(
                "/ingestion/cv",
                files={"file": ("resume.pdf", b"pdf-bytes", "application/pdf")},
            )

        self.assertEqual(response.status_code, 500)
        self.assertEqual(response.json(), {"detail": "CV ingestion failed"})
