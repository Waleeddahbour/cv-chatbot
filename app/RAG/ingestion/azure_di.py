import json
from pathlib import Path

from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.ai.documentintelligence.models import AnalyzeResult, DocumentContentFormat
from azure.core.credentials import AzureKeyCredential

from ...config.settings import settings


class AzureDocumentIntelligence:
    def __init__(self):
        settings.require("azure_ai_services")
        self.client = DocumentIntelligenceClient(
            endpoint=settings.AZURE_AI_SERVICES_ENDPOINT,
            credential=AzureKeyCredential(settings.AZURE_AI_SERVICES_KEY),
        )

    def analyze_document(self, pdf_path: str | Path) -> dict:
        path = Path(pdf_path)
        if not path.exists():
            raise FileNotFoundError(f"PDF not found: {path}")

        with path.open("rb") as document:
            poller = self.client.begin_analyze_document(
                "prebuilt-layout",
                body=document,
                output_content_format=DocumentContentFormat.MARKDOWN,
            )

        return self._to_dict(poller.result())

    def analyze_and_save(self, pdf_path: str | Path, output_path: str | Path) -> dict:
        result = self.analyze_document(pdf_path)
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(result, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return result

    @staticmethod
    def _to_dict(result: AnalyzeResult) -> dict:
        if hasattr(result, "as_dict"):
            return result.as_dict()
        return dict(result)


if __name__ == "__main__":
    if not settings.DATA_FILES:
        raise FileNotFoundError(f"No PDF files found in {settings.DATA_DIR}")

    client = AzureDocumentIntelligence()
    pdf_path = settings.DATA_FILES[0]
    output_path = settings.DATA_DIR / "processed" / "raw" / f"{pdf_path.stem}-di.json"
    client.analyze_and_save(pdf_path, output_path)
    print(f"Saved raw DI result to {output_path}")
