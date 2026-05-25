import os
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "app" / "data"


def env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer.") from exc


def env_str(name: str, default: str) -> str:
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    return value


@dataclass(frozen=True)
class Settings:
    CAPABILITY_REQUIREMENTS: ClassVar[dict[str, tuple[str, ...]]] = {
        "azure_openai": (
            "AZURE_COGNITIVE_ENDPOINT",
            "AZURE_OPENAI_API_KEY",
            "AZURE_OPENAI_CHAT_DEPLOYMENT_NAME",
            "AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME",
        ),
        "azure_search": (
            "AZURE_SEARCH_ENDPOINT",
            "AZURE_SEARCH_ADMIN_KEY",
            "AZURE_SEARCH_QUERY_KEY",
            "AZURE_SEARCH_INDEX_NAME",
        ),
        "azure_ai_services": (
            "AZURE_AI_SERVICES_ENDPOINT",
            "AZURE_AI_SERVICES_KEY",
        ),
        "azure_blob": (
            "AZURE_BLOB_CONNECTION_STRING",
            "AZURE_BLOB_CONTAINER_NAME",
        ),
        "azure_cosmos": (
            "AZURE_COSMOS_ENDPOINT",
            "AZURE_COSMOS_KEY",
            "AZURE_COSMOS_DATABASE_NAME",
            "AZURE_COSMOS_CONTAINER_NAME",
        ),
        "tavily": ("TAVILY_API_KEY",),
    }

    AZURE_COGNITIVE_ENDPOINT: str | None
    AZURE_OPENAI_API_KEY: str | None
    AZURE_OPENAI_CHAT_DEPLOYMENT_NAME: str | None
    AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME: str | None
    AZURE_OPENAI_EMBEDDING_MODEL_NAME: str
    AZURE_OPENAI_API_VERSION: str | None
    AZURE_AI_SERVICES_ENDPOINT: str | None
    AZURE_AI_SERVICES_KEY: str | None
    AZURE_SEARCH_ENDPOINT: str | None
    AZURE_SEARCH_ADMIN_KEY: str | None
    AZURE_SEARCH_QUERY_KEY: str | None
    AZURE_SEARCH_INDEX_NAME: str | None
    AZURE_INDEX_VECTOR_DIMENSION: int | None
    AZURE_SEARCH_VECTOR_PROFILE_NAME: str
    AZURE_SEARCH_VECTOR_ALGORITHM_NAME: str
    AZURE_SEARCH_SEMANTIC_CONFIG_NAME: str
    AZURE_SEARCH_TOP_K: int | None
    AZURE_BLOB_CONNECTION_STRING: str | None
    AZURE_BLOB_CONTAINER_NAME: str | None
    AZURE_COSMOS_ENDPOINT: str | None
    AZURE_COSMOS_KEY: str | None
    AZURE_COSMOS_DATABASE_NAME: str | None
    AZURE_COSMOS_CONTAINER_NAME: str | None
    DATA_DIR: Path
    DATA_FILES: tuple[Path, ...]
    OCR_ENABLED: bool
    OCR_LANGUAGE: str
    OCR_DPI: int
    OCR_MIN_TEXT_CHARS: int
    MAX_CHAT_HISTORY_MESSAGES: int
    MAX_HISTORY_TOKENS: int
    TAVILY_API_KEY: str | None

    @staticmethod
    def discover_data_files(data_dir: Path) -> tuple[Path, ...]:
        if not data_dir.exists():
            return ()

        return tuple(
            sorted(
                file
                for file in data_dir.iterdir()
                if file.is_file() and file.suffix.lower() == ".pdf"
            )
        )

    def require(self, *capabilities: str) -> None:
        missing_capabilities = [
            capability
            for capability in capabilities
            if capability not in self.CAPABILITY_REQUIREMENTS
        ]
        if missing_capabilities:
            raise ValueError(f"Unknown capabilities: {', '.join(missing_capabilities)}")

        for capability in capabilities:
            self._require_capability(capability)

    @property
    def azure_openai_base_url(self) -> str:
        endpoint = (self.AZURE_COGNITIVE_ENDPOINT or "").rstrip("/")
        return f"{endpoint}/openai/v1/"

    @staticmethod
    def _require_values(values: dict[str, str | None]) -> None:
        missing = [name for name, value in values.items() if not value]
        if missing:
            raise ValueError(f"Missing required settings: {', '.join(missing)}")

    def _require_capability(self, capability: str) -> None:
        required_fields = self.CAPABILITY_REQUIREMENTS[capability]
        self._require_values({field: getattr(self, field) for field in required_fields})

    @classmethod
    def from_env(cls) -> "Settings":
        data_dir = Path(os.getenv("CV_DATA_DIR", DATA_DIR))

        return cls(
            AZURE_COGNITIVE_ENDPOINT=os.getenv("AZURE_COGNITIVE_ENDPOINT"),
            AZURE_OPENAI_API_KEY=os.getenv("AZURE_OPENAI_API_KEY"),
            AZURE_OPENAI_CHAT_DEPLOYMENT_NAME=os.getenv(
                "AZURE_OPENAI_CHAT_DEPLOYMENT_NAME"
            ),
            AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME=os.getenv(
                "AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME"
            ),
            AZURE_OPENAI_EMBEDDING_MODEL_NAME=env_str(
                "AZURE_OPENAI_EMBEDDING_MODEL_NAME",
                "text-embedding-3-large",
            ),
            AZURE_OPENAI_API_VERSION=os.getenv("AZURE_OPENAI_API_VERSION"),
            AZURE_AI_SERVICES_ENDPOINT=os.getenv("AZURE_AI_SERVICES_ENDPOINT"),
            AZURE_AI_SERVICES_KEY=os.getenv("AZURE_AI_SERVICES_KEY"),
            AZURE_SEARCH_ENDPOINT=os.getenv("AZURE_SEARCH_ENDPOINT"),
            AZURE_SEARCH_ADMIN_KEY=os.getenv("AZURE_SEARCH_ADMIN_KEY"),
            AZURE_SEARCH_QUERY_KEY=os.getenv("AZURE_SEARCH_QUERY_KEY"),
            AZURE_SEARCH_INDEX_NAME=os.getenv("AZURE_SEARCH_INDEX_NAME"),
            AZURE_INDEX_VECTOR_DIMENSION=env_int("AZURE_INDEX_VECTOR_DIMENSION", 3072),
            AZURE_SEARCH_VECTOR_PROFILE_NAME=env_str(
                "AZURE_SEARCH_VECTOR_PROFILE_NAME", "cv-vector-profile"
            ),
            AZURE_SEARCH_VECTOR_ALGORITHM_NAME=env_str(
                "AZURE_SEARCH_VECTOR_ALGORITHM_NAME", "cv-hnsw"
            ),
            AZURE_SEARCH_SEMANTIC_CONFIG_NAME=env_str(
                "AZURE_SEARCH_SEMANTIC_CONFIG_NAME", "cv-semantic-config"
            ),
            AZURE_SEARCH_TOP_K=env_int("AZURE_SEARCH_TOP_K", 3),
            AZURE_BLOB_CONNECTION_STRING=os.getenv("AZURE_BLOB_CONNECTION_STRING"),
            AZURE_BLOB_CONTAINER_NAME=os.getenv("AZURE_BLOB_CONTAINER_NAME"),
            AZURE_COSMOS_ENDPOINT=os.getenv("AZURE_COSMOS_ENDPOINT"),
            AZURE_COSMOS_KEY=os.getenv("AZURE_COSMOS_KEY"),
            AZURE_COSMOS_DATABASE_NAME=os.getenv("AZURE_COSMOS_DATABASE_NAME"),
            AZURE_COSMOS_CONTAINER_NAME=os.getenv("AZURE_COSMOS_CONTAINER_NAME"),
            TAVILY_API_KEY=os.getenv("TAVILY_API_KEY"),
            DATA_DIR=data_dir,
            DATA_FILES=cls.discover_data_files(data_dir),
            OCR_ENABLED=env_bool("CV_OCR_ENABLED", default=True),
            OCR_LANGUAGE=os.getenv("CV_OCR_LANGUAGE", "eng"),
            OCR_DPI=env_int("CV_OCR_DPI", 250),
            OCR_MIN_TEXT_CHARS=env_int("CV_OCR_MIN_TEXT_CHARS", 25),
            MAX_CHAT_HISTORY_MESSAGES=env_int("MAX_CHAT_HISTORY_MESSAGES", 10),
            MAX_HISTORY_TOKENS=env_int("MAX_HISTORY_TOKENS", 120_000),
        )


settings = Settings.from_env()
