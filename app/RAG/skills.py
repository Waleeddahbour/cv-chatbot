from azure.search.documents.indexes.models import (
    AzureOpenAIEmbeddingSkill,
    InputFieldMappingEntry,
    OutputFieldMappingEntry,
)

from ..config.settings import settings


def azure_openai_resource_url() -> str:
    return (settings.AZURE_COGNITIVE_ENDPOINT or "").rstrip("/")


def build_embedding_skill() -> AzureOpenAIEmbeddingSkill:
    return AzureOpenAIEmbeddingSkill(
        name="embed-cv-chunks",
        description="Generate embeddings for processed CV JSONL chunks",
        # JSON Lines parsing makes each chunk the root /document.
        context="/document",
        resource_url=azure_openai_resource_url(),
        api_key=settings.AZURE_OPENAI_API_KEY,
        deployment_name=settings.AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME,
        model_name=settings.AZURE_OPENAI_EMBEDDING_MODEL_NAME,
        dimensions=settings.AZURE_INDEX_VECTOR_DIMENSION,
        inputs=[
            InputFieldMappingEntry(
                name="text",
                source="/document/text",
            ),
        ],
        outputs=[
            OutputFieldMappingEntry(name="embedding", target_name="embedding"),
        ],
    )


def build_skills() -> list[AzureOpenAIEmbeddingSkill]:
    return [
        build_embedding_skill(),
    ]
