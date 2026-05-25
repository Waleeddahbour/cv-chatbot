from azure.core.credentials import AzureKeyCredential
from azure.cosmos.aio import CosmosClient
from azure.search.documents.aio import SearchClient
from azure.search.documents.indexes import SearchIndexerClient
from azure.storage.blob import BlobServiceClient
from openai import AsyncOpenAI

from .settings import settings


def get_azure_openai_client_kwargs() -> dict[str, str]:
    settings.require("azure_openai")
    return {
        "api_key": settings.AZURE_OPENAI_API_KEY,
        "api_version": settings.AZURE_OPENAI_API_VERSION,
        "deployment_name": settings.AZURE_OPENAI_CHAT_DEPLOYMENT_NAME,
        "endpoint": settings.AZURE_COGNITIVE_ENDPOINT,
    }


def create_openai_embeddings_client() -> AsyncOpenAI:
    settings.require("azure_openai")
    return AsyncOpenAI(
        api_key=settings.AZURE_OPENAI_API_KEY,
        base_url=settings.azure_openai_base_url,
    )


def create_search_query_client() -> tuple[SearchClient, AzureKeyCredential]:
    settings.require("azure_search")
    credential = AzureKeyCredential(settings.AZURE_SEARCH_QUERY_KEY)
    client = SearchClient(
        endpoint=settings.AZURE_SEARCH_ENDPOINT,
        index_name=settings.AZURE_SEARCH_INDEX_NAME,
        credential=credential,
    )
    return client, credential


def create_search_indexer_client() -> tuple[SearchIndexerClient, AzureKeyCredential]:
    settings.require("azure_search")
    credential = AzureKeyCredential(settings.AZURE_SEARCH_ADMIN_KEY)
    client = SearchIndexerClient(
        endpoint=settings.AZURE_SEARCH_ENDPOINT,
        credential=credential,
    )
    return client, credential


def create_blob_service_client() -> BlobServiceClient:
    settings.require("azure_blob")
    return BlobServiceClient.from_connection_string(
        settings.AZURE_BLOB_CONNECTION_STRING
    )


def get_blob_container_name() -> str:
    settings.require("azure_blob")
    return settings.AZURE_BLOB_CONTAINER_NAME


def create_cosmos_client() -> CosmosClient:
    settings.require("azure_cosmos")
    return CosmosClient(
        url=settings.AZURE_COSMOS_ENDPOINT,
        credential=settings.AZURE_COSMOS_KEY,
    )
