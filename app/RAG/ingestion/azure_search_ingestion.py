from azure.search.documents.indexes.models import (
    FieldMapping,
    IndexingParameters,
    SearchIndexerDataContainer,
    SearchIndexerDataSourceConnection,
    SearchIndexer,
    SearchIndexerSkillset,
)

from ...config.settings import settings
from ...config.service_bootstrap import create_search_indexer_client
from ..search.index_schema import create_or_update_index_schema
from ..skills import build_skills
from .azure_blob import AzureBlobService


class AzureSearchIngestion:
    def __init__(self):
        settings.require("azure_blob", "azure_openai")

        self.indexer_client, self.credential = create_search_indexer_client()
        self.blob_service = AzureBlobService()
        self.data_source_name = self.blob_service.container_name
        self.skillset_name = "cv-skillset"
        self.indexer_name = f"{settings.AZURE_SEARCH_INDEX_NAME}-indexer"

    def create_index(self):
        return create_or_update_index_schema()

    def create_data_source(self) -> SearchIndexerDataSourceConnection:
        container = SearchIndexerDataContainer(
            name=self.blob_service.container_name,
            # Only pull processed JSONL files; raw PDFs are handled by Python using DI.
            query="processed",
        )
        data_source_connection = SearchIndexerDataSourceConnection(
            name=self.data_source_name,
            type="azureblob",
            connection_string=settings.AZURE_BLOB_CONNECTION_STRING,
            container=container,
        )
        return self.indexer_client.create_or_update_data_source_connection(
            data_source_connection
        )

    def create_skills(self) -> SearchIndexerSkillset:
        skillset = SearchIndexerSkillset(
            name=self.skillset_name,
            description="Generate embeddings for processed CV JSONL chunks",
            skills=build_skills(),
        )
        return self.indexer_client.create_or_update_skillset(skillset)

    def create_indexer(self) -> SearchIndexer:
        indexer = SearchIndexer(
            name=self.indexer_name,
            description="Pull processed CV JSONL chunks from Blob and embed them",
            data_source_name=self.data_source_name,
            skillset_name=self.skillset_name,
            target_index_name=settings.AZURE_SEARCH_INDEX_NAME,
            field_mappings=[
                FieldMapping(
                    source_field_name="chunk_id", target_field_name="chunk_id"
                ),
                FieldMapping(
                    source_field_name="parent_id", target_field_name="parent_id"
                ),
                FieldMapping(
                    source_field_name="source_file",
                    target_field_name="source_file",
                ),
                FieldMapping(source_field_name="section", target_field_name="section"),
                FieldMapping(source_field_name="text", target_field_name="text"),
                FieldMapping(
                    source_field_name="chunk_index",
                    target_field_name="chunk_index",
                ),
                FieldMapping(
                    source_field_name="page_number",
                    target_field_name="page_number",
                ),
                FieldMapping(
                    source_field_name="metadata_json",
                    target_field_name="metadata_json",
                ),
                FieldMapping(
                    source_field_name="indexed_at",
                    target_field_name="indexed_at",
                ),
            ],
            output_field_mappings=[
                FieldMapping(
                    # Embedding skill outputs a vector collection under
                    # /document/embedding; map the vector values into the index.
                    source_field_name="/document/embedding/*",
                    target_field_name="embedding",
                ),
            ],
            parameters=IndexingParameters(
                configuration={
                    "dataToExtract": "contentAndMetadata",
                    # Each JSONL line is already a complete chunk document.
                    "parsingMode": "jsonLines",
                    "indexedFileNameExtensions": ".jsonl",
                },
            ),
        )
        return self.indexer_client.create_or_update_indexer(indexer)

    def run_indexer(self) -> None:
        self.indexer_client.run_indexer(self.indexer_name)

    def reset_indexer(self) -> None:
        self.indexer_client.reset_indexer(self.indexer_name)

    def get_indexer_status(self):
        return self.indexer_client.get_indexer_status(self.indexer_name)

    def run_pipeline(self) -> None:
        self.create_index()
        self.create_data_source()
        self.create_skills()
        self.create_indexer()
        self.run_indexer()


if __name__ == "__main__":
    ingestion = AzureSearchIngestion()
    ingestion.run_pipeline()
    print(f"Started indexer: {ingestion.indexer_name}")
