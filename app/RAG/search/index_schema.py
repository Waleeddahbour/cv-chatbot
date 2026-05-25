from azure.search.documents.indexes.models import (
    SearchIndex,
    SimpleField,
    SearchableField,
    SearchField,
    SearchFieldDataType,
    VectorSearch,
    HnswAlgorithmConfiguration,
    HnswParameters,
    VectorSearchProfile,
    VectorSearchAlgorithmMetric,
    SemanticConfiguration,
    SemanticPrioritizedFields,
    SemanticField,
    SemanticSearch,
)
from azure.core.credentials import AzureKeyCredential
from azure.search.documents.indexes import SearchIndexClient
from ...config.settings import settings


def build_index_schema() -> SearchIndex:
    settings.require("azure_search")

    fields = [
        SearchableField(
            name="chunk_id",
            type=SearchFieldDataType.String,
            key=True,
            filterable=True,
            searchable=True,
            analyzer_name="keyword",
        ),
        SimpleField(
            name="parent_id",
            type=SearchFieldDataType.String,
            filterable=True,
            facetable=False,
        ),
        SimpleField(
            name="source_file",
            type=SearchFieldDataType.String,
            filterable=True,
            facetable=True,
        ),
        SimpleField(
            name="section",
            type=SearchFieldDataType.String,
            filterable=True,
            facetable=True,
        ),
        SearchableField(
            name="text",
            type=SearchFieldDataType.String,
            searchable=True,
            filterable=False,
            sortable=False,
            facetable=False,
            retrievable=True,
        ),
        SimpleField(
            name="chunk_index",
            type=SearchFieldDataType.Int32,
            filterable=True,
            sortable=True,
            facetable=False,
        ),
        SimpleField(
            name="page_number",
            type=SearchFieldDataType.Int32,
            filterable=True,
            sortable=True,
            facetable=False,
        ),
        SearchField(
            name="metadata_json",
            type=SearchFieldDataType.String,
            searchable=False,
            filterable=False,
            sortable=False,
            facetable=False,
            retrievable=True,
        ),
        SimpleField(
            name="indexed_at",
            type=SearchFieldDataType.DateTimeOffset,
            filterable=True,
            sortable=True,
            facetable=False,
        ),
        SearchField(
            name="embedding",
            type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
            searchable=True,
            retrievable=False,
            vector_search_dimensions=settings.AZURE_INDEX_VECTOR_DIMENSION,
            vector_search_profile_name=settings.AZURE_SEARCH_VECTOR_PROFILE_NAME,
        ),
    ]

    vector_search = VectorSearch(
        algorithms=[
            HnswAlgorithmConfiguration(
                name=settings.AZURE_SEARCH_VECTOR_ALGORITHM_NAME,
                parameters=HnswParameters(
                    m=4,
                    ef_construction=400,
                    ef_search=500,
                    metric=VectorSearchAlgorithmMetric.COSINE,
                ),
            )
        ],
        profiles=[
            VectorSearchProfile(
                name=settings.AZURE_SEARCH_VECTOR_PROFILE_NAME,
                algorithm_configuration_name=settings.AZURE_SEARCH_VECTOR_ALGORITHM_NAME,
            )
        ],
    )

    semantic_search = SemanticSearch(
        configurations=[
            SemanticConfiguration(
                name=settings.AZURE_SEARCH_SEMANTIC_CONFIG_NAME,
                prioritized_fields=SemanticPrioritizedFields(
                    content_fields=[
                        SemanticField(field_name="text"),
                    ],
                    keywords_fields=[
                        SemanticField(field_name="section"),
                        SemanticField(field_name="source_file"),
                    ],
                ),
            )
        ]
    )

    return SearchIndex(
        name=settings.AZURE_SEARCH_INDEX_NAME,
        fields=fields,
        vector_search=vector_search,
        semantic_search=semantic_search,
    )


# Create or update the index schema in Azure Search


def create_or_update_index_schema() -> SearchIndex:
    settings.require("azure_search")

    credential = AzureKeyCredential(settings.AZURE_SEARCH_ADMIN_KEY)
    client = SearchIndexClient(
        endpoint=settings.AZURE_SEARCH_ENDPOINT, credential=credential
    )

    index_schema = build_index_schema()
    return client.create_or_update_index(index_schema)


if __name__ == "__main__":
    search_index = create_or_update_index_schema()
    print(f"Index schema created or updated successfully. Index {search_index}")
