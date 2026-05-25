from typing import Any
from typing import Literal
import logging
from azure.search.documents.models import VectorizedQuery

from ...config.settings import settings
from ...config.service_bootstrap import (
    create_openai_embeddings_client,
    create_search_query_client,
)

logger = logging.getLogger(__name__)


class AzureSearchRetrieval:
    def __init__(self):
        self.search_client, self.credential = create_search_query_client()
        self.get_embedding_client = create_openai_embeddings_client()
        self.SEARCH_SELECTED_FIELDS = [
            "chunk_id",
            "parent_id",
            "text",
            "section",
            "source_file",
            "chunk_index",
            "page_number",
        ]
        self.TOP_K = settings.AZURE_SEARCH_TOP_K
        self.KNN = 10

    async def close(self) -> None:
        await self.search_client.close()
        await self.get_embedding_client.close()

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, traceback) -> None:
        await self.close()

    @staticmethod
    def build_section_filter(section_filter: list[str] | None) -> str | None:
        if not section_filter:
            return None

        # Azure Search search.in expects all allowed values in one delimited
        # string, for example: search.in(section, 'Skills,Experience', ',').
        escaped_sections = [
            section.replace("'", "''")
            for section in dict.fromkeys(section_filter)
            if section
        ]
        if not escaped_sections:
            return None

        return f"search.in(section, '{','.join(escaped_sections)}', ',')"

    @staticmethod
    def normalize_search_results(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
        normalized = []
        for result in results:
            normalized.append(
                {
                    "chunk_id": result.get("chunk_id"),
                    "parent_id": result.get("parent_id"),
                    "text": result.get("text", ""),
                    "section": result.get("section", ""),
                    "source_file": result.get("source_file", ""),
                    "chunk_index": result.get("chunk_index"),
                    "page_number": result.get("page_number"),
                    "score": result.get("@search.score", 0),
                    "reranked_score": result.get("@search.reranker_score", 0),
                }
            )
        return normalized

    async def create_text_embedding(self, text: str) -> list[float]:
        embeddings = await self.get_embedding_client.embeddings.create(
            model=settings.AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME,
            input=text,
        )
        return embeddings.data[0].embedding

    async def _collect_results(self, results) -> list[dict[str, Any]]:
        return [result async for result in results]

    def _resolve_top_k(self, top_k: int | None) -> int:
        return self.TOP_K if top_k is None else top_k

    def _build_search_kwargs(
        self,
        *,
        filter: list[str] | None,
        top_k: int,
        include_total_count: bool,
    ) -> dict[str, Any]:
        return {
            "filter": self.build_section_filter(filter),
            "select": self.SEARCH_SELECTED_FIELDS,
            "top": top_k,
            "include_total_count": include_total_count,
        }

    async def _build_vector_query(self, query: str) -> VectorizedQuery:
        embedding = await self.create_text_embedding(query)
        return VectorizedQuery(
            vector=embedding,
            k_nearest_neighbors=self.KNN,
            fields="embedding",
            kind="vector",
        )

    async def _execute_search(
        self,
        *,
        query: str,
        mode: Literal["keyword", "vector", "hybrid", "semantic_hybrid"],
        filter: list[str] | None,
        top_k: int | None,
    ) -> list[dict[str, Any]]:
        resolved_top_k = self._resolve_top_k(top_k)
        logger.debug(
            "Executing Azure Search. mode=%s top_k=%s filter=%s",
            mode,
            resolved_top_k,
            filter,
        )
        use_vector = mode in {"vector", "hybrid", "semantic_hybrid"}
        use_semantic = mode == "semantic_hybrid"
        include_total_count = mode != "keyword"
        search_text = "" if mode == "vector" else query

        search_kwargs = self._build_search_kwargs(
            filter=filter,
            top_k=resolved_top_k,
            include_total_count=include_total_count,
        )

        if use_vector:
            vector_source_query = query
            vector_query = await self._build_vector_query(vector_source_query)
            search_kwargs["vector_queries"] = [vector_query]

        if use_semantic:
            search_kwargs["query_type"] = "semantic"
            search_kwargs["semantic_configuration_name"] = (
                settings.AZURE_SEARCH_SEMANTIC_CONFIG_NAME
            )

        results = await self.search_client.search(
            search_text=search_text,
            **search_kwargs,
        )
        return self.normalize_search_results(await self._collect_results(results))

    async def keyword_search(
        self,
        query: str,
        top_k: int = None,
        filter: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        return await self._execute_search(
            query=query,
            mode="keyword",
            filter=filter,
            top_k=top_k,
        )

    async def vector_search(
        self,
        query: str,
        top_k: int = None,
        filter: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        return await self._execute_search(
            query=query,
            mode="vector",
            filter=filter,
            top_k=top_k,
        )

    async def hybrid_search(
        self,
        query: str,
        top_k: int = None,
        filter: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        return await self._execute_search(
            query=query,
            mode="hybrid",
            filter=filter,
            top_k=top_k,
        )

    async def semantic_hybrid_search(
        self,
        query: str,
        top_k: int = None,
        filter: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        return await self._execute_search(
            query=query,
            mode="semantic_hybrid",
            filter=filter,
            top_k=top_k,
        )

    async def retrieve_relevant_chunks(
        self,
        query: str,
        top_k: int = None,
        filter: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        top_k = self._resolve_top_k(top_k)
        try:
            results = await self.semantic_hybrid_search(
                query=query,
                top_k=top_k,
                filter=filter,
            )
            logger.info(
                "Semantic hybrid search returned %s results. query=%r",
                len(results),
                query,
            )
        except Exception as error:
            logger.warning(
                "Semantic hybrid search failed. Falling back to hybrid search. query=%r error=%s",
                query,
                error,
            )
            results = await self.hybrid_search(query=query, top_k=top_k, filter=filter)

        return results


async def main() -> None:
    import sys

    query = sys.argv[1] if len(sys.argv) > 1 else "what is waleed's education?"
    print(sys.argv)
    print(f"Query: {query}")
    async with AzureSearchRetrieval() as search_service:
        documents = await search_service.retrieve_relevant_chunks(query=query, top_k=3)

    for document in documents:
        print("-" * 80)
        print(f"chunk_id: {document.get('chunk_id')}")
        print(f"parent_id: {document.get('parent_id')}")
        print(f"source_file: {document.get('source_file')}")
        print(f"score: {document.get('score')}")
        print(f"reranked_score: {document.get('reranked_score')}")
        print(f"chunk_index: {document.get('chunk_index')}")
        print(f"page_number: {document.get('page_number')}")
        print(f"section: {document.get('section')}")
        print(("text:\n" + (document.get("text") or "")))


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
