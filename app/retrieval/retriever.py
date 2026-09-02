from dataclasses import dataclass

from app.embeddings.service import EmbeddingService
from app.vector_store.faiss_store import (
    ChunkMetadata,
    FAISSVectorStore,
)


@dataclass
class RetrievedChunk:
    metadata: ChunkMetadata
    score: float


class Retriever:
    """Retrieve relevant document chunks for a user query."""

    def __init__(
        self,
        embedding_service: EmbeddingService,
        vector_store: FAISSVectorStore,
        similarity_threshold: float = 0.0,
    ) -> None:
        if not 0.0 <= similarity_threshold <= 1.0:
            raise ValueError(
                "similarity_threshold must be between 0 and 1"
            )

        self.embedding_service = embedding_service
        self.vector_store = vector_store
        self.similarity_threshold = similarity_threshold

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[RetrievedChunk]:
        if not query.strip():
            raise ValueError("query must not be empty")

        if top_k <= 0:
            raise ValueError(
                "top_k must be greater than 0"
            )

        query_embedding = self.embedding_service.embed(
            [query]
        )[0]

        results = self.vector_store.search(
            query_embedding,
            top_k=top_k,
        )

        filtered_results = [
            (metadata, score)
            for metadata, score in results
            if score >= self.similarity_threshold
        ]

        return [
            RetrievedChunk(
                metadata=metadata,
                score=score,
            )
            for metadata, score in filtered_results
        ]