from dataclasses import dataclass

from sentence_transformers import CrossEncoder

from app.retrieval.retriever import RetrievedChunk


@dataclass
class RerankedChunk:
    chunk: RetrievedChunk
    rerank_score: float


class CrossEncoderReranker:
    """Rerank retrieved chunks using a cross-encoder model."""

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        model=None,
    ) -> None:
        self.model_name = model_name
        self.model = model or CrossEncoder(model_name)

    def rerank(
        self,
        query: str,
        chunks: list[RetrievedChunk],
        top_k: int | None = None,
    ) -> list[RerankedChunk]:
        if not query.strip():
            raise ValueError("query must not be empty")

        if not chunks:
            return []

        pairs = [
            (query, chunk.metadata.text)
            for chunk in chunks
        ]

        scores = self.model.predict(pairs)

        reranked = [
            RerankedChunk(
                chunk=chunk,
                rerank_score=float(score),
            )
            for chunk, score in zip(chunks, scores)
        ]

        reranked.sort(
            key=lambda item: item.rerank_score,
            reverse=True,
        )

        if top_k is not None:
            if top_k <= 0:
                raise ValueError(
                    "top_k must be greater than 0"
                )

            reranked = reranked[:top_k]

        return reranked

    def is_relevant(
        self,
        reranked_chunk: RerankedChunk,
        threshold: float,
    ) -> bool:
        """Return whether a reranked chunk passes the relevance threshold."""

        return reranked_chunk.rerank_score >= threshold