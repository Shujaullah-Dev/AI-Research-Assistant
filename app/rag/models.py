from dataclasses import dataclass

from app.retrieval.retriever import RetrievedChunk


@dataclass
class Citation:
    """A source reference for an answer."""

    source_id: int
    document_name: str
    page_number: int
    chunk_id: int
    score: float


@dataclass
class RAGResponse:
    """Final response returned by the RAG pipeline."""

    answer: str
    citations: list[Citation]

    @classmethod
    def from_retrieved_chunks(
        cls,
        answer: str,
        chunks: list[RetrievedChunk],
    ) -> "RAGResponse":
        citations = [
            Citation(
                source_id=index,
                document_name=chunk.metadata.document_name,
                page_number=chunk.metadata.page_number,
                chunk_id=chunk.metadata.chunk_id,
                score=chunk.score,
            )
            for index, chunk in enumerate(
                chunks,
                start=1,
            )
        ]

        return cls(
            answer=answer,
            citations=citations,
        )