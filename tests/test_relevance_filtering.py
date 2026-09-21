from app.retrieval.reranker import CrossEncoderReranker
from app.retrieval.retriever import Retriever

from tests.retrieval_test_helpers import (
    retrieval_test_components,
)


RERANKER_NAME = (
    "cross-encoder/ms-marco-MiniLM-L-6-v2"
)


def test_unrelated_question_score(
    retrieval_test_components,
):
    """
    Inspect reranker behavior for an unrelated question.
    """

    embedding_service, vector_store = (
        retrieval_test_components
    )

    retriever = Retriever(
        embedding_service=embedding_service,
        vector_store=vector_store,
        similarity_threshold=0.0,
    )

    reranker = CrossEncoderReranker(
        model_name=RERANKER_NAME
    )

    question = "What is the capital of France?"

    retrieved_chunks = retriever.retrieve(
        question,
        top_k=5,
    )

    reranked_chunks = reranker.rerank(
        question,
        retrieved_chunks,
        top_k=5,
    )

    assert reranked_chunks

    best = reranked_chunks[0]

    print("\n" + "=" * 70)
    print("UNRELATED QUESTION RELEVANCE CHECK")
    print("=" * 70)

    print(f"Question: {question}")

    print(
        f"Best chunk: "
        f"{best.chunk.metadata.chunk_id}"
    )

    print(
        f"Reranker score: "
        f"{best.rerank_score:.4f}"
    )

    print(
        f"Text: "
        f"{best.chunk.metadata.text}"
    )

    assert best.rerank_score < 0.0