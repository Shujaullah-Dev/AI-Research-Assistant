from app.embeddings.service import EmbeddingService
from app.retrieval.reranker import CrossEncoderReranker
from app.retrieval.retriever import Retriever
from app.vector_store.faiss_store import FAISSVectorStore


MODEL_NAME = "BAAI/bge-small-en-v1.5"
RERANKER_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
INDEX_DIRECTORY = "data/vector_store_bge"


def test_unrelated_question_score():
    """Inspect the reranker score for an unrelated question."""

    embedding_service = EmbeddingService(
        model_name=MODEL_NAME
    )

    vector_store = FAISSVectorStore.load(
        INDEX_DIRECTORY
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
    print(f"Best chunk: {best.chunk.metadata.chunk_id}")
    print(f"Reranker score: {best.rerank_score:.4f}")
    print(f"Text: {best.chunk.metadata.text}")

    assert best.rerank_score < 0.0