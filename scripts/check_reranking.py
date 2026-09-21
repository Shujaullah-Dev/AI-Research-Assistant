from app.embeddings.service import EmbeddingService
from app.retrieval.reranker import CrossEncoderReranker
from app.retrieval.retriever import Retriever
from app.vector_store.faiss_store import FAISSVectorStore


MODEL_NAME = "BAAI/bge-small-en-v1.5"
RERANKER_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"


QUESTIONS = [
    "What dataset did the researchers use to train the model?",
    "Which dataset was used for training the model?",
    "What architecture does the model use?",
    "What type of architecture does the model use?",
    "What mechanism does the transformer architecture use?",
    "What framework was used to implement the experiment?",
    "Which framework was used in the experiment?",
    "What classification accuracy did the final model achieve?",
    "How accurate was the final model?",
    "Which model did the researchers compare their proposed model with?",
    "What model was used for comparison?",
]


def main():
    print("=" * 80)
    print("BGE + CROSS-ENCODER RERANKING")
    print("=" * 80)

    print("\nLoading BGE embedding service...")

    embedding_service = EmbeddingService(
        model_name=MODEL_NAME
    )

    print(
        f"Embedding model: {MODEL_NAME}"
    )

    print("\nLoading BGE vector store...")

    vector_store = FAISSVectorStore.load(
        "data/vector_store_bge"
    )

    retriever = Retriever(
        embedding_service=embedding_service,
        vector_store=vector_store,
        similarity_threshold=0.0,
    )

    print("\nLoading reranker...")

    reranker = CrossEncoderReranker(
        model_name=RERANKER_NAME
    )

    print(
        f"Reranker: {RERANKER_NAME}"
    )

    for question in QUESTIONS:
        print("\n" + "=" * 80)
        print(f"QUESTION: {question}")
        print("=" * 80)

        retrieved = retriever.retrieve(
            question,
            top_k=5,
        )

        print("\nFAISS ranking:")

        for rank, chunk in enumerate(
            retrieved,
            start=1,
        ):
            print(
                f"  Rank {rank}: "
                f"chunk {chunk.metadata.chunk_id} "
                f"(FAISS={chunk.score:.4f})"
            )

        reranked = reranker.rerank(
            question,
            retrieved,
            top_k=5,
        )

        print("\nAfter reranking:")

        for rank, item in enumerate(
            reranked,
            start=1,
        ):
            print(
                f"  Rank {rank}: "
                f"chunk {item.chunk.metadata.chunk_id} "
                f"(rerank={item.rerank_score:.4f}, "
                f"FAISS={item.chunk.score:.4f})"
            )


if __name__ == "__main__":
    main()