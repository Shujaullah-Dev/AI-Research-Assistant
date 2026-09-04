from app.embeddings.service import EmbeddingService
from app.vector_store.faiss_store import FAISSVectorStore


def main():
    print("Loading embedding service...")
    embedding_service = EmbeddingService()

    print("Loading vector store...")
    vector_store = FAISSVectorStore.load(
        "data/vector_store"
    )

    questions = [
        "What dataset did the researchers use to train the model?",
        "What architecture does the model use?",
        "What framework was used to implement the experiment?",
        "What classification accuracy did the final model achieve?",
        "Which model did the researchers compare their proposed model with?",
        "Which dataset was used for training the model?",
    ]

    for question in questions:
        print("\n" + "=" * 70)
        print(f"QUESTION: {question}")
        print("=" * 70)

        query_embedding = embedding_service.embed(
            [question]
        )[0]

        results = vector_store.search(
            query_embedding,
            top_k=5,
        )

        for rank, (metadata, score) in enumerate(
            results,
            start=1,
        ):
            print(f"\nRank {rank}")
            print(f"Chunk ID: {metadata.chunk_id}")
            print(f"Page: {metadata.page_number}")
            print(f"Similarity: {score:.4f}")
            print(f"Text: {metadata.text}")


if __name__ == "__main__":
    main()