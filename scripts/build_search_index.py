from app.embeddings.service import EmbeddingService
from app.vector_store.faiss_store import (
    ChunkMetadata,
    FAISSVectorStore,
)


def main():
    texts = [
        "The researchers trained the model using the CIFAR-10 dataset.",
        "The transformer architecture uses self-attention mechanisms.",
        "The experiment was implemented using PyTorch.",
        "The final model achieved 94 percent classification accuracy.",
        "The researchers compared the proposed model with ResNet.",
    ]

    embedding_service = EmbeddingService()

    embeddings = embedding_service.embed(texts)

    vector_store = FAISSVectorStore(
        dimension=len(embeddings[0])
    )

    metadata = [
        ChunkMetadata(
            chunk_id=index,
            page_number=index + 1,
            text=text,
            document_name="demo_paper.pdf",
        )
        for index, text in enumerate(texts)
    ]

    vector_store.add(
        embeddings,
        metadata,
    )
    vector_store.save(
        "data/vector_store"
    )

    query = (
        "What dataset was used to train the model?"
    )

    query_embedding = embedding_service.embed(
        [query]
    )[0]

    results = vector_store.search(
        query_embedding,
        top_k=3,
    )

    print("\nQuery:")
    print(query)

    print("\nResults:")

    for metadata, score in results:
        print(
            f"\nScore: {score:.4f}"
        )
        print(
            f"Page: {metadata.page_number}"
        )
        print(
            f"Text: {metadata.text}"
        )


if __name__ == "__main__":
    main()