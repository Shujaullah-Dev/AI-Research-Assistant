from pathlib import Path

from app.embeddings.service import EmbeddingService
from app.vector_store.faiss_store import (
    ChunkMetadata,
    FAISSVectorStore,
)


MODEL_NAME = "BAAI/bge-small-en-v1.5"

SOURCE_METADATA = Path(
    "data/vector_store/metadata.json"
)

OUTPUT_DIRECTORY = Path(
    "data/vector_store_bge"
)


def main():
    print("=" * 70)
    print("BUILDING BGE EMBEDDING INDEX")
    print("=" * 70)

    print(f"\nEmbedding model: {MODEL_NAME}")

    print("\nLoading embedding service...")
    embedding_service = EmbeddingService(
        model_name=MODEL_NAME
    )

    print(
        f"Embedding dimension: "
        f"{embedding_service.dimension}"
    )

    print(
        f"\nReading metadata from: "
        f"{SOURCE_METADATA}"
    )

    import json

    metadata_data = json.loads(
        SOURCE_METADATA.read_text(
            encoding="utf-8"
        )
    )

    metadata = [
        ChunkMetadata(**item)
        for item in metadata_data
    ]

    texts = [
        item.text
        for item in metadata
    ]

    print(
        f"Number of chunks: {len(texts)}"
    )

    print("\nGenerating embeddings...")

    embeddings = embedding_service.embed(
        texts
    )

    print("Embeddings generated.")

    vector_store = FAISSVectorStore(
        dimension=embedding_service.dimension
    )

    vector_store.add(
        embeddings=embeddings,
        metadata=metadata,
    )

    print(
        f"\nSaving index to: "
        f"{OUTPUT_DIRECTORY}"
    )

    vector_store.save(
        OUTPUT_DIRECTORY
    )

    print("\nIndex created successfully.")
    print(
        f"Vectors stored: "
        f"{vector_store.index.ntotal}"
    )


if __name__ == "__main__":
    main()