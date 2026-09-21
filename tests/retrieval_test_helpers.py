from pathlib import Path

import pytest

from app.embeddings.service import EmbeddingService
from app.vector_store.faiss_store import (
    ChunkMetadata,
    FAISSVectorStore,
)


MODEL_NAME = "BAAI/bge-small-en-v1.5"


TEST_DOCUMENT_NAME = "ci_test_document.pdf"


TEST_CHUNKS = [
    "The researchers trained the model using the CIFAR-10 dataset.",
    "The transformer architecture uses self-attention mechanisms.",
    "The experiment was implemented using PyTorch.",
    "The final model achieved 94 percent classification accuracy.",
    "The researchers compared their proposed model with ResNet.",
]


@pytest.fixture(scope="session")
def retrieval_test_components(tmp_path_factory):
    """
    Build a temporary FAISS index for CI tests.

    The index is created during the test run and is not
    committed to the Git repository.
    """

    temporary_directory = tmp_path_factory.mktemp(
        "retrieval_test_data"
    )

    vector_store_directory = (
        Path(temporary_directory) / "vector_store"
    )

    embedding_service = EmbeddingService(
        model_name=MODEL_NAME
    )

    metadata = [
        ChunkMetadata(
            chunk_id=chunk_id,
            page_number=chunk_id + 1,
            text=text,
            document_name=TEST_DOCUMENT_NAME,
        )
        for chunk_id, text in enumerate(TEST_CHUNKS)
    ]

    embeddings = embedding_service.embed(
        TEST_CHUNKS
    )

    vector_store = FAISSVectorStore(
        dimension=embedding_service.dimension
    )

    vector_store.add(
        embeddings=embeddings,
        metadata=metadata,
    )

    vector_store.save(
        vector_store_directory
    )

    loaded_vector_store = FAISSVectorStore.load(
        vector_store_directory
    )

    return embedding_service, loaded_vector_store