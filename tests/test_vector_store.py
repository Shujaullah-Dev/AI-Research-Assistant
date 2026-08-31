from app.vector_store.faiss_store import (
    ChunkMetadata,
    FAISSVectorStore,
)


def test_add_and_search():
    store = FAISSVectorStore(dimension=3)

    embeddings = [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ]

    metadata = [
        ChunkMetadata(
            chunk_id=0,
            page_number=1,
            text="Machine learning research.",
            document_name="paper.pdf",
        ),
        ChunkMetadata(
            chunk_id=1,
            page_number=2,
            text="Deep learning models.",
            document_name="paper.pdf",
        ),
        ChunkMetadata(
            chunk_id=2,
            page_number=3,
            text="Computer vision experiments.",
            document_name="paper.pdf",
        ),
    ]

    store.add(
        embeddings,
        metadata,
    )

    results = store.search(
        [1.0, 0.0, 0.0],
        top_k=2,
    )

    assert len(results) == 2
    assert results[0][0].chunk_id == 0

def test_search_returns_metadata():
    store = FAISSVectorStore(dimension=2)

    store.add(
        [
            [1.0, 0.0],
        ],
        [
            ChunkMetadata(
                chunk_id=10,
                page_number=5,
                text="Important research result.",
                document_name="research.pdf",
            )
        ],
    )

    results = store.search(
        [1.0, 0.0],
        top_k=1,
    )

    metadata, score = results[0]

    assert metadata.chunk_id == 10
    assert metadata.page_number == 5
    assert metadata.text == "Important research result."
    assert metadata.document_name == "research.pdf"
    assert score > 0.99

import pytest


def test_dimension_mismatch_raises_error():
    store = FAISSVectorStore(dimension=3)

    with pytest.raises(ValueError):
        store.add(
            [[1.0, 0.0]],
            [
                ChunkMetadata(
                    chunk_id=0,
                    page_number=1,
                    text="test",
                    document_name="test.pdf",
                )
            ],
        )

def test_store_can_be_saved_and_loaded(tmp_path):
    store = FAISSVectorStore(dimension=2)

    store.add(
        [[1.0, 0.0]],
        [
            ChunkMetadata(
                chunk_id=1,
                page_number=4,
                text="Saved chunk.",
                document_name="paper.pdf",
            )
        ],
    )

    store.save(tmp_path)

    loaded_store = FAISSVectorStore.load(tmp_path)

    results = loaded_store.search(
        [1.0, 0.0],
        top_k=1,
    )

    assert len(results) == 1
    assert results[0][0].text == "Saved chunk."