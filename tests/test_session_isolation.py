from pathlib import Path

from app.api import routes
from app.vector_store.faiss_store import (
    ChunkMetadata,
    FAISSVectorStore,
)


def test_session_vector_stores_are_isolated(
    tmp_path,
    monkeypatch,
):
    base_directory = tmp_path / "sessions"

    class TestSessionStorage:
        def __init__(self, session_id: str):
            self.session_id = session_id
            self.session_directory = (
                base_directory / session_id
            )
            self.upload_directory = (
                self.session_directory / "uploads"
            )
            self.vector_store_directory = (
                self.session_directory / "vector_store"
            )

        def create_directories(self):
            self.upload_directory.mkdir(
                parents=True,
                exist_ok=True,
            )
            self.vector_store_directory.mkdir(
                parents=True,
                exist_ok=True,
            )

    monkeypatch.setattr(
        routes,
        "SessionStorage",
        TestSessionStorage,
    )

    session_a_store = FAISSVectorStore(
        dimension=2,
    )

    session_a_store.add(
        embeddings=[
            [1.0, 0.0],
        ],
        metadata=[
            ChunkMetadata(
                chunk_id=0,
                page_number=1,
                text="This belongs to Session A.",
                document_name="document_a.pdf",
            )
        ],
    )

    session_a_store.save(
        base_directory
        / "session-a"
        / "vector_store"
    )

    session_b_store = FAISSVectorStore(
        dimension=2,
    )

    session_b_store.add(
        embeddings=[
            [0.0, 1.0],
        ],
        metadata=[
            ChunkMetadata(
                chunk_id=0,
                page_number=1,
                text="This belongs to Session B.",
                document_name="document_b.pdf",
            )
        ],
    )

    session_b_store.save(
        base_directory
        / "session-b"
        / "vector_store"
    )

    loaded_a = routes.get_session_vector_store(
        session_id="session-a",
    )

    loaded_b = routes.get_session_vector_store(
        session_id="session-b",
    )

    results_a = loaded_a.search(
        query_embedding=[1.0, 0.0],
        top_k=1,
    )

    results_b = loaded_b.search(
        query_embedding=[0.0, 1.0],
        top_k=1,
    )

    assert len(results_a) == 1
    assert len(results_b) == 1

    metadata_a, _ = results_a[0]
    metadata_b, _ = results_b[0]

    assert metadata_a.document_name == "document_a.pdf"
    assert metadata_a.text == "This belongs to Session A."

    assert metadata_b.document_name == "document_b.pdf"
    assert metadata_b.text == "This belongs to Session B."

    assert metadata_a.document_name != metadata_b.document_name
    assert metadata_a.text != metadata_b.text