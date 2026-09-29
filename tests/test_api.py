from fastapi.testclient import TestClient

from app.main import app
from app.api import routes


client = TestClient(app)


def test_root():
    response = client.get("/")

    assert response.status_code == 200
    assert response.json()["version"] == "0.1.0"


def test_health():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_upload_rejects_non_pdf(tmp_path):
    text_file = tmp_path / "document.txt"
    text_file.write_text("Not a PDF.")

    with text_file.open("rb") as file:
        response = client.post(
            "/documents/upload",
            files={
                "file": (
                    "document.txt",
                    file,
                    "text/plain",
                )
            },
            data={"session_id": "test_session_123"},
        )

    assert response.status_code == 400


def test_upload_replaces_existing_document_with_same_filename(
    tmp_path,
    monkeypatch,
):
    session_id = "same_filename_test_session"

    first_document_text = "The first document used the CIFAR-10 dataset."
    replacement_document_text = "The replacement document used the MNIST dataset."

    class FakePage:
        def __init__(self, text):
            self.text = text
            self.page_number = 1
            self.document_name = "paper.pdf"

    def fake_ingest_pdf(file_path):
        content = file_path.read_text(encoding="utf-8")
        return [FakePage(content)]

    class FakeEmbeddingService:
        dimension = 3

        def embed(self, texts):
            embeddings = []

            for text in texts:
                if "CIFAR-10" in text:
                    embeddings.append([1.0, 0.0, 0.0])
                elif "MNIST" in text:
                    embeddings.append([0.0, 1.0, 0.0])
                else:
                    embeddings.append([0.0, 0.0, 1.0])

            return embeddings

    monkeypatch.setattr(
        routes.ingestion_service,
        "ingest_pdf",
        fake_ingest_pdf,
    )

    monkeypatch.setattr(
        routes,
        "get_embedding_service",
        lambda: FakeEmbeddingService(),
    )

    first_response = client.post(
        "/documents/upload",
        files={
            "file": (
                "paper.pdf",
                first_document_text.encode("utf-8"),
                "application/pdf",
            )
        },
        data={"session_id": session_id},
    )

    assert first_response.status_code == 200

    second_response = client.post(
        "/documents/upload",
        files={
            "file": (
                "paper.pdf",
                replacement_document_text.encode("utf-8"),
                "application/pdf",
            )
        },
        data={"session_id": session_id},
    )

    assert second_response.status_code == 200

    vector_store = routes.get_session_vector_store(session_id)

    assert len(vector_store.metadata) == 1
    assert vector_store.metadata[0].document_name == "paper.pdf"
    assert "MNIST" in vector_store.metadata[0].text
    assert "CIFAR-10" not in vector_store.metadata[0].text