from fastapi.testclient import TestClient

from app.main import app


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
        )

    assert response.status_code == 400