import pytest

from app.storage.session_storage import SessionStorage


def test_session_storage_creates_session_directories(tmp_path):
    storage = SessionStorage(
        session_id="session-123",
        base_directory=str(tmp_path / "sessions"),
    )

    storage.create_directories()

    assert storage.session_directory == (
        tmp_path / "sessions" / "session-123"
    )

    assert storage.upload_directory == (
        tmp_path / "sessions" / "session-123" / "uploads"
    )

    assert storage.vector_store_directory == (
        tmp_path / "sessions" / "session-123" / "vector_store"
    )

    assert storage.upload_directory.exists()
    assert storage.vector_store_directory.exists()


def test_different_sessions_have_different_directories(tmp_path):
    base_directory = tmp_path / "sessions"

    session_a = SessionStorage(
        session_id="session-a",
        base_directory=str(base_directory),
    )

    session_b = SessionStorage(
        session_id="session-b",
        base_directory=str(base_directory),
    )

    assert session_a.session_directory != session_b.session_directory

    assert session_a.upload_directory != session_b.upload_directory

    assert (
        session_a.vector_store_directory
        != session_b.vector_store_directory
    )


def test_empty_session_id_is_rejected(tmp_path):
    with pytest.raises(ValueError):
        SessionStorage(
            session_id="",
            base_directory=str(tmp_path / "sessions"),
        )