from pathlib import Path


class SessionStorage:
    """
    Manage filesystem paths for a single application session.
    """

    def __init__(
        self,
        session_id: str,
        base_directory: str = "data/sessions",
    ):
        if not session_id or not session_id.strip():
            raise ValueError("session_id cannot be empty.")

        self.session_id = session_id.strip()
        self.base_directory = Path(base_directory)

    @property
    def session_directory(self) -> Path:
        """
        Return the root directory for this session.
        """
        return self.base_directory / self.session_id

    @property
    def upload_directory(self) -> Path:
        """
        Return the directory where this session's PDFs are stored.
        """
        return self.session_directory / "uploads"

    @property
    def vector_store_directory(self) -> Path:
        """
        Return the directory where this session's FAISS
        vector store is stored.
        """
        return self.session_directory / "vector_store"

    def create_directories(self) -> None:
        """
        Create the session's required directories.
        """
        self.upload_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.vector_store_directory.mkdir(
            parents=True,
            exist_ok=True,
        )