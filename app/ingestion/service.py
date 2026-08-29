from pathlib import Path

from app.ingestion.pdf_loader import PDFLoader
from app.ingestion.schemas import PageDocument


class IngestionService:
    """Coordinates document ingestion."""

    def __init__(self) -> None:
        self.pdf_loader = PDFLoader()

    def ingest_pdf(self, file_path: str | Path) -> list[PageDocument]:
        return self.pdf_loader.load(file_path)