from pathlib import Path

import fitz

from app.ingestion.schemas import PageDocument


class PDFLoader:
    """Extract text from a PDF while preserving page information."""

    def load(self, file_path: str | Path) -> list[PageDocument]:
        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(f"PDF not found: {path}")

        if path.suffix.lower() != ".pdf":
            raise ValueError("The provided file must be a PDF.")

        documents: list[PageDocument] = []

        with fitz.open(path) as pdf:
            for page_number, page in enumerate(pdf, start=1):
                text = page.get_text("text").strip()

                if not text:
                    continue

                documents.append(
                    PageDocument(
                        document_name=path.name,
                        page_number=page_number,
                        text=text,
                    )
                )

        return documents