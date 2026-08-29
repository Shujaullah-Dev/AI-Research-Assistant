from pathlib import Path

import pytest

from app.ingestion.pdf_loader import PDFLoader


def test_pdf_file_must_exist():
    loader = PDFLoader()

    with pytest.raises(FileNotFoundError):
        loader.load("does_not_exist.pdf")


def test_file_must_be_pdf(tmp_path: Path):
    text_file = tmp_path / "document.txt"
    text_file.write_text("This is not a PDF.")

    loader = PDFLoader()

    with pytest.raises(ValueError):
        loader.load(text_file)


def test_pdf_extraction():
    pdf_path = Path("data/test_paper.pdf")

    if not pdf_path.exists():
        pytest.skip("Test PDF not available.")

    loader = PDFLoader()
    pages = loader.load(pdf_path)

    assert len(pages) > 0
    assert pages[0].page_number == 1
    assert pages[0].document_name == "test_paper.pdf"
    assert pages[0].text