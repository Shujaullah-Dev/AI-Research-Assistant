from app.rag.citations import format_citations
from app.rag.models import Citation


def test_format_citations():
    citations = [
        Citation(
            source_id=1,
            document_name="paper.pdf",
            page_number=7,
            chunk_id=12,
            score=0.9234,
        ),
        Citation(
            source_id=2,
            document_name="paper.pdf",
            page_number=9,
            chunk_id=18,
            score=0.8512,
        ),
    ]

    result = format_citations(citations)

    assert "Sources:" in result
    assert "[1]" in result
    assert "[2]" in result
    assert "paper.pdf" in result
    assert "Page 7" in result
    assert "Page 9" in result


def test_format_empty_citations():
    result = format_citations([])

    assert result == "No sources available."