from app.chunking.chunker import TextChunker


def test_text_is_split_into_chunks():
    chunker = TextChunker(
        chunk_size=10,
        chunk_overlap=2,
    )

    text = "abcdefghijklmnopqrstuvwxyz"

    chunks = chunker.chunk_page(
        text=text,
        page_number=1,
    )

    assert len(chunks) > 1
    assert chunks[0].text == "abcdefghij"
    assert chunks[0].page_number == 1

def test_chunks_have_overlap():
    chunker = TextChunker(
        chunk_size=10,
        chunk_overlap=2,
    )

    text = "abcdefghijklmnopqrstuvwxyz"

    chunks = chunker.chunk_page(
        text=text,
        page_number=1,
    )

    assert chunks[0].text[-2:] == chunks[1].text[:2]

def test_empty_text_returns_no_chunks():
    chunker = TextChunker()

    chunks = chunker.chunk_page(
        text="",
        page_number=1,
    )

    assert chunks == []

def test_whitespace_returns_no_chunks():
    chunker = TextChunker()

    chunks = chunker.chunk_page(
        text="   \n\t ",
        page_number=1,
    )

    assert chunks == []

import pytest

from app.chunking.chunker import TextChunker


def test_overlap_cannot_equal_chunk_size():
    with pytest.raises(ValueError):
        TextChunker(
            chunk_size=100,
            chunk_overlap=100,
        )

def test_overlap_cannot_be_larger_than_chunk_size():
    with pytest.raises(ValueError):
        TextChunker(
            chunk_size=100,
            chunk_overlap=150,
        )