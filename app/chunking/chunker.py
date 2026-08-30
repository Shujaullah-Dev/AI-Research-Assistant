from dataclasses import dataclass


@dataclass
class TextChunk:
    text: str
    chunk_id: int
    page_number: int


class TextChunker:
    """Split extracted document text into overlapping chunks."""

    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
    ) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be greater than 0")

        if chunk_overlap < 0:
            raise ValueError("chunk_overlap cannot be negative")

        if chunk_overlap >= chunk_size:
            raise ValueError(
                "chunk_overlap must be smaller than chunk_size"
            )

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk_page(
        self,
        text: str,
        page_number: int,
    ) -> list[TextChunk]:
        text = text.strip()

        if not text:
            return []

        chunks: list[TextChunk] = []

        start = 0
        chunk_id = 0

        while start < len(text):
            end = start + self.chunk_size

            chunk_text = text[start:end].strip()

            if chunk_text:
                chunks.append(
                    TextChunk(
                        text=chunk_text,
                        chunk_id=chunk_id,
                        page_number=page_number,
                    )
                )

                chunk_id += 1

            if end >= len(text):
                break

            start = end - self.chunk_overlap

        return chunks