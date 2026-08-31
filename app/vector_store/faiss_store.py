from dataclasses import asdict, dataclass
import json
from pathlib import Path

import faiss
import numpy as np


@dataclass
class ChunkMetadata:
    chunk_id: int
    page_number: int
    text: str
    document_name: str


class FAISSVectorStore:
    """Store and search text embeddings using FAISS."""

    def __init__(self, dimension: int) -> None:
        if dimension <= 0:
            raise ValueError("dimension must be greater than 0")

        self.dimension = dimension
        self.index = faiss.IndexFlatIP(dimension)
        self.metadata: list[ChunkMetadata] = []

    def add(
        self,
        embeddings: list[list[float]],
        metadata: list[ChunkMetadata],
    ) -> None:
        if len(embeddings) != len(metadata):
            raise ValueError(
                "Number of embeddings must match number of metadata records"
            )

        if not embeddings:
            return

        vectors = np.asarray(
            embeddings,
            dtype=np.float32,
        )

        if vectors.ndim != 2:
            raise ValueError(
                "Embeddings must be a 2-dimensional list"
            )

        if vectors.shape[1] != self.dimension:
            raise ValueError(
                f"Expected embedding dimension {self.dimension}, "
                f"got {vectors.shape[1]}"
            )

        self.index.add(vectors)
        self.metadata.extend(metadata)

    def search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
    ) -> list[tuple[ChunkMetadata, float]]:
        if top_k <= 0:
            raise ValueError("top_k must be greater than 0")

        if self.index.ntotal == 0:
            return []

        query = np.asarray(
            [query_embedding],
            dtype=np.float32,
        )

        if query.shape[1] != self.dimension:
            raise ValueError(
                f"Expected query dimension {self.dimension}, "
                f"got {query.shape[1]}"
            )

        k = min(top_k, self.index.ntotal)

        scores, indices = self.index.search(query, k)

        results = []

        for score, index in zip(scores[0], indices[0]):
            if index == -1:
                continue

            results.append(
                (
                    self.metadata[index],
                    float(score),
                )
            )

        return results

    def save(self, directory: str | Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)

        index_path = directory / "index.faiss"
        metadata_path = directory / "metadata.json"

        faiss.write_index(
            self.index,
            str(index_path),
        )

        metadata_data = [
            asdict(item)
            for item in self.metadata
        ]

        metadata_path.write_text(
            json.dumps(
                metadata_data,
                indent=2,
            ),
            encoding="utf-8",
        )

    @classmethod
    def load(
        cls,
        directory: str | Path,
    ) -> "FAISSVectorStore":
        directory = Path(directory)

        index_path = directory / "index.faiss"
        metadata_path = directory / "metadata.json"

        if not index_path.exists():
            raise FileNotFoundError(
                f"FAISS index not found: {index_path}"
            )

        if not metadata_path.exists():
            raise FileNotFoundError(
                f"Metadata file not found: {metadata_path}"
            )

        index = faiss.read_index(
            str(index_path)
        )

        metadata_data = json.loads(
            metadata_path.read_text(
                encoding="utf-8"
            )
        )

        store = cls(index.d)

        store.index = index

        store.metadata = [
            ChunkMetadata(**item)
            for item in metadata_data
        ]

        return store