import numpy as np

from app.embeddings.service import EmbeddingService


class FakeEmbeddingModel:
    def encode(
        self,
        texts,
        normalize_embeddings=True,
    ):
        return np.array(
            [
                [1.0, 0.0],
                [0.0, 1.0],
            ]
        )


def test_embedding_service_returns_vectors():
    service = EmbeddingService(
        model=FakeEmbeddingModel()
    )

    result = service.embed(
        [
            "first sentence",
            "second sentence",
        ]
    )

    assert len(result) == 2
    assert len(result[0]) == 2
    assert len(result[1]) == 2