from app.retrieval.retriever import Retriever


class FakeEmbeddingService:
    def embed(self, texts):
        return [[1.0, 0.0]]


class FakeVectorStore:
    def search(self, query_embedding, top_k=5):
        return [
            (
                type(
                    "Metadata",
                    (),
                    {
                        "chunk_id": 1,
                        "page_number": 2,
                        "text": "Highly relevant text.",
                        "document_name": "paper.pdf",
                    },
                )(),
                0.90,
            ),
            (
                type(
                    "Metadata",
                    (),
                    {
                        "chunk_id": 2,
                        "page_number": 3,
                        "text": "Weakly relevant text.",
                        "document_name": "paper.pdf",
                    },
                )(),
                0.30,
            ),
        ]


def test_retriever_filters_low_similarity_results():
    retriever = Retriever(
        embedding_service=FakeEmbeddingService(),
        vector_store=FakeVectorStore(),
        similarity_threshold=0.50,
    )

    results = retriever.retrieve(
        "What is the main finding?",
        top_k=5,
    )

    assert len(results) == 1
    assert results[0].score == 0.90
    assert results[0].metadata.chunk_id == 1


def test_retriever_rejects_invalid_threshold():
    try:
        Retriever(
            embedding_service=FakeEmbeddingService(),
            vector_store=FakeVectorStore(),
            similarity_threshold=1.5,
        )
        assert False
    except ValueError:
        assert True