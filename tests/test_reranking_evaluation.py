from app.embeddings.service import EmbeddingService
from app.retrieval.reranker import CrossEncoderReranker
from app.retrieval.retriever import Retriever
from app.vector_store.faiss_store import FAISSVectorStore
from tests.retrieval_test_helpers import (
    retrieval_test_components,
)

MODEL_NAME = "BAAI/bge-small-en-v1.5"
RERANKER_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
INDEX_DIRECTORY = "data/vector_store_bge"


EVALUATION_CASES = [
    (
        "What dataset did the researchers use to train the model?",
        0,
    ),
    (
        "Which dataset was used for training the model?",
        0,
    ),
    (
        "What architecture does the model use?",
        1,
    ),
    (
        "What type of architecture does the model use?",
        1,
    ),
    (
        "What mechanism does the transformer architecture use?",
        1,
    ),
    (
        "What framework was used to implement the experiment?",
        2,
    ),
    (
        "Which framework was used in the experiment?",
        2,
    ),
    (
        "What classification accuracy did the final model achieve?",
        3,
    ),
    (
        "How accurate was the final model?",
        3,
    ),
    (
        "Which model did the researchers compare their proposed model with?",
        4,
    ),
    (
        "What model was used for comparison?",
        4,
    ),
]


def test_reranking_top_1_accuracy(
    retrieval_test_components,
):
    """
    Evaluate reranking using the temporary CI index.
    """

    embedding_service, vector_store = (
        retrieval_test_components
    )

    retriever = Retriever(
        embedding_service=embedding_service,
        vector_store=vector_store,
        similarity_threshold=0.0,
    )

    reranker = CrossEncoderReranker(
        model_name=RERANKER_NAME
    )

    correct = 0

    for question, expected_chunk_id in EVALUATION_CASES:
        retrieved_chunks = retriever.retrieve(
            question,
            top_k=5,
        )

        reranked_chunks = reranker.rerank(
            question,
            retrieved_chunks,
            top_k=5,
        )

        assert reranked_chunks, (
            f"No chunks were returned for question: "
            f"{question}"
        )

        top_chunk = reranked_chunks[0].chunk

        if (
            top_chunk.metadata.chunk_id
            == expected_chunk_id
        ):
            correct += 1

        else:
            print("\nFAILED QUESTION:")
            print(question)

            print(
                f"Expected chunk: "
                f"{expected_chunk_id}"
            )

            print(
                f"Retrieved chunk: "
                f"{top_chunk.metadata.chunk_id}"
            )

    accuracy = correct / len(EVALUATION_CASES)

    print("\n" + "=" * 70)
    print("RERANKING EVALUATION")
    print("=" * 70)

    print(
        f"Top-1 accuracy: "
        f"{correct}/{len(EVALUATION_CASES)} "
        f"= {accuracy:.2%}"
    )

    assert accuracy == 1.0