from tests.retrieval_test_helpers import (
    retrieval_test_components,
)


EVALUATION_CASES = [
    {
        "question": "What dataset did the researchers use to train the model?",
        "expected_chunk_id": 0,
    },
    {
        "question": "Which dataset was used for training the model?",
        "expected_chunk_id": 0,
    },
    {
        "question": "What architecture does the model use?",
        "expected_chunk_id": 1,
    },
    {
        "question": "What type of architecture does the model use?",
        "expected_chunk_id": 1,
    },
    {
        "question": "What mechanism does the transformer architecture use?",
        "expected_chunk_id": 1,
    },
    {
        "question": "What framework was used to implement the experiment?",
        "expected_chunk_id": 2,
    },
    {
        "question": "Which framework was used in the experiment?",
        "expected_chunk_id": 2,
    },
    {
        "question": "What classification accuracy did the final model achieve?",
        "expected_chunk_id": 3,
    },
    {
        "question": "How accurate was the final model?",
        "expected_chunk_id": 3,
    },
    {
        "question": "Which model did the researchers compare their proposed model with?",
        "expected_chunk_id": 4,
    },
    {
        "question": "What model was used for comparison?",
        "expected_chunk_id": 4,
    },
]


def test_retrieval_evaluation(
    retrieval_test_components,
):
    """
    Evaluate retrieval using a temporary CI-safe index.
    """

    embedding_service, vector_store = (
        retrieval_test_components
    )

    top_1_correct = 0
    top_3_correct = 0
    reciprocal_ranks = []

    for case in EVALUATION_CASES:
        question = case["question"]
        expected_chunk_id = case["expected_chunk_id"]

        query_embedding = embedding_service.embed(
            [question]
        )[0]

        results = vector_store.search(
            query_embedding,
            top_k=5,
        )

        retrieved_chunk_ids = [
            metadata.chunk_id
            for metadata, score in results
        ]

        if expected_chunk_id in retrieved_chunk_ids:
            rank = (
                retrieved_chunk_ids.index(
                    expected_chunk_id
                )
                + 1
            )

            reciprocal_rank = 1 / rank

        else:
            rank = None
            reciprocal_rank = 0.0

        if rank == 1:
            top_1_correct += 1

        if rank is not None and rank <= 3:
            top_3_correct += 1

        reciprocal_ranks.append(
            reciprocal_rank
        )

        print("\nQuestion:")
        print(question)

        print(
            f"Expected chunk: {expected_chunk_id}"
        )

        print(
            f"Correct chunk rank: "
            f"{rank if rank is not None else 'NOT FOUND'}"
        )

        print("Retrieved:")

        for current_rank, (
            metadata,
            score,
        ) in enumerate(
            results,
            start=1,
        ):
            marker = (
                " <-- CORRECT"
                if metadata.chunk_id
                == expected_chunk_id
                else ""
            )

            print(
                f"  Rank {current_rank}: "
                f"chunk {metadata.chunk_id} "
                f"(similarity={score:.4f})"
                f"{marker}"
            )

        print(
            f"Reciprocal rank: "
            f"{reciprocal_rank:.4f}"
        )

    total = len(EVALUATION_CASES)

    top_1_accuracy = top_1_correct / total
    top_3_accuracy = top_3_correct / total

    mean_reciprocal_rank = (
        sum(reciprocal_ranks) / total
    )

    print("\n" + "=" * 80)
    print("RETRIEVAL EVALUATION")
    print("=" * 80)

    print(
        f"Top-1 accuracy: "
        f"{top_1_correct}/{total} "
        f"= {top_1_accuracy:.2%}"
    )

    print(
        f"Top-3 accuracy: "
        f"{top_3_correct}/{total} "
        f"= {top_3_accuracy:.2%}"
    )

    print(
        f"MRR: "
        f"{mean_reciprocal_rank:.4f}"
    )

    assert total > 0