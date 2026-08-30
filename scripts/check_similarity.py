import numpy as np

from app.embeddings.service import EmbeddingService


def main():
    service = EmbeddingService()

    texts = [
        "Transformers use self-attention.",
        "Self-attention helps models understand token relationships.",
        "Chocolate cake requires flour and sugar.",
    ]

    embeddings = np.array(
        service.embed(texts)
    )

    similarity = embeddings @ embeddings.T

    print("Texts:")
    for index, text in enumerate(texts):
        print(f"{index}: {text}")

    print("\nSimilarity matrix:")
    print(similarity)


if __name__ == "__main__":
    main()