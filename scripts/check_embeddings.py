from app.embeddings.service import EmbeddingService


def main():
    service = EmbeddingService()

    texts = [
        "Transformers use self-attention.",
        "Self-attention helps models understand relationships.",
        "Chocolate cake requires flour and sugar.",
    ]

    embeddings = service.embed(texts)

    print("Number of texts:", len(texts))
    print("Number of embeddings:", len(embeddings))
    print("Embedding dimension:", len(embeddings[0]))


if __name__ == "__main__":
    main()