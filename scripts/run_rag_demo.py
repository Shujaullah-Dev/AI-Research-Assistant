from app.embeddings.service import EmbeddingService
from app.llm.ollama_client import OllamaLLM
from app.rag.pipeline import RAGPipeline
from app.retrieval.retriever import Retriever
from app.vector_store.faiss_store import FAISSVectorStore


def main():
    print("Loading embedding service...")

    embedding_service = EmbeddingService()

    print("Loading vector store...")

    vector_store = FAISSVectorStore.load(
        "data/vector_store"
    )

    retriever = Retriever(
        embedding_service=embedding_service,
        vector_store=vector_store,
    )

    llm = OllamaLLM(
        model="llama3.2:3b"
    )

    pipeline = RAGPipeline(
        retriever=retriever,
        llm=llm,
    )

    print("\nAI Research Assistant")
    print("---------------------")
    print("Type 'exit' to quit.\n")

    while True:
        question = input("Question: ").strip()

        if question.lower() == "exit":
            break

        if not question:
            continue

        try:
            response = pipeline.answer(
                question,
                top_k=5,
            )

            print("\nAnswer:")
            print(response.answer)

            print("\nSources:")

            for source in response.sources:
                metadata = source.metadata

                print(
                    f"- {metadata.document_name}, "
                    f"page {metadata.page_number}, "
                    f"score={source.score:.4f}"
                )

            print()

        except Exception as exc:
            print(f"\nError: {exc}\n")


if __name__ == "__main__":
    main()