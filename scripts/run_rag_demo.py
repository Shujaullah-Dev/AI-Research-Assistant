from app.embeddings.service import EmbeddingService
from app.llm.ollama_client import OllamaLLM
from app.rag.citations import format_citations
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
        similarity_threshold=0.40,
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

            print()

            print(
                format_citations(
                    response.citations
                )
            )

            print()

        except Exception as exc:
            print(f"\nError: {exc}\n")


if __name__ == "__main__":
    main()