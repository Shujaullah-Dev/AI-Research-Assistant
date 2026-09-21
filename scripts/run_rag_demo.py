from app.embeddings.service import EmbeddingService
from app.llm.ollama_client import OllamaLLM
from app.rag.citations import format_citations
from app.rag.pipeline import RAGPipeline
from app.retrieval.reranker import CrossEncoderReranker
from app.retrieval.retriever import Retriever
from app.vector_store.faiss_store import FAISSVectorStore


EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
VECTOR_STORE_DIRECTORY = "data/vector_store_bge"
LLM_MODEL = "llama3.2:3b"
RERANKER_THRESHOLD = 0.0


def main():
    print("Loading embedding service...")
    embedding_service = EmbeddingService(
        model_name=EMBEDDING_MODEL
    )

    print("Loading BGE vector store...")
    vector_store = FAISSVectorStore.load(
        VECTOR_STORE_DIRECTORY
    )

    retriever = Retriever(
        embedding_service=embedding_service,
        vector_store=vector_store,
        similarity_threshold=0.0,
    )

    print("Loading cross-encoder reranker...")
    reranker = CrossEncoderReranker(
        model_name=RERANKER_MODEL
    )

    print("Loading Ollama LLM...")
    llm = OllamaLLM(
        model=LLM_MODEL
    )

    pipeline = RAGPipeline(
        retriever=retriever,
        llm=llm,
        reranker=reranker,
        reranker_threshold=RERANKER_THRESHOLD,
    )

    print("\nAI Research Assistant")
    print("---------------------")
    print(f"Embedding model: {EMBEDDING_MODEL}")
    print(f"Reranker: {RERANKER_MODEL}")
    print(f"Reranker threshold: {RERANKER_THRESHOLD}")
    print(f"LLM: {LLM_MODEL}")
    print("\nType 'exit' to quit.\n")

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
                final_k=5,
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