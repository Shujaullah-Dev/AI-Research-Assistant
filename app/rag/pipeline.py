from app.llm.ollama_client import OllamaLLM
from app.rag.models import RAGResponse
from app.retrieval.retriever import RetrievedChunk, Retriever


class RAGPipeline:
    """Retrieve relevant chunks and generate a grounded answer."""

    def __init__(
        self,
        retriever: Retriever,
        llm: OllamaLLM,
    ) -> None:
        self.retriever = retriever
        self.llm = llm

    def answer(
        self,
        question: str,
        top_k: int = 5,
    ) -> RAGResponse:
        if not question.strip():
            raise ValueError(
                "question must not be empty"
            )

        sources = self.retriever.retrieve(
            question,
            top_k=top_k,
        )

        if not sources:
            return RAGResponse(
                answer=(
                    "I could not find enough relevant "
                    "information in the provided documents."
                ),
                citations=[],
            )

        context = self._build_context(
            sources
        )

        prompt = self._build_prompt(
            question=question,
            context=context,
        )

        answer = self.llm.generate(prompt)

        return RAGResponse.from_retrieved_chunks(
            answer=answer,
            chunks=sources,
        )

    @staticmethod
    def _build_context(
        sources: list[RetrievedChunk],
    ) -> str:
        context_parts = []

        for index, source in enumerate(
            sources,
            start=1,
        ):
            metadata = source.metadata

            context_parts.append(
                f"[Source {index}]\n"
                f"Document: {metadata.document_name}\n"
                f"Page: {metadata.page_number}\n"
                f"Chunk ID: {metadata.chunk_id}\n"
                f"Similarity: {source.score:.4f}\n"
                f"Text: {metadata.text}"
            )

        return "\n\n".join(context_parts)

    @staticmethod
    def _build_prompt(
        question: str,
        context: str,
    ) -> str:
        return f"""
You are an AI research assistant.

Your job is to answer the user's question
using ONLY the provided research context.

Important rules:

1. Do not invent facts.
2. Do not use outside knowledge.
3. If the context does not contain enough
   information to answer the question,
   clearly say so.
4. When making a factual claim, include
   the corresponding source number such as
   [Source 1] or [Source 2].
5. Keep the answer concise and precise.

Research context:

{context}

User question:

{question}

Answer:
""".strip()