from dataclasses import dataclass

from app.llm.ollama_client import OllamaLLM
from app.retrieval.retriever import RetrievedChunk, Retriever


@dataclass
class RAGResponse:
    answer: str
    sources: list[RetrievedChunk]


class RAGPipeline:
    """Retrieve relevant chunks and generate an answer."""

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
                    "I could not find relevant information "
                    "in the provided documents."
                ),
                sources=[],
            )

        context = self._build_context(sources)

        prompt = self._build_prompt(
            question=question,
            context=context,
        )

        answer = self.llm.generate(prompt)

        return RAGResponse(
            answer=answer,
            sources=sources,
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

Answer the user's question using ONLY the
provided research context.

If the answer cannot be determined from
the context, say that the information is
not available in the provided documents.

Do not invent facts.

Research context:

{context}

User question:

{question}

Answer clearly and concisely.
""".strip()