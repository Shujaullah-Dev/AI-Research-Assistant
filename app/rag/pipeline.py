from dataclasses import dataclass

from app.llm.ollama_client import OllamaLLM
from app.rag.models import RAGResponse
from app.retrieval.reranker import CrossEncoderReranker
from app.retrieval.retriever import RetrievedChunk, Retriever


@dataclass
class RAGPipeline:
    """Generate answers using retrieval, reranking, and an LLM."""

    retriever: Retriever
    llm: OllamaLLM
    reranker: CrossEncoderReranker | None = None
    reranker_threshold: float | None = None

    def answer(
        self,
        question: str,
        top_k: int = 5,
        final_k: int | None = None,
    ) -> RAGResponse:
        """Answer a question using retrieved and optionally reranked chunks."""

        if not question.strip():
            raise ValueError("question must not be empty")

        if top_k <= 0:
            raise ValueError(
                "top_k must be greater than 0"
            )

        if final_k is not None and final_k <= 0:
            raise ValueError(
                "final_k must be greater than 0"
            )

        if (
            self.reranker_threshold is not None
            and self.reranker is None
        ):
            raise ValueError(
                "reranker_threshold requires a reranker"
            )

        # ---------------------------------------------------------
        # Step 1: Retrieve candidate chunks using FAISS
        # ---------------------------------------------------------
        retrieved_chunks = self.retriever.retrieve(
            question,
            top_k=top_k,
        )

        # ---------------------------------------------------------
        # Step 2: No retrieved chunks
        # ---------------------------------------------------------
        if not retrieved_chunks:
            return self._no_information_response()

        # ---------------------------------------------------------
        # Step 3: Optionally rerank candidates
        # ---------------------------------------------------------
        final_chunks: list[RetrievedChunk]

        if self.reranker is not None:
            reranked_chunks = self.reranker.rerank(
                question,
                retrieved_chunks,
                top_k=final_k,
            )

            if not reranked_chunks:
                return self._no_information_response()

            # -----------------------------------------------------
            # Step 4: Relevance check
            # -----------------------------------------------------
            if self.reranker_threshold is not None:
                best_chunk = reranked_chunks[0]

                if not self.reranker.is_relevant(
                    best_chunk,
                    self.reranker_threshold,
                ):
                    return self._no_information_response()

            final_chunks = [
                item.chunk
                for item in reranked_chunks
            ]

        else:
            final_chunks = retrieved_chunks

            if final_k is not None:
                final_chunks = final_chunks[:final_k]

        # ---------------------------------------------------------
        # Step 5: Build context
        # ---------------------------------------------------------
        context_parts = []

        for index, chunk in enumerate(
            final_chunks,
            start=1,
        ):
            context_parts.append(
                f"[Source {index}]\n"
                f"Document: {chunk.metadata.document_name}\n"
                f"Page: {chunk.metadata.page_number}\n"
                f"Chunk: {chunk.metadata.chunk_id}\n"
                f"Similarity: {chunk.score:.4f}\n"
                f"Text: {chunk.metadata.text}"
            )

        context = "\n\n".join(context_parts)

        # ---------------------------------------------------------
        # Step 6: Build grounded prompt
        # ---------------------------------------------------------
        prompt = f"""
You are an AI research assistant.

Answer the user's question using ONLY the research context
provided below.

Rules:
- Do not use outside knowledge.
- Do not invent facts.
- If the context does not contain enough information, clearly say so.
- Cite factual claims using [Source N].
- Keep the answer concise and directly answer the question.

Research context:
{context}

User question:
{question}

Answer:
""".strip()

        # ---------------------------------------------------------
        # Step 7: Generate answer
        # ---------------------------------------------------------
        answer = self.llm.generate(prompt)

        # ---------------------------------------------------------
        # Step 8: Return answer + citations
        # ---------------------------------------------------------
        return RAGResponse.from_retrieved_chunks(
            answer=answer,
            chunks=final_chunks,
        )

    @staticmethod
    def _no_information_response() -> RAGResponse:
        """Return a grounded response when evidence is insufficient."""

        return RAGResponse(
            answer=(
                "I could not find enough relevant information "
                "in the indexed documents to answer this question."
            ),
            citations=[],
        )