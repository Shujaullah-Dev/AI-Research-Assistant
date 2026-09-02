from app.rag.pipeline import RAGPipeline
from app.retrieval.retriever import RetrievedChunk
from app.vector_store.faiss_store import ChunkMetadata


class FakeRetriever:
    def retrieve(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[RetrievedChunk]:
        return [
            RetrievedChunk(
                metadata=ChunkMetadata(
                    chunk_id=1,
                    page_number=7,
                    text=(
                        "The model was trained using "
                        "the CIFAR-10 dataset."
                    ),
                    document_name="paper.pdf",
                ),
                score=0.95,
            )
        ]


class FakeLLM:
    def __init__(self):
        self.last_prompt = None

    def generate(self, prompt: str) -> str:
        self.last_prompt = prompt
        return "The model was trained using CIFAR-10."


def test_rag_pipeline_generates_answer():
    retriever = FakeRetriever()
    llm = FakeLLM()

    pipeline = RAGPipeline(
        retriever=retriever,
        llm=llm,
    )

    response = pipeline.answer(
        "What dataset was used?"
    )

    assert (
        response.answer
        == "The model was trained using CIFAR-10."
    )

    assert len(response.sources) == 1

    assert (
        response.sources[0].metadata.page_number
        == 7
    )


def test_rag_prompt_contains_context():
    retriever = FakeRetriever()
    llm = FakeLLM()

    pipeline = RAGPipeline(
        retriever=retriever,
        llm=llm,
    )

    pipeline.answer(
        "What dataset was used?"
    )

    assert llm.last_prompt is not None

    assert "CIFAR-10" in llm.last_prompt

    assert "paper.pdf" in llm.last_prompt

    assert "Page: 7" in llm.last_prompt


def test_empty_question_raises_error():
    retriever = FakeRetriever()
    llm = FakeLLM()

    pipeline = RAGPipeline(
        retriever=retriever,
        llm=llm,
    )

    try:
        pipeline.answer("")
        assert False
    except ValueError:
        assert True