from functools import lru_cache
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

from app.embeddings.service import EmbeddingService
from app.ingestion.service import IngestionService
from app.llm.ollama_client import OllamaLLM
from app.rag.pipeline import RAGPipeline
from app.retrieval.reranker import CrossEncoderReranker
from app.retrieval.retriever import Retriever
from app.vector_store.faiss_store import FAISSVectorStore


router = APIRouter()

ingestion_service = IngestionService()


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
VECTOR_STORE_DIRECTORY = "data/vector_store_bge"
LLM_MODEL = "llama3.2:3b"
RERANKER_THRESHOLD = 0.0


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------

class AskRequest(BaseModel):
    """Request body for the /ask endpoint."""

    question: str = Field(
        ...,
        min_length=1,
        description="Research question to answer.",
        examples=[
            "What is the main contribution of this research paper?"
        ],
    )

    top_k: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Number of chunks retrieved before reranking.",
    )

    final_k: int = Field(
        default=5,
        ge=1,
        le=10,
        description="Number of chunks used after reranking.",
    )


class CitationResponse(BaseModel):
    """Citation information returned with an answer."""

    source_id: int
    document_name: str
    page_number: int
    chunk_id: int
    score: float


class AskResponse(BaseModel):
    """Response returned by the /ask endpoint."""

    answer: str
    citations: list[CitationResponse]


# ---------------------------------------------------------------------------
# RAG pipeline
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def get_pipeline() -> RAGPipeline:
    """Create and cache the RAG pipeline."""

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

    return RAGPipeline(
        retriever=retriever,
        llm=llm,
        reranker=reranker,
        reranker_threshold=RERANKER_THRESHOLD,
    )


# ---------------------------------------------------------------------------
# Document upload endpoint
# ---------------------------------------------------------------------------

@router.post("/documents/upload")
async def upload_document(
    file: UploadFile = File(...)
):
    """Upload and process a PDF document."""

    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="A filename is required.",
        )

    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are supported.",
        )

    upload_dir = Path("data/uploads")
    upload_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    file_path = upload_dir / file.filename

    content = await file.read()
    file_path.write_bytes(content)

    pages = ingestion_service.ingest_pdf(
        file_path
    )

    return {
        "document_name": file.filename,
        "pages_extracted": len(pages),
        "status": "processed",
    }


# ---------------------------------------------------------------------------
# Ask endpoint
# ---------------------------------------------------------------------------

@router.post(
    "/ask",
    response_model=AskResponse,
)
def ask(request: AskRequest) -> AskResponse:
    """Answer a research question using the RAG pipeline."""

    try:
        pipeline = get_pipeline()

        response = pipeline.answer(
            question=request.question,
            top_k=request.top_k,
            final_k=request.final_k,
        )

        return AskResponse(
            answer=response.answer,
            citations=[
                CitationResponse(
                    source_id=citation.source_id,
                    document_name=citation.document_name,
                    page_number=citation.page_number,
                    chunk_id=citation.chunk_id,
                    score=citation.score,
                )
                for citation in response.citations
            ],
        )

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate answer: {exc}",
        ) from exc