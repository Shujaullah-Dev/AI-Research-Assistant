from functools import lru_cache
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

from app.chunking.chunker import TextChunker
from app.embeddings.service import EmbeddingService
from app.ingestion.service import IngestionService
from app.llm.ollama_client import OllamaLLM
from app.rag.pipeline import RAGPipeline
from app.retrieval.reranker import CrossEncoderReranker
from app.retrieval.retriever import Retriever
from app.vector_store.faiss_store import ChunkMetadata, FAISSVectorStore


router = APIRouter()


# -------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------

EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
VECTOR_STORE_DIRECTORY = "data/vector_store_bge"
LLM_MODEL = "llama3.2:3b"
RERANKER_THRESHOLD = 0.0

UPLOAD_DIRECTORY = Path("data/uploads")
MAX_UPLOAD_SIZE = 20 * 1024 * 1024 # 20MB

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50


# -------------------------------------------------------------------
# Services
# -------------------------------------------------------------------

ingestion_service = IngestionService()

chunker = TextChunker(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
)


# -------------------------------------------------------------------
# Request / response models
# -------------------------------------------------------------------

class AskRequest(BaseModel):
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
    source_id: int
    document_name: str
    page_number: int
    chunk_id: int
    score: float


class AskResponse(BaseModel):
    answer: str
    citations: list[CitationResponse]


# -------------------------------------------------------------------
# Cached shared services
# -------------------------------------------------------------------

@lru_cache(maxsize=1)
def get_embedding_service() -> EmbeddingService:
    print("Loading embedding service...")

    return EmbeddingService(
        model_name=EMBEDDING_MODEL
    )


@lru_cache(maxsize=1)
def get_vector_store() -> FAISSVectorStore:
    print("Loading BGE vector store...")

    return FAISSVectorStore.load(
        VECTOR_STORE_DIRECTORY
    )


@lru_cache(maxsize=1)
def get_pipeline() -> RAGPipeline:
    embedding_service = get_embedding_service()
    vector_store = get_vector_store()

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


# -------------------------------------------------------------------
# Document upload + indexing
# -------------------------------------------------------------------

@router.post("/documents/upload")
async def upload_document(
    file: UploadFile = File(...)
):
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

    # Keep only the filename itself.
    safe_filename = Path(file.filename).name

    UPLOAD_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    file_path = UPLOAD_DIRECTORY / safe_filename

    # ---------------------------------------------------------------
    # 1. Save uploaded PDF
    # ---------------------------------------------------------------

    content = await file.read()

    if not content:
        raise HTTPException(
            status_code=400,
            detail="The uploaded PDF is empty.",
        )

    if len(content) > MAX_UPLOAD_SIZE:
        raise HTTPException(
            status_code=413,
            detail="The uploaded PDF is too large. Maximum size is 20MB.",
        )

    file_path.write_bytes(content)

    # ---------------------------------------------------------------
    # 2. Extract pages
    # ---------------------------------------------------------------

    try:
        pages = ingestion_service.ingest_pdf(file_path)
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to process PDF: {exc}",
        ) from exc

    if not pages:
        raise HTTPException(
            status_code=400,
            detail="No readable pages were found in the PDF.",
        )

    # ---------------------------------------------------------------
    # 3. Convert pages into chunks
    # ---------------------------------------------------------------

    chunks = []
    chunk_counter = 0

    for page in pages:
        page_chunks = chunker.chunk_page(
            text=page.text,
            page_number=page.page_number,
        )

        for chunk in page_chunks:
            chunks.append(
                ChunkMetadata(
                    chunk_id=chunk_counter,
                    page_number=chunk.page_number,
                    text=chunk.text,
                    document_name=page.document_name,
                )
            )

            chunk_counter += 1

    if not chunks:
        raise HTTPException(
            status_code=400,
            detail="No text chunks could be created from the PDF.",
        )

    # ---------------------------------------------------------------
    # 4. Create BGE embeddings
    # ---------------------------------------------------------------

    embedding_service = get_embedding_service()

    texts = [
        chunk.text
        for chunk in chunks
    ]

    try:
        embeddings = embedding_service.embed(texts)
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to create embeddings: {exc}",
        ) from exc

    # ---------------------------------------------------------------
    # 5. Add chunks + embeddings to FAISS
    # ---------------------------------------------------------------

    vector_store = get_vector_store()

    try:
        vector_store.add(
            embeddings=embeddings,
            metadata=chunks,
        )

        vector_store.save(
            VECTOR_STORE_DIRECTORY
        )

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to update vector store: {exc}",
        ) from exc

    # ---------------------------------------------------------------
    # 6. Return indexing information
    # ---------------------------------------------------------------

    return {
        "document_name": safe_filename,
        "pages_extracted": len(pages),
        "chunks_indexed": len(chunks),
        "embedding_model": EMBEDDING_MODEL,
        "vector_store": VECTOR_STORE_DIRECTORY,
        "status": "processed",
    }


# -------------------------------------------------------------------
# Question answering
# -------------------------------------------------------------------

@router.post(
    "/ask",
    response_model=AskResponse,
)
def ask(request: AskRequest) -> AskResponse:
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