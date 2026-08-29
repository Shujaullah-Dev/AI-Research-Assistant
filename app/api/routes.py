from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile

from app.ingestion.service import IngestionService

router = APIRouter()

ingestion_service = IngestionService()


@router.post("/documents/upload")
async def upload_document(file: UploadFile = File(...)):
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
    upload_dir.mkdir(parents=True, exist_ok=True)

    file_path = upload_dir / file.filename

    content = await file.read()
    file_path.write_bytes(content)

    pages = ingestion_service.ingest_pdf(file_path)

    return {
        "document_name": file.filename,
        "pages_extracted": len(pages),
        "status": "processed",
    }