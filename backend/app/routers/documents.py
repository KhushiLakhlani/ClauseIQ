"""Contract document upload and management endpoints."""

from fastapi import APIRouter, HTTPException, UploadFile, File
from pydantic import BaseModel

router = APIRouter()


class DocumentMeta(BaseModel):
    doc_id: str
    filename: str
    char_count: int
    word_count: int
    status: str


@router.post("/documents/upload", response_model=DocumentMeta)
async def upload_document(file: UploadFile = File(...)) -> DocumentMeta:
    """Accept a plain-text or PDF contract file and return basic metadata.

    Full preprocessing and clause segmentation are enqueued as a background
    task (not yet implemented).
    """
    allowed = {"text/plain", "application/pdf"}
    if file.content_type not in allowed:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported media type '{file.content_type}'. Upload .txt or .pdf.",
        )
    content: bytes = await file.read()
    text = content.decode("utf-8", errors="replace")
    return DocumentMeta(
        doc_id="placeholder-id",
        filename=file.filename or "unknown",
        char_count=len(text),
        word_count=len(text.split()),
        status="received",
    )


@router.get("/documents", response_model=list[DocumentMeta])
async def list_documents() -> list[DocumentMeta]:
    """List all uploaded documents (stub — wire to DB in production)."""
    return []


@router.get("/documents/{doc_id}", response_model=DocumentMeta)
async def get_document(doc_id: str) -> DocumentMeta:
    """Retrieve metadata for a single document by ID."""
    raise HTTPException(status_code=404, detail=f"Document '{doc_id}' not found")
