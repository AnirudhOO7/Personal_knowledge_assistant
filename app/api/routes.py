from fastapi import APIRouter, HTTPException, UploadFile, File
from pathlib import Path
import shutil
import tempfile

from app.schemas.query import QueryRequest, QueryResponse, SourceDocument
from app.services.query_service import query as run_query
from app.services.ingest_service import ingest_file

router = APIRouter()


@router.post("/query", response_model=QueryResponse)
def query(request: QueryRequest):
    result = run_query(request.question)
    if not result["sources"]:
        raise HTTPException(status_code=404, detail="No relevant documents found")

    sources = [
        SourceDocument(
            content="",
            source=s["file"],
            page=s["page"] if s["page"] != "?" else None,
        )
        for s in result["sources"]
    ]
    return QueryResponse(answer=result["answer"], sources=sources)


@router.post("/ingest")
def ingest(file: UploadFile = File(...)):
    suffix = Path(file.filename).suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    original_path = Path(tmp_path).parent / file.filename
    Path(tmp_path).rename(original_path)

    try:
        count = ingest_file(str(original_path))
    finally:
        Path(original_path).unlink(missing_ok=True)

    return {"message": f"Ingested {count} chunks from {file.filename}"}


@router.get("/health")
def health():
    return {"status": "ok"}
