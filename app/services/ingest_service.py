from app.ingestion.loader import load_file, load_directory
from app.ingestion.parser import clean_documents
from app.ingestion.chunker import chunk_documents
from app.vectorstore.qdrant_store import add_documents
from app.core.logging import get_logger

logger = get_logger(__name__)


def ingest_file(file_path: str) -> int:
    """Load, clean, chunk, and index a single file. Returns number of chunks added."""
    docs = load_file(file_path)
    docs = clean_documents(docs)
    chunks = chunk_documents(docs)
    add_documents(chunks)
    logger.info(f"Ingested {len(chunks)} chunks from {file_path}")
    return len(chunks)


def ingest_directory(dir_path: str) -> int:
    """Load, clean, chunk, and index all supported files in a directory."""
    docs = load_directory(dir_path)
    if not docs:
        logger.warning("No documents found to ingest")
        return 0
    docs = clean_documents(docs)
    chunks = chunk_documents(docs)
    add_documents(chunks)
    logger.info(f"Ingested {len(chunks)} chunks from {dir_path}")
    return len(chunks)
