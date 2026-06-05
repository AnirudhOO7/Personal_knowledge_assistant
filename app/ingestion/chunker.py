from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)

def chunk_documents(docs: list[Document]) -> list[Document]:
    """Split documents into chunks while preserving metadata."""
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", ".", " ", ""]
    )

    chunks = splitter.split_documents(docs)

    logger.info(f"Split {len(docs)} documents into {len(chunks)} chunks")
    return chunks