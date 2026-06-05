from langchain_core.documents import Document
from app.vectorstore.qdrant_store import get_vectorstore
from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)


def retrieve(query: str, top_k: int | None = None) -> list[Document]:
    """Retrieve the top-k most relevant chunks for a query."""
    k = top_k or settings.top_k
    vectorstore = get_vectorstore()
    logger.info(f"Retrieving top {k} chunks for query: {query!r}")
    results = vectorstore.similarity_search(query, k=k)
    logger.info(f"Retrieved {len(results)} chunks")
    return results


def format_context(docs: list[Document])-> str:
    """Format retrived docs into a context string for llm"""
    context_parts=[]

    for i, doc in enumerate(docs):
        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page", "?")

        context_parts.append(
            f"[Source {i+1}: {source} | Page {page}]\n{doc.page_content}"
        )

    return "\n\n".join(context_parts)

