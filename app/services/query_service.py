from langchain_core.documents import Document
from app.retrieval.retriever import retrieve
from app.llm.generator import generate, reflect
from app.core.config import settings
from app.core.logging import get_logger
from pathlib import Path

logger = get_logger(__name__)

MAX_REFLECTION_ATTEMPTS = 2

def query(question:str)->dict:
    """
    Full RAG pipeline with self-reflection loop.
    
    Flow:
        retrieve → reflect → sufficient? → generate
                                ↓ no
                           re-retrieve → reflect → generate
    """

    logger.info(f"Query received: '{question}'")

    docs = retrieve(question)

    attempt = 1

    while attempt<=MAX_REFLECTION_ATTEMPTS:
        logger.info(f"Reflection attempt {attempt}/{MAX_REFLECTION_ATTEMPTS}")

        reflection = reflect(question, docs)
        if reflection.get("is_sufficient"):
            logger.info("Context sufficient — generating answer")
            break

        if attempt == MAX_REFLECTION_ATTEMPTS:
            logger.warning("Max reflection attempts reached — generating with available context")
            break

        # Re-retrieve with missing info appended to original question
        missing = reflection.get("missing")
        if missing:
            refined_query = f"{question} {missing}"
            logger.info(f"Refining query with missing info: '{refined_query}'")
            docs = retrieve(refined_query)

        attempt += 1

    answer = generate(question, docs)

    return {
        "question": question,
        "answer": answer,
        "sources": _extract_sources(docs),
        "reflection_attempts": attempt
    }

def _extract_sources(docs: list[Document]) -> list[dict]:
    """Extract clean source citations from retrieved documents."""
    sources = []
    seen = set()

    for doc in docs:
        source = Path(doc.metadata.get("source", "unknown")).name
        page = doc.metadata.get("page", "?")
        key = f"{source}_{page}"

        if key not in seen:
            seen.add(key)
            sources.append({
                "file": source,
                "page": page
            })

    return sources