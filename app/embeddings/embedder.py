from langchain_huggingface import HuggingFaceEmbeddings
from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)

_embedder = None

def get_embedder() -> HuggingFaceEmbeddings:
    """Return a singleton embedder instance"""
    global _embedder

    if _embedder is None:
        logger.info(f"Loading embedding model: {settings.embedding_model}")
        _embedder = HuggingFaceEmbeddings(
            model_name=settings.embedding_model,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True}
        )
        logger.info("Embedding model loaded")

    return _embedder





