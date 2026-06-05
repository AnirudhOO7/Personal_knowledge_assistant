from langchain_qdrant import QdrantVectorStore
from langchain_core.documents import Document
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from app.core.config import settings
from app.embeddings.embedder import get_embedder
from app.core.logging import get_logger

logger = get_logger(__name__)

_client = None
_vectorstore = None

def get_client() -> QdrantClient:
    """Return a singleton Qdrant client."""
    global _client

    if _client is None:
        logger.info(f"Connecting to Qdrant at {settings.qdrant_url}")
        _client = QdrantClient(url=settings.qdrant_url)
        logger.info("Qdrant client connected")

    return _client


def get_vectorstore() -> QdrantVectorStore:
    """Return a singleton vectorstore instance."""
    global _vectorstore

    if _vectorstore is None:
        client = get_client()
        embedder = get_embedder()

        # Create collection if it doesn't exist
        existing = [c.name for c in client.get_collections().collections]

        if settings.collection_name not in existing:
            logger.info(f"Creating collection: {settings.collection_name}")
            client.create_collection(
                collection_name=settings.collection_name,
                vectors_config=VectorParams(
                    size=384,  # all-MiniLM-L6-v2 dimension
                    distance=Distance.COSINE
                )
            )
            logger.info("Collection created")
        else:
            logger.info(f"Collection '{settings.collection_name}' already exists")

        _vectorstore = QdrantVectorStore(
            client=client,
            collection_name=settings.collection_name,
            embedding=embedder
        )

    return _vectorstore


def add_documents(docs: list[Document]) -> None:
    """Add documents to the vectorstore."""
    vectorstore = get_vectorstore()
    logger.info(f"Adding {len(docs)} chunks to Qdrant")
    vectorstore.add_documents(docs)
    logger.info("Documents added successfully")