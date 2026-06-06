"""Drop and rebuild the Qdrant collection from raw data."""
from qdrant_client import QdrantClient
from app.core.config import settings
from app.core.logging import get_logger
from app.services.ingest_service import ingest_directory

logger = get_logger(__name__)


def main():
    client = QdrantClient(url=settings.qdrant_url)
    existing = [c.name for c in client.get_collections().collections]

    if settings.collection_name in existing:
        logger.info(f"Deleting existing collection: {settings.collection_name}")
        client.delete_collection(settings.collection_name)

    count = ingest_directory(settings.raw_data_path)
    logger.info(f"Index rebuilt with {count} chunks")


if __name__ == "__main__":
    main()
