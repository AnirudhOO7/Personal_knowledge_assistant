"""Ingest a file or directory into the RAG index."""
import argparse
from app.services.ingest_service import ingest_file, ingest_directory
from app.core.logging import get_logger

logger = get_logger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Ingest documents into the RAG index")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--file", type=str, help="Path to a single file to ingest")
    group.add_argument("--dir", type=str, help="Path to a directory to ingest")
    args = parser.parse_args()

    if args.file:
        count = ingest_file(args.file)
        print(f"Ingested {count} chunks from {args.file}")
    else:
        count = ingest_directory(args.dir)
        print(f"Ingested {count} chunks from {args.dir}")


if __name__ == "__main__":
    main()
