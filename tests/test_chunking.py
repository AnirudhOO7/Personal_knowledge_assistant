from langchain_core.documents import Document
from app.ingestion.chunker import chunk_documents


def _make_doc(text: str) -> Document:
    return Document(page_content=text, metadata={"source": "test.txt"})


def test_chunk_splits_large_document():
    long_text = "word " * 500
    docs = [_make_doc(long_text)]
    chunks = chunk_documents(docs)
    assert len(chunks) > 1


def test_chunk_preserves_metadata():
    docs = [_make_doc("Hello world")]
    chunks = chunk_documents(docs)
    assert all(c.metadata["source"] == "test.txt" for c in chunks)


def test_chunk_empty_input():
    chunks = chunk_documents([])
    assert chunks == []
