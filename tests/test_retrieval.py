from unittest.mock import patch, MagicMock
from langchain_core.documents import Document
from app.retrieval.retriever import retrieve


def _make_doc(text: str) -> Document:
    return Document(page_content=text, metadata={"source": "test.txt"})


@patch("app.retrieval.retriever.get_vectorstore")
def test_retrieve_returns_docs(mock_get_vs):
    mock_vs = MagicMock()
    mock_vs.similarity_search.return_value = [_make_doc("relevant chunk")]
    mock_get_vs.return_value = mock_vs

    results = retrieve("test query", top_k=1)

    assert len(results) == 1
    assert results[0].page_content == "relevant chunk"
    mock_vs.similarity_search.assert_called_once_with("test query", k=1)


@patch("app.retrieval.retriever.get_vectorstore")
def test_retrieve_uses_settings_top_k_by_default(mock_get_vs):
    mock_vs = MagicMock()
    mock_vs.similarity_search.return_value = []
    mock_get_vs.return_value = mock_vs

    from app.core.config import settings
    retrieve("query")
    mock_vs.similarity_search.assert_called_once_with("query", k=settings.top_k)
