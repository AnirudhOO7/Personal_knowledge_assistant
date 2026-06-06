import json
from unittest.mock import patch, MagicMock
from langchain_core.documents import Document
from app.llm.generator import reflect, generate
from app.retrieval.retriever import format_context


def _make_doc(text: str, source: str = "test.pdf", page: int = 1) -> Document:
    return Document(page_content=text, metadata={"source": source, "page": page})


# ── format_context ────────────────────────────────────────────────────────────

def test_format_context_includes_source_and_page():
    docs = [_make_doc("hello world", source="notes.pdf", page=3)]
    result = format_context(docs)
    assert "notes.pdf" in result
    assert "Page 3" in result
    assert "hello world" in result


def test_format_context_multiple_docs_separated():
    docs = [_make_doc("chunk one"), _make_doc("chunk two")]
    result = format_context(docs)
    assert "chunk one" in result
    assert "chunk two" in result
    assert result.index("chunk one") < result.index("chunk two")


def test_format_context_empty_list():
    assert format_context([]) == ""


# ── reflect ───────────────────────────────────────────────────────────────────

@patch("app.llm.generator.get_llm")
def test_reflect_returns_parsed_json(mock_get_llm):
    mock_llm = MagicMock()
    payload = {"is_sufficient": True, "reason": "all there", "missing": None}
    mock_llm.invoke.return_value = MagicMock(content=json.dumps(payload))
    mock_get_llm.return_value = mock_llm

    result = reflect("What is X?", [_make_doc("X is Y")])
    assert result["is_sufficient"] is True
    assert result["reason"] == "all there"


@patch("app.llm.generator.get_llm")
def test_reflect_defaults_to_sufficient_on_bad_json(mock_get_llm):
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = MagicMock(content="not json at all")
    mock_get_llm.return_value = mock_llm

    result = reflect("What is X?", [_make_doc("X is Y")])
    assert result["is_sufficient"] is True
    assert result["reason"] == "parse error"


# ── generate ──────────────────────────────────────────────────────────────────

@patch("app.llm.generator.get_llm")
def test_generate_returns_string(mock_get_llm):
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = MagicMock(content="The answer is 42.")
    mock_get_llm.return_value = mock_llm

    result = generate("What is the answer?", [_make_doc("The answer is 42.")])
    assert result == "The answer is 42."
