import re
from langchain_core.documents import Document


def clean_text(text: str) -> str:
    """Remove excessive whitespace and non-printable characters."""
    text = re.sub(r"[ \t]+", " ", text)          # collapse horizontal whitespace
    text = re.sub(r"\n{3,}", "\n\n", text)        # max two consecutive newlines
    text = re.sub(r"[^\x20-\x7E\n]", "", text)   # strip non-printable ASCII
    return text.strip()


def clean_documents(docs: list[Document]) -> list[Document]:
    """Apply text cleaning to a list of LangChain Documents in-place."""
    for doc in docs:
        doc.page_content = clean_text(doc.page_content)
    return [doc for doc in docs if doc.page_content]  # drop empty docs
