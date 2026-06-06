from abc import ABC, abstractmethod
from langchain_core.documents import Document


class BaseVectorStore(ABC):
    """Abstract interface for vector store backends."""

    @abstractmethod
    def add_documents(self, docs: list[Document]) -> None:
        """Embed and store a list of documents."""
        ...

    @abstractmethod
    def similarity_search(self, query: str, k: int) -> list[Document]:
        """Return the k most similar documents to the query."""
        ...
