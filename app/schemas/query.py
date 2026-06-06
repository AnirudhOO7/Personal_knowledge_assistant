from pydantic import BaseModel
from typing import Optional


class QueryRequest(BaseModel):
    question: str
    top_k: Optional[int] = None


class SourceDocument(BaseModel):
    content: str
    source: str
    page: Optional[int] = None


class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceDocument]
