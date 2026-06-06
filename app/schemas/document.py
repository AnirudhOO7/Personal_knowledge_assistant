from pydantic import BaseModel
from typing import Optional


class DocumentMetadata(BaseModel):
    source: str
    page: Optional[int] = None
    chunk_index: Optional[int] = None


class DocumentChunk(BaseModel):
    content: str
    metadata: DocumentMetadata
