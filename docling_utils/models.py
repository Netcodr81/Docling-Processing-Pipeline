from pydantic import BaseModel
from typing import Any, Dict
from uuid import uuid4


class DocumentChunk(BaseModel):
    id: str
    text: str
    metadata: Dict[str, Any]

    @staticmethod
    def create(text: str, metadata: Dict[str, Any]) -> "DocumentChunk":
        return DocumentChunk(id=str(uuid4()), text=text, metadata=metadata)
