from enum import Enum

from pydantic import BaseModel, Extra


class ComponentType(str, Enum):
    LLM = "llm"
    RETRIEVER = "retriever"
    MEMORY = "memory"


class Metadata(BaseModel, extra=Extra.allow):
    conversation_id: str
    user_id: str
    pdf_id: str


class ChatArgs(BaseModel, extra=Extra.allow):
    conversation_id: str
    pdf_id: str
    metadata: Metadata
    streaming: bool
