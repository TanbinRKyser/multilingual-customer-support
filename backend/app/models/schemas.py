from typing import Literal

from pydantic import BaseModel, Field, field_validator


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=2_000)
    explain_method: Literal["lime", "ig"] | None = None

    @field_validator("message")
    @classmethod
    def message_must_have_content(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("message must not be blank")
        return value


class Source(BaseModel):
    source: str
    chunk: int = 0
    snippet: str | None = None


class ExplanationItem(BaseModel):
    token: str
    weight: float


class ChatResponse(BaseModel):
    original_message: str
    detected_language: str
    language_name: str
    intent: str
    confidence: float
    response: str
    resolution: Literal["intent", "knowledge_base", "handoff"]
    sources: list[Source] = Field(default_factory=list)
    explanation: list[ExplanationItem] = Field(default_factory=list)


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"
    knowledge_documents: int
