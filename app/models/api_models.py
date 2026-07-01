
from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field, field_validator


class Role(str, Enum):
    user = "user"
    assistant = "assistant"


class ChatMessage(BaseModel):
    """A single turn in the stateless conversation history."""

    role: Role
    content: str = Field(min_length=1, max_length=4000)


class ChatRequest(BaseModel):
    """POST /chat request body."""

    messages: list[ChatMessage] = Field(min_length=1)

    @field_validator("messages")
    @classmethod
    def last_message_must_be_user(cls, v: list[ChatMessage]) -> list[ChatMessage]:
        if v[-1].role != Role.user:
            raise ValueError("The last message in the conversation must be from the user.")
        return v


class Recommendation(BaseModel):
    """A single recommended assessment. Exactly the three fields required."""

    name: str
    url: str
    test_type: str

    model_config = {"extra": "forbid"}


class ChatResponse(BaseModel):
    """POST /chat response body - schema is fixed by the assignment spec."""

    reply: str
    recommendations: list[Recommendation] = Field(default_factory=list)
    end_of_conversation: bool = False

    model_config = {"extra": "forbid"}


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"
