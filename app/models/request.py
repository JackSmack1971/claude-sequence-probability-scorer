"""Request models for scoring endpoints."""
from typing import List, Literal, Optional

from pydantic import BaseModel, Field


class Msg(BaseModel):
    """Single chat message exchanged with the model."""

    role: Literal["system", "user", "assistant", "tool"]
    content: str


class PromptContext(BaseModel):
    """Conversation context shared across all candidates."""

    system: Optional[str] = None
    messages: List[Msg]


class Candidate(BaseModel):
    """Candidate response to score."""

    id: str
    text: str
    model_for_scoring: str = Field(
        ..., description="e.g., openai/gpt-4o-mini, mistralai/mixtral-8x7b-instruct"
    )
    tokenizer: Optional[str] = None


class ScoreMode(BaseModel):
    """Configuration for selecting the scoring mode."""

    mode: Literal["echo_completions", "chat_regenerate"] = "echo_completions"


class ScoreRequest(BaseModel):
    """Request payload for the /score endpoint."""

    prompt_context: PromptContext
    candidates: List[Candidate]
    return_top_logprobs: int = Field(0, ge=0, le=20)
    scoring: ScoreMode = ScoreMode()
