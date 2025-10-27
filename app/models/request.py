"""Request models for scoring endpoints."""
from typing import List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


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

    # Provide an OpenAPI example to document the contract for clients.
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "prompt_context": {
                    "system": "You are a helpful assistant that evaluates responses for accuracy.",
                    "messages": [
                        {
                            "role": "user",
                            "content": "Summarize the key points from the following article.",
                        },
                        {
                            "role": "assistant",
                            "content": "Sure — please provide the article text so I can summarize it.",
                        },
                    ],
                },
                "candidates": [
                    {
                        "id": "candidate-1",
                        "text": "The article highlights three main trends: increased AI adoption, emphasis on data privacy, and the rise of edge computing.",
                        "model_for_scoring": "openrouter/gpt-4o-mini",
                        "tokenizer": "openrouter/gpt-4o-mini",
                    },
                    {
                        "id": "candidate-2",
                        "text": "AI adoption is accelerating, privacy rules are tightening, and companies are pushing computation closer to users with edge devices.",
                        "model_for_scoring": "mistralai/mixtral-8x7b-instruct",
                        "tokenizer": None,
                    },
                ],
                "return_top_logprobs": 5,
                "scoring": {"mode": "echo_completions"},
            }
        }
    )
