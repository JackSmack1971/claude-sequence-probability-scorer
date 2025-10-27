"""Response models for scoring endpoints."""
from typing import List, Optional

from pydantic import BaseModel, ConfigDict


class ScoreResult(BaseModel):
    """Individual candidate score details."""

    id: str
    sequence_logprob: float
    sequence_probability: float
    avg_logprob: float
    token_count: int
    model: str
    tokenizer: Optional[str] = None
    scoring_mode: str
    notes: Optional[str] = None


class ScoreResponse(BaseModel):
    """Aggregated scoring response."""

    results: List[ScoreResult]

    # Provide an OpenAPI example to document the contract for clients.
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "results": [
                    {
                        "id": "candidate-1",
                        "sequence_logprob": -12.34,
                        "sequence_probability": 4.4e-06,
                        "avg_logprob": -0.61,
                        "token_count": 20,
                        "model": "openrouter/model-name",
                        "tokenizer": "openrouter/tokenizer-name",
                        "scoring_mode": "echo_completions",
                        "notes": None,
                    }
                ]
            }
        }
    )
