"""Response models for scoring endpoints."""
from typing import List, Optional

from pydantic import BaseModel


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
