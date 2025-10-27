"""API endpoints for scoring candidates."""
from fastapi import APIRouter, Depends

from app.core.config import Settings, get_settings
from app.models.request import ScoreRequest
from app.models.response import ScoreResponse
from app.services.scorer import score_candidates

router = APIRouter()


@router.post("/score", response_model=ScoreResponse)
async def score(
    request: ScoreRequest, settings: Settings = Depends(get_settings)
) -> ScoreResponse:
    """Score candidate responses using the configured OpenRouter models."""

    results = await score_candidates(request, settings)
    return ScoreResponse(results=results)
