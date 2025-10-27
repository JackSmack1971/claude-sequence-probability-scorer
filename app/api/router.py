"""API router composition."""
from fastapi import APIRouter

from app.api.endpoints import score

api_v1_router = APIRouter(prefix="/v1")
api_v1_router.include_router(score.router, tags=["score"])

__all__ = ["api_v1_router"]
