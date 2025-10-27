"""API router composition."""
from fastapi import APIRouter

from app.api.endpoints import score

api_router = APIRouter()
api_router.include_router(score.router)

__all__ = ["api_router"]
