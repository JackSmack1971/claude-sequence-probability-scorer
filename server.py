"""Compatibility module for legacy imports.

Prefer running the FastAPI app via ``uvicorn app.main:app``.
"""
from app.main import app

__all__ = ["app"]
