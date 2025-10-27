"""FastAPI application entry point."""
from fastapi import FastAPI

from app.api.router import api_router

app = FastAPI(title="sequence_scorer (OpenRouter)", version="1.0.0")


@app.get("/health")
async def health() -> dict[str, str]:
    """Return service health information."""

    return {"status": "ok"}


# Mount versioned API routers.
app.include_router(api_router)
