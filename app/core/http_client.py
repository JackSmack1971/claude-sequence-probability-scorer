"""Shared HTTP client helpers for communicating with OpenRouter."""
from __future__ import annotations

from typing import Any, Dict

import httpx
from fastapi import HTTPException

from app.core.config import Settings


async def post_openrouter(path: str, payload: Dict[str, Any], settings: Settings) -> Dict[str, Any]:
    """POST JSON payloads to OpenRouter with consistent headers and error handling."""

    if not path.startswith("/"):
        raise ValueError("OpenRouter path must start with '/' to prevent request forgery.")

    base_url = settings.openrouter_base.rstrip("/")
    url = f"{base_url}{path}"
    headers = {
        # Authorization header uses secret stored in settings.
        "Authorization": f"Bearer {settings.openrouter_api_key.get_secret_value()}",
        "Content-Type": "application/json",
    }
    # Optional metadata headers for OpenRouter attribution.
    if settings.app_url:
        headers["HTTP-Referer"] = settings.app_url
    headers["X-Title"] = settings.app_name

    try:
        async with httpx.AsyncClient(timeout=settings.http_timeout) as client:
            response = await client.post(url, headers=headers, json=payload)
    except httpx.HTTPError as exc:
        # Surface transport issues as HTTP 503 for the API.
        raise HTTPException(503, f"OpenRouter request failed: {exc}") from exc

    if response.status_code >= 400:
        raise HTTPException(response.status_code, f"OpenRouter error: {response.text}")

    try:
        return response.json()
    except ValueError as exc:
        raise HTTPException(502, "Invalid JSON response from OpenRouter") from exc
