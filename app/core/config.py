"""Application configuration settings.

This module centralizes environment-driven configuration using pydantic's
settings management so that dependencies can inject configuration at runtime.
"""

from functools import lru_cache
from typing import Optional

from pydantic import AliasChoices, Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings sourced from environment variables or a .env file."""

    openrouter_api_key: SecretStr = Field(
        ..., validation_alias=AliasChoices("OPENROUTER_API_KEY"), description="API key for OpenRouter authentication"
    )
    openrouter_base: str = Field(
        "https://openrouter.ai/api/v1",
        validation_alias=AliasChoices("OPENROUTER_BASE"),
        description="Base URL for OpenRouter API calls",
    )
    http_timeout: float = Field(
        60.0,
        validation_alias=AliasChoices("HTTP_TIMEOUT"),
        description="HTTP client timeout (seconds) for outbound OpenRouter requests",
        ge=0,
    )
    app_name: str = Field(
        "sequence_scorer",
        validation_alias=AliasChoices("OPENROUTER_APP_NAME"),
        description="Application name sent to OpenRouter for observability",
    )
    app_url: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices("OPENROUTER_APP_URL"),
        description="Application URL sent to OpenRouter for observability",
    )

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")


@lru_cache
def get_settings() -> Settings:
    """Return a cached Settings instance for dependency injection usage."""

    return Settings()
