"""Tests for application settings configuration loading behavior."""

from pathlib import Path
import sys

# Ensure the application package is importable when tests run from the repository root.
sys.path.append(str(Path(__file__).resolve().parents[1]))

import pytest
from pydantic import ValidationError

from app.core.config import Settings


@pytest.fixture(autouse=True)
def _clear_settings_env(monkeypatch):
    """Ensure configuration-related environment variables are isolated per test."""

    for key in {
        "OPENROUTER_API_KEY",
        "OPENROUTER_BASE",
        "HTTP_TIMEOUT",
        "OPENROUTER_APP_NAME",
        "OPENROUTER_APP_URL",
    }:
        monkeypatch.delenv(key, raising=False)


def test_settings_apply_defaults_when_overrides_absent(monkeypatch):
    """Settings should fall back to default values when optional overrides are unset."""

    monkeypatch.setenv("OPENROUTER_API_KEY", "test-secret")

    settings = Settings()

    assert settings.openrouter_base == "https://openrouter.ai/api/v1"
    assert settings.http_timeout == pytest.approx(60.0)
    assert settings.app_name == "sequence_scorer"
    assert settings.app_url is None


def test_settings_respect_environment_overrides(monkeypatch):
    """Environment variables must override defaults for all configurable fields."""

    monkeypatch.setenv("OPENROUTER_API_KEY", "override-secret")
    monkeypatch.setenv("OPENROUTER_BASE", "https://override.example")
    monkeypatch.setenv("HTTP_TIMEOUT", "15.5")
    monkeypatch.setenv("OPENROUTER_APP_NAME", "custom_app")
    monkeypatch.setenv("OPENROUTER_APP_URL", "https://app.example")

    settings = Settings()

    assert settings.openrouter_api_key.get_secret_value() == "override-secret"
    assert settings.openrouter_base == "https://override.example"
    assert settings.http_timeout == pytest.approx(15.5)
    assert settings.app_name == "custom_app"
    assert settings.app_url == "https://app.example"


def test_missing_api_key_raises_validation_error():
    """The API key is required and should raise a validation error when absent."""

    with pytest.raises(ValidationError):
        Settings()


def test_env_file_values_used_when_provided(tmp_path: Path):
    """Values from a supplied .env file should populate the settings object."""

    env_file = tmp_path / ".env"
    env_file.write_text(
        "\n".join(
            [
                "OPENROUTER_API_KEY=file-secret",
                "OPENROUTER_BASE=https://file.example",
                "HTTP_TIMEOUT=42.1",
                "OPENROUTER_APP_NAME=file_app",
                "OPENROUTER_APP_URL=https://file.example/url",
            ]
        ),
        encoding="utf-8",
    )

    settings = Settings(_env_file=env_file)

    assert settings.openrouter_api_key.get_secret_value() == "file-secret"
    assert settings.openrouter_base == "https://file.example"
    assert settings.http_timeout == pytest.approx(42.1)
    assert settings.app_name == "file_app"
    assert settings.app_url == "https://file.example/url"


def test_environment_has_priority_over_env_file(monkeypatch, tmp_path: Path):
    """Environment variables should take precedence over values in the .env file."""

    env_file = tmp_path / ".env"
    env_file.write_text(
        "\n".join(
            [
                "OPENROUTER_API_KEY=file-secret",
                "OPENROUTER_BASE=https://file.example",
            ]
        ),
        encoding="utf-8",
    )

    monkeypatch.setenv("OPENROUTER_API_KEY", "env-secret")
    monkeypatch.setenv("OPENROUTER_BASE", "https://env.example")

    settings = Settings(_env_file=env_file)

    assert settings.openrouter_api_key.get_secret_value() == "env-secret"
    assert settings.openrouter_base == "https://env.example"
