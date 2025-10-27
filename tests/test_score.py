"""Tests for the versioned scoring endpoint."""
from collections.abc import Iterator
from typing import List

import pytest
from fastapi.testclient import TestClient

from app.core.config import get_settings
from app.main import app
from app.models.request import ScoreRequest
from app.models.response import ScoreResult

client = TestClient(app)

EXPECTED_SCORE_REQUEST_EXAMPLE = {
    "prompt_context": {
        "system": "You are a helpful assistant that evaluates responses for accuracy.",
        "messages": [
            {
                "role": "user",
                "content": "Summarize the key points from the following article.",
            },
            {
                "role": "assistant",
                "content": "Sure — please provide the article text so I can summarize it.",
            },
        ],
    },
    "candidates": [
        {
            "id": "candidate-1",
            "text": (
                "The article highlights three main trends: increased AI adoption, emphasis on data "
                "privacy, and the rise of edge computing."
            ),
            "model_for_scoring": "openrouter/gpt-4o-mini",
            "tokenizer": "openrouter/gpt-4o-mini",
        },
        {
            "id": "candidate-2",
            "text": (
                "AI adoption is accelerating, privacy rules are tightening, and companies are pushing "
                "computation closer to users with edge devices."
            ),
            "model_for_scoring": "mistralai/mixtral-8x7b-instruct",
        },
    ],
    "return_top_logprobs": 5,
    "scoring": {"mode": "echo_completions"},
}


@pytest.fixture(autouse=True)
def configure_settings(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Ensure required environment configuration is present for each test."""

    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture()
def sample_payload() -> dict:
    """Return a reusable valid request payload."""

    return {
        "prompt_context": {
            "system": "Score the assistant reply.",
            "messages": [
                {"role": "user", "content": "Provide a haiku about the sea."},
            ],
        },
        "candidates": [
            {
                "id": "cand-1",
                "text": "Ocean whispers soft\nMoonlight dances on the waves\nNight cradles the deep",
                "model_for_scoring": "test/model",
                "tokenizer": "test/tokenizer",
            }
        ],
        "return_top_logprobs": 3,
        "scoring": {"mode": "echo_completions"},
    }


def test_v1_score_endpoint_returns_score(monkeypatch: pytest.MonkeyPatch, sample_payload: dict) -> None:
    """The versioned endpoint should wrap scoring results in the documented schema."""

    async def fake_score_candidates(_: ScoreRequest, __) -> List[ScoreResult]:
        return [
            ScoreResult(
                id="cand-1",
                sequence_logprob=-12.0,
                sequence_probability=6.1e-06,
                avg_logprob=-0.6,
                token_count=20,
                model="test/model",
                tokenizer="test/tokenizer",
                scoring_mode="echo_completions",
                notes=None,
            )
        ]

    monkeypatch.setattr("app.api.endpoints.score.score_candidates", fake_score_candidates)

    response = client.post("/v1/score", json=sample_payload)

    assert response.status_code == 200
    body = response.json()
    assert body["results"][0]["id"] == "cand-1"
    assert body["results"][0]["scoring_mode"] == "echo_completions"


def test_v1_score_endpoint_requires_candidates(sample_payload: dict) -> None:
    """Missing required fields should surface validation errors with field paths."""

    invalid_payload = sample_payload.copy()
    invalid_payload.pop("candidates")

    response = client.post("/v1/score", json=invalid_payload)

    assert response.status_code == 422
    detail = response.json()["detail"][0]
    assert detail["loc"] == ["body", "candidates"]


def test_legacy_score_route_removed(sample_payload: dict) -> None:
    """The legacy /score route should no longer be available."""

    response = client.post("/score", json=sample_payload)

    assert response.status_code == 404


def test_openapi_documents_score_request_schema() -> None:
    """The OpenAPI schema should expose the score request contract and example."""

    response = client.get("/openapi.json")

    assert response.status_code == 200
    schema = response.json()

    score_path = schema["paths"]["/v1/score"]["post"]
    request_ref = score_path["requestBody"]["content"]["application/json"]["schema"]
    assert request_ref == {"$ref": "#/components/schemas/ScoreRequest"}

    score_request_schema = schema["components"]["schemas"]["ScoreRequest"]
    assert score_request_schema["title"] == "ScoreRequest"
    assert score_request_schema["required"] == ["prompt_context", "candidates"]
    assert set(score_request_schema["properties"].keys()) == {
        "prompt_context",
        "candidates",
        "return_top_logprobs",
        "scoring",
    }

    logprob_field = score_request_schema["properties"]["return_top_logprobs"]
    assert logprob_field["minimum"] == 0.0
    assert logprob_field["maximum"] == 20.0
    assert logprob_field["default"] == 0

    assert score_request_schema["example"] == EXPECTED_SCORE_REQUEST_EXAMPLE
