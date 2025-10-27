"""Scoring service containing reusable business logic for the API layer."""
from __future__ import annotations

import math
from typing import List

from fastapi import HTTPException

from app.core.config import Settings
from app.core.http_client import post_openrouter
from app.models.request import Candidate, Msg, PromptContext, ScoreRequest
from app.models.response import ScoreResult


def build_prompt_from_messages(ctx: PromptContext, candidate_text: str) -> str:
    """Create an OpenRouter-friendly prompt from conversation messages."""

    parts: List[str] = []
    if ctx.system:
        parts.append(f"<|system|>\n{ctx.system.strip()}\n")
    for message in ctx.messages:
        parts.append(f"<|{message.role}|>\n{message.content.strip()}\n")
    parts.append(f"<|assistant|>\n{candidate_text.strip()}\n")
    return "\n".join(parts)


def sum_logprobs_for_response_segment(tokens, token_logprobs, response_start_idx: int):
    """Aggregate log probabilities for the assistant response portion of an echo call."""

    seq_logs = token_logprobs[response_start_idx:]
    filtered = [logprob for logprob in seq_logs if logprob is not None]
    if not filtered:
        return float("-inf"), 0
    return float(sum(filtered)), len(filtered)


async def call_openrouter_completions_echo(
    model: str, prompt: str, top_logprobs: int, settings: Settings
):
    """Call the OpenRouter /completions endpoint with echo mode enabled."""

    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": 0,
        "echo": True,
        "logprobs": max(1, int(top_logprobs)),
        "temperature": 0,
    }
    return await post_openrouter("/completions", payload, settings)


async def call_openrouter_chat_generate(
    model: str, messages: List[Msg], top_logprobs: int, settings: Settings
):
    """Call the OpenRouter chat completions endpoint for regeneration scoring."""

    payload = {
        "model": model,
        "messages": [{"role": msg.role, "content": msg.content} for msg in messages],
        "logprobs": True,
        "top_logprobs": int(top_logprobs),
        "temperature": 0,
        "max_tokens": 1024,
    }
    return await post_openrouter("/chat/completions", payload, settings)


async def score_candidate_echo(
    request: ScoreRequest, candidate: Candidate, settings: Settings
) -> ScoreResult:
    """Score a candidate using the echo completions strategy."""

    # Obtain context token count by scoring prompt without the candidate text.
    context_only_prompt = build_prompt_from_messages(
        PromptContext(system=request.prompt_context.system, messages=request.prompt_context.messages),
        candidate_text="",
    )
    context_echo = await call_openrouter_completions_echo(
        candidate.model_for_scoring, context_only_prompt, request.return_top_logprobs, settings
    )
    try:
        context_choice = context_echo["choices"][0]
        context_logprobs = context_choice.get("logprobs", {})
        context_tokens = context_logprobs.get("tokens") or [
            item.get("token") for item in context_logprobs.get("content", [])
        ]
    except Exception as exc:  # noqa: BLE001 - propagate parsing issues as API errors
        raise HTTPException(500, f"Cannot parse echo (context): {exc}") from exc

    context_token_count = len(context_tokens)

    flat_prompt = build_prompt_from_messages(request.prompt_context, candidate.text)
    full_echo = await call_openrouter_completions_echo(
        candidate.model_for_scoring, flat_prompt, request.return_top_logprobs, settings
    )

    try:
        full_choice = full_echo["choices"][0]
        full_logprobs = full_choice.get("logprobs", {})
        tokens_full = full_logprobs.get("tokens") or [
            item.get("token") for item in full_logprobs.get("content", [])
        ]
        token_logprobs_full = full_logprobs.get("token_logprobs") or [
            item.get("logprob") for item in full_logprobs.get("content", [])
        ]
    except Exception as exc:  # noqa: BLE001 - propagate parsing issues as API errors
        raise HTTPException(500, f"Cannot parse echo (full): {exc}") from exc

    seq_logprob, segment_length = sum_logprobs_for_response_segment(
        tokens_full, token_logprobs_full, context_token_count
    )
    seq_probability = math.exp(seq_logprob) if math.isfinite(seq_logprob) else 0.0
    avg_logprob = (seq_logprob / segment_length) if segment_length > 0 else float("-inf")

    return ScoreResult(
        id=candidate.id,
        sequence_logprob=seq_logprob,
        sequence_probability=seq_probability,
        avg_logprob=avg_logprob,
        token_count=segment_length,
        model=candidate.model_for_scoring,
        tokenizer=candidate.tokenizer,
        scoring_mode=request.scoring.mode,
        notes=None,
    )


async def score_candidate_chat_regenerate(
    request: ScoreRequest, candidate: Candidate, settings: Settings
) -> ScoreResult:
    """Score a candidate by regenerating the text through chat completions."""

    regen_messages = request.prompt_context.messages.copy() + [
        Msg(role="user", content=f"Repeat exactly the following text:\n\n{candidate.text}")
    ]
    raw_response = await call_openrouter_chat_generate(
        candidate.model_for_scoring, regen_messages, request.return_top_logprobs, settings
    )

    try:
        choice = raw_response["choices"][0]
        generated_content = choice["message"]["content"]
        logprobs = choice.get("logprobs", {})
        content_items = logprobs.get("content") or []
        token_logprobs = [item.get("logprob") for item in content_items if item.get("logprob") is not None]
    except Exception as exc:  # noqa: BLE001 - propagate parsing issues as API errors
        raise HTTPException(500, f"Cannot parse chat logprobs: {exc}") from exc

    seq_logprob = float(sum(token_logprobs)) if token_logprobs else float("-inf")
    seq_probability = math.exp(seq_logprob) if math.isfinite(seq_logprob) else 0.0
    avg_logprob = (seq_logprob / len(token_logprobs)) if token_logprobs else float("-inf")
    notes = None
    if generated_content.strip() != candidate.text.strip():
        notes = "approximate: regenerated text diverged"

    return ScoreResult(
        id=candidate.id,
        sequence_logprob=seq_logprob,
        sequence_probability=seq_probability,
        avg_logprob=avg_logprob,
        token_count=len(token_logprobs),
        model=candidate.model_for_scoring,
        tokenizer=candidate.tokenizer,
        scoring_mode="chat_regenerate",
        notes=notes,
    )


async def score_candidates(request: ScoreRequest, settings: Settings) -> List[ScoreResult]:
    """Public entry point for routers to score candidates."""

    results: List[ScoreResult] = []
    for candidate in request.candidates:
        if request.scoring.mode == "echo_completions":
            result = await score_candidate_echo(request, candidate, settings)
        else:
            result = await score_candidate_chat_regenerate(request, candidate, settings)
        results.append(result)
    return results
