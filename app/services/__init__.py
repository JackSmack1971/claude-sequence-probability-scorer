"""Service layer modules."""

from .scorer import (
    build_prompt_from_messages,
    call_openrouter_chat_generate,
    call_openrouter_completions_echo,
    score_candidate_chat_regenerate,
    score_candidate_echo,
    score_candidates,
    sum_logprobs_for_response_segment,
)

__all__ = [
    "build_prompt_from_messages",
    "call_openrouter_chat_generate",
    "call_openrouter_completions_echo",
    "score_candidate_chat_regenerate",
    "score_candidate_echo",
    "score_candidates",
    "sum_logprobs_for_response_segment",
]
