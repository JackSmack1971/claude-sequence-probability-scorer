#!/usr/bin/env bash
set -euo pipefail
curl -s -X POST http://localhost:8000/score \  -H 'Content-Type: application/json' \  -d '{
    "prompt_context": {
      "system": "You are sampling tails.",
      "messages": [{"role":"user","content":"Name a capital of France"}]
    },
    "candidates": [
      {"id":"c1","text":"Paris.","model_for_scoring":"openai/gpt-4o-mini"},
      {"id":"c2","text":"Lyon.","model_for_scoring":"openai/gpt-4o-mini"}
    ],
    "return_top_logprobs": 0,
    "scoring": {"mode":"echo_completions"}
  }' | jq .
