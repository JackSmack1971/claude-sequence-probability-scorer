# claude-sequence-probability-scorer

FastAPI tool server that computes **sequence probabilities** (and log-probabilities) for candidate responses using **OpenRouter**.
Designed to be registered as a **Claude tool** to enable **tail sampling** (e.g., keep only responses with `P(r) < 0.10`).

## Features
- `POST /score` — batch-score candidate responses w/ `sequence_logprob`, `sequence_probability`, `avg_logprob`, `token_count`.
- Two scoring modes:
  - **echo_completions** (preferred): uses OpenRouter **Completions** with `echo: true` to get logprobs for prompt tokens and sum over the assistant segment.
  - **chat_regenerate** (fallback): uses OpenRouter **Chat Completions** with `logprobs: true` and asks the model to repeat the candidate.
- `GET /health` — simple healthcheck.

## Quickstart

```bash
# 1) Install
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2) Configure env
cp .env.example .env
# edit .env and set OPENROUTER_API_KEY

# 3) Run
uvicorn server:app --host 0.0.0.0 --port 8000

# 4) Smoke test
bash examples/curl_smoketest.sh
```

## Environment

Create `.env` from the example and fill your keys:
```
OPENROUTER_API_KEY=sk-or-...
OPENROUTER_APP_URL=https://yourapp.example.com
OPENROUTER_APP_NAME=sequence_scorer
```

## Claude tool registration

See `examples/claude_messages_request.json` for a minimal Messages request registering the tool and prompting Claude to call it.

## Notes
- Provider responses vary; this server parses both normalized `tokens`/`token_logprobs` arrays and the `content[]` structure.
- Use `echo_completions` whenever supported by the chosen model on OpenRouter for exact conditional likelihood of the response segment.
- For long candidates, prefer inspecting `avg_logprob` alongside the full sequence probability.

## License
MIT
