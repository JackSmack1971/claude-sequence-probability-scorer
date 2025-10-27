import math
import os
from typing import List, Literal, Optional

import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
if not OPENROUTER_API_KEY:
    raise RuntimeError("Missing OPENROUTER_API_KEY env var")

OPENROUTER_BASE = "https://openrouter.ai/api/v1"
HTTP_TIMEOUT = 60.0

# ---------- Request / Response Schemas ----------

class Msg(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: str

class PromptContext(BaseModel):
    system: Optional[str] = None
    messages: List[Msg]

class Candidate(BaseModel):
    id: str
    text: str
    model_for_scoring: str = Field(..., description="e.g., openai/gpt-4o-mini, mistralai/mixtral-8x7b-instruct")
    tokenizer: Optional[str] = None

class ScoreMode(BaseModel):
    mode: Literal["echo_completions", "chat_regenerate"] = "echo_completions"

class ScoreRequest(BaseModel):
    prompt_context: PromptContext
    candidates: List[Candidate]
    return_top_logprobs: int = Field(0, ge=0, le=20)
    scoring: ScoreMode = ScoreMode()

class ScoreResult(BaseModel):
    id: str
    sequence_logprob: float
    sequence_probability: float
    avg_logprob: float
    token_count: int
    model: str
    tokenizer: Optional[str] = None
    scoring_mode: str
    notes: Optional[str] = None

class ScoreResponse(BaseModel):
    results: List[ScoreResult]

# ---------- Utilities ----------

def build_prompt_from_messages(ctx: PromptContext, candidate_text: str) -> str:
    parts = []
    if ctx.system:
        parts.append(f"<|system|>\n{ctx.system.strip()}\n")
    for m in ctx.messages:
        role_tag = m.role
        parts.append(f"<|{role_tag}|>\n{m.content.strip()}\n")
    parts.append(f"<|assistant|>\n{candidate_text.strip()}\n")
    return "\n".join(parts)

def sum_logprobs_for_response_segment(tokens, token_logprobs, response_start_idx: int):
    seq_logs = token_logprobs[response_start_idx:]
    filtered = [lp for lp in seq_logs if lp is not None]
    if not filtered:
        return float("-inf"), 0
    return float(sum(filtered)), len(filtered)

async def call_openrouter_completions_echo(model: str, prompt: str, top_logprobs: int):
    url = f"{OPENROUTER_BASE}/completions"
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": os.getenv("OPENROUTER_APP_URL", ""),
        "X-Title": os.getenv("OPENROUTER_APP_NAME", "sequence_scorer"),
    }
    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": 0,
        "echo": True,
        "logprobs": max(1, int(top_logprobs)),
        "temperature": 0,
    }
    async with httpx.AsyncClient(timeout=HTTP_TIMEOUT) as client:
        r = await client.post(url, headers=headers, json=payload)
        if r.status_code >= 400:
            raise HTTPException(r.status_code, f"OpenRouter /completions error: {r.text}")
        return r.json()

async def call_openrouter_chat_generate(model: str, messages: List[Msg], top_logprobs: int):
    url = f"{OPENROUTER_BASE}/chat/completions"
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": os.getenv("OPENROUTER_APP_URL", ""),
        "X-Title": os.getenv("OPENROUTER_APP_NAME", "sequence_scorer"),
    }
    chat_messages = [{"role": m.role, "content": m.content} for m in messages]
    payload = {
        "model": model,
        "messages": chat_messages,
        "logprobs": True,
        "top_logprobs": int(top_logprobs),
        "temperature": 0,
        "max_tokens": 1024,
    }
    async with httpx.AsyncClient(timeout=HTTP_TIMEOUT) as client:
        r = await client.post(url, headers=headers, json=payload)
        if r.status_code >= 400:
            raise HTTPException(r.status_code, f"OpenRouter /chat/completions error: {r.text}")
        return r.json()

# ---------- FastAPI app ----------

app = FastAPI(title="sequence_scorer (OpenRouter)", version="1.0.0")

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.post("/score", response_model=ScoreResponse)
async def score(req: ScoreRequest):
    results: List[ScoreResult] = []

    for cand in req.candidates:
        model = cand.model_for_scoring
        scoring_mode = req.scoring.mode

        if scoring_mode == "echo_completions":
            # Context-only token count
            ctx_only_prompt = build_prompt_from_messages(
                PromptContext(system=req.prompt_context.system, messages=req.prompt_context.messages),
                candidate_text=""
            )
            ctx_echo = await call_openrouter_completions_echo(model, ctx_only_prompt, req.return_top_logprobs)
            try:
                choice = ctx_echo["choices"][0]
                logprobs_obj = choice.get("logprobs", {})
                tokens_ctx = logprobs_obj.get("tokens") or [x.get("token") for x in logprobs_obj.get("content", [])]
            except Exception as e:
                raise HTTPException(500, f"Cannot parse echo (context): {e}")
            context_token_count = len(tokens_ctx)

            # Full echo (context + candidate)
            flat_prompt = build_prompt_from_messages(req.prompt_context, cand.text)
            full_echo = await call_openrouter_completions_echo(model, flat_prompt, req.return_top_logprobs)
            try:
                choice_full = full_echo["choices"][0]
                lp_full = choice_full.get("logprobs", {})
                tokens_full = lp_full.get("tokens") or [x.get("token") for x in lp_full.get("content", [])]
                token_logprobs_full = lp_full.get("token_logprobs") or [x.get("logprob") for x in lp_full.get("content", [])]
            except Exception as e:
                raise HTTPException(500, f"Cannot parse echo (full): {e}")

            seq_logprob, seg_len = sum_logprobs_for_response_segment(tokens_full, token_logprobs_full, context_token_count)
            seq_prob = math.exp(seq_logprob) if math.isfinite(seq_logprob) else 0.0
            avg_lp = (seq_logprob / seg_len) if seg_len > 0 else float("-inf")

            results.append(ScoreResult(
                id=cand.id,
                sequence_logprob=seq_logprob,
                sequence_probability=seq_prob,
                avg_logprob=avg_lp,
                token_count=seg_len,
                model=model,
                tokenizer=cand.tokenizer,
                scoring_mode=scoring_mode,
                notes=None
            ))

        else:  # chat_regenerate
            base_msgs = req.prompt_context.messages.copy()
            regen_msgs = base_msgs + [Msg(role="user", content=f"Repeat exactly the following text:\n\n{cand.text}")]
            raw = await call_openrouter_chat_generate(model, regen_msgs, req.return_top_logprobs)
            try:
                ch = raw["choices"][0]
                generated = ch["message"]["content"]
                lp = ch.get("logprobs", {})
                content_items = lp.get("content") or []
                token_logprobs = [x.get("logprob") for x in content_items if x.get("logprob") is not None]
                seq_logprob = float(sum(token_logprobs)) if token_logprobs else float("-inf")
                seq_prob = math.exp(seq_logprob) if math.isfinite(seq_logprob) else 0.0
                avg_lp = (seq_logprob / len(token_logprobs)) if token_logprobs else float("-inf")
                note = None
                if generated.strip() != cand.text.strip():
                    note = "approximate: regenerated text diverged"
                results.append(ScoreResult(
                    id=cand.id,
                    sequence_logprob=seq_logprob,
                    sequence_probability=seq_prob,
                    avg_logprob=avg_lp,
                    token_count=len(token_logprobs),
                    model=model,
                    tokenizer=cand.tokenizer,
                    scoring_mode="chat_regenerate",
                    notes=note
                ))
            except Exception as e:
                raise HTTPException(500, f"Cannot parse chat logprobs: {e}")

    return ScoreResponse(results=results)
