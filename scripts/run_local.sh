#!/usr/bin/env bash
set -euo pipefail
export $(grep -v '^#' .env | xargs -d '\n' -0 2>/dev/null || true)
uvicorn app.main:app --host 0.0.0.0 --port 8000
