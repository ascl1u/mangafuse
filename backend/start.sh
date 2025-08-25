#!/usr/bin/env bash
set -Eeuo pipefail

: "${PORT:=8000}"
: "${LOG_LEVEL:=info}"
: "${UVICORN_WORKERS:=1}"

echo "Starting Uvicorn (workers=$UVICORN_WORKERS)"

uvicorn app.main:app \
  --host 0.0.0.0 \
  --port "$PORT" \
  --workers "$UVICORN_WORKERS" \
  --log-level "$LOG_LEVEL"


