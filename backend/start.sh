#!/usr/bin/env bash
set -Eeuo pipefail

: "${PORT:=8000}"
: "${LOG_LEVEL:=info}"
: "${UVICORN_WORKERS:=1}"
: "${CELERY_CONCURRENCY:=1}"
: "${CELERY_PREFETCH_MULTIPLIER:=1}"

echo "Starting Uvicorn (workers=$UVICORN_WORKERS) and Celery (concurrency=$CELERY_CONCURRENCY)"

uvicorn app.main:app \
  --host 0.0.0.0 \
  --port "$PORT" \
  --workers "$UVICORN_WORKERS" \
  --log-level "$LOG_LEVEL" &
WEB_PID=$!

celery -A app.worker.celery_app worker \
  --loglevel="$LOG_LEVEL" \
  --concurrency="$CELERY_CONCURRENCY" \
  --prefetch-multiplier="$CELERY_PREFETCH_MULTIPLIER" &
WORKER_PID=$!

term_handler() {
  echo "Shutting down..."
  kill -TERM "$WEB_PID" "$WORKER_PID" 2>/dev/null || true
  wait "$WEB_PID" "$WORKER_PID" || true
}
trap term_handler SIGTERM SIGINT

wait -n "$WEB_PID" "$WORKER_PID"
kill -TERM "$WEB_PID" "$WORKER_PID" 2>/dev/null || true
wait || true


