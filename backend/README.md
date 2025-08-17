# MangaFuse Backend — Phase 1 (Backend, DB & Worker) Verification

This guide documents how to set up, run, and verify the Phase 1 backend implementation on Windows (PowerShell). It covers the API service, Redis, Celery worker, and an end-to-end task flow.

## Prerequisites
- Python 3.11+
- Docker Desktop (for Redis)
- PostgreSQL (DB URL available, or run via Docker)
- PowerShell

## 1) Clone and create a virtual environment
```powershell
cd backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -U pip
pip install -r requirements.txt
```

## 2) Configure environment
Create `backend/.env` with the following contents:
```dotenv
# backend/.env
APP_ENV=development
LOG_LEVEL=info

# Redis URL used for both Celery broker and result backend in Phase 1
REDIS_URL=redis://127.0.0.1:6379/0

# Celery task limit (seconds)
CELERY_TASK_TIME_LIMIT=120

# Phase 2 (local AI script) — set this for translation stage
# GOOGLE_API_KEY=

# Database URL (PostgreSQL)
# Either of the following env names are accepted:
# - DATABASE_URL=postgresql+psycopg://USER:PASS@HOST:5432/DBNAME
# If you omit "+psycopg" (e.g., postgresql://...), it will be coerced at runtime.
DATABASE_URL=postgresql+psycopg://postgres:postgres@127.0.0.1:5432/mangafuse
```
Notes:
- The app will automatically load `backend/.env` (see `app/core/config.py`).
- You may also set `CELERY_BROKER_URL` and `CELERY_RESULT_BACKEND`, but by default they inherit from `REDIS_URL`.

## 3) Start Redis (Docker)
```powershell
docker run --name mangafuse-redis -p 6379:6379 -d redis:7-alpine
```
Verify Redis is running:
```powershell
docker ps | Select-String mangafuse-redis
```

## 4) (Optional) Start PostgreSQL via Docker
If you don't have a PostgreSQL instance already, you can run one locally:
```powershell
docker run --name mangafuse-postgres -e POSTGRES_PASSWORD=postgres -e POSTGRES_USER=postgres -e POSTGRES_DB=mangafuse -p 5432:5432 -d postgres:16-alpine
```

## 5) Apply database migrations
From the `backend/` directory:
```powershell
alembic -c alembic.ini upgrade head
```
This creates the `users`, `projects`, and `project_artifacts` tables.

## 6) Start the FastAPI server
In a new PowerShell window (keep the venv active):
```powershell
cd backend
.\.venv\Scripts\Activate.ps1
uvicorn app.main:app --reload
```
The server should be available at `http://127.0.0.1:8000` and docs at `http://127.0.0.1:8000/docs`.

## 7) Start the Celery worker
In another PowerShell window (keep the venv active):
```powershell
cd backend
.\.venv\Scripts\Activate.ps1
celery -A app.worker.celery_app.celery_app worker --pool=solo --concurrency=1 --loglevel=INFO
```
Notes:
- `--pool=solo` ensures compatibility on Windows.
- Logs are JSON-formatted for readability and future ingest.

## 8) Verify endpoints (liveness/readiness/hello/database)
Use either the browser or PowerShell. Examples below use PowerShell.

- Hello World
```powershell
irm http://127.0.0.1:8000/api/v1/
# Expected: { "message": "Hello World" }
```

- Liveness
```powershell
irm http://127.0.0.1:8000/api/v1/healthz
# Expected: { "status": "ok" }
```

- Readiness (requires Redis)
```powershell
irm http://127.0.0.1:8000/api/v1/readyz
# Expected: { "status": "ready" } when Redis is running and REDIS_URL is set
# If Redis is down or misconfigured: HTTP 503
```

- Database Readiness (requires DATABASE_URL and a reachable DB)
```powershell
irm http://127.0.0.1:8000/api/v1/dbz
# Expected: { "status": "ready" } when the DB is reachable
# If DB is down or misconfigured: HTTP 503
```

## 9) End-to-end job flow (upload → process → edits → exports)

Start a processing job by uploading an image using `multipart/form-data` and optional form fields.

```powershell
# Upload an image (depth: cleaned|full)
$form = @{
  file = Get-Item ..\assets\samples\test_page.jpg
  depth = 'full'
  debug = 'false'
  force = 'false'
}
$resp = iwr -UseBasicParsing -Method POST -Form $form http://127.0.0.1:8000/api/v1/process
$taskId = ($resp.Content | ConvertFrom-Json).task_id
"Task ID: $taskId"

# Poll until done
while ($true) {
  $r = irm "http://127.0.0.1:8000/api/v1/process/$taskId"
  $r
  if ($r.state -eq 'SUCCESS' -or $r.state -eq 'FAILURE') { break }
  Start-Sleep -Seconds 1
}

# Apply edits (persist + re-typeset in background)
$edits = @{ edits = @(@{ id = 1; en_text = 'Hello!'; font_size = 18 }) } | ConvertTo-Json
$resp2 = iwr -UseBasicParsing -Method POST -ContentType 'application/json' -Body $edits "http://127.0.0.1:8000/api/v1/jobs/$taskId/edits"
$editsTask = ($resp2.Content | ConvertFrom-Json).task_id

# Poll edits task if desired
while ($true) {
  $r = irm "http://127.0.0.1:8000/api/v1/process/$editsTask"
  $r
  if ($r.state -eq 'SUCCESS' -or $r.state -eq 'FAILURE') { break }
  Start-Sleep -Seconds 1
}

# Fetch export URLs
irm "http://127.0.0.1:8000/api/v1/jobs/$taskId/exports"

# Download packaged artifacts (zip)
irm -OutFile "mangafuse_$taskId.zip" "http://127.0.0.1:8000/api/v1/jobs/$taskId/download"
```

## Phase 1 Exit Criteria (Checklist)
- API serves:
  - `GET /api/v1/` → Hello World
  - `GET /api/v1/healthz` → liveness
  - `GET /api/v1/readyz` → readiness (200 when Redis reachable, 503 otherwise)
  - `GET /api/v1/dbz` → database readiness (200 when DB reachable, 503 otherwise)
- `POST /api/v1/process` → enqueues Celery job from an uploaded image, returns `{ task_id }`
- `POST /api/v1/jobs/{task_id}/edits` → persists edits and enqueues re-typeset
- `GET /api/v1/jobs/{task_id}/exports` → returns artifact URLs (final image, text layer)
  - `GET /api/v1/process/{task_id}` → polls task state/result
- Celery worker processes jobs and returns results.
- Redis runs locally via Docker.
- Database is provisioned and reachable; Alembic migrations have been applied.
- Configuration via `.env` is documented (this file).

If all the above pass, Phase 1 is complete and you can proceed to Phase 2 in `../roadmap.md`.

## Troubleshooting
- Readiness returns 503:
  - Ensure Redis is running (`docker ps`).
  - Confirm `REDIS_URL` is set in `backend/.env`.
- Celery connects to wrong broker:
  - Ensure `.env` is loaded and `REDIS_URL` is correct.
  - Worker start logs include the broker/backend in JSON; verify values.
- Windows worker issues:
  - Use `--pool=solo --concurrency=1` as shown above.

## Clean up
```powershell
docker stop mangafuse-redis
docker rm mangafuse-redis
```


## Phase 2 Step 2.0 — Local AI Setup (no code execution here)

This repository now includes placeholders and a separate dependency file for the AI pipeline per `plan.md` Step 2.0. Do not mix these into the Phase 1 runtime environment.

What was added:
- `backend/requirements-ai.txt` — pinned AI deps (CPU-only PyTorch, Ultralytics, manga-ocr, simple-lama-inpainting, google-genai)
- `backend/scripts/` — location for the pipeline script `mangafuse_pipeline.py` (to be implemented in later steps)
- Directory markers to keep structure under version control: `assets/`, `artifacts/`, and subfolders

Local setup guide (PowerShell, summarized; see `plan.md` for exact commands):
1. Create and activate a separate venv, then install AI deps:
   - `cd backend`
   - `python -m venv .venv-ai`
   - `.\.venv-ai\Scripts\Activate.ps1`
   - `pip install -U pip`
   - `pip install --index-url https://download.pytorch.org/whl/cpu -r requirements-ai.txt`
2. Place assets:
   - Fonts: `assets/fonts/animeace2_reg.ttf`
   - Model: `assets/models/speech_bubble_yolov8m_seg.pt`
   - Samples: `assets/samples/test_page.jpg`
3. Create `backend/.env` (use `GOOGLE_API_KEY=...` for translation).
4. Outputs will go to `artifacts/` when the pipeline is implemented in later steps.

Note: We are not executing any commands here. This section documents the expected local setup only, per Step 2.0.

