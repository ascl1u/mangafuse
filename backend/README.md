# MangaFuse Backend — Phase 1 (Backend, DB & Worker) Verification

This guide documents how to set up, run, and verify the backend implementation on Windows (PowerShell). It covers the API service and an end-to-end job flow.

## Prerequisites
- Python 3.11+
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

# Translation API key (Gemini)
# GOOGLE_API_KEY=

# Database URL (PostgreSQL)
# Either of the following env names are accepted:
# - DATABASE_URL=postgresql+psycopg://USER:PASS@HOST:5432/DBNAME
# If you omit "+psycopg" (e.g., postgresql://...), it will be coerced at runtime.
DATABASE_URL=postgresql+psycopg://postgres:postgres@127.0.0.1:5432/mangafuse
```
Notes:
- The app will automatically load `backend/.env` (see `app/core/config.py`).

## 3) (Removed) Redis and Celery
Redis and Celery were removed in favor of a GPU inference service. See `plan.md` and `app/core/gpu_client.py`.

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

## 7) GPU service
Provision the GPU service separately (see repository Dockerfile and `app/gpu_service/main.py`). The backend submits jobs and receives a webhook callback upon completion.

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

- Readiness
```powershell
irm http://127.0.0.1:8000/api/v1/readyz
# Expected: { "status": "ready" } when DB is reachable and GPU base URL is configured
# If dependencies are down or misconfigured: HTTP 503
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

# Apply edits (persist + re-typeset)
$edits = @{ edits = @(@{ id = 1; en_text = 'Hello!'; font_size = 18 }) } | ConvertTo-Json
$resp2 = iwr -UseBasicParsing -Method POST -ContentType 'application/json' -Body $edits "http://127.0.0.1:8000/api/v1/jobs/$taskId/edits"

# Fetch export URLs
irm "http://127.0.0.1:8000/api/v1/jobs/$taskId/exports"

# Download packaged artifacts (zip)
irm -OutFile "mangafuse_$taskId.zip" "http://127.0.0.1:8000/api/v1/jobs/$taskId/download"
```

## Backend Checklist
- API serves:
  - `GET /api/v1/` → Hello World
  - `GET /api/v1/healthz` → liveness
  - `GET /api/v1/readyz` → readiness (200 when DB reachable and GPU base URL configured)
  - `GET /api/v1/dbz` → database readiness (200 when DB reachable, 503 otherwise)
- Job flow uses GPU service submission and webhook callback; CPU backend performs translation + typesetting.
- Database is provisioned and reachable; Alembic migrations have been applied.
- Configuration via `.env` is documented (this file).

## Troubleshooting
- Readiness returns 503:
  - Ensure the database is reachable and GPU base URL is set.
- Windows issues:
  - Ensure `uvicorn` is started as described; consult logs for errors.

## Clean up
```powershell
# If you used Dockerized Postgres
docker stop mangafuse-postgres
docker rm mangafuse-postgres
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

