# MachineSense backend (Stage App-1)

A FastAPI walking skeleton that serves `ml.inference.diagnose()` from the **approved, hash-verified model bundle**
as the evidence record in `backend/schemas.py`. Research demonstration; development results; not validated for real
machines. CPU only; no language model and no API key.

## Setup, run, test
Run from `SystemCode/` with the project venv.

1. **Bundle path.** Point the backend at the approved bundle (it is never rebuilt here):
   `export MACHINESENSE_BUNDLE_DIR=/path/to/MachineSense-Project/SystemCode/artifacts/bundle`
   (default: `SystemCode/artifacts/bundle`). The bundle is loaded with `ml.bundle.load_bundle()` at startup; every
   file hash and the packaged-config hash are verified, and the server refuses to start on any mismatch.
   Dataset recordings are read from `SystemCode/data/raw_npy/` (in a git worktree, link it to the main checkout's).
2. **API:** `uvicorn backend.app:create_app --factory --port 8000`
   Endpoints: `GET /api/health`, `GET /api/recordings` (development pool only), `POST /api/diagnose`
   (`{"rec_id": ...}` or a multipart CSV upload in field `file`: headerless, 8 columns, 250,000 rows = 5 s at 50 kHz),
   `GET /api/cases/{case_id}`, `GET /api/cases/{case_id}/report` (fixed template, no language model).
3. **Frontend:** `cd frontend && npm ci && npm run dev` → http://localhost:5173 (Vite proxies `/api` to port 8000).
4. **Tests:** `python -m pytest tests/backend` (needs `MACHINESENSE_BUNDLE_DIR`; skipped without the bundle).
