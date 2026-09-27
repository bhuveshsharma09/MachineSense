# MachineSense backend (Stages App-1, App-2a)

A FastAPI service that serves `ml.inference.diagnose()` from the **approved, hash-verified model bundle** as the evidence
record in `backend/schemas.py`, plus measured evidence views computed with existing `ml` functions. Research
demonstration; development results; not validated for real machines. CPU only; no language model and no API key.

## Paths and settings
Every path is an environment variable with a default relative to `SystemCode/` (`backend/settings.py`):

| Setting | Env var | Default |
| --- | --- | --- |
| approved model bundle (never rebuilt here) | `MACHINESENSE_BUNDLE_DIR` | `SystemCode/artifacts/bundle` |
| case history (SQLite; survives restarts) | `MACHINESENSE_CASES_DB` | `SystemCode/backend/data/cases.db` |
| uploaded arrays (for the evidence views) | `MACHINESENSE_UPLOADS_DIR` | `SystemCode/backend/data/uploads` |
| reserved-recording fingerprint index | `MACHINESENSE_GUARD_INDEX` | `SystemCode/backend/data/reserved_fingerprints.json` |
| showcase list (committed) | `MACHINESENSE_SHOWCASE` | `SystemCode/backend/showcase.json` |
| raw development recordings | fixed by `ml.paths.data_dir()` | `SystemCode/data/raw_npy` |
| fail (not skip) tests without the bundle | `MACHINESENSE_REQUIRE_BUNDLE=1` | off |

`SystemCode/backend/data/` is gitignored. The `app-skeleton` branch will be merged into `main` at a stable point, so a
single checkout holds the bundle, the data and the app; until then, a worktree needs `MACHINESENSE_BUNDLE_DIR` and
links for `SystemCode/data/` pointing at the main checkout.

## Run
From `SystemCode/` with the project venv:

1. API: `uvicorn backend.app:create_app --factory --port 8000`. At startup the bundle is verified with
   `ml.bundle.load_bundle()` (every file hash and the packaged-config hash); on any mismatch the server refuses to start.
   On the first start the reserved-fingerprint index is built read-only from `data/raw_npy` (~15 s), or run
   `python -m backend.guard build` ahead of time.
2. Frontend: `cd frontend && npm ci && npm run dev` → http://localhost:5173 (Vite proxies `/api` to port 8000).
3. Showcase (optional; the committed `showcase.json` is used): `python -m backend.showcase build` sweeps `diagnose()`
   over the 804 development-pool recordings (~15 min) and picks one real recording per app state, with the reason.

## Endpoints
- `GET /api/health`, `GET /api/recordings` (development pool only), `GET /api/showcase`.
- `POST /api/diagnose`: `{"rec_id": ...}` or a multipart CSV upload in field `file` (headerless, 8 columns,
  250,000 rows = 5 s at 50 kHz). 422 with a clear message for invalid uploads; 403 for reserved rec_ids and for
  uploads that match a reserved recording's fingerprint.
- `GET /api/cases`, `GET /api/cases/{id}`, `GET /api/cases/{id}/activity` (the deterministic pipeline steps with their
  measured timings), `GET /api/cases/{id}/report` (fixed template, markdown plus sections tagged Measured / Model /
  Guidance / Missing; no language model).
- `GET /api/cases/{id}/channels`: per accelerometer axis and the microphone, R1 feature values (mean over the 9
  windows: RMS, kurtosis, 1× and 2× order amplitudes, envelope lines at BPFO / BPFI / 2×BSF / FTF on the radial axes)
  and the same values for the healthy **training** recording nearest in measured speed (named). The 1–2 channels with
  the largest change factor are marked: descriptive, not a calibrated anomaly test. Per-channel autoencoder error is
  not used as channel evidence.
- `GET /api/cases/{id}/signals`: per channel, a 0.5 s cleaned waveform (min/max per bucket) and the order spectrum
  (`ml.features.order_spectrum`) with physics cursors at the measured speed.

## Upload guard (a guard, not a guarantee)
After an upload is parsed, its fingerprint (sha256 of the float32 array; and a signature of the first 1,000 samples
rounded to the dataset's 5 significant figures, confirmed by per-channel mean and standard deviation) is compared with
an index of the 1,147 reserved test recordings. A match is refused with 403 "this appears to be a reserved test
recording". It catches reserved recordings uploaded as-is or re-exported at CSV precision; a modified copy (cropped,
resampled, rescaled, noise added) is not recognised.

## Tests
`MACHINESENSE_REQUIRE_BUNDLE=1 python -m pytest tests/backend` — use this mode before every commit: without the
bundle or the raw recordings the bundle-dependent tests fail instead of skipping. Without the variable they skip.
