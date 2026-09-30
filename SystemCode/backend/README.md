# MachineSense backend and app (Stages App-1 to App-4.5)

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
| typical healthy variation (built on first start) | `MACHINESENSE_VARIATION` | `SystemCode/backend/data/healthy_variation.json` |
| raw development recordings | fixed by `ml.paths.data_dir()` | `SystemCode/data/raw_npy` |
| fail (not skip) tests without the bundle | `MACHINESENSE_REQUIRE_BUNDLE=1` | off |

`SystemCode/backend/data/` is gitignored. The app lives on `main` (merged from `app-skeleton`), so one checkout holds
the code, the bundle and the data: with the defaults above, no environment variable and no data links are needed.

## Setup (one checkout)
What must exist in `SystemCode/` before the first start (none of it is committed):
- `artifacts/bundle/`: the approved bundle (rebuild order in CLAUDE.md §7; never rebuilt by the app);
- `data/raw_npy/`: the converted recordings (`scripts/00_convert.py`), read for dataset recordings and to build the
  reserved-fingerprint index;
- `.venv/` with `requirements.txt`, and Node 18+ for the frontend.
The frontend needs no internet connection: its fonts are self-hosted (`frontend/src/fonts/`, SIL OFL 1.1).

## Run
From `SystemCode/` with the project venv (`source .venv/bin/activate`):

1. API: `uvicorn backend.app:create_app --factory --port 8000`. At startup the bundle is verified with
   `ml.bundle.load_bundle()` (every file hash and the packaged-config hash); on any mismatch the server refuses to start.
   On the first start the reserved-fingerprint index is built read-only from `data/raw_npy` (~15 s), or run
   `python -m backend.guard build` ahead of time.
2. Frontend: `cd frontend && npm ci && npm run dev` → http://localhost:5173 (Vite proxies `/api` to port 8000).
3. Showcase (optional; the committed `showcase.json` is used): `python -m backend.showcase build` sweeps `diagnose()`
   over the 804 development-pool recordings (~15 min) and picks one real recording per app state, with the reason.

## Endpoints
- `GET /api/health` (the bundle path relative to `SystemCode/`), `GET /api/recordings` (development pool only), `GET /api/showcase`.
- `POST /api/diagnose`: `{"rec_id": ...}` or a multipart CSV upload in field `file` (headerless, 8 columns,
  250,000 rows = 5 s at 50 kHz). 422 with a clear message for invalid uploads; 403 for reserved rec_ids and for
  uploads that match a reserved recording's fingerprint.
- Report and UI share `backend/checks.py`: each check's relation to its threshold (≤ / > and ≥ / <), margin (value −
  threshold, absolute and relative; for MSP relative to 1 − τ_p), "narrow" within 0.1 % (then "narrowly above/below"
  instead of "checks disagree"), and "Checks disagree" only when the rule variants disagree for a non-narrow reason or
  CNN-6 ≠ LR. Scores are never shown as 1.0000 ("> 0.9999").
- `GET /api/cases`, `GET /api/cases/{id}`, `GET /api/cases/{id}/activity` (the deterministic pipeline steps with their
  measured timings), `GET /api/cases/{id}/report` (fixed template, markdown plus sections tagged Measured / Model /
  Guidance / Missing; no language model).
- `GET /api/cases/{id}/channels`: per accelerometer axis and the microphone, R1 feature values (mean over the 9
  windows: RMS, excess kurtosis (normal ≈ 0), 1× and 2× order amplitudes, envelope lines at BPFO / BPFI / 2×BSF / FTF on
  the radial axes) and the same values for the healthy **training** recording nearest in measured speed (named), with
  each metric's direction (up / down), change factor max(ratio, 1/ratio) and typical healthy variation. Descriptive,
  not a calibrated anomaly test; per-channel autoencoder error is not used as channel evidence.
  - **Typical healthy variation** (`backend/variation.py`, `python -m backend.variation build`): the 95th percentile of
    the change factor over all pairs of healthy TRAINING recordings within ±3 Hz (20 recordings, 35 pairs); never
    calibration or reserved recordings. Volt metrics are compared only if at least one side is ≥ 0.005 V (the median
    healthy over-radial 1× amplitude); excess kurtosis is never ratioed.
  - **Marker rule** (Bhuvesh's decision, App-2a.2): no marker on a plain healthy verdict; healthy-with-caution, known
    fault and unknown verdicts mark up to 2 channels beyond their band. Each recording is compared on ~20 channel ×
    metric values, so a healthy recording often exceeds at least one band by chance (2 of the 5 calibration normals do;
    reported descriptively by the tests, the band is not tuned on them).
- `GET /api/cases/{id}/signals`: per channel, a 0.5 s cleaned waveform (min/max per bucket) and the order spectrum
  (`ml.features.order_spectrum`) with physics cursors at the measured speed.
- `GET /api/cases/{id}/explanation` (404 until generated) and `POST /api/cases/{id}/explanation`: the plain-language
  explanation (App-3; a local language model behind guards, or the fixed template).
- `GET /api/cases/{id}/trace`: the read-only pipeline trace of a saved case (below).

## Pipeline trace (schema)
`GET /api/cases/{id}/trace` (`backend/trace.py`), **trace_version "1.1"**. A derived, versioned, read-only view of a saved
case: it never re-runs inference, never changes anything, and **the evidence record (`backend/schemas.py`) is unchanged**.
- `{case_id, trace_version, bundle_matches_record, bundle_note, value_sources, edge_kinds, blocks, edges}`; ten blocks in
  the order of `ml.inference.diagnose` (input, validation, signal_quality, preparation, representations → physics_features
  / order_spectrogram, models → cnn6 / class6 / mahalanobis / ae6 / b1s / lr, decision, evidence_record, explanation,
  presentation), each with `purpose`, `operation` (what the step always does), `operation_case` (wording that depends on
  this case, or null; added in 1.1), `input` / `output` / `technical`, `status`, `timing_ms`, `derived`, `ports_in` /
  `ports_out`; `edges` are the real data dependencies (`from`, `to`, `kind`, `label`).
- Every value carries its source (`record:`, `activity:`, `config:`, `manifest:`, `explanation:`, `store:`, or `derived:`
  for a value computed for this view), or is `{"value": null, "reason": ...}` when it was never recorded. `timing_ms` is
  copied from the stored activity log or null. `bundle_matches_record` is false when the loaded bundle is not the one that
  produced the case; config and manifest values then describe the loaded (current) bundle.
- One example: `frontend/tests/fixtures/trace_known_fault.json`.

## Pipeline page (App-4.2 to App-4.5)
`#/case/<case_id>/pipeline` in the app, reached from a diagnosis by "How this diagnosis was made". Loading it is one
`GET /trace`: it reads the saved case and runs no inference.
- **Two modes.** "This diagnosis" (the default from a case): what happened to this recording, with each step's status as
  a word and a dot, its values, the verdict exactly as saved ("further inspection needed" verbatim; a predicted fault
  marked "model prediction"), the rule comparison as a secondary, read-only box, and "not recorded: <reason>" for any
  null. "General architecture": the same graph with no case values.
- **Two reading levels.** Mechanic: plain-language outcomes, no thresholds, raw scores, hashes or model names. Developer:
  the numbers with their thresholds, each model's aggregation note, provenance, and "computed for this view" on derived
  values. Both show the same verdict and evidence.
- **Previews** (This diagnosis only, fetched only on an explicit "Load" in a step's drawer): the cleaned waveform and the
  order spectrum come from `/signals`, computed on demand and never part of the saved diagnosis (dashed, hatched frame,
  "computed now"); the envelope spectrum is the saved record's (plain frame, "saved evidence").
- **Screens.** A pan-and-zoom canvas from 900 px wide (keyboard: Tab through the steps in order, Enter opens a step's
  drawer, Escape closes it, arrow keys pan, + and − zoom); below 900 px, a stacked list of the steps with the same values
  and drawer content. Error states (unknown case, network failure, a malformed response) show a plain message, never a
  partial diagram.
- **Audit against the real case history:** `cd frontend && npm run audit:cases -- <directory of saved /trace responses>`
  renders every case in both modes, both levels and at phone width, and checks every shown value against its trace.

## Upload guard (a guard, not a guarantee)
After an upload is parsed, its fingerprint (sha256 of the float32 array; and a signature of the first 1,000 samples
rounded to the dataset's 5 significant figures, confirmed by per-channel mean and standard deviation) is compared with
an index of the 1,147 reserved test recordings. A match is refused with 403 "this appears to be a reserved test
recording". It catches reserved recordings uploaded as-is or re-exported at CSV precision; a modified copy (cropped,
resampled, rescaled, noise added) is not recognised.

## Tests
Frontend: `cd frontend && npm test` — production build, then a server-side render of every main screen from recorded real
API responses (`frontend/tests/fixtures`, re-recorded with `PYTHONPATH=. python frontend/tests/make_fixtures.py`).
It also runs the pipeline-page checks (value walker against the trace fixtures, previews, phone width, error states) and
`tests/pipeline-requests.mjs` in headless Chrome (one `/trace` on load, previews only on request, focus and keyboard).
The WebGL fallback frame `frontend/public/rig-fallback.png` (+ marker anchors `.json`) is a rendered frame of the app's
own 3D rig (`?capture=1`), not a photo.

Backend:
`MACHINESENSE_REQUIRE_BUNDLE=1 python -m pytest tests/backend` — use this mode before every commit: without the
bundle or the raw recordings the bundle-dependent tests fail instead of skipping. Without the variable they skip.

Everything, from `SystemCode/` in the main checkout: `MACHINESENSE_REQUIRE_BUNDLE=1 python -m pytest tests` (expected:
no skips; the `ml` bundle tests also read `artifacts/bundle`) and `cd frontend && npm test`.
