# MachineSense backend and app (Stages App-1 to App-7)

A FastAPI service that serves `ml.inference.diagnose()` from the **approved, hash-verified model bundle** as the evidence
record in `backend/schemas.py`, plus measured evidence views computed with existing `ml` functions. Research
demonstration; development results; not validated for real machines. CPU only; no language model and no API key are
needed: the optional explanation (App-3) and assistant (App-7) use a local model when one is reachable and otherwise fall
back to fixed text.

## Paths and settings
Every path is an environment variable with a default relative to `SystemCode/` (`backend/settings.py`):

| Setting | Env var | Default |
| --- | --- | --- |
| approved model bundle (never rebuilt here) | `MACHINESENSE_BUNDLE_DIR` | `SystemCode/artifacts/bundle` |
| case history (SQLite; survives restarts) | `MACHINESENSE_CASES_DB` | `SystemCode/backend/data/cases.db` |
| uploaded arrays (for the evidence views) | `MACHINESENSE_UPLOADS_DIR` | `SystemCode/backend/data/uploads` |
| reserved-recording fingerprint index | `MACHINESENSE_GUARD_INDEX` | `SystemCode/backend/data/reserved_fingerprints.json` |
| showcase list (committed) | `MACHINESENSE_SHOWCASE` | `SystemCode/backend/showcase.json` |
| unseen-fault selection (committed, App-6) | `MACHINESENSE_UNSEEN` | `SystemCode/backend/unseen.json` |
| unseen-fault sweep cache (App-6) | `MACHINESENSE_UNSEEN_SWEEP` | `SystemCode/backend/data/unseen_sweep.json` |
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
- Report section 04 ("What to inspect next", Stage App-5): steps from the curated entries in `knowledge/entries/`,
  selected by `ml.guidance.select` for the verdict (the fault class, `normal`, or `unknown`) and the available evidence
  (the bundle's feature dictionary names, `speed_hz`, `decision.decision`, and `quality.warnings` when the record has a
  warning). One `Guidance` item per step: the text ends "(source: <entry id>)", plus "AI-drafted, not yet checked against
  a published source" or "Citation not yet checked against the full text" for those verification levels; `cite` holds the
  id and `hover` the entry title and full source (and URL). For an unknown verdict `further-inspection-needed` comes
  first. `needs_paywalled_source` stubs are never offered. A step written as `{text, when_warning}` is shown only when one
  of its substrings appears (case-insensitive) in a recorded warning, and an entry left with no steps is not offered, so
  the acquisition entry shows only the steps for the warnings actually recorded (the calibration-speed note has none).
  When nothing is selected: for a healthy verdict one `Model` item, "No inspection steps are offered for a healthy
  verdict."; for any other condition one `Missing` item naming the condition. No generic advice is generated.
  `backend.report.guidance_items` is the only caller of `select`; the Mechanic view shows section 04's items as given,
  and the explanation's `next_step` is the same items, one per line, filled in by code (below).
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
  explanation (App-3; a local language model behind guards, or the fixed template). Its `next_step` is never written by
  the model: the prompt asks for an empty string and `explain_case` sets it from report section 04's items
  (`plain.next_step`), for the model's text and the template alike (App-5 addendum).
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

## Explore an unseen fault (App-6)
A research demonstration page (`#/unseen`, "Explore an unseen fault" in the top bar and the phone header): a saved
**leave-one-class-out CNN-6** (`artifacts/final/loco/<class>/`, seed 42, CLAUDE.md §12), trained with one fault class
withheld, is run on a recording of that class. It is a **simulated unseen fault inside MAFAULDA, not evidence that the
system recognises genuinely novel real-world faults**; the page says so in its own copy. The main app, its bundle, its
scoring and its verdicts are unchanged.
- **Works on a clean clone.** `artifacts/final/` is gitignored, so the page serves the committed selection
  `backend/unseen.json` (every displayed value). When `artifacts/final/` verifies (`ml.final.load_final`, every hash) and
  `data/raw_npy` exists, the API recomputes live (`source: "live"`); otherwise it returns the saved values (`"saved"`).
  `tests/backend/test_app6.py` checks live = saved (1e-9 relative, identical decisions), skipping with the reason when
  the weights or the raw recordings are absent.
- **Build** (where the weights exist): `python -m backend.unseen build` sweeps the 804 eligible recordings through the
  nine models (~17 min; cached in `backend/data/unseen_sweep.json`, gitignored) and writes `backend/unseen.json`: per
  hidden class a successful rejection, an honest failure and a familiar comparison, with reasons. A category with no
  eligible recording is reported in `why_missing`, never substituted (four bearing classes reject every eligible
  withheld recording, so they have no honest failure).
- **Scoring and decision:** `ml.final.recording_inputs` → `ml.final.score_cnn` with that class's CNN-6, normalisation and
  the **shared** Mahalanobis (τ_d is calibrated for it), plus the shared AE-6 and B1s; τ_p, τ_d, τ_e, τ_b from the
  class's `thresholds.json`; the verdict is `ml.arbitrator.decide("B_with_caution")` with `rule_comparison`; the check
  pass / fail is `ml.rejection.failed_checks`. AE-6 and B1s appear only as a caution; they do not gate the verdict.
- **Eligibility, one predicate** (`backend.unseen.may_show`, used by the build and every request): `split_closed` train or
  val, and neither `split_speed_block` nor `split_severity` test (the F1 partitions stay sealed); the withheld example is
  of the hidden class; the familiar example is a learned class and not one of that run's calibration or
  threshold-defining recordings. The eligible examples therefore exclude the largest severities and the fastest speeds:
  the page shows the easier part of the unfamiliar-fault problem.
- **Endpoints:** `GET /api/unseen` (the curated examples, without results, and whether live values are available);
  `GET /api/unseen/run?withheld=<class>&rec_id=<id>&role=withheld|familiar` (the result; 403 for a reserved or
  ineligible recording; 404 for a non-curated example when the weights are absent).
- Screenshots: `design/app6_screenshots/`.

## Assistant (App-7)
One tool-using agent answers a question about one diagnosed case (CLAUDE.md §11). Code, not the model, decides what is
shown.
- **Tools** (`backend/agent/tools.py`; read-only, over the stored case; no model is run): `diagnose` (verdict, fault class
  and its location according to the dataset label, each check's score and threshold, caution, two most probable classes;
  from the record and `backend.checks.assessment`), `check_signal` (speed in Hz and rpm, stability, warnings, inside the
  training / calibration speed ranges), `get_evidence` (the report's channel evidence, `Engine.channels`; optional
  `channel`; channels ranked by change beyond the healthy band — factor ÷ band, the report's marked order — with `rank`,
  `rank_basis` and `largest_change`), `lookup_guidance` (section 04's items from `backend.report.guidance_items`).
  Values are rounded as the report displays them (check scores and thresholds `_num`, probabilities `fmt_score`, Hz 2
  decimals, rpm 0, change factors and bands 1; raw channel values the UI's `f3`). Every fact is a
  `{"value", "ref"}` leaf; record fields keep their path as ref (`diagnosis.scores.msp`), derived values end in their own
  key (`evidence.over_radial.rms.change_factor`), guidance items are `guidance.<entry id>.<k>`. The model-facing schemas
  take no case id: every call is bound to the case asked about. No tool exposes the CNN gradient map.
- **Loop** (`backend/agent/loop.py`): `OpenAICompatibleProvider.chat` (httpx; reasoning fields dropped on receipt), one
  system prompt (`backend/agent/prompt.py`, `AGENT_PROMPT_VERSION` stored on every answer), no chat memory between
  questions. Final answer: `{"answer", "claims": [{"text", "refs"}], "citations": [entry ids]}`.
- **Budget** (per question): `MACHINESENSE_AGENT_MAX_TOOL_CALLS` (6), `MACHINESENSE_AGENT_MAX_REVISIONS` (1),
  `MACHINESENSE_AGENT_TIMEOUT_S` (90 s wall clock; also each model request's timeout). Exhausted → fallback.
- **Evidence review** (`backend/agent/review.py`; reuses the explanation guard): schema; numbers only from this turn's
  tool results (numbers the mechanic typed do not count; 1 and 2 allowed as in the guard); claim refs returned this turn;
  citations returned by `lookup_guidance` and every stated inspection step cited; no `guard.FORBIDDEN` phrase; fault
  location only "according to the dataset label" and never from sensor or channel evidence; ranking words (most,
  largest, highest, biggest, strongest, first / least, smallest, lowest) in a claim or answer sentence only with a cited
  rank ref of rank 1 (or `largest_change`) / the last rank. Numbers are matched against the rounded values, so a number
  quoted more precisely than displayed fails. A failure gets one revision
  with the failed checks and the entry ids `lookup_guidance` returned; a second failure gives the fallback. Rejected text is never shown or stored.
- **Fallback:** "The assistant could not give an answer supported by the case record. …" plus report sections 01 and
  04. Also returned at once when the model is disabled (`MACHINESENSE_LLM_ENABLED=false`, no network call) or unreachable.
- **Endpoints:** `POST /api/cases/{id}/ask` `{"question"}` (1–500 characters) → `answer`, `claims`, `citations`
  (`[{id, title}]`), `actions` (tool, arguments, one-line summary), `review` (`passed | revised | fallback | off`, failed
  checks), `source` (`agent | fallback`), `reason`, `prompt_version`, `model`, `latency_ms`, `tool_calls_used`; never
  reasoning. `GET /api/cases/{id}/questions`: every stored question and response of the case (table `questions`).
- **D7 switch:** `MACHINESENSE_AGENT_REVIEW=false` shows the model's first answer as written (`review.status = "off"`) and
  stores what the review would have rejected. Pilot only; never offered in the UI.
- **UI:** the chat panel (`frontend/src/components/AgentChat.jsx`): suggestions, question box, answers with citation chips
  (entry title on hover), an Actions list and a badge ("Checked against the case record", "Revised once", "Could not
  answer with evidence"); an unavailable model is one line pointing to the report.

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
