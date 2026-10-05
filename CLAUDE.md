# CLAUDE.md — MachineSense

Guidance for coding agents working in this repository. Read it fully before the first change.
The full plan (rationale, experiment ladder, review findings) is the "MachineSense Project Plan" doc;
this file is the operational subset. When the two disagree, stop and ask.

## 1. What we are building

MachineSense is a multi-sensor fault-diagnosis system for rotating machinery, built on the MAFAULDA dataset.
It is Bhuvesh Kumar's solo practice-module project for NUS-ISS Pattern Recognition Systems (MTech AI Systems).

Given one 5-second, 8-channel recording, the system answers three questions:

1. **Is the machine healthy?** Healthy-only anomaly detection: an autoencoder trained on normal recordings only.
2. **If faulty, which known fault?** A 10-class closed-set classifier, plus a 6-class view.
3. **Is this a fault we have never seen?** Open-set rejection: the verdict is healthy, known fault or unknown.

Every verdict carries explanations: calibrated confidence, per-sensor evidence, and an order spectrum and
envelope spectrum with physics cursors at the measured shaft speed.

**Added 27 Sep (Bhuvesh's decision):** the product goal widens from verdicts to inspection decisions.
MachineSense helps a mechanic move from a sensor reading to an evidence-supported inspection decision,
adding two questions on top of the three above: what evidence supports or contradicts the conclusion, and
what to inspect next. A bounded multi-agent investigation assistant (D6) and a pilot evaluation of its
research question (D7) are **planned deliverables, not future work**. The diagnostic models, protocols,
frozen splits and reserved-test rules are unchanged and remain the assistant's only source of numbers.
See §11. Out of scope: remaining-life prediction, or proving that a repair will fix the problem.

### Deliverables and dates

| Date | Deliverable |
| --- | --- |
| 30 Sep 2026 | proposal |
| 1 Oct | data + splits **frozen** |
| 6 Oct | first presentation: goals, data, techniques, progress (deck frozen 4 Oct) |
| 8 Oct | features frozen |
| 11 Oct | checkpoint 1: CNN + autoencoder trained; app walking skeleton serves `/api/diagnose` |
| 18 Oct | checkpoint 2: `manifest.json` v1 frozen |
| 24 Oct | **code freeze** (after this: bug fixes and docs only) |
| 31 Oct | submission ZIP on Canvas |
| 1 Nov | final presentation + peer review |

Agent-stage targets for D6/D7 (11, 18 and 24 Oct) are **proposed, not committed**, and collide with
checkpoint 2, the app week and the writing week. See §11 and the plan's Timeline before scheduling work.

Rubric: the project must visibly use supervised + unsupervised learning, classical ML + deep learning,
hybrid/ensemble, and intelligent sensing (multi-sensor fusion). Pretrained models count only if fine-tuned.

## 2. Hardware and stack

Training runs on a **Mac mini, Apple M4 Pro, 48 GB unified memory, macOS Tahoe**, and uses the GPU through PyTorch MPS.
Graders run the app on a Linux or Windows CPU.

- Python **3.12**, in a venv at `SystemCode/.venv`. Dependencies are pinned in `SystemCode/requirements.txt`
  (torch 2.14.0, torchvision 0.29.0, numpy 2.5.3, pandas 3.0.6, pyarrow 25.0.1, scikit-learn 1.9.1,
  lightgbm 4.7.0, mlflow 3.16.1, fastapi 0.141.1, pydantic 2.13.5).
  - Every pin ships an arm64 wheel.
  - Do not add a dependency without adding a pinned line and a one-line reason.
- LightGBM needs `brew install libomp`.
- Backend: FastAPI + Uvicorn + Pydantic v2.
- Frontend: React 18 + Vite 5 + Tailwind 3 + Radix/shadcn + Recharts + Axios, in the same structure as
  the PropertyLens project.
- MLflow tracking lives at `sqlite:///mlruns.db`.

### Mac / MPS rules (non-negotiable)

- `os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")` must run **before** `import torch`.
  `ml/__init__.py` does this.
- **MPS has no float64.** Cast every NumPy or scikit-learn output to `float32` before `.to(device)`.
- **No DataLoader worker processes.** The needed windows of the spectrogram cache (float16) are loaded once as one
  tensor on the device: in development runs only the development-pool windows (≈ 0.32 GB); reserved-test windows are
  never loaded onto the device. Batches are index slices of that tensor, cast to float32 on the device.
- STFT and feature extraction run on **CPU** with `torch.stft`/NumPy, once, and are cached.
  No librosa or torchaudio in the hot path.
- fp32 training, no AMP. Use batch 256 for the scratch CNN and the autoencoder, 64 for ResNet-18.
- Call `torch.mps.set_per_process_memory_fraction(0.7)` at training start, and `torch.mps.empty_cache()` between runs.
- Run one GPU job at a time. LightGBM and scikit-learn may run on the CPU alongside it.
- LightGBM and PyTorch are never imported in the same Python process (duplicate OpenMP runtimes hang or crash
  on macOS). LightGBM lives only in `scripts/05_baselines.py` and notebook 05, which import no torch.
  Anything that compares LightGBM with a torch model reads saved scores from `results/scores/<run_name>.parquet`
  (columns: `rec_id`, `window_idx`, `split`, `y_true`, and per-class scores). The app, `ml/inference.py` and the
  manifest never load LightGBM; the app's walking-skeleton stub uses a scikit-learn model.
- Anything longer than a few minutes is a **script** (`caffeinate -i python scripts/...`), not a notebook cell.
  Notebooks read the results back from MLflow.
- Device selection comes from the `MACHINESENSE_DEVICE` env var (`mps` / `cuda` / `cpu`).
  Inference, tests and the app default to **CPU**.
- MPS is not bit-deterministic. Seed everything anyway (`ml.utils.seed_everything`, including `torch.mps.manual_seed`).

## 3. Dataset facts (verify, don't assume)

- `mafaulda/` at the repo root holds the Kaggle CSV tree: **1,951 CSV files**, each 250,000 rows × 8 columns,
  no header, 50 kHz, 5 s, about 16 MB (31 GB total).
  - Check whether the tree is nested (`mafaulda/mafaulda/...`) and set the data root in config. Never hard-code it.
- `archive.zip` is the original download. Never read it in code, never commit it.
- Columns:

  | Index | Signal |
  | --- | --- |
  | 0 | tachometer (once-per-rev pulse) |
  | 1–3 | underhang accelerometer: axial, radial, tangential |
  | 4–6 | overhang accelerometer: axial, radial, tangential |
  | 7 | microphone |

- Folders map to labels:

  | Folder | Class | Recordings |
  | --- | --- | --- |
  | `normal/` | normal | 49 |
  | `imbalance/<mass>g/` | imbalance | 333 |
  | `horizontal-misalignment/<mm>/` | horizontal misalignment | 197 |
  | `vertical-misalignment/<mm>/` | vertical misalignment | 301 |
  | `underhang/{ball_fault,cage_fault,outer_race}/<mass>g/` | underhang ball / cage / outer | 186 / 188 / 184 |
  | `overhang/{ball_fault,cage_fault,outer_race}/<mass>g/` | overhang ball / cage / outer | 137 / 188 / 188 |

  In the bearing folders, `0g/6g/20g/35g` is **added imbalance mass**, not damage size.
- Label views (`ml/labels.py`):
  - **class10**: `normal`, `imbalance`, `horizontal_misalignment`, `vertical_misalignment`, `underhang_ball`,
    `underhang_cage`, `underhang_outer`, `overhang_ball`, `overhang_cage`, `overhang_outer`.
  - **class6**: `normal`, `imbalance`, `horizontal_misalignment`, `vertical_misalignment`, `underhang_bearing`,
    `overhang_bearing`. The three bearing faults at each position collapse into one class.
- The file name is the nominal shaft frequency in Hz (×60 = rpm). The measured speed is about 2.3 % lower.
- Bearing ratios, as multiples of shaft rate: BPFO 2.998, BPFI 5.002, BSF 1.871, FTF 0.375.
  - BPFO and BPFI coincide with the 3rd and 5th shaft orders, so **bearing evidence comes only from the envelope
    spectrum**. Raw-spectrum orders are for imbalance and misalignment.
  - The bearing-defect line is 2×BSF, not BSF.
- Known quirks:
  - Tacho glitches (displaced pulses): 228 files have at least one, 94 are severe (> 20 % of revolutions
    glitched), and 63 of the severe ones are underhang ball. Speed is corrected in all of them. Keep these files
    and never feed quality flags to a model.
  - Microphone events flagged by the burst threshold in 168 files: single-sample spikes up to ~−200 V with symmetric
    filter ringing, almost all in imbalance 20–35 g and underhang cage 6–35 g (plus one overhang cage 35 g). They are
    class-correlated, so they are removed (rule 10 in §5).
  - Accelerometer saturation in 9 overhang-ball 6 g files at 47–55 Hz (underhang accelerometers on the rail).
  - The overhang accelerometers carry DC up to 2.2 V; DC is removed per window.
  - Full details: `SystemCode/results/audit_report.md`.
  - The ~23 kHz "resonance" is a narrowband line at ~22,028 Hz with ±60/±120 Hz sidebands on every channel, consistent
    with electrical interference. Its level is folder-linked (a shortcut risk), so a **fixed band-stop** removes it in
    `ml.preprocess.clean_recording` (not an adaptive high-Q notch). Never low-pass before envelope analysis.
  - `cage_fault`: no detectable inner-race signature under this analysis; the underhang-ball positive control failed,
    which limits sensitivity. **Keep the folder labels.**
  - Channel groups: bearing faults and misalignment respond most on the opposite-named accelerometer group. This is an
    unresolved observation; channel names do not change.
- **Preprocessing decisions (26 Sep 2026):**
    Stage 3B autoencoder reading: nearly every fault class, including the overhang faults, is reconstructed worst on an
    overhang channel. This weakens a simple swapped-channel explanation; it does not establish a mounting or sensor-type
    explanation. Per-channel reconstruction error is not usable as fault-location evidence.
  - **App wording:** the app shows "predicted fault location according to the dataset labels"; the classifier does not
    independently verify the physical sensor mapping.
  - **Unknown-fault wording (Stage 3C.1; report and app):** "On these development folds, embedding distance separates withheld bearing classes more successfully than withheld imbalance and vertical misalignment. When either of those two classes is withheld, the model frequently assigns the other label. The existing class labels are kept. This is a limitation of unfamiliar-fault detection, not a new physical diagnosis or merged class." "Withheld horizontal misalignment is sometimes declared healthy (see the healthy-rule variants)." "Development results on one partition; reserved tests sealed." App healthy rule pending Bhuvesh's
    decision after 3C.1 (see §7).
  - **Q3 wording (Stage ML-4, replaces any stronger phrasing):** "The tested scores and rules do not consistently distinguish the imbalance / vertical-misalignment pair when one class is withheld. The classifier distinguishes them well when both are represented in training. This does not establish that no representation, training objective or decision method could help." Not attributed to missing cross-axis
    information (hypothesis only). Stage 3D: E2 did not test whether combining LR and CNN evidence helps (τ_s
    non-restrictive in 67 of 81 runs); "D reduces unfamiliar faults called healthy (seed 42: 19 → 2) at a cost in healthy acceptance (184 → 157 of 225 instances from 25 distinct recordings; τ_b from 2–3 calibration normals per run). Vibration-only B1s alone had 6/25 false alarms in the Q1 evaluation. Candidate only; not selected."
  - **Primary envelope band: 1–5 kHz for every track**, fixed from the sensor specifications (inside both sensors'
    rated ranges). The per-track bands in `results/demod_bands.json` are a separately reported comparison only.
  - **Order axis: per-window speed** from `ml.preprocess.window_speeds`, computed from the untouched tachometer,
    falling back to the recording speed. Fallback indicators and speed-variability statistics are never model inputs.
  - **22 kHz line: a fixed band-stop** (above), set from `split_closed` train normal recordings only.
  - **Microphone cleaning is a provisional mitigation**; it is re-checked on the microphone representation in Stage 2C.
  - **Primary pipeline: the 10 kHz-capped variant r1cap10 (decided 26 Sep 2026).** A pragmatic bandwidth limit that reduces
    exposure to the identified high-frequency interference; it does not guarantee calibrated measurements (the 604B31
    is rated to 5 kHz) or remove all artefacts. The choice uses engineering considerations and the permitted development
    comparison (−0.005 [−0.012, +0.001], LightGBM). `base.yaml` carries it; the band-stop-only table (r1) is kept for the
    record via `configs/variants/r1_bandstop_only.yaml`.
- **Decisions of 27 Sep 2026:**
  - **Representation:** candidate C is the LEADING candidate, not yet frozen: order-domain resampling, 8 revolutions ×
    512 samples/rev, Kaiser anti-alias at 205 × speed, 128 log order bins 0.5–150, T = 25, plus the 128-bin microphone
    log-Mel aligned to the same frame times. B is the fallback; A is rejected (it did not separate 0.5× from 1× at 12 Hz).
    C's lower speed-from-sharpness score (logistic regression; 0.668 vs A 0.943, B 0.761; permutation ≈ 0.32) shows less
    speed-dependent resolution; it does not show C is free of shortcuts, and speed is real physical information here.
  - **Microphone:** the current cleaning policy stays. The burst comparison has no speed-matched controls (flags begin near
    20 Hz within each affected folder); residue is untested, not shown to be absent.
  - **Classical reference:** logistic regression on the capped table, "a simple reference with competitive development
    performance; paired advantage over capped LightGBM +0.046 [+0.011, +0.095] on these development folds". LightGBM stays
    a reported rung. The walking-skeleton stub will be logistic regression. The chosen representation still needs its
    own model evaluation later.
  - **Tacho fallback:** all 94 fallback recordings (no measurable window speed) run at ≥ 38.5 Hz (median 49 Hz); 92 are
    `split_speed_block` test and none is in the development pool. Speed fallback matters mainly for the reserved
    speed-block test and for fast machines in the app.
- **Development-evaluation limits (Stage 2B.1):** folders are session proxies; normal is evaluated within its single
  folder; speeds are below 40 Hz; cleaning settings were inherited from earlier development and are not re-evaluated per
  fold. The 2B.1 results are the classical benchmark; no feature redesign around the difficult folders.
- "Imbalance" means imbalance-only. About three-quarters of the bearing recordings also carry imbalance mass.

## 4. Repository layout (target)

```
MachineSense-Project/                 repo root
├── CLAUDE.md
├── .gitignore                        mafaulda/, archive.zip, data/, *.npy, mlruns.db, mlruns/, artifacts/**/*.pt,
│                                     *.safetensors, .venv/, node_modules/, frontend/dist/ (unless release), .env
├── mafaulda/                         raw CSV tree (gitignored, read-only for code)
├── archive.zip                       original download (gitignored)
├── ProjectReport/                    report PDF, reflection
├── Miscellaneous/                    slides, video, images
└── SystemCode/
    ├── pyproject.toml  requirements.txt  Makefile  HOW_TO_RUN.md  .env.example
    ├── configs/                      base.yaml (+ cnn.yaml, ae.yaml, resnet.yaml later)
    ├── metadata.csv                  COMMITTED: one row per recording, labels, speed, quality flags, split columns
    ├── data/                         gitignored: raw_npy/, windows.parquet, cache/rep_<hash>.npy
    ├── ml/                           importable package — ALL reusable logic lives here
    │   ├── __init__.py               sets PYTORCH_ENABLE_MPS_FALLBACK
    │   ├── config.py                 Pydantic PreprocessConfig/TrainConfig + YAML loader + config sha256
    │   ├── paths.py                  resolves data root, repo root, artifact dirs from config/env
    │   ├── labels.py                 folder path → labels (class10, class6, severity, fr_file_hz)
    │   ├── preprocess.py             load_csv, quality_gate, estimate_speed, window
    │   ├── features.py               physics features (R1), envelope comb
    │   ├── spectro.py                order spectrogram + mic log-Mel (R2)
    │   ├── splits.py                 split assignment
    │   ├── datasets.py  models/  arbitrator.py  bundle.py  inference.py  train.py  utils.py
    ├── scripts/                      thin CLIs over ml/: smoke_env, 00_convert, 01_index, 02_split, ...
    ├── notebooks/                    01_data_layer/ … 05_xai_layer/ (see §6)
    ├── tests/                        pytest
    ├── artifacts/  results/  demo/
    ├── backend/                      FastAPI (later stages)
    └── frontend/                     React (later stages)
```

`scripts/smoke_env.py`, `requirements.txt` and `HOW_TO_RUN.md` already exist from planning.

## 5. Rules that protect the results (the project's credibility depends on these)

1. **Split by recording, never by window.** Assign every split in `metadata.csv` before any window exists.
   Neighbouring windows of one recording must never cross train/val/test.
2. **Fit on train only.** Normalisation statistics, scalers, feature selection and thresholds all come from
   training (or validation) data. Recompute them per leave-one-class-out fold.
3. **The test split is evaluated once per model**, at the end. Model selection uses validation only.
4. **The unknown class never tunes anything.** Open-set thresholds are quantiles of *known-class* validation data
   (for example, 95 % acceptance). Never pick a threshold by H-score on data that contains the withheld class.
5. **Decisions are made per recording.** Average the 9 window scores (mean log-p, mean reconstruction error,
   mean distance), then apply the rule once.
6. **Physics choices come from physics or train data.** Examples: cut-offs, the demodulation band, the order range.
   Never inspect test recordings to make them.
7. **Global per-channel normalisation**, never per-window z-scores. Absolute levels carry the signal.
8. **Frozen means frozen.** After 1 Oct, split columns in `metadata.csv` change only with an explicit note in
   `results/CHANGELOG.md` and a re-run of everything downstream.
9. Report metrics with uncertainty: bootstrap over test recordings, Clopper–Pearson for rates from few recordings.
10. **Microphone events flagged by the burst threshold are removed** with `ml.preprocess.remove_mic_bursts` before
    every representation, at training and at serving. No mic peak, crest-factor or kurtosis features. The threshold
    comes from `split_closed` TRAIN normal recordings only, and the padding from event-aligned residual ringing.
    This is a **conservative policy**: the flagged events look electrical, but that is not proven, and the cleaning
    itself may leave a trace, so the ban on mic peak statistics stays even after cleaning. The cleaning is a
    provisional mitigation, re-checked on the microphone representation in Stage 2C.
11. **`clean_recording` is applied to the whole recording before windowing, at training and serving.**
12. **Selection protocol.** Model and variant choices use the folder-grouped development scheme (the 2B.1 pool, K = 3).
    Closed-set ties are broken by simplicity and latency. Reserved tests are never used for selection.
13. **Evaluation protocol (27 Sep 2026, `ml/protocol.py`).** Consistent scoring: every evaluated recording is scored
    exactly once, by the model of its own outer dev fold (no averaging over fold models). Calibration is separate from
    evaluation: thresholds come from calibration recordings only, and rates at a threshold come from recordings not used to
    choose it or fit the model. Primary summaries are per-fold threshold-free metrics (AUROC; OSCR for Q3), mean and
    spread across folds; metrics on pooled raw scores are supplementary. Score directions are fixed before running.
    Every Q1 result states that the healthy sample is 25 recordings from one folder.

## 6. Working method: exploration → development

Exploration happens in Jupyter notebooks, so each step is visible. Development promotes the code that worked
into `ml/`, the scripts and the app.

- **Promotion rule:** a function used in two places, or needed at serving time, lives in `ml/`.
  - Notebooks `from ml... import ...` after `pip install -e .`.
  - Never `sys.path.append`, never copy-paste logic between notebooks.
- Notebook layout (ten notebooks):

  | Folder | Notebooks |
  | --- | --- |
  | `01_data_layer/` | `01_index_audit_splits`, `02_signal_exploration` |
  | `02_feature_layer/` | `03_windows_physics_features`, `04_representations` |
  | `03_ml_layer/` | `05_baselines`, `06_cnn_scratch`, `07_pretrained_finetune`, `08_autoencoder_healthy` |
  | `04_openset_layer/` | `09_openset_loco` |
  | `05_xai_layer/` | `10_xai_ablations_selection` |

- Every notebook follows the same shape:
  1. **Purpose** (markdown)
  2. **Inputs** (paths and config hash)
  3. The work, in short cells with a markdown line above each explaining *what we are looking at and why*
  4. **Findings**: 3–6 bullets with numbers
  5. **Outputs written**
- **Every code cell gets its own light-blue "What this code does" note directly above it** (a separate markdown cell,
  after any section heading). It says in plain language what the code does, for a reader who doesn't read Python:
  what it loads, computes or plots, and how to read the result. Use exactly this HTML (the text colour keeps it
  readable in dark themes; GitHub's preview drops the styling):
  ```html
  <div style="background-color:#e3f2fd; color:#0d2538; padding:10px 14px; border-radius:6px; border-left:4px solid #64b5f6; margin:4px 0;"><b>What this code does:</b> …</div>
  ```
  Inside the div, use `<code>` and `<b>`, not Markdown backticks or `**` (Markdown isn't rendered inside HTML).
  Check before committing that every code cell is preceded by such a note.
- A notebook must run top to bottom from a fresh kernel, under a few minutes. Heavy computation calls a cached
  artifact or a script.
- Figures: matplotlib, labelled axes with units, saved to `results/figures/` when they are report material.
- Strip outputs on commit (`nbstripout`). Before the final ZIP, run all notebooks and export them to HTML.

## 7. Current stage: Stage App-7 — the investigation assistant: one agent with tools (D6)

The chat panel is connected: one agent answers a mechanic's question about one diagnosed case, using only four read-only
tools, and code reviews its answer before the mechanic sees it (CLAUDE.md §11, dated edit of 5 Oct). No model, threshold,
scoring path, `EvidenceRecord` field, golden evidence record or report fixture changes; reserved tests sealed; no new
dependency. The D7 pilot is the next stage; this stage builds only its switch.

**App-7 protocol (declared before writing code).**
0. **Tool calling checked first (5 Oct, by hand):** the configured LM Studio model `google/gemma-4-e4b` returned proper
   `tool_calls` for one tool in 3 of 3 runs (`finish_reason: tool_calls`) and a plain answer after the tool result. Its
   messages carry a `reasoning_content` field; it is dropped on receipt and never stored, returned or fed back (§11 rule 4).
1. **Tools (`backend/agent/tools.py`).** `diagnose`, `check_signal`, `get_evidence`, `lookup_guidance` are plain functions
   over a case stored by `/api/diagnose`; they read the stored record and never run a model. Every fact a claim may rest on
   is returned as a leaf `{"value": …, "ref": "<path>"}`: numbers always, and also the verdict, labels, warnings and each
   guidance item. Refs are record paths (`diagnosis.scores.msp`, `diagnosis.thresholds.tau_p`,
   `signal_quality.shaft_speed_hz`, `physics.shaft_speed_rpm`), `evidence.<channel>.<metric>.<field>` for channel evidence,
   and `guidance.<entry id>.<k>` for the k-th section 04 item of an entry. Builders reused, nothing reimplemented:
   `diagnose` ← the record and `backend.checks.assessment` (scores and their thresholds, pass / fail, caution, the two most
   probable classes, the location text of `backend.report.location` "according to the dataset label");
   `check_signal` ← `signal_quality` (+ `physics.shaft_speed_rpm`); inside / outside the training and calibration speed
   ranges computed from the record's own ranges; `get_evidence` ← the channel evidence the report uses
   (`backend.evidence.channel_evidence`, moved behind one engine method so the app endpoint and the tool share it), optional
   `channel`; `lookup_guidance` ← `backend.report.guidance_items` only (the bundle's feature names as the report uses; never
   `ml.guidance.select`). No tool exposes the CNN gradient map (§11 rule 11). An unknown case id or channel returns
   `{"error": …}`. **The model-facing tool schemas take no case id**: the loop binds every call to the case being asked
   about, so the model cannot read another case; `get_evidence` takes an optional `channel` enum.
2. **Loop (`backend/agent/loop.py`).** `OpenAICompatibleProvider.chat(messages, tools)` over httpx (no SDK) returns the
   message's content and tool calls (reasoning dropped). One versioned system prompt, `AGENT_PROMPT_VERSION` = sha256 of
   the prompt, the tool schemas and the answer schema (first 16 hex), stored on every answer. Final answer = the model's
   content when it makes no tool call, a JSON object `{"answer": str, "claims": [{"text": str, "refs": [str]}],
   "citations": [entry ids]}`. Each question is answered on its own (no chat memory). **Budget** (settings):
   `MACHINESENSE_AGENT_MAX_TOOL_CALLS` 6 (tool calls per question, counting calls to unknown tools; a turn that would exceed
   it ends the loop), `MACHINESENSE_AGENT_MAX_REVISIONS` 1, `MACHINESENSE_AGENT_TIMEOUT_S` 60 (wall clock per question; also
   the HTTP timeout of each model request); exhausting any → fallback. The loop takes the review as a parameter (wired to
   `backend.agent.review` in step 3).
3. **Evidence review (`backend/agent/review.py`, code only; reuses `guard.parse`, `guard.NUMBER`, `guard._matches`,
   `guard.FORBIDDEN`).** Checks, each reported by name when it fails: (1) `schema` — parses; exactly the three keys;
   `answer` a non-empty string, `claims` a list of `{text, refs}` with a non-empty `refs` list of strings, `citations` a list
   of strings; (2) `numbers` — every number in `answer` and in each claim's text matches (guard._matches) a number returned
   by a tool in this turn: a numeric leaf, or a number inside a returned string (a warning, a guidance step); 1 and 2 are
   allowed as in the explanation guard ("1×", "2×"); numbers the mechanic typed do not count; (3) `refs` — every claim ref is
   a ref returned in this turn; (4) `citations` — every cited id is an entry id `lookup_guidance` returned in this turn, and
   every sentence that states an inspection step (pattern `ml`-free, in `review.INSPECTION`: inspect, examine, look for,
   record again / re-record / re-run, check the / for / that / whether, measure the, escalate, align, rebalance, tighten,
   lubricate) is in an answer with at least one citation, or, for a claim, the claim has a `guidance.` ref; (5) `forbidden` —
   no `guard.FORBIDDEN` match (replace, confirmed, root cause, urgency words, percentages …); (6) `location` — §11 rule 5: a
   sentence with fault-location wording (`located`, `location`, "the fault is in / at / on") must say "according to the
   dataset label(s)" and must not rest on sensor or channel evidence (a sensor, channel, axis, accelerometer, microphone or
   reconstruction-error word in the same sentence). On failure: one revision (the failed checks are sent back as a user
   message; tool calls in the revision count toward the budget). Still failing → **fallback**: the fixed message "The
   assistant could not give an answer supported by the case record. The fixed report's findings and inspection steps are
   below." plus sections 01 and 04 of `backend.report.build_sections` for the case (feature names as the report). The
   rejected text is never shown and never stored.
4. **API and storage.** `POST /api/cases/{case_id}/ask` `{"question": str}` (1–500 characters; else 422; unknown case
   404) → `answer`, `claims`, `citations` (`[{id, title}]`, titles from the entries), `actions` (one per tool call: tool,
   arguments, a one-line summary built by code), `review` (`{status: passed | revised | fallback | off, failed: [...]}`),
   `source` (agent | fallback), `reason` (fallback only), `prompt_version`, `model`, `latency_ms`, `tool_calls_used`; never
   reasoning text. Every question and response is stored in the case store (table `questions`);
   `GET /api/cases/{case_id}/questions` lists them oldest first. LLM disabled or unreachable → the fallback immediately
   (`source: "fallback"`, reason "language model disabled" / "language model unreachable (…)"), no network call when
   disabled. The rest of the app is unchanged.
5. **D7 switch.** `MACHINESENSE_AGENT_REVIEW` (default on). Off: the review still runs but does not gate — the first final
   answer is shown as is (no revision), `review.status = "off"` and `review.failed` records what it would have rejected; the
   budget and the unavailable fallback still apply. Not offered in the UI.
6. **Frontend (`AgentChat.jsx`).** Working chat for the open case: suggestions, text box, Send; each answer with citation
   chips (title on hover), an "Actions" list, and a badge — passed "Checked against the case record", revised "Revised
   once", fallback "Could not answer with evidence" (no badge for `off`). The fallback shows the fixed message and the report
   items; unavailable → one line pointing to the report. Previous questions of the case are loaded from
   `GET …/questions`. Launcher, focus handling and Escape unchanged. Reasoning is never rendered.
7. **Tests** as the brief lists (scripted httpx transport, no live model); golden evidence records and report fixtures
   unchanged. **Hand run:** 5 questions on development cases with the local model, review on; per check, how often it fails
   is reported. A check that looks too strict is reported, not loosened.
8. **Docs:** `backend/README.md` "Assistant"; §11 rules 2 and 3 lines.

## 7-prev9. Previous stage: Stage App-5 — wire the guidance library into the report

Section 04 of the diagnostic report ("What to inspect next") stops being a placeholder and shows inspection steps
retrieved from `SystemCode/knowledge/entries/*.yaml`, each cited by its entry `id`. Nothing else changes: no model,
threshold, scoring path or `EvidenceRecord` field; reserved tests sealed; no evaluation re-run; the agent chat panel is not
touched. Entry prose is Bhuvesh's (relabelled in 29de432, corrected in 5737f15) and is never edited here.

**App-5 protocol (declared before writing code).**
1. **`ml/guidance.py` (pure).** `load_entries(path)` parses every entry with a Pydantic model mirroring the README schema
   exactly (`extra="forbid"`; `drafted_with_ai: bool` required; `verification` ∈ {internal, ai_drafted, open_verified,
   citation_unverified, needs_paywalled_source}; `applies_to` a list of class10 labels or `any`); an `id` different from its
   file stem, or a duplicate `id`, is an error. `select(entries, *, condition, available_evidence, warnings)`: (1) drop
   `needs_paywalled_source`; (2) keep if `applies_to` is `any` or contains `condition`; (3) keep only if every
   `evidence_required` name is in `available_evidence`; (4) drop if any `excludes` string matches a recorded warning —
   **matching = case-insensitive substring of the warning text** (Bhuvesh, 4 Oct; no entry uses `excludes` today); order:
   internal, open_verified, citation_unverified, ai_drafted, then `id`. No formatting.
2. **Available evidence (Bhuvesh, 4 Oct; no schema change).** `EvidenceRecord` does not list the features a recording
   produced, and every scored recording produces every R1 feature (no required feature is NaN in any of the 17,559 windows of
   the R1 table), so available = the names in the bundle's packaged, hash-verified `feature_dictionary.csv` ∪ {`speed_hz`,
   `decision.decision`} (always in the record) ∪ {`quality.warnings`} only when the record carries at least one warning.
   Condition = the arbitrator's verdict: the fault class for a known fault, `normal` for healthy, `unknown` otherwise (only
   `any` entries apply).
3. **`backend/report.py`.** `GUIDANCE_PLACEHOLDER` deleted; section 04 = one `Guidance` item per step of every selected entry,
   in `select`'s order, except that for an `unknown` verdict `further-inspection-needed` is shown first (the verdict's own
   instruction before generic acquisition checks). Each item: the step text, the entry id shown visibly in the text
   ("source: <id>"), `cite` = the id, `hover` = the entry title and its full `source` (and `source_url`). Visible notes in the
   item text: `ai_drafted` → "AI-drafted, not yet checked against a published source"; `citation_unverified` → "Citation not
   yet checked against the full text". Nothing selected → one `Missing` item: no curated guidance covers this finding
   (naming the condition); never generic advice. `build_sections` receives the evidence names from the caller (the app passes
   the bundle's feature dictionary); the markdown report and `EvidenceRecord` are unchanged.
4. **Frontend:** only the report caption that calls Guidance items placeholders is corrected; report fixtures re-recorded.
5. **Tests:** `tests/test_guidance.py` and `tests/backend/test_app5.py` as the brief lists (loading and stems, no
   gradient-attribution / saliency name in `evidence_required` (§11 rule 11), stubs empty, evidence gate, stub dropped,
   excludes, deterministic order, schema failures; bearing citation, unknown first, envelope-absent → Missing, ai_drafted
   note on every imbalance-1x step); golden evidence-record tests unchanged.

6. **Addendum (4 Oct 2026, before the report change; Bhuvesh's decision).** `further-inspection-needed` had `applies_to: any`
   with `evidence_required: [decision.decision]`, which every record carries, so it would have been offered on healthy and
   known-fault reports where its steps are wrong. `applies_to` now also accepts the verdict label `unknown` (the condition
   passed for a further-inspection verdict); Bhuvesh changes that entry to `applies_to: [unknown]` and the README's
   `applies_to` line. `ml.guidance` validates class10 labels plus `unknown`; the selection rule is otherwise unchanged.

7. **Addendum (4 Oct 2026, before any code; Bhuvesh's brief "warning-matched steps, the healthy case, and the two
   remaining placeholders"; his entry change 9fd72ab).** Scope: `ml/guidance.py`, `backend/report.py`, `backend/explain/`
   (`plain.py`, and by Bhuvesh's decision of 4 Oct also `prompt.py`, `guard.py`, `service.py`), `backend/app.py` (passing
   items), `frontend/src/lib/mechanic.js` and the Mechanic row of `Aside.jsx`, their tests and docs. No model, threshold,
   scoring path, `EvidenceRecord` field or explanation SCHEMA key changes; reserved tests sealed.
   (a) **Warning-matched steps.** A step is a string or a mapping `{text, when_warning}` (non-empty list of substrings;
   `extra="forbid"`). `select` keeps its entry rule and order, then filters each kept entry's steps: a string step is always
   kept; a `when_warning` step only if one of its substrings is a case-insensitive substring of a recorded warning (the
   `excludes` matching); an entry left with no steps is dropped. `select` returns the entries with their steps filtered, so
   no filtered-out step reaches the report. Coverage of the emitted warnings (`ml.inference._signal_quality`): every one
   matches a substring except the calibration-speed note ("healthy thresholds were not calibrated at this speed …"), which
   by design gets no acquisition step.
   (b) **Healthy case.** For a `healthy` verdict with nothing selected, section 04 is one `Model` item, "No inspection steps
   are offered for a healthy verdict."; other conditions with nothing selected keep the `Missing` item. No entry for
   `normal` is invented.
   (c) **One selection, three displays.** Section 04's items (`backend.report.guidance_items`, the only caller of
   `select`) are reused unchanged: the Mechanic view renders `report.sections` 04 items (citation and verification note
   kept); the explanation's `next_step` is set by code from the same items, one step per line (`plain.next_step(items)`),
   for both the template and the language-model text: the model is no longer asked to write `next_step` (prompt changed,
   `PROMPT_VERSION` changes), the guard no longer compares it to the placeholder, and `service.explain_case` receives the
   items from `app._explain`. `GUIDANCE_TEXT` is deleted in `plain.py` and `mechanic.js`; a frontend test fails if
   "not yet connected" reappears.
   **Recorded after implementation:** Bhuvesh approved editing `prompt.py`, `guard.py` and `service.py` for this change
   only. `PROMPT_VERSION` daa7a46bda2172aa → f154953912c969cc (the `next_step` rule and the removal of
   `guidance_connected` / `guidance_text` from DATA); explanations stored before this change carry the old version and the
   placeholder `next_step`. The explanation SCHEMA keys are unchanged.

## 7-prev8. Previous stage: Stage R7 — diagnostics on saved models: where the open-set failures come from (no training)

Exploratory, development data only. **No model is trained**: saved checkpoints are run forward and their outputs analysed.
Reserved partitions sealed; F1 already run; nothing here is an independent claim; the bundle, rules and thresholds are
unchanged. Deliverable: `notebooks/06_research_layer/17_diagnostics.ipynb` (question, protocol, figures, tables, verdict).
Forward passes (≈ 10 min) run in `scripts/37_r7_forward.py`, which only reproduces and saves model outputs; pure helpers
in `ml/diagnostics.py` (tested). Outputs `results/research/r7_diagnostics.json`, `r7_d1_mild_hm.csv`, `r7_d2_probe.csv`,
`r7_d3_gates.csv`, figures `results/figures/r7_*.png`.

**Stage R7 protocol (declared before computing anything; changes only as dated addenda with a reason).**
1. **Reproduction first.** For the 81 LOCO runs (9 withheld classes × 3 outer folds × seeds 42 / 43 / 44;
   `artifacts/loco/cnn6_w-<c>_s<seed>_loco/fold<k>.pt`, normalisation `results/loco/cnn6_w-<c>_s<seed>_loco/fold<k>_norm.json`,
   roles from `results/q1q3/q3_fit_cal_eval_assignment.csv`), every window of every role is run forward on the device the runs
   were trained on (MPS); window logits and embeddings must match the saved Stage 3C / 3C.1 window outputs
   (`results/scores/cnn6_w-<c>_s<seed>_loco_fold<k>_{logits,embeddings}`) and the recording MSP and shared Mahalanobis must
   match the saved recording scores (seed 42 also `cnn6ae6_s42_loco_decisions.parquet`) within |Δ MSP| ≤ 1e-5, |Δ maha| ≤
   1e-4 × maha, with identical top classes; maximum differences are reported. Seed-42 τ_p, τ_d recomputed from the saved
   calibration scores (`ml.protocol.order_statistic_threshold`, 0.975; MSP on its negation) must equal
   `results/loco/decision_thresholds.csv` within 1e-12; seeds 43 / 44 thresholds are computed the same way (τ_e = maximum AE
   error of the run's calibration normals; B1s and τ_b from `b1s_nomic_loco_q3fit_scores.parquet`, seed-independent).
   **Any failure stops the stage and is reported.** After a pass, the saved (verified) outputs are used.
   Conventions unchanged: recording probabilities = mean window log-probability, renormalised; Mahalanobis = per window the
   minimum over classes with one shared within-class Ledoit–Wolf covariance from the fit windows, mean over windows;
   recording embedding = mean of the 9 window embeddings.
2. **D1 — does the gate catch mild horizontal misalignment?** Population: the 25 development-pool horizontal misalignment
   0.5 mm recordings (none reserved), with the 14 that R6's A0 missed at seed 42 flagged. **Gate verdict (Bhuvesh, 1 Oct):**
   the closed-set development CNN-6 models have no valid calibration (every recording outside fold k trained fold k's
   model), so the default rule `ml.arbitrator.decide("B_with_caution")` is applied with the saved **LOCO runs that learned
   horizontal misalignment** (withheld ≠ horizontal misalignment: 8 runs per fold per seed), for which each 0.5 mm recording
   of fold k is out of sample (role foldk_known), with each run's own τ_p, τ_d, τ_e, τ_b. Per recording and seed: the 8
   verdicts (healthy incl. caution / horizontal misalignment / another fault / further inspection); a recording counts as
   **called healthy** on a seed if at least 4 of its 8 runs say healthy (the strict "any run" count is reported too).
   Threshold-free context: the closed-set development CNN-6 (seeds 42 / 43 / 44, its own fold) top class, top probability,
   and shared Mahalanobis (fitted on that fold model's training windows, forward pass only); horizontal ↔ vertical
   misalignment confusion counted for both. **Reading:** covered by the gate if, on every seed, ≤ 1 recording is called
   healthy; system blind spot if ≥ 3 are on any seed; otherwise unclear.
3. **D2 — feature collapse (linear probe), per run and seed.** Descriptors (fixed now): `f_under_radial_ratio_2x_1x`,
   `f_over_radial_ratio_2x_1x`, `f_under_ratio_axial_radial_1x`, `f_over_ratio_axial_radial_1x`, `f_under_axial_o1`,
   `f_under_axial_o2`, `f_under_radial_o1`, `f_under_radial_o2`, `f_under_tangential_o1`, `f_under_tangential_o2`,
   `f_over_axial_o1`, `f_over_axial_o2`, `f_over_radial_o1`, `f_over_radial_o2`, `f_over_tangential_o1`,
   `f_over_tangential_o2` (16; recording value = mean over the 9 windows of the capped R1 table). Descriptors and embeddings
   are standardised with the run's **fit** recordings (known classes only); one ridge (multi-output, a single α) from z to
   the 16 descriptors is fitted on the fit recordings with α chosen by 5-fold cross-validation on those same recordings
   (α ∈ logspace(−3, 4, 15), shuffled with the run's seed). R² per descriptor (sklearn `r2_score` against each evaluated
   set's own mean) and averaged, on (a) known-class evaluation recordings (roles eval and foldk_known) and (b) withheld
   recordings (foldk_withheld); RMSE in standardised units reported beside R² (R² on a narrow-range set is sensitive to
   that set's variance). The probe never sees a withheld recording. Per run: withheld-class Mahalanobis AUROC (foldk_known
   vs foldk_withheld). **Reading:** collapse supported if, for withheld vertical misalignment, mean withheld R² < 0.5 while
   known R² ≥ 0.8 on all three seeds, AND across the 27 runs of each seed Spearman(withheld R², Mahalanobis AUROC) > 0.4;
   not supported if withheld R² ≥ 0.8 for vertical misalignment on all seeds; otherwise unclear.
4. **D3 — is the probability half saturated? (logit-based gates).** Per recording: max-logit = mean over the 9 windows of
   the window's maximum logit; energy = mean over the 9 windows of −logsumexp(logits). Unknown scores u1 = −max-logit,
   u2 = energy (higher = more unknown); thresholds by `order_statistic_threshold` at 0.975 on the run's calibration
   recordings, exactly as τ_p. Gates, all with the unchanged Mahalanobis term (maha ≤ τ_d): G0 MSP ≥ τ_p; G1 u1 ≤ τ1;
   G2 u2 ≤ τ2. Verdict as rule B (confident and top class normal → healthy; confident fault → that fault; else unknown).
   Primary: unknown recall on foldk_withheld at a **matched known-side rejection count** (the logit-term threshold of G1 / G2
   moved, τ_d fixed, so the gate rejects exactly as many foldk_known recordings as G0; reported if unreachable); also at the
   calibrated thresholds; withheld called healthy; per withheld class; the mildest setting of each family; saturation = the
   share of known recordings (cal and foldk_known) with top probability > 0.999. **Reading:** G1 / G2 helps only if matched
   unknown recall ≥ G0 + 5 points on all three seeds with withheld called healthy not higher on any seed; does not help if
   the seed-mean difference ≤ 0 or healthy calls rise on any seed; otherwise too thin to tell. R0's tie rule carries forward.
5. **Decision (last cell):** D2 supports collapse → an auxiliary physics-descriptor head is worth declaring as Stage R8;
   D2 does not support collapse and D3 does not help → no further training experiment before code freeze, the open-set gap
   documented; D3 helps → a development finding only (changing the deployed gate is Bhuvesh's decision). D1's reading goes
   into the write-up beside R6.

## 7-prev7. Previous stage: Stage R6 — autoencoder experiments for healthy versus faulty separation (Q1, development data only)

Exploratory, development data only; seeds 42, 43, 44 throughout. The bundle, decision rules, thresholds, `artifacts/final/`,
`results/final/`, `metadata.csv` and the app are untouched; no reserved recording is read. Runs after Stage R5 (done).
Deliverable: `notebooks/06_research_layer/16_ae_experiments.ipynb` (defines the protocol, presents every result, writes
`results/research/r6_ae_experiments.json` and companion CSVs and the figures). Training (≈ 50 min in total, too long for a
notebook cell) runs in `scripts/36_r6_train.py`, which only saves per-recording scores; the notebook loads and analyses
them. Reusable code: `ml/ae_experiments.py` (pure, typed, tested in `tests/test_ae_experiments.py`). Weights gitignored.

**Stage R6 protocol (declared before training; changes only as dated addenda with a reason).**
1. **Partitions: the Q1 development folds** (Stage 2D.1 / 3B; `results/q1q3/q1_fit_cal_assignment.csv`, `ml.q1.fold_ids`):
   per outer fold k, **11 healthy fit normals and 5–6 calibration normals** (from folds ≠ k), evaluation = every
   development-pool recording of fold k; over the 3 folds every one of the **25 development normals** is evaluated once out
   of sample, and every fault recording once. (Not the shared / final AE-6: 20 fit / 5 cal; not the LOCO runs: ≈ 11 / 2–3.)
2. **Measurement limit (stated before results):** 25 healthy recordings from one folder: a false-alarm rate moves in 4-point
   steps; no folder-grouped interval exists on the healthy side (Clopper–Pearson, n = 25, always stated); speed and
   recording identity are confounded (R0). All 49 healthy recordings are from one session: augmentation cannot create the
   missing session and operating diversity.
3. **Correction to the brief (recorded):** the existing gain jitter (±1 dB, independent per example and per channel) is
   applied to the input AND the target (`AeConfig.reconstruct = augmented_input`), so it adds input diversity but does not
   teach level invariance. It is already per-channel, so it already contains per-channel mismatch. The frame-level AE sees
   each time frame independently, so a time shift within the recording is a no-op and is not used.
4. **Experiment 1 arms** (healthy fit normals only; the AE-6 architecture, normalisation, epochs budget, error and recording
   score unchanged; vibration channels 0–5):
   - **A0 baseline:** the AE-6 recipe, retrained by the same code as the other arms (reproduction against Stage 3B's saved
     seed-42 AE-6 scores reported; MPS is not bit-deterministic).
   - **A1 control:** V = max(5, ceil(100 / n_fit)) = 10 materialised recording-level views per fit normal (110 views per
     fold), each a fixed per-channel offset vector drawn once from U(±1 dB) (the same distribution as the jitter);
     stochastic jitter off; target = the view (as A0). Same number of optimiser steps as A0 (epochs = 150 / V = 15).
     **Prediction recorded before running: no difference from A0.**
   - **A2 denoising:** A0's stochastic jitter on the input; target = the **un-augmented** normalised input.
   - **A3 wider plausible set, A2 target:** a common gain offset U(±1 dB) over all channels + a per-channel mismatch
     N(0, σ_mm) with σ_mm estimated from the fit normals (the across-recording SD of each channel's mean dB level relative
     to the recording's mean over channels, pooled over channels) + additive noise at the measured floor: in power units,
     10^(x/10) + u · 10^(F_c/10), u ~ Exp(1) per bin and frame, F_c = the 5th percentile of channel c's cached dB values over
     the fit normals' frames. Nothing outside the allowed list (overall gain, per-channel gain mismatch, measured noise
     floor; time shift is a no-op here).
   Views and augmentations stay linked to their source recording, are never counted as recordings, and never enter
   calibration, evaluation or any uncertainty calculation; tables report original recordings and generated views apart.
5. **Experiment 2, fault-only AE (F):** the AE-6 architecture and recipe trained on the **fault** recordings of folds ≠ k
   (all development-pool faults outside fold k; 506–531 per fold), normalised on exactly those recordings (`norm_stats`
   development mode; fold k and the calibration normals excluded); scored by the same recording procedure; higher error =
   more healthy-like. Threshold on fold k from the calibration normals only: τ_F = their minimum error; alarm (fault) iff
   error < τ_F (ties no alarm). **Prior recorded before running: expected to fail** (a fault recording contains the healthy
   machine plus a signature, and R4 showed these AEs reconstruct what they never saw). **Level controls:** Spearman of the
   F error with the recording's input level (`ml.ae_experiments.recording_level`, as R4); the comparison repeated on a
   level-corrected score log e − β L (R5 convention: L in natural-log power units; β as in item 7); and healthy vs the **three
   quietest fault classes** (lowest median input level among the fit recordings) separately from the loud ones. No
   leave-one-class-out analysis is run for F (not required; none is reported).
6. **Scores on a common scale:** per fold, z = log e − log τ_k for the healthy-AE arms (τ_k = maximum calibration-normal
   error) and z = log τ_F − log e for F; alarm iff z > 0 (ties no alarm); pooled over folds. **Primary operating point:**
   per seed, K_s = A0's pooled healthy false alarms at its calibrated thresholds; every arm's pooled threshold on z is set
   to flag exactly K_s of the 25 test normals (`ml.speed_calibration.matched_threshold`); detection = fault test
   recordings flagged. Also reported: K = 0…5 (0–20 %), each arm at its own calibrated thresholds, AUROC (pooled z and per
   fold), distributions by class, speed band (< 20, 20–30, ≥ 30 Hz) and severity, the mildest setting of each family
   (`anomaly.mildest`).
7. **Exploratory two-score combination (H + F, separate):** each score standardised by its own calibration normals (z minus
   their mean, divided by their SD, in log units, per fold), combined as the mean of A0 and F; same operating points.
   Level-corrected F (item 5): β = pooled within-class OLS slope of log e_F on L over the F model's fit fault recordings
   (class-demeaned), applied as log e_F − β L before the z transform.
8. **Uncertainty:** seeds 42 / 43 / 44 (spread across seeds) and, per seed, sampling: Clopper–Pearson for healthy rates
   (n = 25); for detection differences vs A0 a stratified mixed-unit bootstrap (fault folders within class, normals
   individually; `ml.evaluation.mixed_bootstrap_draws`, the 3A settings) over recordings, never windows, frames or views.
   Never merged.
9. **Acceptance (vs A0 at the matched point):** an arm **helps** only if (1) detection is higher on all three seeds, (2)
   the mean improvement exceeds 4 percentage points, and (3) at the mildest setting of every family the seed-mean
   detection is not lower than A0's. **Does not help** if the mean improvement ≤ 0 or it is not higher on any seed;
   otherwise **too thin to tell**. Nothing is selected or deployed.
10. **Cost:** training seconds per model, parameters, CPU ms per recording (9 windows). Time guard: the script projects the
   total after its first fault-AE fold and stops above 60 min (§10). One GPU job at a time.

## 7-prev6. Previous stage: Stage App-6 — "Explore an unseen fault" research demonstration page

A page that runs a saved **leave-one-class-out CNN-6** (artifacts/final/loco/<class>/, seed 42, §12) on a recording of
the class it never trained on, and shows the checks and the verdict step by step. It is a **simulated unseen fault inside
MAFAULDA**, not evidence that the system recognises genuinely novel real-world faults; the page says so in its own copy.
The normal application, its ten-class bundle, its scoring path, its verdicts, `EvidenceRecord` and `ml/` are unchanged.

**App-6 protocol (declared before writing code).**
1. **Architecture (the showcase pattern).** `artifacts/final/` is gitignored, so the page must work without it:
   `python -m backend.unseen build` (run where the weights and raw recordings exist) sweeps every eligible recording through
   the real models and writes `backend/data/unseen_sweep.json` (gitignored) and the curated `backend/unseen.json`
   (committed, every displayed value). The page serves `backend/unseen.json`; when `artifacts/final/` verifies and the raw
   recordings exist, the API recomputes live and serves the live values (`source` = "live" or "saved"). Nothing is trained at
   request time or in a test.
2. **Eligibility — one predicate, `backend.unseen.may_show`, called by the build step and the request path.** A recording may
   be shown only if `metadata.csv` has `split_closed` ∈ {train, val} AND `split_speed_block` ≠ test AND `split_severity` ≠
   test (the partitions consumed by F1 stay sealed). Role "withheld": class10 = the selected model's withheld class (unseen
   by construction). Role "familiar": class10 ≠ the withheld class, and not in that run's `calibration_scores.csv` nor its
   `thresholds.json → defining_recordings` (as `backend.showcase.threshold_defining` does for the bundle). The predicate lives
   in `backend/`, so `ml/` gains nothing.
3. **Scoring and decision.** Per (withheld class, recording): `ml.final.load_final` (verifies every hash), the raw
   recording via `ml.inference.load_recording` (refuses reserved ids), `ml.final.recording_inputs` → `ml.final.score_cnn`
   with that class's CNN-6, normalisation and the **shared** Mahalanobis (τ_d is calibrated for it; the per-class file is not
   used), plus the shared AE-6 and B1s. τ_p, τ_d, τ_e, τ_b from that class's `thresholds.json`. The verdict is
   `ml.arbitrator.decide("B_with_caution", …)` and `ml.arbitrator.rule_comparison`; the check pass / fail shown is
   `ml.rejection.failed_checks` (the rule's own inequalities, ties accepted), tested equal to the decision. AE-6 and B1s are
   shown only as a possible caution; the page states that they do not gate the verdict under the default rule.
4. **Live vs saved agreement:** a test recomputes every curated example when the weights and raw data are present and
   requires |live − saved| ≤ 1e-9 × max(1, |saved|) per score and identical decisions (ML-4.2 found repeated CPU scoring
   bit-identical); skipped with the reason when either is absent.
5. **Curated set** (deterministic, reasons recorded, like `showcase.py`): per hidden class, a **successful rejection** (a
   withheld recording whose checks fail → "further inspection needed"), an **honest failure** (a withheld recording that
   passes both checks and receives a confident familiar label — a wrong fault or healthy, recorded which), and a **familiar
   comparison** (a learned class, accepted with its correct label, no caution, all rule variants agreeing). Preference: no
   speed-range warning, then the median Mahalanobis distance within the category (a typical case, not the most extreme),
   then rec_id. A category with no eligible recording is reported in `categories_without_an_eligible_recording`, never
   substituted; a missing required category is a stop-and-ask item.
6. **Page:** route `#/unseen`, entry "Explore an unseen fault" in the navigation (desktop and phone). Default hidden class
   imbalance. Steps revealed one at a time after "Run investigation": proposed label → top probability vs τ_p → familiarity
   distance vs τ_d → which checks passed → verdict → the true dataset label (behind "Reveal actual condition", marked as
   hidden from this model during fitting). Mechanic / Developer views as in the main app (technical names in Developer only).
   Required copy: what "further inspection needed" means and does not mean; simulated unseen fault in one dataset from one
   rig; the eligible examples exclude the largest severities and fastest speeds (reserved), so the page shows the easier
   part of the problem. A collapsible "Experiment details": withheld class, seed, recording id, final manifest sha256, source.
7. **Tests:** the predicate rejects a split_closed-test, a split_speed_block-test and a split_severity-test recording; no
   reserved id in `backend/unseen.json`; the API refuses a reserved id; per hidden class the withheld class is absent from the
   model's classes and calibration ids; live = saved (as item 4); the decision shown = `ml.arbitrator.decide` on the same
   scores; the curated set has a rejection and a failure case; the golden and main app tests unchanged.
8. **Addendum (1 Oct, after the build sweep, Bhuvesh's decisions):** four hidden classes have no eligible honest failure (the
   LOCO model rejected every eligible withheld recording: underhang cage 72 / 72, underhang outer 73 / 73, overhang ball
   69 / 69, overhang outer 74 / 74); this is reported plainly (`why_missing`, the example list, Experiment details), never
   substituted. Where a class has both wrong-fault and healthy acceptances (only horizontal misalignment: 1 wrong fault, 36
   healthy of 73), a second honest failure "called healthy" (its median-distance case) is shown beside the declared one.

## 7-prev5. Previous stage: Stage R5 — decision design for the specialist bank (Q3, development data only)

Exploratory further analysis of existing development data. The reserved partitions stay sealed and are not reopened; F1
has already been run, so nothing here can become an independent final claim. Seed 42 only for the screen; seeds 43 / 44
only for a variant that passes, and only after Bhuvesh approves (§10). `metadata.csv`, `artifacts/bundle/`,
`artifacts/final/`, `results/final/` and the app are untouched. Code: pure helpers in `ml/specialists.py`,
`scripts/35_decision_design.py` (no training); notebook `notebooks/06_research_layer/15_decision_design.ipynb`; outputs
`results/research/r5_decision_design.json`, `r5_per_variant.csv`, `r5_per_recording.csv`, figures
`results/figures/r5_level_correction.png` (and `r5_background_difference.png` if E3 runs).

**Stage R5 protocol (declared before running anything; changes only as dated addenda with a reason).**
1. **Data (no training in Step 0, E1, E2).** R4's saved per-recording table `results/research/r4_ambiguity.csv` (11,548 rows
   = 27 runs × their cal / eval / foldk_known / foldk_withheld recordings; per-specialist scores s_u, `level`, speed,
   severity), with the Stage 3C saved scores (`cnn6ae6_s42_loco_decisions.parquet`: CNN-6 top class, MSP, Mahalanobis,
   AE-6 error), B1s and τ_b (`b1s_nomic_loco_q3fit_scores.parquet`) and τ_p, τ_d, τ_e (`results/loco/decision_thresholds.csv`).
   Same 27 runs, roles and leave-one-class-out discipline as R4.
2. **Step 0, the floor (reported first).** Over the 779 withheld instances (one per distinct recording): how many pass the
   arbitrator's healthy gates, AE-6 ≤ τ_e AND B1s ≤ τ_b. The gates come from `ml.arbitrator.decide` itself (helper
   `ml.specialists.healthy_gates`: decide("B_with_caution") with the classifier outcome fixed to a confident normal, so the
   verdict is healthy and the caution is None exactly when both gates pass); R0's tie rule (`ml.rejection.snap_ties`, 1e-12)
   is applied first. Context: the healthy counts of the full rules A, B, C, D, B_with_caution (`decide` on the saved top
   class, MSP and Mahalanobis).
3. **Two compositions.**
   - **Bank composition (R4):** all nine specialists including normal; verdicts as R4.
   - **Gated composition (E1):** the healthy gates decide healthy; a recording failing them goes to the bank of the **eight
     fault specialists only** (normal removed from A(x)): no fault specialist accepts → unfamiliar; otherwise the accepting
     fault set (named fault; ambiguous if several).
   Baselines: **M0**, Mahalanobis-only as in R4 (maha > τ_d → unknown; else the CNN-6 top class, normal → healthy), for bank-
   composition variants; **M1**, Mahalanobis in the gated composition (gates pass → healthy; else maha > τ_d → unknown; else
   the CNN-6 top class if it is a fault; top class normal with failed gates → unknown), for gated variants. The E1 healthy
   count equals the floor and equals M1's by construction; that is stated wherever criterion 2 is read.
4. **Scores.** Unfamiliarity score of a variant = the minimum over its specialists of the (possibly corrected) s_u; τ
   recalibrated per run and per variant as in R4: `ml.protocol.order_statistic_threshold` at 0.975 over ALL the run's
   calibration recordings (never R4's τ_S). Sensitivity (secondary, declared): for gated variants τ also over the run's
   known-FAULT calibration recordings only (the recordings the fault bank actually judges).
5. **E2, level correction.** L = level × ln(10) / 10 (the cached dB level in natural-log power units, so that β = 1 gives
   s − L = log(e / power), the plain energy-relative error); s'_u = s_u − β L. **E2a** β = 1. **E2b** β fitted per run on the
   run's known-class calibration recordings only: the within-class pooled OLS slope of the recording's OWN-class specialist
   score on L (class-demeaned; never a withheld recording). Caveat (declared): e is a squared error of dB values, already
   scale-free in amplitude, so the correction is empirical. Each E2 variant runs in both compositions (E2x-bank vs M0;
   E1+E2x vs M1). Reported per withheld class and by severity, with the loud classes (imbalance, ball faults) called out.
6. **Variants:** R4 (reference, recalibrated), E1, E2a-bank, E2b-bank, E1+E2a, E1+E2b.
7. **Evaluation (every variant):** primary, unknown recall (withheld called unfamiliar / unknown) at a **matched known-side
   rejection count**: per run the variant's threshold is placed so that it calls exactly as many foldk_known recordings
   unfamiliar as its baseline does at τ_d (gated variants: among known recordings failing the gates; if the count cannot be
   reached it is reported); also both at their own calibrated thresholds, labelled each time; withheld called healthy;
   unknown recall at the mildest severity of each family (`anomaly.mildest`); per withheld class; AUROC and OSCR per fold of
   the variant's score (known vs withheld; OSCR correctness = CNN-6 top class); CPU ms per recording (the specialists used:
   9 AEs for bank variants; the gated variants need the AE-6 for the gate and the 8 fault AEs, also 9). Sampling
   uncertainty: Clopper–Pearson for rates; a class-stratified bootstrap over the 779 withheld recordings (2,000 draws, seed
   42, thresholds fixed) for the paired matched-recall difference. One seed: no seed spread yet, and none is merged.
8. **Acceptance bar (vs the composition's Mahalanobis baseline, matched known-side rejection count):** a variant **helps** only
   if (1) pooled unknown recall ≥ baseline + 5 percentage points AND (2) withheld called healthy ≤ baseline. **Does not
   help** if the recall difference ≤ 0 or (2) fails; otherwise **too thin to tell**. Nothing is selected or deployed; no app
   change is proposed unless a variant clears the bar at three seeds.
9. **E3 deferred (Bhuvesh, 1 Oct):** the background autoencoder on the pooled fit set costs ≈ 80 min for seed 42 (285–335
   recordings per run at 0.556 s per recording, from R4's timings), not ≈ 10 min; it is declared in a dated addendum after
   Step 0, E1 and E2 are reported. **E4** runs only if a variant helps, with its own criteria and time budget declared first.

10. **Addendum (1 Oct 2026) — E1 confirmed on seeds 43 and 44 (Bhuvesh; not a new stage; the R5 protocol is unchanged).**
   E1 is the only variant of R1–R7 that cleared its bar, at seed 42 only. **Approved above the §10 hour:** the eight fault
   specialists per run are trained for seeds 43 and 44 with exactly R4's recipe (`ml.train_ae.train_ae`, `AeConfig` from
   base.yaml, the run's fit recordings of the class only, its own normalisation, the withheld class excluded everywhere),
   seed set to 43 / 44 (≈ 72–78 min per seed); R4's time guard is kept per seed (project after the first run; stop that seed
   above 120 min); one GPU job at a time. The normal specialist is the saved 3C AE-6 of the same run and seed (not
   retrained; used only by the ungated R4 bank). `scripts/34_train_specialists.py` and `scripts/35_decision_design.py`
   gain a `--seed` argument (default 42, identical to the seed-42 run; seed-42 files are never overwritten; seeds 43 / 44
   write seed-suffixed paths). Thresholds for seeds 43 / 44 are calibrated per run on its own calibration recordings with
   `ml.calibration.thresholds_from_scores` (τ_p, τ_d the 0.975 order statistic; τ_e, τ_b the maximum over the calibration
   normals; checked to reproduce seed 42's saved thresholds), saved to `results/loco/decision_thresholds_s<seed>.csv`; the
   CNN-6 / AE-6 scores are the saved 3C.1 recording scores (reproduced exactly by R7). Computed per seed through the same
   functions (`ml.specialists`: scoring, calibration, `healthy_gates`, `gated_verdict`; no rule redefined): E1 vs M1 (matched
   and calibrated unknown recall, known-side rejection, withheld called healthy) and R4 vs M0 for context; per withheld
   class, per outer fold, at the mildest setting of each family; CPU ms per recording. E2 is not rerun; E3 stays deferred.
   A test shows seed 42's saved R5 numbers are reproduced exactly after the change. **Reading (three seeds together):** E1
   **helps, confirmed on three seeds** only if on each of seeds 42, 43 and 44 matched unknown recall ≥ M1 + 5 points AND
   withheld called healthy ≤ M1's; otherwise **seed-dependent** (the failing seed and condition named; the seed-42 result
   then described as not robust). Mean gain, worst seed and seed spread are reported; seed spread and sampling uncertainty
   are never merged. Outputs: seed-suffixed R4 per-run files, `results/research/r5_seeds.json`, `r5_per_variant.csv` with a
   `seed` column, notebook 15 extended with a "Seeds 43 and 44" section. Nothing is selected or deployed.

## 7-prev4. Previous stage: Stage R4 — condition-specialist autoencoder bank for unfamiliar-fault detection (Q3, development data only)

Research question: can reconstruction by per-condition specialists identify recordings of a fault class the system never
trained on better than the current Mahalanobis distance (shared Ledoit–Wolf) on the CNN-6 embedding? Exploratory
development experiment, **seed 42 only** (Bhuvesh, 30 Sep: the estimate for one seed is ≈ 87–95 min of MPS training, above
the one-hour line of §10; approved for seed 42, the Stage D1 precedent; single seed is a stated limitation). Reserved tests
stay sealed; `metadata.csv`, `artifacts/bundle/`, `artifacts/final/`, `results/final/` and the app are untouched. Code:
`ml/specialists.py` (pure: run plan, score calibration, combining rule, verdicts), `scripts/34_train_specialists.py`;
notebook `notebooks/06_research_layer/14_specialist_bank.ipynb`; outputs `results/research/r4_specialists.json`,
`r4_per_run.csv`, `r4_ambiguity.csv`, `results/figures/r4_unknown_recall.png`, `r4_specialist_scores.png`; weights
`artifacts/specialists/` (gitignored, state_dict).

**Stage R4 protocol (declared before any training; changes only as dated addenda with a reason).**
1. **Runs.** The 27 Stage 3C runs (9 withheld classes w × outer folds k = 0–2, seed 42) on the saved assignment
   (`results/q1q3/q3_fit_cal_eval_assignment.csv`, `ml.openset.loco_run_ids`); roles fit, cal, eval, foldk_known,
   foldk_withheld as in 3C.
2. **The bank: nine specialists per run** (normal + the eight fault classes other than w; **w has no specialist**). Each
   specialist u is the Stage 3B/3C frame-level autoencoder (`ml.train_ae.train_ae`, `AeConfig` from base.yaml unchanged:
   conv 32/64/64, kernel 5, bottleneck 16, 150 epochs, batch 256, AdamW lr 1e-3, weight decay 1e-4, warm-up 5 epochs +
   cosine, gain jitter ±1 dB, target = the augmented input, seed 42) on cache C, vibration channels 0–5, trained ONLY on
   the run's fit recordings of class u, with its own normalisation from `ml.datasets.norm_stats` in development mode on
   exactly those recordings (context = the run's calibration, evaluation and withheld recordings, which development mode
   refuses as fit ids). **Normal specialist = the saved Stage 3C AE-6 of the run** (identical recipe, the same fit normals,
   its saved normalisation; Bhuvesh, 30 Sep), not retrained. The withheld class is therefore excluded from every
   specialist's training, every normalisation statistic and all calibration. Training set sizes per run: faults 29–67 fit
   recordings (261–603 windows), normal 10–12; every class is trained (none is too small by this recipe's precedent, normal
   at 10–12 recordings being the AE-6 itself), and the sizes are reported.
3. **Specialist error:** recording error e_u(x) = mean over the 9 windows of the per-window reconstruction error (squared
   error in the specialist's normalised units, mean over channels and frames; `window_channel_errors`, as AE-6).
4. **Comparable scores (Bhuvesh, 30 Sep): log-ratio to the specialist's own calibration median.**
   s_u(x) = log e_u(x) − log median{e_u(c) : c a calibration recording of class u in the run}. For a calibration recording of
   class u itself, the median excludes that recording (leave-one-out; normal: the median of the other 1–2). Why not
   `ml.openset.empirical_rank`: it is capped at n/(n + 1) and normal has only 2–3 calibration recordings per run (cap
   0.67–0.75), so under any 97.5 % combined threshold the normal specialist would accept every recording; empirical rank is
   reported as a secondary diagnostic with the cap stated. Per-specialist 97.5 % order-statistic thresholds are unattainable
   (r = ceil(0.975 (n + 1)) > n for n < 39; the class calibration sets have 2–18) — reported as a finding.
5. **Combining rule and calibration.** S(x) = min_u s_u(x) (the most compatible specialist). τ_S =
   `ml.protocol.order_statistic_threshold` of S over ALL the run's calibration recordings (known classes including normal),
   acceptance 0.975 (the 3C convention). The same τ_S applies to every specialist: specialist u accepts x iff s_u(x) ≤ τ_S;
   A(x) = the set of accepting specialists. A(x) empty (equivalently S > τ_S) → **potentially unfamiliar**; otherwise
   compatible with the conditions in A(x), and **the whole set is kept** (|A| ≥ 2 = ambiguous; counted, never collapsed).
   Bank verdict: A = {normal} → healthy; A non-empty and without normal → known fault (ambiguous if |A| ≥ 2); normal ∈ A and
   |A| ≥ 2 → ambiguous healthy / fault; A empty → unfamiliar. R0's tie rule carries forward (a score within 1e-12 of its
   threshold is a tie, accepted).
6. **Comparison: the saved Stage 3C Mahalanobis (shared) check** on the same runs (`cnn6ae6_s42_loco_decisions.parquet`,
   `results/loco/decision_thresholds.csv`, τ_d), known side = foldk_known (fold-k known recordings incl. normals), unknown side
   = foldk_withheld:
   (i) **primary, threshold-free:** per-run AUROC (known vs withheld) and OSCR (`ml.openset.oscr`; correctness = the 3C CNN-6
   top class, for both scores) of S vs Mahalanobis; per outer fold the mean over its 9 runs; mean and spread across folds;
   per withheld class the mean over its 3 folds;
   (ii) unknown recall at a **matched known-side rejection count**: per run, the bank threshold is placed to reject exactly as
   many foldk_known recordings as Mahalanobis rejects at τ_d; also both at their own calibrated thresholds;
   (iii) unfamiliar recordings called healthy vs called a known fault, separately: the bank verdicts vs the Mahalanobis-only
   decision (accepted → the CNN top class: normal → healthy, else that fault); rule B of 3C (MSP and Mahalanobis) as context;
   (iv) ambiguity: how often |A| ≥ 2 (by role and class), which specialists co-accept, and its association with speed and
   severity; per-instance table `r4_ambiguity.csv`;
   (v) speed bands (< 20, 20–30, ≥ 30 Hz) and severity levels within the development pool;
   (vi) cost: training seconds, total parameters of the 9 specialists vs one CNN-6 + Mahalanobis, CPU latency per recording
   (9 windows from the cache);
   (vii) **generalisation check:** per specialist, reconstruction and acceptance of the withheld class (median s_u, acceptance
   rate) against its acceptance of the other foreign known classes, the share of withheld recordings accepted by the normal
   specialist, and the association of S with the recording's input level (mean cached log-spectral level, vibration channels).
7. **Pre-declared reading** (Δ_k = bank − Mahalanobis in fold-k mean AUROC): **helps** only if all hold: (a) Δ_k > 0 in all 3
   folds; (b) mean Δ ≥ +0.02; (c) pooled unknown recall at the matched known-side rejection count higher than Mahalanobis's by
   ≥ 5 percentage points; (d) the per-withheld-class mean AUROC higher for at least 5 of the 9 classes. **Does not help** if
   Δ_k ≤ 0 in all 3 folds or mean Δ ≤ −0.02. Otherwise **too thin to tell**. Secondary, per class: "helps for withheld X"
   only if the bank's AUROC is higher in all 3 folds of X. Unfamiliar recordings called healthy are reported next to every
   verdict.
8. **Time guard:** after the first run the total is projected; above 120 min the stage stops and is reported. One GPU job at
   a time (`artifacts/gpu.lock`).

## 7-prev3. Previous stage: Stage R3 — contrastive encoder for known-condition recognition (Q2, development data only)

Research question: does training the encoder with a supervised contrastive objective, instead of cross-entropy alone,
improve recognition of known conditions across operating settings and speeds, and separate the confusable pairs?
Exploratory development experiment. Reserved tests stay sealed; `metadata.csv`, `artifacts/bundle/`, `artifacts/final/`,
`results/final/` and the app are untouched. Code: `ml/contrastive.py` (pure: loss, pair sampler, embedding model,
training function reusing `ml.train`), `configs/contrastive.yaml` (separate from base.yaml so the packaged bundle config
hash is unchanged), `scripts/33_train_contrastive.py`; notebook `notebooks/06_research_layer/13_contrastive_encoder.ipynb`;
outputs `results/research/r3_contrastive.json`, `r3_per_fold.csv`, `r3_confusion.csv`, `results/figures/r3_confusion.png`,
`r3_embedding.png`; weights `artifacts/contrastive/` (gitignored, state_dict).

**Ceiling, stated before training.** On these folds CNN-6 misclassifies 8 / 11 / 4 of 804 development recordings (seeds
42 / 43 / 44; class10 macro-F1 per fold 0.969–1.000). The named pairs account for 0 errors (imbalance ↔ vertical
misalignment) and 2 (horizontal ↔ vertical misalignment, seed 42); most errors are underhang cage ↔ overhang cage. The
development pool runs below 40 Hz and excludes the largest severity of each family, so this stage cannot measure the
development-to-final gap (0.990 → 0.916); it can only detect small changes in a handful of recordings.

**Stage R3 protocol (declared before any training; nothing here changes after a result; changes only as dated addenda).**
1. **Data and folds.** Development pool, `results/dev_folds.csv` (K = 3, folder-grouped, CLAUDE.md §5 rule 12); cache C
   (the CNN-6 representation, `ml.spectro.stacked_representation`), vibration channels 0–5; normalisation per fold from
   `ml.datasets.fold_norm_stats` (development mode, fit = the other two folds' recordings). Every window of a recording is in
   one partition; every pool recording is scored once, by its own fold's model; decisions per recording (mean
   log-probability over the 9 windows, `ml.evaluation.recording_predictions`, prob floor as 3A).
2. **Encoder = CNN-6 unchanged** (`ml.models.cnn.ScratchCNN`, the `cnn:` settings of base.yaml: widths 32/64/128/128, 3
   pooled blocks, 128-d embedding z = ReLU(BN(Linear)), dropout 0.2 + linear head). **Projection head** (training-time
   loss component only, not part of the encoder, discarded after stage A): Linear 128→128, ReLU, Linear 128→64, output
   L2-normalised. The deployed model is a ScratchCNN with the same parameter count as CNN-6.
3. **Stage A, supervised contrastive pretraining (SupCon, Khosla et al. 2020, "L_out").** For anchor i in a batch,
   positives P(i) = windows of the same class10 from a DIFFERENT recording; the candidate set A(i) = every other window of
   the batch except windows of i's own recording. loss_i = −(1/|P(i)|) Σ_{p∈P(i)} log[exp(s_ip/T) / Σ_{a∈A(i)} exp(s_ia/T)],
   s = cosine similarity of the projections, T = 0.1; anchors without a positive are skipped; loss = mean over anchors.
   **Same-recording positives: excluded entirely (fraction 0)**, both as positives and from the denominator, and there is
   one view per window (no two augmented views of one window), so the objective cannot be met by recognising a recording.
   **Negatives** = every window of a different class10 in the batch. Folders are class-specific, so every negative pair
   is necessarily across folders; negatives are not sub-sampled.
4. **Pair sampler (class-balanced batches).** Per batch: for each of the 10 classes, 24 recordings drawn without
   replacement (all of the class's fit recordings if it has fewer; normal has 16–17 per fold), with probability ∝ 1 /
   (number of fit recordings in that recording's folder), so folders are equally represented and positives cross folders
   where the class has more than one folder (normal has one folder, so its positives never do); one window per recording,
   uniform over its 9 windows. Batches per epoch = ceil(fit windows / 256), as CNN-6. Realised pair statistics are recorded
   for every run: positive pairs total, from different recordings (must be 100 %), different folders, and different
   speeds (|Δ measured speed| ≥ 1 Hz), with the median |Δ speed|.
5. **Augmentations: exactly CNN-6's**, applied per window before the encoder in both stages: gain offset uniform ±1 dB per
   window and channel (before normalisation), circular time shift ±2 frames, one time mask of 0–4 frames. **Gain argued
   explicitly (§5 rule 7):** it is kept only so that the objective is the single difference from CNN-6; ±1 dB is small
   against the level differences between conditions, is independent per channel, and does not alter the order axis or
   the relative spectral shape; normalisation stays global per channel. No augmentation touches the order axis or speed
   (**no speed invariance is encouraged**: positives are not selected to differ in speed, and speed information stays in
   the representation, consistent with R1, where removing the speed trend cost detection), and none simulates a fault.
6. **Stage A settings (fixed):** 100 epochs, AdamW lr 3e-3, weight decay 0.01, linear warm-up 1 epoch then cosine decay
   to 0 (`ml.train.lr_lambda`), fp32 on MPS, no early stopping, nothing chosen on the evaluated fold.
7. **Stage B, classifier head: encoder FROZEN** (no gradients; its BatchNorm layers stay in eval mode), the same for every
   seed and fold. The CNN-6 head (dropout 0.2 + Linear 128→10), newly initialised, is trained with `ml.train.train_model`
   unchanged: cross-entropy with balanced class weights, the same augmentations, AdamW lr 3e-3, weight decay 0.01,
   warm-up 1 epoch + cosine, 40 epochs, batch 256.
8. **Seeds 42, 43, 44** (`ml.utils.seed_everything`, sampler and augmentation generators seeded with the run seed),
   compared with the saved CNN-6 runs of the same seed on the same folds (seed 42: `results/scores/cnn6_dev.parquet`;
   43 / 44: `cnn6_s43_dev_logits.parquet`, `cnn6_s44_dev_logits.parquet`).
9. **Time estimate and guard.** From CNN-6 (≈ 60 s per 40-epoch fold): ≈ 150 s stage A + ≈ 30 s stage B per fold, ≈ 30
   min for 9 fold × seed runs. After the first fold the total is projected; above 60 min the run stops and is reported
   (§10). One GPU job at a time (`artifacts/gpu.lock`).
10. **Evaluation** (recording level; contrastive vs CNN-6, same seed): class10 and class6 macro-F1 and balanced accuracy
    per fold, mean and spread across folds (§5 rule 13); per-class recall; class10 confusion matrices with imbalance ↔
    vertical misalignment and horizontal ↔ vertical misalignment called out; accuracy per folder (operating setting) and
    per speed band (< 20, 20–30, ≥ 30 Hz); seed spread (std over seeds) reported separately from sampling uncertainty
    (paired mixed-unit bootstrap per seed, `ml.evaluation.paired_mixed`, the 3A draws); median CPU latency per recording
    (9 windows from the cache, the 3A method) against CNN-6.
11. **Pre-declared reading** (Δ_s = contrastive − CNN-6 of the mean over folds of class10 macro-F1, seed s):
    **helps** only if all hold: (a) Δ_s > 0 for every seed; (b) mean Δ over seeds ≥ +0.003; (c) in every seed the
    contrastive per-fold macro-F1 is ≥ CNN-6's in at least 2 of 3 folds; (d) class6 mean Δ ≥ −0.003.
    **Does not help** if Δ_s ≤ 0 for every seed, or mean Δ ≤ −0.003. Otherwise **too thin to tell**.
    Pairs (secondary): "separates pair X better" only if its confusions (recordings, both directions) are fewer than
    CNN-6's in every seed; where CNN-6 has none, only a worsening can be observed, and that is said.
12. **Optional Q3 step** (Mahalanobis familiarity on the new embedding, `ml.openset.fit_mahalanobis`, saved Q3
    assignment, withheld class excluded from fitting, normalisation and calibration) runs **only if the reading is
    "helps"**. R0's tie rule (a score within 1e-12 of its threshold is a tie, accepted) carries forward to any threshold
    comparison.

## 7-prev1. Previous stage: Stage R1 — speed-conditioned healthy thresholds (development data only)

Exploratory research experiment testing R0's hypothesis (healthy false alarms concentrate at speeds outside the calibration
normals' range). No training, no new model, no threshold change in the bundle or the app, no reserved read; `artifacts/`
(bundle, final, loco), `results/final/` and `metadata.csv` are untouched; nothing is written to `artifacts/`. Recordings
are never moved between fit, calibration and evaluation. Code: `ml/speed_calibration.py` (pure), the fit-normal re-score
`scripts/32_r1_fit_normal_scores.py`, notebook `notebooks/06_research_layer/12_speed_conditioned_thresholds.ipynb`;
outputs `results/research/r1_fit_normal_scores.parquet`, `r1_speed_thresholds.json`, `r1_per_fold.csv`,
`r1_healthy_outcomes.csv`, figure `results/figures/r1_score_vs_speed.png`.

**Stage R1 protocol (declared before running).**
1. **Data.** The 27 Stage 3C seed-42 LOCO runs on the saved assignment (`results/q1q3/q3_fit_cal_eval_assignment.csv`,
   `ml.openset.loco_run_ids`); saved AE-6 errors (`results/scores/cnn6ae6_s42_loco_decisions.parquet`), saved B1s and τ_b
   (`results/scores/b1s_nomic_loco_q3fit_scores.parquet`), τ_p, τ_d, τ_e (`results/loco/decision_thresholds.csv`); speed =
   `fr_measured_hz`. Per run: **fit normals** (10–12; the AE-6 / B1s training normals), **calibration normals** (2–3; set
   τ_e, τ_b), **evaluation** = roles eval, foldk_known, foldk_withheld (normals: 299 instances of 25 distinct recordings
   pooled with the others as in R0; faults: every fault recording of those roles, the withheld class included).
2. **Fit-normal re-score (Bhuvesh, Option A; strictly limited).** Only each run's healthy FIT normals are scored; never a
   calibration, evaluation, fault or reserved recording. AE-6: inference only, with the saved Stage 3C weights
   (`artifacts/loco/ae6_w-<c>_s42_loco/fold<k>.pt`, read only) and that run's saved AE normalisation
   (`fold<k>_norm.json`), on the dev-pool cache C, exactly the 3C scoring (`window_channel_errors`, frame level, mean over
   channels, mean over the 9 windows), on the device 3C used (MPS; one GPU job, `artifacts/gpu.lock`); no retraining or
   re-initialisation. B1s: the deterministic Stage 3D refit
   (`scripts/09c_loco_complementary.b1s_scores` recipe) is used ONLY after it reproduces that run's saved cal/eval B1s
   scores within R0's tie tolerance 1e-12 (max absolute difference); a run that fails is reported **infeasible** for B1s
   and no approximation is substituted. The same check is applied to AE-6 against the saved cal/eval AE errors (MPS is not
   bit-deterministic, so the max absolute difference is recorded; a run is used for AE only if it is ≤ 1e-5 × the run's
   τ_e, otherwise infeasible for AE). The per-run
   reproduction evidence is recorded in `r1_speed_thresholds.json`. No threshold is changed, no model is created.
3. **In-sample bias (required disclosure).** The fit normals are the recordings AE-6 was trained on and B1s was fitted on,
   so their scores are in-sample and optimistically low. Fitting m(speed), d(speed) on them biases the standardised score
   upward for calibration and evaluation recordings, which works AGAINST the new method: a positive result is therefore
   conservative and meaningful; a negative result is partly attributable to this bias and does not by itself refute
   speed-conditioning; the clean alternative (out-of-fold scores on the fit normals) does not exist in this project. This
   is stated in the Findings, the JSON and beside every headline number.
4. **Transformation** (per run, per detector, AE-6 and B1s separately): s_std = (s − m(v)) / d(v), x = log(speed Hz),
   fitted on the run's fit normals only. m: least-squares polynomial in x, degree 2 (primary cap) and degree 1. d: scale of
   the residuals r = s − m(x), degree 1 in x with a log link, d(x) = exp(a + b x), by maximum likelihood of a Laplace scale
   (minimise Σ [log d + |r| / d]; convex, always positive). x is clipped to the fit normals' range before m and d are
   evaluated (constant extrapolation; no polynomial runs away outside the fitted speeds). Variants: raw; std-m2d1; std-m1d1.
   If m2 beats m1 only marginally (macro AUROC gain < 0.01, or not higher in at least 2 of 3 fold means), degree 1 is
   preferred and that is said.
5. **Threshold:** τ_std = the maximum s_std over the run's calibration normals (`ml.protocol.max_threshold`, the existing
   rule); accept iff s_std ≤ τ_std; R0's tie rule carried forward (a score within 1e-12 of its threshold is a tie, accepted).
6. **Evaluation**, each detector separately, never a combined number:
   (i) **primary:** per-run AUROC healthy (evaluation normals) vs faulty (evaluation faults), raw vs each standardised
   variant; summarised per outer fold (mean over its 9 runs) and as mean and spread across the 3 folds, plus the 27 runs;
   (ii) healthy false alarms (score above its threshold) on the same 299 instances / 25 recordings, raw vs std, by speed
   band (below / inside / above the run's own calibration normals' speed range; the bundle range 18.08–38.34 Hz as a
   secondary band), instances and distinct recordings with Clopper–Pearson intervals;
   (iii) **cost side (mandatory):** fault detection at a matched healthy false-alarm rate: per run, the std threshold is
   placed so that the same number of evaluation normals is flagged as by the raw rule at its calibrated τ; detection per
   fault class and at the mildest settings (`anomaly.mildest`), raw vs std; also detection at the calibrated thresholds;
   (iv) three-way outcomes (healthy / healthy with caution / known fault / unknown) of rules A, B, C, D and B_with_caution
   (`ml.arbitrator.decide`), old (raw scores, τ_e, τ_b) vs new (std scores, τ_std for AE and B1s; MSP and Mahalanobis
   unchanged), on evaluation normals and faults.
7. **Pre-declared reading** (per detector): "speed-conditioning helps" only if (a) the std AUROC fold mean is ≥ raw in all
   3 folds and higher on average, (b) healthy false alarms (distinct recordings) at the calibrated threshold do not rise,
   and (c) pooled fault detection at matched false-alarm rate is not lower; "does not help" if the std AUROC fold mean is
   lower in all 3 folds; otherwise "too thin to tell". Every speed conclusion repeats: 25 recordings from one folder (one
   recording = 4 points); speed and recording identity are completely confounded; only 5 recordings lie outside the
   calibration range; R0's hypothesis remains a hypothesis.

## 7-prev0. Previous stage: Stage R0 — diagnose the existing healthy rejections (development data only)

Exploratory analysis, no training, no new models, no scoring of recordings, no threshold tuning or re-derivation, no fix.
Reserved tests stay sealed (no reserved recording is read); `artifacts/bundle/`, `artifacts/final/` and `metadata.csv`
are untouched; `results/final/` is historical context only, never a tuning target. Question: which checks reject healthy
development recordings, and under what conditions, before any false alarm is attributed to a model. Code:
`ml/rejection.py` (pure); notebook `notebooks/06_research_layer/11_rejection_analysis.ipynb`; outputs
`results/research/r0_rejection_analysis.json`, `results/research/r0_attribution.csv`.

**Stage R0 protocol (declared before running).**
1. **Healthy sample.** The 25 development-pool normals (`ml.splits.development_pool`; 20 closed train, 5 closed val),
   all from one folder (`ml.splits.folder_proxy` = `normal/-`), so a folder breakdown is degenerate and is reported as
   such; the grouping used instead is the saved outer fold / run. Every Q1 number states: 25 recordings from one folder,
   with Clopper–Pearson intervals over distinct recordings (with n = 25, one recording moves a rate by 4 points).
2. **Three evidence sources, each with its own models and its own thresholds (never mixed):**
   - **(A) Deployed bundle**, thresholds from `artifacts/bundle/manifest.json`, scores from
     `artifacts/bundle/calibration_scores.csv`: only the 5 calibration normals are scored. They SET τ_e and τ_b (each the
     maximum over them) and are among the 144 that set τ_p and τ_d: **in-sample, not a false-alarm rate**. The 20
     bundle-training normals have no saved bundle scores; they are reported as not available (this stage scores nothing).
   - **(B) Q1 development folds** (Stage 2D.1 / 3B, `results/q1q3/q1_fit_cal_assignment.csv`): standalone B1s
     (`b1s_nomic` = the bundle's vibration-only variant; `b1s_all` for comparison) and standalone AE-6, each against its
     own per-fold threshold. Roles: `cal` (17 instances; set the threshold; in-sample) and `test` (25 instances, each
     normal once; used for neither fitting nor thresholds; out of sample).
   - **(C) Stage 3C LOCO runs** (seed 42, 27 runs, `results/q1q3/q3_fit_cal_eval_assignment.csv`): per-run CNN-6 top class,
     MSP and Mahalanobis (shared) and AE-6 (`results/scores/cnn6ae6_s42_loco_decisions.parquet`), B1s with its τ_b
     (`results/scores/b1s_nomic_loco_q3fit_scores.parquet`), and the run's τ_p, τ_d, τ_e (`results/loco/decision_thresholds.csv`).
     The **full rule** is `ml.arbitrator.decide` (B_with_caution) on these scores and thresholds; standalone checks are
     reported separately. Roles: `cal` (71 instances; set the run's thresholds; in-sample) versus not used: `eval` (74,
     seen-folder) and `foldk_known` (225, the unseen outer fold). Each normal recurs across runs, so rates are given per
     distinct recording (primary) and per instance.
3. **Failed checks** use the rule's own inequalities (ties accepted): msp < τ_p, maha > τ_d, ae > τ_e, b1s > τ_b, plus a
   fifth cause, **top class not normal** (a confident non-normal top class is a known-fault verdict with no failed check).
   `ml/rejection.py` does not import `backend` (the dependency runs backend → ml); a test shows it agrees with
   `backend.checks.checks` and with `ml.arbitrator.decide` on the same scores.
4. **Outcomes** of the full rule on a normal, reported separately: plain healthy; healthy with caution (ae and/or b1s
   above its limit); abstention (msp and/or maha fails); known fault (top class not normal). For every non-accepted
   normal, the failed checks and their **joint pattern** (combination) are recorded, with counts.
5. **Breakdown** by measured speed (`fr_measured_hz`), against the calibration-normal speed range in the bundle manifest
   and against each run's own calibration normals' speed range; and by outer fold / withheld-class run.
6. **Attribution statement:** a model is named only if the evidence singles it out; otherwise "it does not separate".
   Hypotheses go to Findings; nothing is fixed or tuned in this stage.
7. **Addendum (found while implementing item 3, before any Finding was written):** in the saved Stage 3C files the
   recording that defines a run's τ_p (the order statistic) is stored one floating-point step (1.1e-16) below the stored
   τ_p, so a plain re-comparison would reject it although the run accepted it as a tie. A score within 1e-12 of its
   threshold is therefore treated as a tie (`ml.rejection.snap_ties`; accepted; the snapped instances are recorded).
   With it, the attribution reproduces the saved 3C msp_ok / maha_ok flags on all 11,548 rows and rule A reproduces the
   saved classifier-first decisions exactly. Thresholds are unchanged.

## 7-prev. Previous stage: Stage D1 — nested learned unfamiliarity combiner (development data only)

Exploratory development experiment, seed 42 only; the candidate was designed after seeing the 3D and ML-5 results. F1
stays unrun; reserved tests stay sealed; the bundle, `artifacts/final/`, §12 and the app are unchanged. Nothing enters F1
unless Bhuvesh amends §12 before F1. Code: `scripts/31_d1_combiner.py`; outputs `results/loco/d1/`.

**Stage D1 protocol (declared before any training).** For each of the 27 Stage 3C LOCO runs (withheld class w, outer
fold k, seed 42; the saved Stage 2D.1 Q3 assignment, `ml.openset.loco_run_ids`) with fit set F, calibration set C and the
3C evaluation groups: (a) known-fault recordings of seen folders (role `eval`), (b) known-fault recordings of the unseen
outer fold k (`foldk_known`), (u) withheld-class recordings of fold k (`foldk_withheld`), and the normals of fold k and of
the seen folders.
1. **Inner data (w is never used anywhere).** F is split into halves H1 / H2 by folder: within each known class its
   folders (every fault class has ≥ 2 folders in every F) are put in a seeded random order (seed 42) and assigned
   alternately to H1 and H2; **normal-folder exception:** normal has a single folder, so its recordings are split by
   recording in the same way. For each fault class u ≠ w (8 per run): an inner CNN-6 (Stage 3A recipe and fixed
   settings, seed 42, vibration channels, `norm_stats` in development mode on its training recordings) and an inner LR
   (the classical reference, `baselines.logreg` on the capped R1 table) are trained on H1 minus u. They score the H2
   known-class recordings excluding u (**inner-known rows**, target 0) and every recording of u in F (**inner-unfamiliar
   rows**, target 1). 8 inner models per run × 27 runs = 216 CNN-6 + 216 LR. **Time guard:** after the first run the
   total GPU time is estimated; above 3 h the stage stops and is reported.
2. **Features** (identical for inner and outer rows; the scoring model = the inner models for inner rows, the 3C outer
   CNN-6 and the 2D.1 outer LR of the run for outer rows): CNN 1 − MSP; CNN Mahalanobis with one shared within-class
   Ledoit–Wolf covariance fitted on the scoring CNN's own training windows (`ml.openset.mahalanobis_min`); LR 1 − MSP;
   AE-6 error and B1s, both from the run's OUTER healthy-only models for every row (the 3C AE-6 of the run; B1s refitted
   on the run's AE fit normals exactly as Stage 3D); CNN top-1 minus top-2 recording probability gap; CNN ≠ LR top-class
   flag (0/1); measured recording speed. **The signal-quality warning flag of the brief is dropped (Bhuvesh, 28 Sep):
   CLAUDE.md §3 forbids quality flags as model inputs.** Every continuous feature (all except the 0/1 flag) is
   rank-normalised with the 3D convention r(x) = #{reference ≤ x} / (n + 1) against the scoring model's own held-out
   known reference: the H2 known rows of that inner model for inner rows, C for outer rows. Caveat (declared): the AE-6
   and B1s were fitted on the fit normals, so their values on F's normals (inner rows) are in-sample.
3. **Combiner (one per run):** L2 logistic regression, C = 1.0, scikit-learn defaults otherwise (lbfgs, no class
   weights, intercept), fixed; target inner-unfamiliar = 1, inner-known = 0; fitted only on that run's inner rows (all 8
   inner models pooled). It never sees C, the outer evaluation sets or any w recording.
4. **Decision rule D1:** the combiner scores C (outer features); τ_s = the 97.5 % calibration order statistic (3D
   convention; accept iff score ≤ τ_s, ties accepted); C is used only for τ_s. With the outer CNN's top class: normal and
   score ≤ τ_s → healthy; a fault and score ≤ τ_s → that fault; otherwise unknown. The combiner check REPLACES the MSP and
   Mahalanobis checks; no AE override. Caution flags as in B_with_caution: a healthy verdict with AE > τ_e and/or
   B1s > τ_b (τ_e, τ_b: the run's maximum calibration-normal values, as 3C / 3D).
5. **Evaluation** (instances and distinct recordings; each recording counted in its own run only): per withheld class
   and macro (mean over folds per class, then over classes): per-run AUROC and OSCR (fold-k known vs withheld; OSCR
   correctness = the outer CNN's top class) of the combiner score vs CNN Mahalanobis (shared); three-way outcomes on
   (u); false-unknown rates on (a) and (b) (fault recordings); normals healthy / unknown. Comparison rows: rules B and
   B_with_caution (identical decisions; caution counted) from the 3C scores, seed 42 (`ml.loco_eval.decide`). Paired
   bootstrap as in Stage 3C.1: one stratified mixed-unit draw over the whole development pool weights every run's
   recordings (runs lacking a group in a draw are dropped; the caveat of 3C.1 applies). Combiner coefficients per run,
   with imbalance and vertical misalignment called out.
6. **Pre-declared pass criteria (reported pass/fail; nothing is selected):** (i) macro AUROC of the combiner > CNN
   Mahalanobis (shared), with the paired-bootstrap 95 % lower bound of the difference > 0; AND (ii) fewer (u) instances
   passed as a known fault than rule B, with the pooled false-unknown rate on (a) and on (b) each ≤ rule B's + 2
   percentage points.
7. **Wording:** exploratory development results, seed 42 only; candidates designed after seeing 3D / ML-5. A fail means
   this combiner did not improve these results, not that the representation is the proven limit. If it passes, a final
   version would generate inner-unfamiliar rows from each final LOCO model's fit set in the same way, set τ_s on its
   calibration set, and be declared as a secondary Q3 decision layer, only if Bhuvesh amends §12 before F1.
- **Tests in this stage** run with `MACHINESENSE_REQUIRE_BUNDLE=1 MACHINESENSE_NO_RESERVED_READ=1`: no reserved raw array
  is read (the backend guard tests use the existing `backend/data/reserved_fingerprints.json` as is, never rebuilt; the
  two tests that open a reserved signal are skipped and listed).

## 7a. Previous stage: Stage ML-5 — ResNet-18 transfer learning (Track A, development experiment)

Question: does an ImageNet-pretrained ResNet-18 on the order spectrogram match CNN-6 on closed-set diagnosis (Q2), and
does it separate the imbalance / vertical-misalignment pair better when one is withheld (targeted Q3)? It does **not**
change the frozen bundle or the app; ResNet-18 is not added to the bundle (Bhuvesh decides after the report).

**Stage ML-5 protocol (declared before running; code `ml/models/resnet.py`, `configs/resnet.yaml`,
`scripts/20_resnet.py`).**
- **Input (as CNN-6):** cache C, the 6 vibration channels × 128 log order bins × 25 frames; global per-channel
  normalisation from `norm_stats` in development mode fitted on the training recordings of each fold only.
- **Architecture:** learnable 1×1 conv adapter 6 → 3 channels (with bias) → bilinear resize to 224 × 224
  (align_corners = False) → ImageNet mean/std normalisation → torchvision ResNet-18 with
  `ResNet18_Weights.IMAGENET1K_V1` (file `resnet18-f37072fd.pth`, sha256
  f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec; torchvision 0.29.0) → the 512-d penultimate
  embedding z (after global average pooling) → dropout 0.2 → new linear head (10 classes). The dropout matches CNN-6's head.
- **Kept from CNN-6 (Stage 3A):** cross-entropy with balanced class weights n / (n_classes × count_c) from the training
  windows; augmentation on training windows only (gain jitter ±1 dB per window and channel before normalisation; circular
  time shift ±2 frames and one time mask of 0–4 frames after normalisation; applied to the 6 × 128 × 25 input, before
  the adapter); AdamW with weight decay 0.01; linear warm-up for 1 epoch, then cosine decay to 0; fixed epochs, no
  early stopping, no epoch or checkpoint selection on any evaluation data; window → recording aggregation = mean
  log-probability over the 9 windows (`ml.evaluation.recording_predictions`, prob floor as 3A); every pool recording
  scored exactly once by its own outer-fold model; fp32 on MPS, batch 64 (evaluation batch 256).
- **Variants:** R18-probe: backbone frozen (no gradients; BatchNorm layers kept in eval mode with the ImageNet running
  statistics); the adapter and the head are trained, lr 1e-3, 30 epochs. R18-ft: full fine-tune, backbone lr 1e-4,
  adapter and head lr 1e-3, 30 epochs.
- **Time guard:** R18-ft seed 42 runs first; after the first epoch of fold 0, the B1 seed-42 total (both variants,
  3 folds) is estimated as 2 × (R18-ft estimate) (the probe costs no more than the fine-tune); if the estimate exceeds
  90 minutes the run stops and is reported. Seeds 43 and 44 of R18-ft run only if seed 42 (both variants) took under
  1 hour in total.
- **B1 (Q2, dev folds K = 3, `results/dev_folds.csv`):** class10 and class6 macro-F1 and balanced accuracy with the 3A
  mixed-unit bootstrap (same draws for every model); paired differences R18 − CNN-6 on the same folds and seeds (CNN-6
  seed 42: the Stage 3A outputs; seeds 43 / 44: Stage 3B); per-class F1 and the class10 confusion matrix, with imbalance
  vs vertical misalignment and horizontal misalignment at 0.5 mm reported separately. Seed variation and bootstrap
  intervals are reported separately, never merged.
- **B2 (targeted Q3, R18-ft seed 42 only, 6 runs):** withheld ∈ {imbalance, vertical_misalignment} × outer folds 0–2 on
  the SAVED Stage 2D.1 Q3 assignment (`results/q1q3/q3_fit_cal_eval_assignment.csv`, `ml.openset.loco_run_ids`). The
  withheld class is excluded from training, normalisation (fit ids = fit recordings; context as 3C), Mahalanobis
  fitting and calibration. Scores: MSP (lower = more unknown) and Mahalanobis on the 512-d embedding of the fit windows
  with **per-class Ledoit–Wolf covariances** (minimum over the known classes per window, mean over windows; the brief's
  primary). Stage 3C's CNN-6 used one **shared** within-class Ledoit–Wolf covariance, so both Mahalanobis variants are
  reported for both models (CNN-6 per-class recomputed from its saved 3C seed-42 embeddings). Threshold-free per-fold
  AUROC and OSCR, fold k known vs fold k withheld, vs CNN-6 seed 42 on the same 6 runs; a withheld → assigned-class table
  (the argmax over known classes). **Pre-declared reading:** "helps for class X" only if R18-ft's Mahalanobis (per-class)
  or MSP AUROC for withheld X is higher than CNN-6's in at least 2 of 3 folds; exploratory: 6 runs cannot establish a
  general Q3 improvement.
- **B3 (cost):** training seconds per run, total and trainable parameters, median CPU latency for one recording (9
  windows from the cached representation, as 3A).
- Outputs: `ml.paths.RunPaths` (family "resnet", variants r18probe / r18ft; LOCO family "loco", variant r18ft, withheld in
  the name); results under `results/resnet/`; weights (state_dict) under `artifacts/resnet/` and `artifacts/loco/`
  (gitignored). The split hash is unchanged; reserved tests stay sealed; one GPU job at a time (`artifacts/gpu.lock`).

## 7a2. Earlier stage: Stage ML-4.2 — recalibrate on the deployment scoring path (Track A)

Decision (Bhuvesh, after ML-4.1 item 3): recalibrate the **unchanged** models on the exact CPU scoring path the
application uses. No retraining, no change to model weights, rule variants or the default rule (B_with_caution).

**Stage ML-4.2 protocol (declared before running).**
1. **One shared scoring function** (`ml.inference.score_recording`): from the raw recording and its speed, cleaning,
   windowing, R1 features, the cache-C representation, float32 on the CPU, the CNN-6 / AE-6 / B1s / LR scores and the
   window → recording aggregation. Both `diagnose()` and the calibration script call it; no second path computes MSP,
   Mahalanobis, AE error or B1s.
2. **Recalibration** (`scripts/19_recalibrate.py`) on the same 144 calibration recordings (closed val ∩ development pool,
   including the same 5 normals), with the existing formulas: τ_p and τ_d = the 97.5 % order statistic over the 144
   (r = ceil(0.975 × 145) = 142; MSP on its negation); τ_e and τ_b = the maximum over the 5 normals. Nothing is optimised
   against fault outcomes. The per-recording calibration scores are saved in the bundle (`calibration_scores.csv`, hash
   pinned); the previous thresholds are kept in `thresholds.json` as `previous` for the record.
3. **Exact comparison first** (ties accepted): each threshold is verified against its order statistic or maximum and its
   defining recording is named; the marginal acceptance of the MSP and Mahalanobis checks and their measured joint
   acceptance are reported (the joint is not required to equal either marginal target); the ML-4.1 per-normal table under
   all variants. The 5 calibration normals define τ_e and τ_b, so they are expected to pass both healthy-detector checks
   and, where rule B says healthy, to carry no caution; caution on other healthy verdicts is intended behaviour.
4. **Tolerance only if needed:** if repeated CPU runs of the shared function are not bit-identical, a per-score tolerance
   justified by the measured differences is declared, applied only in the accepting direction, identically in calibration
   reporting and inference, recorded in the manifest and here; otherwise none. A tolerance cannot guarantee identical
   behaviour on every CPU.
5. **Development-results screen (read-only):** evaluation instances in the saved 3C / 3C.1 / 3D decision files whose
   scores lie within the ML-4.1 measured cross-path difference (MSP 3.3e-8, Mahalanobis 2.0e-5, AE 1.8e-7, B1s 1.8e-15)
   of their run's thresholds are listed. That difference is a **screening tolerance, not a proven numerical bound** for the
   separately trained development models; nearby scores identify cases worth investigating, and finding none does not
   establish numerical stability. Development results are not recomputed.
6. **Regression tests:** the three boundary recordings (normal__18.432, overhang__outer_race__20g__18.2272,
   overhang__cage_fault__6g__13.1072) produce the declared decisions; the shared function's scores equal the saved
   calibration scores; values clearly on either side of each threshold. Numeric score comparisons may use justified
   tolerances; decision outcomes satisfy the declared boundary rule exactly.
7. **Regenerate:** the manifest (revision and unrelated_worktree_changes), the golden reference
   (`scripts/16_update_golden.py`), and the sanity report, labelled "calibration recordings, not an independent test;
   expected to be optimistic".

**Stage ML-4.2 outcome (recorded):** repeated CPU runs of `score_recording` are bit-identical, so **no tolerance** is
declared; all threshold comparisons are exact with ties accepted (`thresholds.json` → `tolerance: null`). A tolerance
could not guarantee identical behaviour on every CPU either; other CPUs/BLAS builds may differ in the last digits.
**Bundle rebuild order:** `scripts/14_train_bundle.py` (models; thresholds left pending) → `scripts/15_bundle_manifest.py`
→ `scripts/19_recalibrate.py` (thresholds on the deployment scoring path; rebuilds the manifest) →
`scripts/16_update_golden.py` → `scripts/18_calibration_checks.py`.

**Previous stage (ML-4.1, completed except item 3, which this stage completes):**
**Stage ML-4.1 protocol (declared before running).**
1. **Configuration drift.** The complete inference configuration (the full `Config`: preprocessing, cleaning, windowing,
   representation / spectrogram, feature, normalisation and model settings) and the feature dictionary are packaged in
   the bundle (`inference_config.json`, `feature_dictionary.csv`); inference uses the packaged configuration, never the
   repo's current `configs/` (the repo config is used only to refuse reserved rec_ids when no bundle is present). The
   manifest records `inference_config_sha256` (config_sha256 of the packaged Config); load_bundle recomputes it and
   refuses on mismatch, in addition to the per-file sha256 check. The packaged preprocess hash must equal the one recorded
   when the models were trained. Tests: a changed repo config value does not change diagnose(); a tampered packaged
   config is refused.
2. **Golden reference.** The golden test compares against the stored reference for the current manifest and FAILS when
   it is missing or belongs to another manifest. Updating it is an explicit step, `scripts/16_update_golden.py`, which
   prints a diff summary (the latency measurement moves to `scripts/17_inference_latency.py`).
3. **Calibration consistency.** All 144 calibration recordings through diagnose(); the joint acceptance of the MSP and
   Mahalanobis checks must equal the threshold-setting count and rate exactly; each of the 5 calibration normals must
   pass AE ≤ τ_e and B1s ≤ τ_b, and rule B healthy must carry no caution. The threshold-setting scores are recomputed from
   the bundle weights on the path used when setting thresholds (representation cache, the training device; B1s from the
   feature cache) and compared with the inference path. **If anything differs, stop and report the cause before changing
   anything** (no threshold, rule or model change without Bhuvesh's approval).
4. **Speed warnings.** In addition to the training-range warning, a warning when the measured speed lies outside the
   calibration-normal speed range (from the 5 calibration normals, recorded in the manifest): "healthy thresholds were not
   calibrated at this speed; a healthy verdict is less reliable".
5. **Test eligibility** per task as stated above, recorded in the manifest (`test_eligibility`) and the limitations.
6. **Six-class output:** class6 probabilities = sums of the class10 probabilities by the existing class10 → class6
   mapping (`ml.evaluation.CLASS10_TO_6`).
7. **Explanation labels:** CNN map "exploratory gradient attribution; an occlusion check did not confirm band-level
   importance", display_priority "advanced"; LR "feature contributions of the secondary logistic-regression model",
   display_priority "primary".
8. **Bundle sanity report:** class10 / class6 macro-F1 and the confusion matrix on the 144 calibration recordings (closed
   val within the development pool only), labelled: "not an independent test — these recordings were used for
   calibration (and epoch selection, if applicable); the bundle's training and calibration sets together use the full
   development pool; the only independent evaluation left is the reserved tests." They were **not** used for epoch
   selection (fixed 40 / 150 epochs).
9. **Manifest provenance:** rebuilt after items 1–8 are committed, recording the committed implementation revision
   (`git_sha`, HEAD without a "-dirty" suffix) and `unrelated_worktree_changes` (e.g. the untouched notebook
   `01_data_layer/01_index_audit_splits.ipynb`); the checkout is never described as clean if it is not, and that notebook
   is never stashed or discarded. The golden reference is then regenerated with the new script.

**Golden-reference procedure (standing):** after any bundle rebuild, run `python scripts/16_update_golden.py`, read its
diff summary, and commit the new reference with a message that says why the outputs changed.

**Bundle and app facts (Stage ML-4, kept):**
**Bundle data (Bhuvesh's decision, 27 Sep, before any training):** the bundle trains on **closed train ∩ development
pool** (660 recordings: `split_closed` = train and no reserved-test column = test; 20 normals) and calibrates on **closed
val ∩ development pool** (144 recordings; **5 normals**, so τ_e and τ_b each come from 5 healthy recordings). No
reserved recording (any of `split_closed`, `split_severity`, `split_speed_block` = test) is used for training,
calibration or inference. **Test eligibility (per task, Stage ML-4.1):** closed-set known-condition classification (Q2),
healthy-vs-fault detection (Q1), and the severity and speed-block extrapolation tests on their declared tasks: eligible,
since no reserved recording was used; the leave-one-class-out unfamiliar-fault evaluation (Q3): **not eligible**, because
the bundle was trained on all ten classes — the Q3 final evaluation needs the separate LOCO models trained without the
withheld class. Training speed
range (measured recording speed) 11.80–39.75 Hz; severities present: imbalance 6–30 g, horizontal misalignment 0.5–1.5 mm,
vertical misalignment 0.51–1.78 mm, bearing faults 0 / 6 / 20 g added mass (the largest severity of each family is
reserved). The existing 3A CNN-6 and 3B AE-6 artifacts were trained on development-pool folds that include closed-val
recordings, so they do not satisfy this rule and are retrained.

**Stage ML-4 protocol (declared before running).**
- **Models (bundle v1, seed 42, fixed settings from `configs/base.yaml`):** CNN-6 (Stage 3A architecture and settings,
  10 classes, vibration only) on the bundle-train windows; normalisation from `norm_stats` in development mode (fit_ids =
  bundle train; context: calibration = bundle val). AE-6 (frame level, Stage 3B settings) on the bundle-train normals
  (normalisation on those normals). R1 logistic regression (the classical reference: capped R1 table, all features,
  `baselines.logreg`) on bundle train, secondary evidence. B1s (speed-aware polynomial, Stage 2D.1 settings,
  vibration-only features as in Stage 3D) on the bundle-train normals. scikit-learn / numpy only outside torch.
- **Thresholds on bundle val (calibration only), as in 3C:** τ_p and τ_d at 97.5 % marginal targets over all known
  classes including normal (order statistic r = ceil(0.975 (n + 1)); ties accepted), Mahalanobis with class centroids and
  one shared Ledoit–Wolf within-class covariance fitted on bundle-train window embeddings; τ_e = maximum AE error and
  τ_b = maximum B1s score of the 5 bundle-val normals.
- **App healthy rule (provisional demo default, NOT a final rule selection; chosen after the Stage 3D results; the final
  choice is made later under stated assumptions about fault frequency and costs):** configurable
  `healthy_rule ∈ {A, B, C, D, B_with_caution}`, default **B_with_caution**. B_with_caution = rule B's decisions
  (classifier-first, no AE override); when B says healthy and B1s > τ_b and/or AE > τ_e, caution = "b1s_disagrees",
  "ae_disagrees" or "both", display label "healthy — caution: a healthy-baseline check disagrees; consider a recheck".
  D (strict, selectable) = B, and healthy also requires B1s ≤ τ_b. Unknown is shown as "further inspection needed".
  Every diagnosis reports `rule_comparison` for A, B, C, D. Logic lives in `ml/arbitrator.py`, reusing
  `ml.openset.three_way`.
- **Inference (`ml/inference.py`):** `diagnose(rec_id | raw 8-channel array, 50 kHz, 5 s)` → plain dict (no Pydantic):
  signal_quality, diagnosis, physics, xai, limitations, provenance; every number from existing `ml/` functions. A rec_id
  is accepted only if it is in the development pool; every reserved-test rec_id is refused with a clear error (a raw
  array cannot be checked and is the user's own upload). Signal-quality warning when the recording's measured speed lies
  outside the training range: "speed outside the range the model was trained on; results less reliable". Limitations
  include: "calibrated on 5 healthy recordings; trained on speeds below 40 Hz and on the development pool only", the Q3
  wording below, "decision rule is a provisional demo default", "research demonstration; development results; not
  validated for real machines". CPU latency = median over 20 development recordings.
- **Bundle v1 (`ml/bundle.py`, schema "1.0"):** `artifacts/bundle/` (gitignored): safetensors weights (state_dict),
  documented joblib files for LR and B1s, thresholds, rule config, `manifest.json` with a sha256 per file plus the data
  facts above; `load_bundle()` verifies every hash and refuses on mismatch. Only code and a manifest example are committed.
- **XAI (minimal):** Grad-CAM on the CNN's last convolution block (input × gradient if unstable) → top channels and order
  bands and a coarse channels × order-bin map; LR: top R1 features by coefficient × transformed value for the recording,
  with feature-dictionary names. Labelled "what the model attended to", not proof of fault location.
- **Tests:** golden (3 fixed development recordings: normal, a bearing fault, a misalignment; identical output on CPU
  across runs, timestamps excluded), bundle hash mismatch, reserved-recording refusal, out-of-range speed warning, every
  arbitrator branch including ties.
- **Stretch (only after the above is committed):** ResNet-18 fine-tune, reported next to CNN-6, not added to the bundle.

**Wording and open items recorded in this stage.**
- Q3: "The tested scores and rules do not consistently distinguish the imbalance / vertical-misalignment pair when one
  class is withheld. The classifier distinguishes them well when both are represented in training. This does not
  establish that no representation, training objective or decision method could help." (No attribution to missing
  cross-axis information; that is a hypothesis only.)
- **Open item — 3C.1 bootstrap:** suggestive, conditional evidence. Each draw drops runs lacking withheld or known
  recordings (mean 7, max 13 of 27) and averages the remaining folds within class, so draws use different fold
  combinations. The handling of missing folds needs validation before any publication claim.
- Stage 3D E2 did not test whether combining LR and CNN evidence helps (its combined check was non-restrictive in most
  runs); any redesigned combination is a separately declared experiment. D is a candidate only, not selected.

**Standing scope rules (Stages 3A/3B):**
- **Q2 only for the CNNs.** They are trained on all ten classes; their embeddings and scores are for Q2 analysis ONLY and
  are tagged "Q2-only; trained on all 10 classes; not valid for LOCO". LOCO experiments (Stage 3C) must retrain the
  encoder without the withheld fault class and exclude that class from normalisation, calibration and threshold
  selection. Stage 3A/3B CNN outputs must not be reused for Q3.
- **Early fusion only.** Late fusion (separate sensor branches combined later) remains on the roadmap; Stages 3A/3B do not
  complete the fusion study. CNN-7 vs CNN-6 is an early-fusion preview.
- **Seeds.** Stage 3A used one seed (42); Stage 3B adds seeds 43 and 44. Seed variation and bootstrap intervals are
  different sources of uncertainty: report both, never merge them.
- **Autoencoder.** It may not outperform the classical detector (B1s, per-fold AUROC 0.990); its value (an alternative
  detector and sensor anomaly evidence) must be demonstrated by results, not assumed.

**Notes of record:**
- Only development-pool windows are loaded onto the device; reserved windows never are.
- Two scripts share the number 06 (`06_bandstop.py`, `06_train_cnn.py`); they are not renamed.
- Latency: about 4 ms is CNN prediction from an already prepared representation; building the representation adds about
  0.5 s, so upload-to-result time is longer.
- **Artifact paths.** Every existing seed-42 artifact of Stage 3A stays unchanged at its original path. From Stage 3B on,
  every score, embedding, weight, normalisation-stat and result path, and every MLflow run name and tag, carries the
  variant, the seed and (where applicable) the fold and the withheld class (`ml.paths.RunPaths`).
- **Q1 alarm combination (deferred):** an alarm-level combination of B1s and AE-7 is uninformative on these recordings
  (B1s flags every fault; AE-7 flags no healthy recording, so OR reproduces B1s and AND reproduces AE-7). Deferred until
  more healthy data exist.

Items: 1 configuration drift; 2 golden enforcement; 3 calibration consistency; 4 speed warnings; 5 test eligibility;
6 six-class output; 7 explanation labels; 8 sanity report; 9 manifest provenance.

## 7b. Plan notes (planned controls)

Recorded 26 Sep 2026 (Stage 2B.1); status updated in Stage 3A.
- **Q1 (detection):** report the detection rate at the **mildest severities** of each fault family, and compare the
  detector **with vs without the microphone**. **Done** on development data (Stage 2D.1, classical detectors).
- **Q2 (severity extrapolation):** the `split_severity` test is the main severity-extrapolation result, across held-out
  folder proxies, **evaluated once at the end**.
- **Q3 (open set):** report the **false-unknown rate on known classes at held-out severities**. Covered on development
  data by control (b) of Stage 2D.1 (unseen-folder known faults); the `split_severity` test stays sealed.
- Development evaluation (Stage 2B.1): folder-grouped folds on the development pool (`split_closed` ∈ {train, val},
  `split_severity` ≠ test, `split_speed_block` ≠ test); folders are proxies for sessions. Model selection uses permitted
  development data only; closed-set ties are broken by simplicity and measured latency; reserved tests are never used for
  selection. LOCO runs exclude the withheld class from fitting, preprocessing selection and calibration, but every class
  has already been seen during development.

## 8. Code conventions

- Type hints on public functions. Docstrings say *what and why* in one or two lines, plus units
  (Hz, orders, g, seconds).
- Constants (`FS = 50_000`, bearing ratios, expected counts) live in `ml/config.py` or `configs/base.yaml`.
  Never scatter magic numbers.
- Pure functions in `ml/`. Side effects (file writes, MLflow) happen only in scripts and notebooks.
- Every cached artifact is named by the sha256 of the config that produced it, so stale caches fail loudly.
- Model weights are saved as `state_dict` or safetensors, never pickled modules.
- Log to MLflow: resolved config, seed, split hash, git sha, cache hash, metrics, train seconds, ms per recording.
- Tests: pytest, fast (< 1 min total), run on CPU, with synthetic data where possible.
  Required from Stage 1: `test_labels`, `test_speed`, `test_splits` (no recording appears in two splits;
  the withheld class is absent from train/val in every LOCO column; counts add up).
- Performance: vectorise with NumPy/torch. Parse CSVs with pyarrow, never `np.loadtxt` or row loops.

## 9. Git

- Small commits with messages that say what changed and why. Commit `metadata.csv`, configs, code and
  stripped notebooks.
- Commits are authored by Bhuvesh only. No Claude `Co-Authored-By` lines or other AI attribution.
- Never commit data, caches, weights over 100 MB, `mlruns.db`, `.env`, or `archive.zip`.
- Don't rewrite history or force-push. Don't push unless asked.

## 10. When to stop and ask Bhuvesh

- A count, shape or speed doesn't match §3.
- A decision would change the splits, labels, representation or evaluation protocol.
- A new dependency, or a run expected to take more than an hour.
- Anything that deletes or moves files under `mafaulda/`.
- The plan and this file disagree.

Otherwise, make the reasonable choice, note it in the notebook's Findings or the commit message, and keep going.

## 11. Agentic application (planned deliverables D6/D7; added 27 Sep)

Status: planned deliverable, decided by Bhuvesh on 27 Sep. Do **not** implement it until a stage brief says
so, and never let it interrupt or change a diagnostic stage. The full design, research question, evaluation
design and schedule conflicts are in the plan doc section "Agentic application and research direction".
The earlier "no login, no chat" exclusion is replaced by this section.

**Product flow.** Display name (optional PIN; a demo identity, not authentication) → upload a CSV or pick a
prepared recording → main screen with a basic interactive 3-D machine view, sensor positions and recording
details → "Investigate this reading" → progress, then: condition and predicted fault; supporting and
conflicting evidence; uncertainty and missing information; next inspections with cited references; a chat
panel for follow-up questions.

**Agent (Bhuvesh's decision, dated edit 5 Oct 2026; replaces the earlier five-agent design).** One tool-using agent
answers a mechanic's question about one diagnosed case. It may use only four read-only tools over the stored case:
`diagnose` (verdict, scores and the thresholds they were compared with, caution, top classes), `check_signal` (speed,
stability, warnings, speed ranges), `get_evidence` (per-channel comparison with a healthy reference) and
`lookup_guidance` (report section 04's cited steps). Every answer passes an evidence review applied by code (not a
second model) before the mechanic sees it. Every investigation has a fixed budget; when the budget runs out or the review
fails, the answer falls back to the fixed report.

**Hard rules (non-negotiable once implementation starts).**
1. Numbers come only from tested tools in `ml/` (through the backend API). Agents never generate, alter or
   replace measurements or model outputs.
2. The fixed diagnostic report works with no language model and no API key; the assistant degrades to it.
   Graders run on a clean CPU machine.
3. Every investigation has a fixed tool-call and revision budget, set in config.
4. The UI shows actions taken and evidence found, never private model reasoning.
5. Location wording: "predicted fault location according to the dataset labels" is kept separate from
   "sensor anomaly evidence". Per-channel reconstruction error is not fault-location evidence.
6. API keys live in `.env` and are never committed. The demo identity stores nothing sensitive.
7. Maintenance references are curated under `knowledge/`, each with its source and licence note, and are
   cited by id. Model memory is never presented as a source.
   Enforced for the report by `ml.guidance.select` (Stage App-5): section 04 shows only steps from entries it selects,
   each cited by entry id; nothing selected gives a Missing item, never generated advice.
8. Data rules still hold: reserved tests stay sealed. D7 development cases are separate from final
   evaluation cases, and a recording and its altered variants stay on the same side.
9. New dependencies (a language-model SDK, a 3-D library) need a pinned line, a one-line reason and approval
   (§2). The LightGBM/torch isolation rule still applies to every process.
10. D1–D5 keep priority whenever they compete with agent work.
11. Agents must not use the gradient highlights (the CNN attribution map) alone to support a repair recommendation;
    they are exploratory and an occlusion check did not confirm band-level importance.

**D7 research question (dated edit 5 Oct 2026).** When diagnostic evidence is uncertain or conflicting, does an
evidence review step reduce unsupported claims and unsupported repair recommendations in a tool-using assistant, while
keeping useful next inspections? The arms are the fixed report, the agent without review, and the agent with review, all
with the same model, tools, references and budget. Measures:
unsupported repair recommendations and useful next inspections (refusing everything must not look
successful), citation correctness, unnecessary refusal, time and cost. Blinded expert review is sought;
without it, usefulness claims stay limited.

Proposed agent kill rule: if the minimum agent loop is not working on 18 Oct, D6 ships as the fixed report plus
chat over that report, and D7 becomes a smaller pilot. Stop and ask Bhuvesh about anything listed as open in
the plan's Decisions (provider, references, reviewer, cuts, dates).

## 12. Final evaluation protocol (declared 28 Sep 2026, Stage F0, before any final-model training)

This section freezes the final evaluation (F1). It is committed before the final models are trained. After F1 starts,
nothing in it changes. No reserved-test recording is read, scored, listed or fingerprinted before F1.

**a. Frozen models** (the Q3 LOCO models and R18-ft are trained in F0 on development data only; they are saved in
`artifacts/final/` with a manifest: sha256 per file, git sha, split hash; they are verified on load, like the bundle).
- **Primary: the ML-4.2 bundle** (`artifacts/bundle/`, manifest sha256 521cc746…), unchanged: CNN-6, AE-6, B1s, LR,
  Mahalanobis and the thresholds τ_p, τ_d, τ_e, τ_b set on the CPU deployment scoring path.
- **Secondary, reported only (never selected on test results): R18-ft**, trained once with the Stage ML-5 recipe
  (CLAUDE.md §7; `configs/resnet.yaml`), seed 42, on the bundle's training subset (closed train within the development
  pool, 660 recordings); normalisation from those recordings; no thresholds (Q2 only).
- **Q3: 9 final CNN-6 LOCO models** (Stage 3C recipe: Stage 3A architecture and fixed settings, seed 42, 9 known
  classes). For withheld class c: **fit** = closed train within the development pool minus class c; **calibration** =
  closed val within the development pool minus class c (the bundle's scheme). Normalisation (`norm_stats`, development
  mode) on the fit recordings. Mahalanobis on the fit-window embeddings: class centroids with **one shared within-class
  Ledoit–Wolf covariance** (primary, as 3C) and **per-class Ledoit–Wolf covariances** (secondary, "added after ML-5,
  declared before F1"). **AE-6 (frame level) and B1s** are refit on the fit normals. Class normal is never withheld, so
  the fit normals (20) and calibration normals (5) are the same for all 9 models: AE-6 and B1s are trained ONCE and shared
  by the 9 LOCO models. Thresholds by the 3C / bundle rules on each model's calibration recordings: τ_p, τ_d = the 97.5 %
  order statistic over all calibration recordings (known classes including normal; MSP on its negation; ties
  accepted); τ_e, τ_b = the maximum over the 5 calibration normals. Nothing is chosen against fault outcomes.
- **Scoring path:** every calibration and test score (bundle, LOCO models, R18-ft) is computed on the CPU from the raw
  recording with its metadata speed, on the deployment path of ML-4.2: `clean_recording` → windows → R1 features and
  the cache-C representation computed online → models (`ml.inference.score_recording` for the bundle; `ml.final` for
  the LOCO models and R18-ft, which reproduces `score_recording` exactly on the bundle, tested). Recording decisions use
  the mean over its 9 windows (rule 5).

**b. Tests and models** (each reserved test is opened once, in F1).
- **split_closed test (290):** bundle for Q1 and Q2; R18-ft for Q2 only.
- **split_severity test (370):** bundle; declared task severity extrapolation: Q2 primary, Q1 detection secondary.
- **split_speed_block test (776):** bundle; declared task speed extrapolation: Q2 primary, Q1 secondary. All its
  recordings run at 40.0–60.7 Hz, above the training speed range (11.80–39.75 Hz), so every one is expected to carry
  the speed warning; warning counts are reported.
- **loco_<class> test × 9:** the matching LOCO model. **Known side** = the recordings with loco_<class> = test (240–270
  per class, all reserved, all of known classes). **Unknown side, primary** = the withheld-class recordings that are
  reserved (in any of the three test columns; 68–189 per class; never seen in development). **Unknown side, secondary,
  labelled** = all withheld-class recordings (adds the 69–144 per class in the development pool; the final LOCO models
  never trained on them, but Stage 3C evaluated them and the rule choice after Stage 3D was informed by them).
- **Breakdown by origin (Bhuvesh, 28 Sep):** every Q3 result is also reported by where each reserved recording comes
  from (split_closed test / split_severity test / split_speed_block test), on both the unknown and the known side, so a
  speed- or severity-driven "unknown" can be told apart from fault-type novelty. The three columns overlap (union 1,147;
  closed ∩ severity 71, closed ∩ speed 125, severity ∩ speed 121, all three 28): the breakdown counts by membership (a
  recording appears once in each column it belongs to) and also reports the 7 exclusive combinations with their counts.
  The headline stays the pooled primary number.

**c. Metrics.**
- **Q2:** class10 and class6 macro-F1 and balanced accuracy, per-class F1, class10 confusion matrix (recording level).
- **Q1:** at the frozen thresholds, per decision rule: detection rate (fault recordings not called healthy) and false-
  alarm rate (normal recordings not called healthy; healthy-with-caution counts as healthy and is reported separately),
  with Clopper–Pearson intervals (few normals: 7 / 7 / 20); AUROC of the AE error and of B1s (normal vs fault).
- **Q3:** per withheld class: threshold-free AUROC and OSCR (known side vs unknown side) for MSP and Mahalanobis
  (shared = primary; per-class Ledoit–Wolf = secondary, "added after ML-5, declared before F1"); three-way outcomes at
  the frozen thresholds (unknown / known fault / healthy) for the unknown side and the known side. **Headline:** the
  macro-average over the 9 withheld classes of the Mahalanobis (shared) AUROC and of the MSP AUROC, and the pooled
  unknown recall of the primary rule (unknown-side recordings called "unknown") with its false-unknown rate on the
  known side, on the primary unknown set.
- **Uncertainty:** stratified mixed-unit bootstrap (fault folders within class, normal recordings individually;
  `ml.evaluation.mixed_bootstrap_draws`, the Stage 3A settings: n and seed from `baselines`) for macro-F1, balanced
  accuracy and AUROC; Clopper–Pearson for rates from few recordings. Every table states the number of instances and of
  distinct recordings (Q3 known-side recordings recur across LOCO runs).

**d. Decision rules.** A, B, C, D and B_with_caution are all reported; the primary is **B_with_caution** (a provisional
research setting, not tuned on any test data).

**e. Once-only rules.** F1 is a single committed script, `scripts/30_final_eval.py`, run once. It refuses to run unless
this §12 is committed and unmodified, the script itself is committed and unmodified, and `artifacts/final/` and the
bundle verify; it refuses if `results/final/` already exists. It writes `results/final/` with provenance (manifest
sha256s, git sha, split hash, run timestamp, its own sha256). No retraining, recalibration, threshold or rule change
after F1 starts. Every result is reported. A bug found afterwards is reported as a deviation with both the original
and the corrected numbers.
- **Dry run (F0 only):** `--dry-run` substitutes development-pool recordings for every reserved set (closed test ←
  closed val ∩ pool; severity test ← the largest development-pool severity of each fault family; speed-block test ←
  pool recordings above 35 Hz; loco known side ← calibration set minus class c; loco unknown side ← pool recordings of
  class c). It writes only to `results/final_dry_run/`, every file labelled "DRY RUN — development data"; its numbers
  are not results and are never mixed with `results/final/`. In dry-run mode the loader refuses any reserved rec_id.
