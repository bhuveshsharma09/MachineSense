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

## 7. Current stage: Stage R0 — diagnose the existing healthy rejections (development data only)

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

**Agents.** Coordinator (chooses the next step, delegates, decides when evidence is enough); signal-quality
agent (format, sensor quality, model applicability); diagnostic agent (calls the classifiers, anomaly
detectors and unknown-fault checks; examines disagreement); maintenance-guidance agent (retrieves curated
references, proposes supported inspection steps); evidence-review agent (checks every conclusion and
recommendation against model results and retrieved sources). They share one structured case record:
measurements, model versions, results, source references, unresolved questions.

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
8. Data rules still hold: reserved tests stay sealed. D7 development cases are separate from final
   evaluation cases, and a recording and its altered variants stay on the same side.
9. New dependencies (a language-model SDK, a 3-D library) need a pinned line, a one-line reason and approval
   (§2). The LightGBM/torch isolation rule still applies to every process.
10. D1–D5 keep priority whenever they compete with agent work.
11. Agents must not use the gradient highlights (the CNN attribution map) alone to support a repair recommendation;
    they are exploratory and an occlusion check did not confirm band-level importance.

**D7 research question.** When diagnostic evidence is uncertain or conflicting, does a multi-agent maintenance
assistant give fewer unsupported repair recommendations than a single-agent assistant, while still
recommending useful next inspections? The conditions are a fixed report, a single agent, multi-agent without
the reviewer, and the full system, all with the same model, tools, references and budgets. Measures:
unsupported repair recommendations and useful next inspections (refusing everything must not look
successful), citation correctness, unnecessary refusal, time and cost. Blinded expert review is sought;
without it, usefulness claims stay limited.

**Proposed targets (not committed; conflicts in the plan's Timeline).**

| Date | Proposed | Conflict |
| --- | --- | --- |
| 11 Oct | Upload/dropdown → diagnosis screen; `/api/diagnose` returns the structured evidence record | Extends the walking skeleton |
| 18 Oct | Minimum multi-agent investigation + chat; curated references; D7 protocol fixed | Checkpoint 2 (ResNet-18, seeds, XAI, manifest v1) |
| 24 Oct | Integrated app, bounded agent workflow, basic 3-D view, reproducible pilot; code freeze | Inference, golden tests, clean-clone check |

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
