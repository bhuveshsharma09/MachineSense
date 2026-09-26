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

## 7. Current stage: Stage 3C — deep unknown-fault test (Q3): LOCO-retrained CNN-6 and the three-way decision rule

Dev pool only; reserved tests sealed; the training process never imports LightGBM; all Mac/MPS, precision, path-scheme
(variant / seed / fold / withheld class in every path) and one-GPU-job rules apply. Stage 3A/3B embeddings and scores are
Q2-only and are NOT used here. Not in this stage: ResNet, late fusion, a learned meta-combiner (deferred), Q1 rule
combination.

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

**Stage 3C protocol (declared before running).**
- **Split:** the SAVED Stage 2D.1 Q3 assignment (`results/q1q3/q3_fit_cal_eval_assignment.csv`): per withheld class c and
  outer fold k, the known-class recordings of folds ≠ k are split within each folder 4:1:1 into fit / calibration /
  seen-folder evaluation. Fold k (known classes and class c) is evaluated once per c.
- **CNN-6 per (c, k):** the Stage 3A architecture and fixed settings, seed 42, 9 outputs (the known classes), trained on
  the fit recordings only. `norm_stats` in development mode: fit_ids = those fit recordings; context excludes calibration,
  seen-folder evaluation, fold k and every class-c recording.
- **AE-6 per (c, k), not per k (deviation from the brief, reason):** the saved Q3 assignment splits the normal folder
  differently for each withheld class, so one AE per k would have been trained on normals that are calibration or
  seen-folder evaluation recordings in other runs. Each (c, k) run therefore retrains the frame-level AE-6 on ITS fit
  normals only; its threshold uses ITS calibration normals.
- **Scores (directions fixed):** MSP of the aggregated class probabilities (lower = more unknown); Mahalanobis distance in
  the embedding (class centroids from fit windows; ONE shared Ledoit–Wolf covariance of fit-window embeddings minus their
  own class centroid; per window the minimum distance over known classes; recording = mean over its 9 windows; higher =
  more unknown); k-NN distance (cosine distance from a window to its 10th nearest fit-window embedding; recording = mean
  over its 9 windows; higher = more unknown); AE reconstruction error (higher = more anomalous). AE error measures
  departure from healthy behaviour, not unfamiliarity among fault types; a familiar severe fault can have a larger error
  than an unfamiliar mild one.
- **Three-way rule thresholds (calibration recordings only):** τ_p (MSP) and τ_d (Mahalanobis) are calibration targets
  of 97.5 % marginal acceptance each on the calibration known recordings (order statistic r = ceil(0.975 (n + 1)); accept
  iff MSP ≥ τ_p, resp. d ≤ τ_d; ties accepted); the joint acceptance is measured, not assumed. τ_e = the maximum
  calibration-normal AE error (healthy iff error ≤ τ_e).
- **Orderings:** PRIMARY = classifier-first (the AE misses some mild faults, and AE-first would declare them healthy);
  AE-first is a comparison only; no choosing the best ordering afterwards. Never fall back to the second-highest class.
- **Classical comparison:** logistic regression / MSP of Stage 2D.1 on the same split and recordings, reproduced per
  recording (its per-fold results must match `results/q1q3/q3_per_fold.csv`). Observed differences only; no winner
  claims without paired intervals.

0. Stage 3B reporting corrections (WAE-6 in every table, wording, pooled paired interval, sensor-evidence reading, app
   wording, Q1 combination note).
1. Deep LOCO: 27 CNN-6 and 27 AE-6 trainings; scores; AUROC/OSCR per class, fold spread, macro.
2. Three-way decision rule: joint acceptance, outcome tables (both orderings), AE override counts, classical comparison.
3. Notebook `09_openset_loco`.

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
