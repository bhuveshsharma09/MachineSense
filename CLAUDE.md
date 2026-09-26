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
- **No DataLoader worker processes.** The whole spectrogram cache (~0.8 GB, float16) is loaded once as one tensor
  on the device. Batches are index slices of that tensor, cast to float32 on the device.
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
  - **Primary envelope band: 1–5 kHz for every track**, fixed from the sensor specifications (inside both sensors'
    rated ranges). The per-track bands in `results/demod_bands.json` are a separately reported comparison only.
  - **Order axis: per-window speed** from `ml.preprocess.window_speeds`, computed from the untouched tachometer,
    falling back to the recording speed. Fallback indicators and speed-variability statistics are never model inputs.
  - **22 kHz line: a fixed band-stop** (above), set from `split_closed` train normal recordings only.
  - **Microphone cleaning is a provisional mitigation**; it is re-checked on the microphone representation in Stage 2C.
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

## 7. Current stage: Stage 2A — shared preprocessing, windowing and physics features

Stages 1, 1b and 1c (data foundation, signal exploration, fold-safe preprocessing rules) are **done**, 26 Sep 2026.

Stage 2A scope, verified on a small training subset only (no full caches, no predictive models, no spectrograms;
the only model allowed is the diagnostic RandomForest of the band-stop shortcut test):

1. `ml.preprocess.clean_recording(x, cfg)`: whole-recording mic burst removal, then the band-stop on channels 1–7;
   the tachometer is returned untouched.
2. The band-stop: set from `split_closed` train normal recordings, checked (not retuned) on other train/val
   recordings, verified (designed response, synthetic signals, real recordings) and shortcut-tested.
3. `ml.preprocess.window`: 1 s windows, 0.5 s hop, 9 per recording, DC removed per window.
4. `ml.features.recording_features(x_raw, fr_hz, cfg)`: the one feature entry point for batch and serving.
   Feature columns start with `f_`; model code selects features by that prefix only.
5. Cache identity: every derived artifact is named by the sha256 of everything that changes its values.
6. Subset run and checks in `notebooks/02_feature_layer/03_windows_physics_features.ipynb` (first half).
7. Blue notes in notebook 00.

Stop after Stage 2A and report. Don't build spectrograms, full caches or models without a go-ahead.

## 7b. Plan notes (planned controls, not yet implemented)

Recorded 26 Sep 2026 (Stage 2B.1). Each runs when its stage comes; none is implemented yet.
- **Q1 (detection):** report the detection rate at the **mildest severities** of each fault family, and compare the
  detector **with vs without the microphone**.
- **Q2 (severity extrapolation):** the `split_severity` test is the main severity-extrapolation result, across held-out
  folder proxies, **evaluated once at the end**.
- **Q3 (open set):** report the **false-unknown rate on known classes at held-out severities**.
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
