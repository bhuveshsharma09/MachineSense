# HOW_TO_RUN — MachineSense

## Quick start: run from a fresh clone

You can run the app from a fresh clone. You do not need to download MAFAULDA.
The repository includes the trained model bundle and a demo pack of 5 recordings.

### Requirements

- Python 3.12
- Node 22 (with npm)
- About 3 GB of free disk space for the Python packages
- Git

### 1. Clone and install

```bash
git clone https://github.com/bhuveshsharma09/MachineSense.git
cd MachineSense/SystemCode
python3.12 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

On Linux or Windows, install the CPU-only PyTorch first. Otherwise pip downloads large CUDA packages.
Run this before `pip install -r requirements.txt`:

```bash
pip install torch==2.14.0 torchvision==0.29.0 --index-url https://download.pytorch.org/whl/cpu
```

On Windows, activate the venv with `.venv\Scripts\activate`.
On macOS, LightGBM needs the OpenMP runtime: `brew install libomp`. The app itself does not use LightGBM.

### 2. Start the backend

In `SystemCode/`, with the venv active:

```bash
uvicorn backend.app:create_app --factory --host 127.0.0.1 --port 8000
```

At startup the backend checks the SHA-256 of every file in the model bundle. It refuses to start if one does not match.
Check it at http://127.0.0.1:8000/api/health. You should see `"bundle_verified": true` and `"mode": "demo-pack"`.

### 3. Start the frontend

In a second terminal:

```bash
cd MachineSense/SystemCode/frontend
npm ci
npm run dev
```

### 4. Open the app

Open http://localhost:5173.
Enter a display name (no account is needed). Then choose **Start investigation**.
The 5 demo recordings are listed:

| Recording | What the app shows |
| --- | --- |
| `normal__24.576` | healthy |
| `imbalance__30g__19.456` | known fault: imbalance |
| `horizontal-misalignment__1.0mm__24.9856` | known fault: horizontal misalignment |
| `underhang__cage_fault__6g__35.0208` | known fault: underhang cage (see the envelope view) |
| `imbalance__10g__13.9264` | further inspection needed |

You can also upload a CSV in the MAFAULDA format. It must have no header, 8 columns and 250,000 rows (5 s at 50 kHz).
The **Explore an unseen fault** page works from saved results.

### About the demo recordings

The demo recordings are development recordings, not test data.
They come from the closed validation split, which set the decision thresholds.
So their verdicts are in-sample.
The reserved test recordings are never shipped, listed or read by the app.
The pack also holds the 4 healthy training recordings that the evidence view compares against.
Details are in `demo/manifest.json` and `demo/README.md`. The pack is rebuilt with `scripts/40_demo_pack.py`.

### Optional: written explanations and the assistant

The app works without a language model. Without one it shows template explanations, and the assistant shows the fixed report.
For AI-written explanations and the investigation assistant, install [LM Studio](https://lmstudio.ai).
Load the model `google/gemma-4-e4b` and start its local server on port 1234.
The backend finds it at `http://localhost:1234/v1`. No API key is needed.

### Optional: the full dataset

The full dataset is needed to reproduce training and the evaluation, and to diagnose every development recording.
1. Download MAFAULDA from Kaggle: [mafaulda-full](https://www.kaggle.com/datasets/vuxuancu/mafaulda-full).
2. Unpack it so that `mafaulda/` sits next to `SystemCode/` (the folder that holds `normal/`, `imbalance/`, and so on).
3. Convert it: `python scripts/00_convert.py` (in `SystemCode/`). This writes `SystemCode/data/raw_npy/`.

When `SystemCode/data/raw_npy/` exists, the app uses the full data instead of the demo pack.
"Full reproduction" below covers training, notebooks and experiment tracking.

### Tests

In `SystemCode/`, with the venv active:

```bash
MACHINESENSE_REQUIRE_BUNDLE=1 python -m pytest tests -q
cd frontend && npm test
```

In a clone without the full dataset, tests that need it are skipped. The demo-pack tests still run.
With the full local data, add `MACHINESENSE_REQUIRE_FULL_DATA=1` so those tests fail instead of skipping.

## Full reproduction (optional, needs MAFAULDA)

Target machine: Mac mini (Apple M4 Pro, 48 GB unified memory, macOS Tahoe 26.4.1). Everything also runs on a
Linux or Windows CPU for grading; only the training speed differs.

### 1. One-time setup on the Mac mini (about 10 minutes, mostly downloads)

```bash
# Homebrew packages: Python 3.12, the OpenMP runtime LightGBM needs, Node for the React build
brew install python@3.12 libomp node@22

cd SystemCode
python3.12 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt        # ~2 GB of wheels; nothing compiles on Apple Silicon
pip install -e .                       # once pyproject.toml exists (Stage 1): makes `import ml` work everywhere
```

If you prefer conda: `conda create -n machinesense python=3.12 && conda activate machinesense`, then the same
`pip install` lines. Do not mix conda-forge PyTorch with the pip wheel.

### 2. Check the machine before anything else

```bash
python scripts/smoke_env.py --data /path/to/mafaulda --resnet
```

It prints PASS / WARN / FAIL for: Python 3.12 on arm64, free disk (about 20 GB needed beyond the CSV tree),
the 1,951-file tree, one real CSV timed with pyarrow, every import, a LightGBM fit (proves libomp),
headless matplotlib, MPS availability, the float64 guard, `torch.stft` on CPU, and timed training steps for
the scratch CNN and ResNet-18 with the data resident on the GPU. Paste the timings into the plan's memory
and speed table; they replace the estimates.

### 3. Environment variables (shell profile or `.env`)

```bash
export PYTORCH_ENABLE_MPS_FALLBACK=1   # unsupported MPS ops fall back to CPU instead of raising
export MACHINESENSE_DEVICE=mps         # training device; the app and the tests default to cpu
export MLFLOW_DISABLE_AGENT_HINT=1     # silences an MLflow banner
```

### 4. How training stays fast on this machine

- The whole spectrogram cache (17,559 windows × 7 channels × 128 × 25, float16 ≈ 0.8 GB) is loaded once as
  one tensor on the GPU. Batches are index slices of it: no DataLoader, no worker processes, no disk reads
  during training. This is the single biggest lever on a Mac; multiprocessing data loaders are what make
  PyTorch on macOS slow or crash.
- Representations are computed once on CPU (`torch.stft`, a few minutes with a process pool) and cached under
  a hash of the preprocessing config, so a config change can never silently reuse a stale cache.
- fp32 only, batch 256 for the scratch CNN and the autoencoder, 64 for ResNet-18. No mixed precision: it
  buys nothing at this size and adds MPS edge cases.
- One GPU job at a time. LightGBM and scikit-learn baselines use the CPU cores and can run alongside it.
- Long runs are scripts, not notebook cells: `caffeinate -i python scripts/06_train_cnn.py --config configs/cnn.yaml`
  keeps the Mac awake overnight; the notebooks read the results back from MLflow.
- Between experiments in a notebook: `del model; torch.mps.empty_cache()`, or restart the kernel. The GPU
  ceiling is capped at 70 % of memory (`torch.mps.set_per_process_memory_fraction(0.7)`) so the browser and
  JupyterLab never get squeezed.
- Cast everything to float32 before `.to("mps")`: MPS has no float64, and NumPy defaults to float64.

### 5. Experiment tracking

```bash
mlflow ui --backend-store-uri sqlite:///mlruns.db --port 5000
```

### 6. Running the app

From `SystemCode/` with the venv active (details, endpoints and settings: `backend/README.md`). Without
`data/raw_npy/` the app runs on the committed demo pack ("Quick start" above):

```bash
uvicorn backend.app:create_app --factory --port 8000   # the API; verifies the model bundle at start, refuses on mismatch
cd frontend && npm ci && npm run dev                    # the app at http://localhost:5173 (Vite proxies /api to :8000)
MACHINESENSE_REQUIRE_BUNDLE=1 python -m pytest tests    # backend and ml tests (golden outputs included)
cd frontend && npm test                                 # frontend: build, server-side renders, browser checks
```

The pipeline page ("How this diagnosis was made") opens from a diagnosis via its link, or at
`http://localhost:5173/#/case/<case_id>/pipeline`; see "Pipeline page" in `backend/README.md`.
