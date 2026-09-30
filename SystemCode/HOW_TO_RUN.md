# HOW_TO_RUN — MachineSense

Target machine: Mac mini (Apple M4 Pro, 48 GB unified memory, macOS Tahoe 26.4.1). Everything also runs on a
Linux or Windows CPU for grading; only the training speed differs.

## 1. One-time setup on the Mac mini (about 10 minutes, mostly downloads)

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

## 2. Check the machine before anything else

```bash
python scripts/smoke_env.py --data /path/to/mafaulda --resnet
```

It prints PASS / WARN / FAIL for: Python 3.12 on arm64, free disk (about 20 GB needed beyond the CSV tree),
the 1,951-file tree, one real CSV timed with pyarrow, every import, a LightGBM fit (proves libomp),
headless matplotlib, MPS availability, the float64 guard, `torch.stft` on CPU, and timed training steps for
the scratch CNN and ResNet-18 with the data resident on the GPU. Paste the timings into the plan's memory
and speed table; they replace the estimates.

## 3. Environment variables (shell profile or `.env`)

```bash
export PYTORCH_ENABLE_MPS_FALLBACK=1   # unsupported MPS ops fall back to CPU instead of raising
export MACHINESENSE_DEVICE=mps         # training device; the app and the tests default to cpu
export MLFLOW_DISABLE_AGENT_HINT=1     # silences an MLflow banner
```

## 4. How training stays fast on this machine

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

## 5. Experiment tracking

```bash
mlflow ui --backend-store-uri sqlite:///mlruns.db --port 5000
```

## 6. Running the app

From `SystemCode/` with the venv active (details, endpoints and settings: `backend/README.md`):

```bash
uvicorn backend.app:create_app --factory --port 8000   # the API; verifies the model bundle at start, refuses on mismatch
cd frontend && npm ci && npm run dev                    # the app at http://localhost:5173 (Vite proxies /api to :8000)
MACHINESENSE_REQUIRE_BUNDLE=1 python -m pytest tests    # backend and ml tests (golden outputs included)
cd frontend && npm test                                 # frontend: build, server-side renders, browser checks
```

The pipeline page ("How this diagnosis was made") opens from a diagnosis via its link, or at
`http://localhost:5173/#/case/<case_id>/pipeline`; see "Pipeline page" in `backend/README.md`.
