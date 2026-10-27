# MachineSense

Multi-sensor fault diagnosis for rotating machinery, built on the MAFAULDA dataset.
NUS-ISS Pattern Recognition Systems practice module (MTech AI Systems).
Research demonstration. Not validated on real machines.

## Quick start

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
Enter a display name. Then choose **Start investigation**.
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
Details are in `SystemCode/demo/manifest.json`. The pack is rebuilt with `scripts/40_demo_pack.py`.

### Optional: written explanations and the assistant

The app works without a language model. Without one it shows template explanations, and the assistant shows the fixed report.
For AI-written explanations and the investigation assistant, install [LM Studio](https://lmstudio.ai).
Load the model `google/gemma-4-e4b` and start its local server on port 1234.
The backend finds it at `http://localhost:1234/v1`. No API key is needed.

### Optional: the full dataset

The full dataset is needed to reproduce training and the evaluation, and to diagnose every development recording.
1. Download MAFAULDA from Kaggle: [Machinery Fault Dataset](https://www.kaggle.com/datasets/uysalserkan/fault-induction-motor-dataset).
2. Unpack it so that `mafaulda/` sits next to `SystemCode/` (the folder that holds `normal/`, `imbalance/`, and so on).
3. Convert it: `python scripts/00_convert.py` (in `SystemCode/`). This writes `SystemCode/data/raw_npy/`.

When `SystemCode/data/raw_npy/` exists, the app uses the full data instead of the demo pack.
See `SystemCode/HOW_TO_RUN.md` for training, notebooks and experiment tracking.

### Tests

In `SystemCode/`, with the venv active:

```bash
MACHINESENSE_REQUIRE_BUNDLE=1 python -m pytest tests -q
cd frontend && npm test
```

In a clone without the full dataset, tests that need it are skipped. The demo-pack tests still run.

## Repository layout

- `SystemCode/`: code, model bundle, demo pack, notebooks, results and tests. See `SystemCode/HOW_TO_RUN.md`.
- `CLAUDE.md`: the project's working rules and stage log.
