# Demo pack

This folder lets a fresh clone run the app without the MAFAULDA dataset.

- `raw_npy/`: 5 demo recordings and the 4 healthy training recordings their evidence view compares against.
  Each file is one recording as a float32 array of 250,000 × 8 (5 s at 50 kHz), the format `ml.inference.load_recording` reads.
- `healthy_reference_values.json`: the evidence values of all 20 healthy training recordings.
  With it, an uploaded CSV at any speed still gets the channel comparison.
- `manifest.json`: every file's SHA-256, its split columns and the verdict the full setup gives.

All recordings are development recordings, not test data. The demo recordings are from the closed validation split,
which set the thresholds, so their verdicts are in-sample. No reserved test recording is included.
`scripts/40_demo_pack.py` builds the pack from the full data and checks every file with the split code and the
reserved-fingerprint guard. `tests/backend/test_demo_pack.py` repeats these checks.

The app uses this pack only when `SystemCode/data/raw_npy/` does not exist (see `backend/settings.py`, `use_demo_pack`).
