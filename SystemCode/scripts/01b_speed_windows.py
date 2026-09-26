"""Add per-window speed variability to metadata.csv (audit columns; never model inputs).

    python scripts/01b_speed_windows.py

Runs after 01_index.py and 02_split.py. For every recording: ml.preprocess.window_speeds (1 s windows, 0.5 s hop,
fallback = fr_measured_hz), then speed_min_hz, speed_max_hz, speed_range_pct (range / fr_measured_hz × 100) and
speed_window_fallbacks (NaN range when every window fell back). Split columns are untouched; the split hash is printed before and after.
"""
from __future__ import annotations

import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from ml.config import PreprocessConfig, load_config
from ml.paths import DEFAULT_CONFIG, SYSTEM_CODE_DIR, data_dir
from ml.preprocess import TACHO, window_speeds
from ml.splits import split_hash

COLUMNS = ["speed_min_hz", "speed_max_hz", "speed_range_pct", "speed_window_fallbacks"]


def one(args: tuple[str, float, PreprocessConfig]) -> dict:
    rec_id, fr, pre = args
    tacho = np.load(data_dir() / "raw_npy" / f"{rec_id}.npy", mmap_mode="r")[:, TACHO]
    sp, n_fb = window_speeds(np.asarray(tacho), pre.fs_hz, cfg=pre, fr_fallback_hz=fr)
    if n_fb == sp.size:  # no window measured: variability is unknown, not zero
        return {"rec_id": rec_id, "speed_min_hz": np.nan, "speed_max_hz": np.nan, "speed_range_pct": np.nan,
                "speed_window_fallbacks": n_fb}
    return {"rec_id": rec_id, "speed_min_hz": sp.min(), "speed_max_hz": sp.max(),
            "speed_range_pct": 100 * (sp.max() - sp.min()) / fr, "speed_window_fallbacks": n_fb}


def main() -> int:
    pre = load_config(DEFAULT_CONFIG).preprocess
    path = SYSTEM_CODE_DIR / "metadata.csv"
    meta = pd.read_csv(path)
    before = split_hash(meta)
    print(f"split hash before {before}")
    with ProcessPoolExecutor(max(1, (os.cpu_count() or 4) - 2)) as pool:
        rows = list(pool.map(one, [(r, f, pre) for r, f in zip(meta.rec_id, meta.fr_measured_hz)], chunksize=8))
    new = pd.DataFrame(rows)
    meta = meta.drop(columns=[c for c in COLUMNS if c in meta.columns]).merge(new, on="rec_id", how="left")
    after = split_hash(meta)
    print(f"split hash after  {after}")
    if after != before:
        print("FAIL: split columns changed; nothing written", file=sys.stderr)
        return 1
    meta.to_csv(path, index=False)
    print(f"wrote {len(COLUMNS)} columns for {len(new)} recordings to metadata.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
