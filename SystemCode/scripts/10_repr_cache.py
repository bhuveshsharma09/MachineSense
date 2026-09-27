"""Build the full representation cache for the FROZEN candidate (Stage 2C-2).

    python scripts/10_repr_cache.py

All 1,951 recordings → data/representations/<candidate>_<key12>.npy (N × 7 × 128 × T, float16: six accelerometer
order-spectrograms + the time-aligned microphone log-Mel), an index parquet (rec_id, window_idx; same row order as the
feature table) and a json with the cache key (ml.cache.representation_cache_key), git sha and provenance. Labels and
splits are never stored; they are joined from metadata.csv.
"""
from __future__ import annotations

import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from ml.cache import representation_cache_key, sources_from_metadata
from ml.config import load_config
from ml.datasets import representation_paths
from ml.paths import DEFAULT_CONFIG, SYSTEM_CODE_DIR, data_dir
from ml.spectro import stacked_representation
from ml.utils import git_sha


stacked = stacked_representation  # (9, 7, 128, T) float16 for one recording: the batch AND serving form (ml.spectro)


def _one(args):
    rid, fr, pre, cand = args
    return rid, stacked(np.load(data_dir() / "raw_npy" / f"{rid}.npy"), fr, pre, cand)


def main() -> int:
    cfg = load_config(DEFAULT_CONFIG)
    pre, rc = cfg.preprocess, cfg.representation
    if not rc.frozen:
        raise SystemExit("no representation is frozen")
    cand = rc.leading_candidate
    meta = pd.read_csv(SYSTEM_CODE_DIR / "metadata.csv").sort_values("rec_id")
    key = representation_cache_key(pre, sources_from_metadata(meta), cand, rc.version)
    out_dir = data_dir() / "representations"
    out_dir.mkdir(parents=True, exist_ok=True)
    npy, idx, info = representation_paths(out_dir, cand, key)
    T = pre.spectro.candidates[cand].frames_per_window
    n_win = pre.window.n_windows
    arr = np.lib.format.open_memmap(npy, mode="w+", dtype=np.float16,
                                    shape=(len(meta) * n_win, 7, pre.spectro.n_orders, T))
    order = {r: i for i, r in enumerate(meta.rec_id)}
    t0 = time.perf_counter()
    workers = max(1, (os.cpu_count() or 4) - 2)
    with ProcessPoolExecutor(workers) as pool:
        for rid, block in pool.map(_one, [(r, f, pre, cand) for r, f in zip(meta.rec_id, meta.fr_measured_hz)],
                                   chunksize=2):
            i = order[rid] * n_win
            arr[i:i + n_win] = block
    arr.flush()
    wall = time.perf_counter() - t0
    pd.DataFrame({"rec_id": np.repeat(meta.rec_id.to_numpy(), n_win),
                  "window_idx": np.tile(np.arange(n_win), len(meta))}).to_parquet(idx, index=False)
    info.write_text(json.dumps({"cache_key": key, "candidate": cand, "version": rc.version, "git_sha": git_sha(),
                                "shape": list(arr.shape), "dtype": "float16", "wall_s": round(wall, 1),
                                "workers": workers}, indent=1))
    print(f"{cand}: {arr.shape} float16, {npy.stat().st_size / 1e9:.2f} GB in {wall:.0f} s on {workers} workers → "
          f"{npy.relative_to(SYSTEM_CODE_DIR)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
