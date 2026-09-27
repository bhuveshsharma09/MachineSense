"""CPU latency of ml.inference.diagnose over 20 development recordings (two calibration recordings per class, chosen by
name), with the bundle loaded once and one warm-up call; also writes the golden outputs for tests/test_inference.py.

    python scripts/16_inference_latency.py
"""
from __future__ import annotations

import json
import os
import platform
import sys
import time

import numpy as np
import pandas as pd

from ml.config import load_config
from ml.inference import diagnose, get_bundle, stable_output
from ml.paths import DEFAULT_CONFIG, SYSTEM_CODE_DIR, results_dir
from ml.splits import development_pool

GOLDEN_RECS = ["normal__18.432", "underhang__outer_race__0g__14.7456", "horizontal-misalignment__0.5mm__15.36"]


def main() -> int:
    import torch

    cfg = load_config(DEFAULT_CONFIG)
    meta = pd.read_csv(SYSTEM_CODE_DIR / "metadata.csv")
    pool = development_pool(meta, cfg.development.allowed_closed, cfg.development.reserved_columns)
    cal = pool[pool.split_closed == "val"].sort_values("rec_id")
    recs = cal.groupby("class10").rec_id.apply(lambda s: list(s)[:2]).sum()
    assert len(recs) == 20
    t0 = time.perf_counter()
    b = get_bundle()
    load_s = time.perf_counter() - t0
    diagnose(recs[0])  # warm-up
    full, no_xai = [], []
    for r in recs:
        t = time.perf_counter(); diagnose(r); full.append(1000 * (time.perf_counter() - t))
        t = time.perf_counter(); diagnose(r, with_xai=False); no_xai.append(1000 * (time.perf_counter() - t))
    out = {"recordings": recs, "device": "cpu", "torch_threads": torch.get_num_threads(), "machine": platform.platform(),
           "processor": platform.processor(), "cpu_count": os.cpu_count(), "bundle_load_s": load_s,
           "median_ms_with_xai": float(np.median(full)), "median_ms_without_xai": float(np.median(no_xai)),
           "range_ms_with_xai": [float(min(full)), float(max(full))],
           "note": "from a raw 5 s, 8-channel recording already in memory as .npy: cleaning, speed, R1 features, the cache-C "
                   "representation, all models, physics and (optionally) XAI; excludes reading a CSV upload"}
    (results_dir() / "bundle" / "latency.json").write_text(json.dumps(out, indent=1))
    outs = {r: stable_output(diagnose(r)) for r in GOLDEN_RECS}
    gold = {"manifest_sha256": b.manifest_sha256, "outputs": outs}
    (SYSTEM_CODE_DIR / "tests" / "golden" / "diagnose_golden.json").write_text(json.dumps(gold, indent=1, sort_keys=True))
    print(json.dumps({k: out[k] for k in ("bundle_load_s", "median_ms_with_xai", "median_ms_without_xai", "range_ms_with_xai")}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
