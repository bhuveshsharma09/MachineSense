"""Choose the envelope demodulation band separately for every evaluation track → results/demod_bands.json.

    python scripts/04_demod_bands.py

Tracks and the training data each one may use:
  closed        split_closed == "train"
  loco_<class>  loco_<class> == "train" (the withheld class is excluded by construction), × 9

The rule is ml.features.select_demod_band, unchanged per track. Test recordings are never read.
"""
from __future__ import annotations

import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from ml.config import PreprocessConfig, config_sha256, load_config
from ml.features import select_demod_band, spectral_kurtosis_table
from ml.paths import DEFAULT_CONFIG, SYSTEM_CODE_DIR, data_dir, results_dir
from ml.splits import FAULT_CLASSES, split_hash
from ml.utils import git_sha


def load(rec_id: str) -> np.ndarray:
    return np.load(data_dir() / "raw_npy" / f"{rec_id}.npy")


def _sk_chunk(args: tuple[list[str], PreprocessConfig]) -> pd.DataFrame:
    ids, pre = args
    return spectral_kurtosis_table(ids, pre, load)


def tracks(meta: pd.DataFrame) -> dict[str, pd.DataFrame]:
    out = {"closed": meta[meta.split_closed == "train"]}
    for cls in FAULT_CLASSES:
        out[f"loco_{cls}"] = meta[meta[f"loco_{cls}"] == "train"]
    return out


def main() -> int:
    cfg = load_config(DEFAULT_CONFIG)
    pre = cfg.preprocess
    sel = pre.envelope.selection
    meta = pd.read_csv(SYSTEM_CODE_DIR / "metadata.csv")
    by_track = tracks(meta)
    assert all((t.split_closed == "train").all() for t in by_track.values()), "a track reads non-train data"

    # Spectral kurtosis once for every recording any track can group (all are split_closed train).
    need = pd.concat(by_track.values()).drop_duplicates("rec_id")
    need = need[(need.class10 == sel.reference_class)
                | (need.class10.isin(sel.fault_classes) & need.severity.isin(sel.severities))]
    ids = sorted(need.rec_id)
    workers = max(1, (os.cpu_count() or 4) - 2)
    t = time.perf_counter()
    with ProcessPoolExecutor(workers) as pool:
        sk = pd.concat(pool.map(_sk_chunk, [(ids[i::workers], pre) for i in range(workers)]))
    print(f"spectral kurtosis for {len(ids)} train recordings in {time.perf_counter() - t:.0f} s")

    result = {
        "rule": "ml.features.select_demod_band",
        "selection_config_sha256": config_sha256(pre.envelope.selection),
        "git_sha": git_sha(),
        "split_hash": split_hash(meta),
        "tracks": {},
    }
    for name, train in by_track.items():
        result["tracks"][name] = {"training_data": "split_closed train" if name == "closed" else f"{name} == train",
                                  "channels": select_demod_band(train, sk, pre)}

    out = results_dir() / "demod_bands.json"
    out.write_text(json.dumps(result, indent=1) + "\n")
    print(f"{'track':34s} " + "  ".join(f"{ch:>30s}" for ch in sel.channels))
    for name, tr in result["tracks"].items():
        cells = []
        for ch in sel.channels:
            c = tr["channels"].get(ch)
            cells.append(f"{c['lo_hz'] / 1e3:5.1f}–{c['hi_hz'] / 1e3:4.1f} kHz "
                         + (f"({c['fault_class']}, {c['contrast']:+.2f})" if "contrast" in c else "(a priori)")
                         if c else "-")
        print(f"{name:34s} " + "  ".join(f"{x:>30s}" for x in cells))
    print(f"wrote {out.relative_to(SYSTEM_CODE_DIR)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
