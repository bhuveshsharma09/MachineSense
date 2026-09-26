"""R1 physics features (Stage 2A).

    python scripts/07_features.py dictionary             # write results/feature_dictionary.csv
    python scripts/07_features.py subset --fraction 0.1  # class-stratified subset of split_closed TRAIN → cache

The subset cache is data/cache/r1_subset_shared_<key12>.parquet, keyed by ml.cache.feature_cache_key (full config,
feature definitions, source npy checksums). Full caches are not built in Stage 2A.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from ml.cache import feature_cache_key, feature_cache_path, write_feature_cache
from ml.config import PreprocessConfig, load_config
from ml.features import batch_features, feature_dictionary
from ml.paths import DEFAULT_CONFIG, SYSTEM_CODE_DIR, data_dir, results_dir
from ml.splits import split_columns, split_hash
from ml.utils import git_sha

SEED = 0


def load(rec_id: str) -> np.ndarray:
    return np.load(data_dir() / "raw_npy" / f"{rec_id}.npy")


def write_dictionary(pre: PreprocessConfig) -> None:
    d = feature_dictionary(pre)
    out = results_dir() / "feature_dictionary.csv"
    d.to_csv(out, index=False)
    print(f"{len(d)} features ({pre.features.version}) → {out.relative_to(SYSTEM_CODE_DIR)}")


def stratified_subset(meta: pd.DataFrame, fraction: float) -> pd.DataFrame:
    """ceil(fraction × n) split_closed TRAIN recordings per class10, fixed seed."""
    train = meta[meta.split_closed == "train"].sort_values("rec_id")
    return pd.concat([g.sample(n=int(np.ceil(fraction * len(g))), random_state=SEED)
                      for _, g in train.groupby("class10")])


def _one(args: tuple[pd.DataFrame, PreprocessConfig, tuple[str, ...]]) -> pd.DataFrame:
    row, pre, keep = args
    t = time.perf_counter()
    df = batch_features(row, pre, load, keep)
    df["compute_s"] = time.perf_counter() - t
    return df


def run_subset(meta: pd.DataFrame, pre: PreprocessConfig, fraction: float, workers: int) -> None:
    sub = stratified_subset(meta, fraction)
    keep = ("class10", "severity", "fr_measured_hz", *split_columns(meta))
    t = time.perf_counter()
    with ProcessPoolExecutor(workers) as pool:
        parts = list(pool.map(_one, [(sub[sub.rec_id == r], pre, keep) for r in sub.rec_id], chunksize=1))
    wall = time.perf_counter() - t
    df = pd.concat(parts, ignore_index=True)
    shas = dict(zip(meta.rec_id, meta.npy_sha256))
    key = feature_cache_key(pre, {r: shas[r] for r in sub.rec_id})
    out = feature_cache_path(data_dir() / "cache", "r1_subset", key)
    write_feature_cache(df, out, key, {"split_hash": split_hash(meta), "git_sha": git_sha(),
                                       "feature_version": pre.features.version, "fraction": fraction,
                                       "workers": workers, "wall_s": f"{wall:.1f}"})
    per = df.groupby("rec_id").compute_s.first()
    print(f"{len(sub)} recordings ({len(df)} windows) in {wall:.1f} s wall with {workers} workers; "
          f"{per.mean():.2f} s per recording (single core) → {out.relative_to(SYSTEM_CODE_DIR)}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("task", choices=["dictionary", "subset"])
    ap.add_argument("--fraction", type=float, default=0.1)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 2))
    args = ap.parse_args()
    pre = load_config(DEFAULT_CONFIG).preprocess
    if args.task == "dictionary":
        write_dictionary(pre)
    else:
        run_subset(pd.read_csv(SYSTEM_CODE_DIR / "metadata.csv"), pre, args.fraction, args.workers)
    return 0


if __name__ == "__main__":
    sys.exit(main())
