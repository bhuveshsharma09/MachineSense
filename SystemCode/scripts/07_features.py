"""R1 physics features (Stage 2A).

    python scripts/07_features.py dictionary             # results/feature_dictionary.csv (primary) or _<version>.csv
    python scripts/07_features.py subset --fraction 0.1  # class-stratified subset of split_closed TRAIN → cache
    python scripts/07_features.py full --name r1cap10    # every recording, primary pipeline → data/features/
    python scripts/07_features.py full --name r1 --config configs/variants/r1_bandstop_only.yaml  # the earlier record
    python scripts/07_features.py lowpass-edge           # measured edge effect of the primary low-pass

The subset cache is data/cache/r1_subset_<key12>.parquet; full tables are data/features/<name>_<key12>.parquet, keyed by ml.cache.feature_cache_key (full config,
feature definitions, source npy checksums and speeds). Full caches are not built in Stage 2A.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from ml.cache import feature_cache_key, feature_cache_path, sources_from_metadata, write_feature_cache
from ml.config import PreprocessConfig, load_config
from ml.features import batch_features, feature_columns, feature_dictionary
from ml.preprocess import lowpass_taps, measure_edge_effect
from pathlib import Path

from ml.paths import DEFAULT_CONFIG, SYSTEM_CODE_DIR, cache_file, data_dir, results_dir
from ml.utils import git_sha

SEED = 0


def load(rec_id: str) -> np.ndarray:
    return np.load(data_dir() / "raw_npy" / f"{rec_id}.npy")


def write_dictionary(pre: PreprocessConfig, primary: bool) -> None:
    d = feature_dictionary(pre)
    out = results_dir() / ("feature_dictionary.csv" if primary else f"feature_dictionary_{pre.features.version}.csv")
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
    keep = ()  # labels and splits are joined from metadata.csv at load time (ml.cache.load_features)
    t = time.perf_counter()
    with ProcessPoolExecutor(workers) as pool:
        parts = list(pool.map(_one, [(sub[sub.rec_id == r], pre, keep) for r in sub.rec_id], chunksize=1))
    wall = time.perf_counter() - t
    df = pd.concat(parts, ignore_index=True)
    key = feature_cache_key(pre, sources_from_metadata(meta, sub.rec_id))
    out = feature_cache_path(data_dir() / "cache", "r1_subset", key)
    write_feature_cache(df, out, key, {"git_sha": git_sha(),
                                       "feature_version": pre.features.version, "fraction": fraction,
                                       "workers": workers, "wall_s": f"{wall:.1f}"})
    per = df.groupby("rec_id").compute_s.first()
    print(f"{len(sub)} recordings ({len(df)} windows) in {wall:.1f} s wall with {workers} workers; "
          f"{per.mean():.2f} s per recording (single core) → {out.relative_to(SYSTEM_CODE_DIR)}")


def run_full(meta: pd.DataFrame, pre: PreprocessConfig, name: str, workers: int) -> None:
    """Every recording (deterministic features; test recordings included, labels untouched) → data/features."""
    t = time.perf_counter()
    rows = meta.sort_values("rec_id")
    with ProcessPoolExecutor(workers) as pool:
        parts = list(pool.map(_one, [(rows[rows.rec_id == r], pre, ()) for r in rows.rec_id], chunksize=4))
    wall = time.perf_counter() - t
    df = pd.concat(parts, ignore_index=True)
    key = feature_cache_key(pre, sources_from_metadata(meta, rows.rec_id))
    out = feature_cache_path(data_dir() / "features", name, key)
    write_feature_cache(df, out, key, {"git_sha": git_sha(), "feature_version": pre.features.version,
                                       "workers": workers, "wall_s": f"{wall:.1f}"})
    fc = feature_columns(df)
    per_rec = df.groupby("rec_id").size()
    fb = df.merge(meta[["rec_id", "class10"]]).groupby("class10").speed_fallback.sum()
    print(f"{name}: {df.rec_id.nunique()} recordings, {len(df)} windows, {len(fc)} features in {wall:.0f} s → "
          f"{out.relative_to(SYSTEM_CODE_DIR)}")
    print(f"  NaN/inf: {int((~np.isfinite(df[fc].to_numpy())).sum())}; recordings without 9 windows: "
          f"{int((per_rec != 9).sum())}; fallback windows by class: {fb[fb > 0].to_dict()}")


def _edge_one(args: tuple[str, PreprocessConfig]) -> dict:
    rec_id, pre = args
    x = load(rec_id)[:, 1:].astype(np.float64)
    return {"rec_id": rec_id, **measure_edge_effect(x, lowpass_taps(pre), pre.fs_hz)}


def run_lowpass_edge(meta: pd.DataFrame, pre: PreprocessConfig, workers: int) -> None:
    """Measured edge effect of the variant's low-pass on one train recording per class."""
    sel = meta[meta.split_closed == "train"].sort_values("rec_id").groupby("class10").sample(n=1, random_state=SEED)
    with ProcessPoolExecutor(workers) as pool:
        df = pd.DataFrame(list(pool.map(_edge_one, [(r, pre) for r in sel.rec_id]))).merge(sel[["rec_id", "class10"]])
    out = cache_file("07_lowpass_edge", pre.lowpass, ext="parquet")
    df.to_parquet(out, index=False)
    print(df.drop(columns="rec_id").describe().loc[["min", "50%", "max"]].to_string())
    print(f"→ {out.relative_to(SYSTEM_CODE_DIR)}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("task", choices=["dictionary", "subset", "full", "lowpass-edge"])
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    ap.add_argument("--name", default="r1cap10")
    ap.add_argument("--fraction", type=float, default=0.1)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 2))
    args = ap.parse_args()
    pre = load_config(args.config).preprocess
    meta = pd.read_csv(SYSTEM_CODE_DIR / "metadata.csv")
    if args.task == "dictionary":
        write_dictionary(pre, primary=args.config == DEFAULT_CONFIG)
    elif args.task == "subset":
        run_subset(meta, pre, args.fraction, args.workers)
    elif args.task == "full":
        run_full(meta, pre, args.name, args.workers)
    else:
        run_lowpass_edge(meta, pre, args.workers)
    return 0


if __name__ == "__main__":
    sys.exit(main())
