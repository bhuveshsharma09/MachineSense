"""Convert the MAFAULDA CSVs to float32 .npy files in data/raw_npy/<rec_id>.npy, then verify them.

    python scripts/00_convert.py [--fraction 0.1] [--workers N]

Files are converted in class-stratified order, so the first 10 % is a balanced subset usable before the rest
finishes. Existing .npy files are kept (the run resumes). Verification re-reads every .npy and writes
convert_order, n_rows_verified and npy_sha256 into metadata.csv. CSVs are never deleted.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from ml.config import load_config
from ml.paths import DEFAULT_CONFIG, SYSTEM_CODE_DIR, data_dir, data_root
from ml.preprocess import load_csv

SEED = 0


def stratified_order(df: pd.DataFrame, seed: int = SEED) -> pd.Series:
    """Rank recordings so any prefix is class-balanced: a random rank within each class, scaled to [0, 1)."""
    rng = np.random.default_rng(seed)
    key = pd.Series(0.0, index=df.index)
    for _, idx in df.groupby("class10").groups.items():
        key[idx] = (rng.permutation(len(idx)) + rng.random()) / len(idx)
    return key.rank(method="first").astype(int) - 1


def convert_one(args: tuple[Path, Path]) -> str:
    csv_path, npy_path = args
    if npy_path.exists():
        return "skipped"
    x = load_csv(csv_path)
    tmp = npy_path.with_suffix(".tmp.npy")
    np.save(tmp, np.ascontiguousarray(x, dtype=np.float32))
    os.replace(tmp, npy_path)  # atomic: a crash never leaves a half-written .npy
    return "converted"


def verify_one(args: tuple[Path, int, int]) -> tuple[int, str]:
    npy_path, n_rows, n_cols = args
    x = np.load(npy_path, mmap_mode="r")
    ok = x.dtype == np.float32 and x.shape == (n_rows, n_cols) and np.isfinite(x).all()
    digest = hashlib.sha256(npy_path.read_bytes()).hexdigest()
    return (x.shape[0] if ok else -1), digest


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    ap.add_argument("--metadata", type=Path, default=SYSTEM_CODE_DIR / "metadata.csv")
    ap.add_argument("--fraction", type=float, default=1.0, help="convert only this leading share of the order")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 2))
    args = ap.parse_args()

    cfg = load_config(args.config)
    root = data_root(cfg)
    out_dir = data_dir() / "raw_npy"
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.metadata)
    df["convert_order"] = stratified_order(df)
    todo = df.sort_values("convert_order")
    todo = todo.iloc[: int(np.ceil(args.fraction * len(todo)))]
    jobs = [(root / p, out_dir / f"{r}.npy") for p, r in zip(todo["path"], todo["rec_id"])]

    t0 = time.perf_counter()
    first_tenth = max(1, len(df) // 10)
    with ProcessPoolExecutor(args.workers) as pool:
        for i, status in enumerate(pool.map(convert_one, jobs, chunksize=2), start=1):
            if i == first_tenth or i == len(jobs):
                print(f"{i}/{len(jobs)} files in {time.perf_counter() - t0:.0f} s")
    convert_s = time.perf_counter() - t0

    t1 = time.perf_counter()
    n = cfg.preprocess.n_rows, cfg.preprocess.n_channels
    have = df[[(out_dir / f"{r}.npy").exists() for r in df["rec_id"]]]
    with ProcessPoolExecutor(args.workers) as pool:
        results = list(pool.map(verify_one, [(out_dir / f"{r}.npy", *n) for r in have["rec_id"]], chunksize=4))
    df.loc[have.index, "n_rows_verified"] = [r[0] for r in results]
    df.loc[have.index, "npy_sha256"] = [r[1] for r in results]
    df["n_rows_verified"] = df["n_rows_verified"].astype("Int64")
    verify_s = time.perf_counter() - t1

    bad = df[df["n_rows_verified"].notna() & (df["n_rows_verified"] != n[0])]
    size_gb = sum(f.stat().st_size for f in out_dir.glob("*.npy")) / 1e9
    print(f"converted in {convert_s:.0f} s, verified {len(have)} files in {verify_s:.0f} s; "
          f"{size_gb:.1f} GB in {out_dir}")
    df.to_csv(args.metadata, index=False)
    if len(bad):
        print(f"FAIL: {len(bad)} files failed verification: {bad['rec_id'].tolist()[:10]}", file=sys.stderr)
        return 1
    print(f"all {len(have)} verified: float32 {n}, finite")
    return 0


if __name__ == "__main__":
    sys.exit(main())
