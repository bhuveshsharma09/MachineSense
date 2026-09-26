"""Index the MAFAULDA CSV tree into metadata.csv: labels, measured speed and quality numbers, one row per file.

    python scripts/01_index.py [--config configs/base.yaml] [--workers N] [--out metadata.csv]

Fails loudly (exit 1, nothing written) if class counts differ from CLAUDE.md §3.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from ml.config import PreprocessConfig, load_config
from ml.labels import CLASS10, parse_labels
from ml.paths import DEFAULT_CONFIG, SYSTEM_CODE_DIR, data_root
from ml.preprocess import (TACHO, choose_speed, load_csv, quality_flags, quality_stats, speed_candidates,
                           tacho_glitch_count)


def rec_id_for(rel: Path) -> str:
    """Stable, human-readable id: the relative path without .csv, '/' → '__'."""
    return "__".join(rel.with_suffix("").parts)


def index_one(args: tuple[Path, Path, PreprocessConfig]) -> dict:
    root, path, cfg = args
    rel = path.relative_to(root)
    lab = parse_labels(rel)
    row = {"rec_id": rec_id_for(rel), "path": rel.as_posix(), **lab.__dict__}
    try:
        x = load_csv(path)
    except Exception as exc:  # noqa: BLE001 — record, never stop the index
        row.update(read_error=f"{type(exc).__name__}: {exc}"[:200], speed_method="failed")
        row["quality_flags"] = ";".join(["read_error", *quality_flags(row, cfg)])
        return row
    cands = speed_candidates(x, lab.fr_file_hz, cfg)
    est = choose_speed(cands, lab.fr_file_hz, cfg)
    row.update(
        fr_measured_hz=est.fr_hz, speed_method=est.method, speed_agreement=est.agreement,
        speed_ratio=est.fr_hz / lab.fr_file_hz,
        **{f"fr_{k}": v for k, v in cands.items()},
        tacho_glitches=tacho_glitch_count(x[:, TACHO], cfg),
        **quality_stats(x, cfg.channels),
    )
    row["quality_flags"] = ";".join(quality_flags(row, cfg))
    return row


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 2))
    ap.add_argument("--out", type=Path, default=SYSTEM_CODE_DIR / "metadata.csv")
    args = ap.parse_args()

    cfg = load_config(args.config)
    root = data_root(cfg)
    files = sorted(root.rglob("*.csv"))
    print(f"indexing {len(files)} files under {root} with {args.workers} workers")

    t = time.perf_counter()
    with ProcessPoolExecutor(args.workers) as pool:
        rows = list(pool.map(index_one, [(root, f, cfg.preprocess) for f in files], chunksize=4))
    print(f"done in {time.perf_counter() - t:.0f} s")

    df = pd.DataFrame(rows).sort_values("rec_id").reset_index(drop=True)
    counts = df["class10"].value_counts().reindex(CLASS10, fill_value=0)
    expected = pd.Series(cfg.preprocess.expected_class_counts).reindex(CLASS10)
    table = pd.DataFrame({"found": counts, "expected": expected, "ok": counts == expected})
    print(table.to_string())
    if not table["ok"].all() or len(df) != cfg.preprocess.expected_total_files:
        print(f"FAIL: class counts differ from CLAUDE.md §3 (total {len(df)}); nothing written", file=sys.stderr)
        return 1

    df.to_csv(args.out, index=False)
    print(f"wrote {args.out} ({len(df)} rows, {df.shape[1]} columns)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
