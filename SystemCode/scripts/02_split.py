"""Add split columns to metadata.csv (CLAUDE.md §7.6) and print balance tables.

    python scripts/02_split.py [--config configs/base.yaml]

Columns: speed_tercile, split_closed, loco_<class> ×9, split_speed_block, split_severity, cv_fold.
Seed comes from the config. After the 1 Oct freeze, re-running this needs a note in results/CHANGELOG.md.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from ml.config import load_config
from ml.labels import CLASS10
from ml.paths import DEFAULT_CONFIG, SYSTEM_CODE_DIR
from ml.splits import FAULT_CLASSES, assign_splits, split_columns, split_hash


def table(df: pd.DataFrame, col: str) -> pd.DataFrame:
    t = pd.crosstab(df["class10"], df[col]).reindex(CLASS10)
    t.loc["total"] = t.sum()
    return t


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    ap.add_argument("--metadata", type=Path, default=SYSTEM_CODE_DIR / "metadata.csv")
    args = ap.parse_args()

    cfg = load_config(args.config)
    df = pd.read_csv(args.metadata)
    df = df.drop(columns=[c for c in split_columns(df) + ["speed_tercile"] if c in df.columns])
    df = assign_splits(df, cfg.seed)

    pd.set_option("display.width", 200)
    for col in ["split_closed", "split_speed_block", "split_severity", "cv_fold"]:
        print(f"\n{col}\n{table(df, col).to_string()}")
    frac = pd.crosstab(df["class10"], df["split_closed"], normalize="index").reindex(CLASS10)
    print(f"\nsplit_closed fractions per class\n{frac.round(3).to_string()}")
    print(f"\nsplit_closed × speed tercile (all classes)\n{pd.crosstab(df['speed_tercile'], df['split_closed'])}")
    loco = pd.DataFrame({c: df[f"loco_{c}"].value_counts() for c in FAULT_CLASSES}).T.fillna(0).astype(int)
    print(f"\nloco columns\n{loco.to_string()}")

    df.to_csv(args.metadata, index=False)
    print(f"\nseed {cfg.seed}; split hash {split_hash(df)}; wrote {args.metadata}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
