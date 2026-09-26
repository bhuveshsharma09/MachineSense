"""R1 physics features (Stage 2A).

    python scripts/07_features.py dictionary   # write results/feature_dictionary.csv
"""
from __future__ import annotations

import argparse
import sys

from ml.config import load_config
from ml.features import feature_dictionary
from ml.paths import DEFAULT_CONFIG, SYSTEM_CODE_DIR, results_dir


def write_dictionary() -> None:
    pre = load_config(DEFAULT_CONFIG).preprocess
    d = feature_dictionary(pre)
    out = results_dir() / "feature_dictionary.csv"
    d.to_csv(out, index=False)
    print(f"{len(d)} features ({pre.features.version}) → {out.relative_to(SYSTEM_CODE_DIR)}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("task", choices=["dictionary"])
    args = ap.parse_args()
    if args.task == "dictionary":
        write_dictionary()
    return 0


if __name__ == "__main__":
    sys.exit(main())
