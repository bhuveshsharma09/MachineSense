"""Typical healthy variation per channel and metric, for the measured-change marker (Stage App-2a.2).

Method (declared before use; TRAINING normals only, never calibration or reserved recordings):
- recordings: the healthy training recordings of the bundle (closed train ∩ development pool, class normal);
- pairs: every pair of them whose measured speeds differ by at most SPEED_WINDOW_HZ;
- for each pair, channel and metric: the change factor max(ratio, 1/ratio) of the two recordings' values (mean over the
  9 windows of the R1 features, as in backend.evidence);
- band = the PERCENTILE-th percentile of those factors over the pairs.
Metrics in volts are compared only when at least one side reaches MIN_AMPLITUDE_V; excess kurtosis is never ratioed.

    python -m backend.variation build     # writes backend/data/healthy_variation.json (also built on first start)
"""
from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

import numpy as np

SPEED_WINDOW_HZ = 3.0
PERCENTILE = 95.0
# Median healthy over-radial 1× amplitude of the training normals (0.0051 V): lines below this on both sides are too small
# for their ratio to mean anything (a 0.004 → 0.08 V change reads as ×20).
MIN_AMPLITUDE_V = 0.005
VOLT_METRICS = ("rms", "order_1x", "order_2x")
NOT_RATIOED = ("kurtosis",)


def comparable(metric: str, a: float, b: float) -> bool:
    """Whether two values of a metric may be compared as a ratio."""
    if metric in NOT_RATIOED or not (a > 0 and b > 0):
        return False
    return not (metric in VOLT_METRICS and max(a, b) < MIN_AMPLITUDE_V)


def factor(a: float, b: float) -> float:
    r = a / b
    return float(max(r, 1 / r))


def training_normals(meta, cfg):
    from ml.splits import development_pool
    pool = development_pool(meta, cfg.development.allowed_closed, cfg.development.reserved_columns)
    return pool[(pool.split_closed == "train") & (pool.class10 == "normal")].sort_values("fr_measured_hz")


def build(meta, cfg, values_for) -> dict:
    """values_for(rec_id) → channel_values dict. Returns the band table with its method."""
    tr = training_normals(meta, cfg)
    speed = dict(zip(tr.rec_id, tr.fr_measured_hz))
    vals = {r: values_for(r) for r in tr.rec_id}
    pairs = [(a, b) for a, b in itertools.combinations(tr.rec_id, 2) if abs(speed[a] - speed[b]) <= SPEED_WINDOW_HZ]
    bands, counts = {}, {}
    for ch, metrics in vals[tr.rec_id.iloc[0]].items():
        bands[ch], counts[ch] = {}, {}
        for m in metrics:
            f = [factor(vals[a][ch][m], vals[b][ch][m]) for a, b in pairs if comparable(m, vals[a][ch][m], vals[b][ch][m])]
            bands[ch][m] = float(np.percentile(f, PERCENTILE)) if f else None
            counts[ch][m] = len(f)
    return {"method": {"recordings": "healthy training recordings (closed train ∩ development pool); never calibration "
                                     "or reserved recordings",
                       "pairs": f"all pairs with measured speeds within ±{SPEED_WINDOW_HZ:g} Hz",
                       "statistic": f"{PERCENTILE:g}th percentile of max(ratio, 1/ratio) over the pairs",
                       "min_amplitude_v": MIN_AMPLITUDE_V, "volt_metrics": list(VOLT_METRICS),
                       "not_ratioed": list(NOT_RATIOED),
                       "min_amplitude_reason": "median healthy over-radial 1× amplitude of the training normals"},
            "n_recordings": len(tr), "n_pairs": len(pairs), "recordings": list(tr.rec_id),
            "bands": bands, "n_pairs_used": counts}


def main() -> int:
    from backend.evidence import training_values
    from backend.settings import bundle_dir, variation_file
    from ml.bundle import load_bundle
    from ml.inference import _metadata

    if sys.argv[1:] != ["build"]:
        print("usage: python -m backend.variation build")
        return 2
    cfg = load_bundle(bundle_dir(), device="cpu").config
    out = build(_metadata().reset_index(), cfg, lambda r: training_values(r, cfg))
    Path(variation_file()).parent.mkdir(parents=True, exist_ok=True)
    Path(variation_file()).write_text(json.dumps(out, indent=1, ensure_ascii=False))
    print(json.dumps(out, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
