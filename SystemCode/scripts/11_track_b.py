"""Healthy-only detectors (Track B classical baselines, Q1), CORRECTED protocol (Stage 2D.1; ml/protocol.py).

    python scripts/11_track_b.py

For each outer dev fold k (results/dev_folds.csv): the outer-training normal recordings (folds ≠ k) are split, sorted by
speed and dealt 2:1 with a fixed seed, into FIT and CALIBRATION (assignment saved). Each detector is fitted on FIT only;
its threshold is the maximum CALIBRATION score of the same fitted detector (score direction fixed: higher = more
anomalous; alarm iff score > threshold, ties do not alarm). The same detector and threshold score fold k only: its normal
recordings (false alarms) and its fault recordings (detection) — every evaluated recording exactly once, by its own
outer fold's model. Recording score = mean of its 9 window scores (kNN-by-speed: recording level).
Primary: per-fold AUROC (healthy vs each fault class and overall), mean and spread across folds. The healthy sample is
25 recordings from ONE folder, so healthy-vs-fault may partly reflect folder proxies.
"""
from __future__ import annotations

import json
import sys

import pandas as pd
from ml.anomaly import RUNGS
from ml.cache import load_features
from ml.config import load_config
from ml.features import feature_columns
from ml.paths import DEFAULT_CONFIG, SYSTEM_CODE_DIR, data_dir, results_dir
from ml.protocol import speed_stratified_split
from ml.q1 import FoldScores, classical_scores, summarise
from ml.splits import development_pool, split_hash
from ml.utils import git_sha

OUT = results_dir() / "q1q3"


def evaluate_rung(rung, ac, df, rec, cols, splits, folds):
    """Fit per outer fold on FIT normals, score CALIBRATION normals and fold k, then the shared Q1 summary."""
    fs = []
    for k in folds:
        normals_tr = df[(df.class10 == "normal") & (df.dev_fold != k)]
        part = splits[k]
        fit = normals_tr[normals_tr.rec_id.isin(part[part == "fit"].index)]
        cal = normals_tr[normals_tr.rec_id.isin(part[part == "cal"].index)]
        fs.append(FoldScores(int(k), fit.rec_id.nunique(), classical_scores(rung, ac, fit, cal, cols),
                             classical_scores(rung, ac, fit, df[df.dev_fold == k], cols)))
    return summarise(fs, rec, ac)[0]


def main() -> int:
    cfg = load_config(DEFAULT_CONFIG)
    ac, pre, dc = cfg.anomaly, cfg.preprocess, cfg.development
    meta = pd.read_csv(SYSTEM_CODE_DIR / "metadata.csv")
    pool = development_pool(meta, dc.allowed_closed, dc.reserved_columns).merge(
        pd.read_csv(results_dir() / dc.folds_file)[["rec_id", "dev_fold"]], on="rec_id")
    path = max((data_dir() / "features").glob(f"{cfg.baselines.feature_table}_*.parquet"), key=lambda p: p.stat().st_mtime)
    df = load_features(path, meta, pre).merge(pool[["rec_id", "folder", "dev_fold"]], on="rec_id")
    df = df.sort_values(["rec_id", "window_idx"]).reset_index(drop=True)
    rec = pool.set_index("rec_id")[["class10", "fault_family", "severity", "folder", "fr_measured_hz", "dev_fold"]]
    folds = sorted(df.dev_fold.unique())
    normals = pool[pool.class10 == "normal"]
    splits, rows = {}, []
    for k in folds:
        part = speed_stratified_split(normals[normals.dev_fold != k], {"fit": 2, "cal": 1}, seed=ac.seed + int(k))
        splits[k] = part
        rows += [{"outer_fold": int(k), "rec_id": r, "part": p} for r, p in part.items()]
    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT / "q1_fit_cal_assignment.csv", index=False)
    all_cols = feature_columns(df)
    no_mic = [c for c in all_cols if not c.startswith("f_mic_")]
    res = {"provenance": {"feature_table": path.name, "split_hash": split_hash(meta), "git_sha": git_sha(),
                          "protocol": "ml/protocol.py; CLAUDE.md rule 13", "n_healthy": int(len(normals)),
                          "healthy_caveat": "25 healthy recordings from ONE folder: healthy-vs-fault may partly reflect "
                                            "folder proxies",
                          "ci_note": "the predictions are held out, but folds share training data, so this binomial "
                                     "interval is approximate, not an exact guarantee for the whole procedure",
                          "acceptance_note": "n/(n+1) is a theoretical reference under exchangeability and continuous "
                                             "scores, not a guarantee for this speed-stratified dataset"},
           "rungs": {}}
    for rung in RUNGS:
        res["rungs"][rung] = {"all": evaluate_rung(rung, ac, df, rec, all_cols, splits, folds)}
        e = res["rungs"][rung]["all"]
        print(f"  {rung:16s} AUROC overall {e['auroc_fold_mean']['overall']:.3f} ± {e['auroc_fold_std']['overall']:.3f} "
              f"| cal {e['calibration_counts']} | FA {e['false_alarms_total']}/{e['heldout_normals_total']} "
              f"(acceptance {e['measured_acceptance_total']:.2f}) | detection {e['detection_overall']:.3f}")
    best2 = sorted(RUNGS, key=lambda r: -res["rungs"][r]["all"]["auroc_fold_mean"]["overall"])[:2]
    for rung in best2:
        res["rungs"][rung]["no_mic"] = evaluate_rung(rung, ac, df, rec, no_mic, splits, folds)
        e = res["rungs"][rung]["no_mic"]
        print(f"  {rung} without mic: AUROC {e['auroc_fold_mean']['overall']:.3f} ± {e['auroc_fold_std']['overall']:.3f}, "
              f"FA {e['false_alarms_total']}/{e['heldout_normals_total']}, detection {e['detection_overall']:.3f}")
    res["best_two_by_fold_mean_auroc"] = best2
    (OUT / "q1_corrected.json").write_text(json.dumps(res, indent=1))
    print(f"fit/cal counts per fold: {[(int(k), int((splits[k] == 'fit').sum()), int((splits[k] == 'cal').sum())) for k in folds]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
