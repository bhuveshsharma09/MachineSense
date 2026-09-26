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

import numpy as np
import pandas as pd
from scipy.stats import beta
from sklearn.metrics import roc_auc_score

from ml.anomaly import RECORDING_LEVEL, RUNGS, knn_speed_scores, window_scores
from ml.cache import load_features
from ml.config import load_config
from ml.features import feature_columns
from ml.labels import CLASS10
from ml.paths import DEFAULT_CONFIG, SYSTEM_CODE_DIR, data_dir, results_dir
from ml.protocol import max_threshold, speed_stratified_split
from ml.splits import development_pool, split_hash
from ml.utils import git_sha

OUT = results_dir() / "q1q3"
FAULTS = CLASS10[1:]


def score(rung, ac, fit: pd.DataFrame, target: pd.DataFrame, cols) -> pd.Series:
    """Recording-level anomaly scores of `target` from a detector fitted on `fit` windows."""
    if rung in RECORDING_LEVEL:
        return knn_speed_scores(ac, fit, target, cols)[0]
    w = window_scores(rung, ac, fit[cols].to_numpy(), fit.speed_hz.to_numpy(), target[cols].to_numpy(),
                      target.speed_hz.to_numpy(), cols)
    return pd.Series(w, index=target.rec_id.to_numpy()).groupby(level=0, sort=False).mean()


def mild_mask(rec: pd.DataFrame, ac) -> dict[str, pd.Series]:
    out = {}
    for fam, sev in ac.mildest.items():
        sel = ((rec.fault_family == fam) if fam == "bearing" else (rec.class10 == fam)) & np.isclose(rec.severity, sev)
        for c in sorted(set(rec.class10[sel])):
            out[f"{c} @ {sev:g}"] = sel & (rec.class10 == c)
    return out


def evaluate_rung(rung, ac, df, rec, cols, splits, folds):
    per_fold, detail = [], []
    for k in folds:
        normals_tr = df[(df.class10 == "normal") & (df.dev_fold != k)]
        part = splits[k]
        fit = normals_tr[normals_tr.rec_id.isin(part[part == "fit"].index)]
        cal_ids = part[part == "cal"].index
        cal = normals_tr[normals_tr.rec_id.isin(cal_ids)]
        test = df[df.dev_fold == k]
        s_cal = score(rung, ac, fit, cal, cols)
        s_test = score(rung, ac, fit, test, cols)
        thr = max_threshold(s_cal.to_numpy())
        r = rec.loc[s_test.index]
        healthy = s_test[r.class10 == "normal"]
        fold = {"fold": int(k), "n_fit": int(fit.rec_id.nunique()), "n_cal": int(len(s_cal)),
                "n_heldout_normal": int(len(healthy)), "threshold": thr,
                "false_alarms": int((healthy > thr).sum()), "measured_acceptance": float((healthy <= thr).mean()),
                "auroc": {}}
        y_all = np.r_[np.zeros(len(healthy)), np.ones(int((r.class10 != "normal").sum()))]
        fold["auroc"]["overall"] = float(roc_auc_score(y_all, np.r_[healthy, s_test[r.class10 != "normal"]]))
        for c in FAULTS:
            s = s_test[r.class10 == c]
            fold["auroc"][c] = float(roc_auc_score(np.r_[np.zeros(len(healthy)), np.ones(len(s))], np.r_[healthy, s]))
        per_fold.append(fold)
        detail.append(pd.DataFrame({"score": s_test, "alarm": s_test > thr, "fold": k}))
    d = pd.concat(detail).join(rec)
    n_h = int((d.class10 == "normal").sum())
    fa = int(d[d.class10 == "normal"].alarm.sum())
    faults = d[d.class10 != "normal"]
    out = {
        "per_fold": per_fold,
        "auroc_fold_mean": {c: float(np.mean([f["auroc"][c] for f in per_fold])) for c in ["overall", *FAULTS]},
        "auroc_fold_std": {c: float(np.std([f["auroc"][c] for f in per_fold])) for c in ["overall", *FAULTS]},
        "auroc_fold_min": {c: float(np.min([f["auroc"][c] for f in per_fold])) for c in ["overall", *FAULTS]},
        "auroc_fold_max": {c: float(np.max([f["auroc"][c] for f in per_fold])) for c in ["overall", *FAULTS]},
        "calibration_counts": [f["n_cal"] for f in per_fold], "fit_counts": [f["n_fit"] for f in per_fold],
        "theoretical_acceptance_reference": [f["n_cal"] / (f["n_cal"] + 1) for f in per_fold],
        "false_alarms_total": fa, "heldout_normals_total": n_h,
        "false_alarm_ci95_approx": [float(beta.ppf(0.025, fa, n_h - fa + 1)) if fa else 0.0,
                                    float(beta.ppf(0.975, fa + 1, n_h - fa)) if fa < n_h else 1.0],
        "measured_acceptance_total": float(1 - fa / n_h),
        "detection_overall": float(faults.alarm.mean()),
        "detection_by_class": {c: float(faults[faults.class10 == c].alarm.mean()) for c in FAULTS},
        "detection_mildest": {k: {"n": int(m.sum()), "detection": float(d.loc[m.reindex(d.index).fillna(False).to_numpy(), "alarm"].mean())}
                              for k, m in mild_mask(d, ac).items()},
    }
    return out


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
