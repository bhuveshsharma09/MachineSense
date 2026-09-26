"""Q1 (healthy-only detection) evaluation under the corrected protocol (Stage 2D.1, ml/protocol.py), shared by the
classical detectors (scripts/11_track_b.py, 11b_b1s_scores.py) and the autoencoder (scripts/08_train_ae.py). No torch.

Per outer fold k: a detector fitted on FIT normals of folds ≠ k scores the CALIBRATION normals (threshold = their maximum;
higher = more anomalous; alarm iff score > threshold, ties do not alarm) and fold k (every evaluated recording once)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import beta
from sklearn.metrics import roc_auc_score

from ml.anomaly import RECORDING_LEVEL, knn_speed_scores, window_scores
from ml.config import AnomalyConfig
from ml.labels import CLASS10
from ml.protocol import max_threshold

FAULTS = CLASS10[1:]


@dataclass
class FoldScores:
    """Recording-level scores of one outer fold's detector: calibration normals and fold-k recordings (index rec_id)."""
    fold: int
    n_fit: int
    cal: pd.Series
    test: pd.Series


def classical_scores(rung: str, ac: AnomalyConfig, fit: pd.DataFrame, target: pd.DataFrame, cols) -> pd.Series:
    """Recording-level anomaly scores of `target` from a classical detector fitted on `fit` windows (mean of the 9
    window scores; kNN-by-speed is recording level)."""
    if rung in RECORDING_LEVEL:
        return knn_speed_scores(ac, fit, target, cols)[0]
    w = window_scores(rung, ac, fit[cols].to_numpy(), fit.speed_hz.to_numpy(), target[cols].to_numpy(),
                      target.speed_hz.to_numpy(), cols)
    return pd.Series(w, index=target.rec_id.to_numpy()).groupby(level=0, sort=False).mean()


def mild_mask(rec: pd.DataFrame, ac: AnomalyConfig) -> dict[str, pd.Series]:
    """Masks of the mildest setting per fault family (bearing 0 g = no added imbalance mass), one per class."""
    out = {}
    for fam, sev in ac.mildest.items():
        sel = ((rec.fault_family == fam) if fam == "bearing" else (rec.class10 == fam)) & np.isclose(rec.severity, sev)
        for c in sorted(set(rec.class10[sel])):
            out[f"{c} @ {sev:g}"] = sel & (rec.class10 == c)
    return out


def fold_result(fs: FoldScores, rec: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    """Per-fold threshold, false alarms, measured acceptance and AUROC (healthy vs each fault class and overall)."""
    thr = max_threshold(fs.cal.to_numpy())
    s_test = fs.test
    r = rec.loc[s_test.index]
    healthy = s_test[r.class10 == "normal"]
    out = {"fold": int(fs.fold), "n_fit": int(fs.n_fit), "n_cal": int(len(fs.cal)), "n_heldout_normal": int(len(healthy)),
           "threshold": thr, "false_alarms": int((healthy > thr).sum()),
           "measured_acceptance": float((healthy <= thr).mean()), "auroc": {}}
    y_all = np.r_[np.zeros(len(healthy)), np.ones(int((r.class10 != "normal").sum()))]
    out["auroc"]["overall"] = float(roc_auc_score(y_all, np.r_[healthy, s_test[r.class10 != "normal"]]))
    for c in FAULTS:
        s = s_test[r.class10 == c]
        out["auroc"][c] = float(roc_auc_score(np.r_[np.zeros(len(healthy)), np.ones(len(s))], np.r_[healthy, s]))
    return out, pd.DataFrame({"score": s_test, "alarm": s_test > thr, "fold": fs.fold})


def summarise(folds: list[FoldScores], rec: pd.DataFrame, ac: AnomalyConfig) -> tuple[dict, pd.DataFrame]:
    """The Stage 2D.1 Q1 summary: per-fold results, fold mean/std/min/max AUROC, calibration counts, false alarms out of
    the held-out normals (approximate Clopper–Pearson), measured acceptance, detection overall, by class and at the
    mildest settings. Also returns the per-recording detail (score, alarm, fold, labels)."""
    per_fold, detail = [], []
    for fs in folds:
        f, d = fold_result(fs, rec)
        per_fold.append(f)
        detail.append(d)
    d = pd.concat(detail).join(rec)
    n_h = int((d.class10 == "normal").sum())
    fa = int(d[d.class10 == "normal"].alarm.sum())
    faults = d[d.class10 != "normal"]
    keys = ["overall", *FAULTS]
    out = {
        "per_fold": per_fold,
        "auroc_fold_mean": {c: float(np.mean([f["auroc"][c] for f in per_fold])) for c in keys},
        "auroc_fold_std": {c: float(np.std([f["auroc"][c] for f in per_fold])) for c in keys},
        "auroc_fold_min": {c: float(np.min([f["auroc"][c] for f in per_fold])) for c in keys},
        "auroc_fold_max": {c: float(np.max([f["auroc"][c] for f in per_fold])) for c in keys},
        "calibration_counts": [f["n_cal"] for f in per_fold], "fit_counts": [f["n_fit"] for f in per_fold],
        "theoretical_acceptance_reference": [f["n_cal"] / (f["n_cal"] + 1) for f in per_fold],
        "false_alarms_total": fa, "heldout_normals_total": n_h,
        "false_alarm_ci95_approx": [float(beta.ppf(0.025, fa, n_h - fa + 1)) if fa else 0.0,
                                    float(beta.ppf(0.975, fa + 1, n_h - fa)) if fa < n_h else 1.0],
        "measured_acceptance_total": float(1 - fa / n_h),
        "detection_overall": float(faults.alarm.mean()),
        "detection_by_class": {c: float(faults[faults.class10 == c].alarm.mean()) for c in FAULTS},
        "detection_mildest": {k: {"n": int(m.sum()),
                                  "detection": float(d.loc[m.reindex(d.index).fillna(False).to_numpy(), "alarm"].mean())}
                              for k, m in mild_mask(d, ac).items()},
    }
    return out, d


def load_assignment(path) -> dict[int, pd.Series]:
    """Saved Stage 2D.1 healthy fit/calibration assignment → {outer fold: Series(rec_id → "fit" | "cal")}."""
    a = pd.read_csv(path)
    return {int(k): g.set_index("rec_id").part for k, g in a.groupby("outer_fold")}


def fold_ids(assignment: dict[int, pd.Series], pool: pd.DataFrame, fold: int):
    """Outer fold `fold` of the Q1 protocol: (fit IDs, calibration IDs, evaluation IDs, NormContext). Fit and calibration
    are the saved Stage 2D.1 healthy parts of the folds ≠ fold; evaluation = every pool recording of `fold`. The context
    excludes calibration and evaluation from normalisation. Raises if the assignment is inconsistent with the pool."""
    from ml.datasets import NormContext

    part = assignment[fold]
    fit = sorted(part[part == "fit"].index)
    cal = sorted(part[part == "cal"].index)
    ev = sorted(pool.loc[pool.dev_fold == fold, "rec_id"])
    normals_tr = set(pool.loc[(pool.class10 == "normal") & (pool.dev_fold != fold), "rec_id"])
    if set(fit) | set(cal) != normals_tr or set(fit) & set(cal):
        raise ValueError(f"fold {fold}: fit/cal assignment does not partition the outer-training normals")
    return fit, cal, ev, NormContext(calibration=frozenset(cal), evaluation=frozenset(ev))
