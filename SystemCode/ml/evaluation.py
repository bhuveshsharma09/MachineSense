"""Recording-level evaluation shared by every model: aggregation of window probabilities, class6 view, metrics with
bootstrap intervals over recordings, paired bootstrap differences, and block permutation importance. No torch."""
from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score

from ml.labels import CLASS6, CLASS10

CLASS10_TO_6 = {c: c if c in CLASS6 else c.split("_")[0] + "_bearing" for c in CLASS10}


def aggregate_recordings(proba: np.ndarray, rec_ids: Sequence[str], floor: float = 1e-6) -> pd.DataFrame:
    """Recording-level class10 probabilities: clip window probabilities to [floor, 1], average their logs over a
    recording's windows, exponentiate and renormalise. Rows = recordings (first-seen order), columns = CLASS10."""
    logp = np.log(np.clip(proba, floor, 1.0))
    df = pd.DataFrame(logp, columns=CLASS10)
    df["rec_id"] = list(rec_ids)
    mean = df.groupby("rec_id", sort=False).mean()
    p = np.exp(mean.to_numpy())
    return pd.DataFrame(p / p.sum(axis=1, keepdims=True), index=mean.index, columns=CLASS10)


def to_class6(proba10: pd.DataFrame) -> pd.DataFrame:
    """Sum recording-level class10 probabilities into class6."""
    return pd.DataFrame({c6: proba10[[c for c in CLASS10 if CLASS10_TO_6[c] == c6]].sum(axis=1) for c6 in CLASS6})


def metrics(y_true: Sequence[str], y_pred: Sequence[str], labels: Sequence[str], n_boot: int, seed: int) -> dict:
    """Macro-F1 with a 95 % bootstrap interval over the rows (recordings), balanced accuracy, per-class F1 and the
    confusion matrix. Only classes present in y_true count towards macro-F1."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    present = [c for c in labels if c in set(y_true)]
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        i = rng.integers(0, len(y_true), len(y_true))
        boots.append(f1_score(y_true[i], y_pred[i], labels=present, average="macro", zero_division=0))
    return {
        "macro_f1": float(f1_score(y_true, y_pred, labels=present, average="macro", zero_division=0)),
        "macro_f1_ci": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "per_class_f1": dict(zip(labels, f1_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
                                 .round(4).tolist())),
        "confusion": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "n": int(len(y_true)),
    }


def paired_bootstrap(y_true: Sequence[str], pred_a: Sequence[str], pred_b: Sequence[str], labels: Sequence[str],
                     n_boot: int, seed: int) -> dict:
    """Macro-F1(a) − macro-F1(b) on the same resampled recordings: point value, 95 % interval, P(diff > 0)."""
    y, a, b = map(np.asarray, (y_true, pred_a, pred_b))
    present = [c for c in labels if c in set(y)]
    f = lambda t, p: f1_score(t, p, labels=present, average="macro", zero_division=0)
    rng = np.random.default_rng(seed)
    d = []
    for _ in range(n_boot):
        i = rng.integers(0, len(y), len(y))
        d.append(f(y[i], a[i]) - f(y[i], b[i]))
    return {"diff": float(f(y, a) - f(y, b)), "ci": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))],
            "p_positive": float(np.mean(np.array(d) > 0))}


def block_permutation_importance(X: np.ndarray, rec_ids: np.ndarray, y_rec: pd.Series,
                                 predict_proba: Callable[[np.ndarray], np.ndarray], groups: dict[str, list[int]],
                                 n_repeats: int, seed: int, floor: float = 1e-6) -> pd.DataFrame:
    """Drop in recording-level macro-F1 when a group of columns is shuffled BETWEEN recordings as whole blocks:
    every window of recording r takes the values of the same window of recording π(r); windows are never shuffled
    individually. X rows must be ordered recording by recording, window_idx ascending, equal windows per recording.
    y_rec: true class10 per recording (index = rec_id)."""
    order = pd.unique(rec_ids)
    n_rec, n_win = len(order), len(rec_ids) // len(order)
    X3 = X.reshape(n_rec, n_win, X.shape[1])

    def score(Xw: np.ndarray) -> float:
        agg = aggregate_recordings(predict_proba(Xw), rec_ids, floor)
        return f1_score(y_rec.loc[agg.index], agg.idxmax(axis=1), average="macro", zero_division=0)

    base = score(X)
    rng = np.random.default_rng(seed)
    rows = []
    for name, cols in groups.items():
        drops = []
        for _ in range(n_repeats):
            perm = rng.permutation(n_rec)
            Xp = X3.copy()
            Xp[:, :, cols] = X3[perm][:, :, cols]
            drops.append(base - score(Xp.reshape(-1, X.shape[1])))
        rows.append({"group": name, "n_columns": len(cols), "importance": float(np.mean(drops)),
                     "std": float(np.std(drops))})
    return pd.DataFrame(rows).sort_values("importance", ascending=False).reset_index(drop=True)
