"""Unknown-fault (Q3) scores and helpers, shared by the classical LOCO baseline (scripts/12_track_c1.py) and the deep
LOCO track (scripts/09_loco_cnn.py). No torch. Every score is oriented so that HIGHER = MORE UNKNOWN, except where a
function says otherwise (MSP itself is returned as a probability: lower = more unknown)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.special import logsumexp
from sklearn.covariance import LedoitWolf


def unknown_scores(model, X: np.ndarray, rec_ids: np.ndarray, floor: float) -> pd.DataFrame:
    """Classical model, per recording: predicted class and three unknown scores (higher = more unknown): −MSP of the
    aggregated probabilities (mean log-probability, renormalised), −max of the mean logits, energy −logsumexp(mean
    logits)."""
    classes = np.asarray(model.classes_)
    lp = pd.DataFrame(np.log(np.clip(model.predict_proba(X), floor, 1.0)), columns=classes).assign(rec_id=rec_ids)
    lp = lp.groupby("rec_id", sort=False).mean()
    prob = np.exp(lp.to_numpy())
    prob /= prob.sum(axis=1, keepdims=True)
    lg = pd.DataFrame(model.decision_function(X), columns=classes).assign(rec_id=rec_ids).groupby("rec_id", sort=False).mean()
    return pd.DataFrame({"pred": classes[prob.argmax(axis=1)], "msp": -prob.max(axis=1),
                         "max_logit": -lg.to_numpy().max(axis=1), "energy": -logsumexp(lg.to_numpy(), axis=1)},
                        index=lp.index)


def oscr(u_known: np.ndarray, known_correct: np.ndarray, u_unknown: np.ndarray) -> float:
    """Area under CCR (knowns accepted AND correctly classified) vs FPR (unknowns accepted) over all thresholds;
    u = unknown score, higher = more unknown."""
    t = np.sort(np.unique(np.r_[u_known, u_unknown]))
    ccr = [0.0] + [float(np.mean((u_known <= x) & known_correct)) for x in t]
    fpr = [0.0] + [float(np.mean(u_unknown <= x)) for x in t]
    return float(np.trapezoid(ccr, fpr))


def recording_msp(logits: np.ndarray, rec_ids, classes, floor: float = 1e-6) -> pd.DataFrame:
    """Per recording from window logits: softmax, clip to [floor, 1], mean log-probability over the recording's windows,
    renormalise; returns the top class `pred` and its probability `msp` (lower = more unknown)."""
    z = logits.astype(np.float64)
    logp = z - logsumexp(z, axis=1, keepdims=True)
    lp = pd.DataFrame(np.log(np.clip(np.exp(logp), floor, 1.0)), columns=list(classes)).assign(rec_id=list(rec_ids))
    lp = lp.groupby("rec_id", sort=False).mean()
    p = np.exp(lp.to_numpy())
    p /= p.sum(axis=1, keepdims=True)
    return pd.DataFrame({"pred": np.asarray(classes)[p.argmax(axis=1)], "msp": p.max(axis=1)}, index=lp.index)


def mahalanobis_min(fit_emb: np.ndarray, fit_labels, emb: np.ndarray) -> tuple[np.ndarray, dict]:
    """Per window of `emb`: the minimum Mahalanobis distance over the known classes. Class centroids = mean fit-window
    embedding per class; ONE shared covariance = Ledoit–Wolf fitted to the fit-window embeddings after subtracting each
    window's own class centroid (within-class variation only; never the overall covariance)."""
    fit_labels = np.asarray(fit_labels)
    classes = sorted(set(fit_labels))
    mu = np.stack([fit_emb[fit_labels == c].mean(axis=0) for c in classes])
    resid = fit_emb - mu[np.searchsorted(classes, fit_labels)]
    lw = LedoitWolf().fit(resid)
    prec = lw.precision_
    d2 = np.stack([np.einsum("ij,jk,ik->i", emb - m, prec, emb - m) for m in mu], axis=1)
    return np.sqrt(np.maximum(d2.min(axis=1), 0.0)), {"classes": classes, "shrinkage": float(lw.shrinkage_)}


def knn_kth_cosine(fit_emb: np.ndarray, emb: np.ndarray, k: int = 10) -> np.ndarray:
    """Per window of `emb`: cosine distance (1 − cosine similarity) to its k-th nearest fit-window embedding."""
    a = fit_emb / np.linalg.norm(fit_emb, axis=1, keepdims=True)
    b = emb / np.linalg.norm(emb, axis=1, keepdims=True)
    d = 1.0 - b @ a.T
    return np.partition(d, k - 1, axis=1)[:, k - 1]


def recording_mean(values: np.ndarray, rec_ids) -> pd.Series:
    """Mean of window values per recording (first-seen order)."""
    return pd.Series(np.asarray(values, dtype=np.float64), index=list(rec_ids)).groupby(level=0, sort=False).mean()


def loco_run_ids(assignment: pd.DataFrame, pool: pd.DataFrame, withheld: str, fold: int) -> dict:
    """Recording IDs of one LOCO run (withheld class, outer fold) from the SAVED Q3 assignment (withheld, outer_fold,
    rec_id, part ∈ fit | cal | eval), plus the development-mode normalisation context: calibration = cal; evaluation =
    seen-folder evaluation ∪ fold k; withheld = every recording of the withheld class."""
    from ml.datasets import NormContext

    a = assignment[(assignment.withheld == withheld) & (assignment.outer_fold == fold)]
    part = a.set_index("rec_id").part
    cls = pool.set_index("rec_id").class10
    ids = {p: sorted(part[part == p].index) for p in ("fit", "cal", "eval")}
    ids["foldk_known"] = sorted(pool.loc[(pool.dev_fold == fold) & (pool.class10 != withheld), "rec_id"])
    ids["foldk_withheld"] = sorted(pool.loc[(pool.dev_fold == fold) & (pool.class10 == withheld), "rec_id"])
    all_c = set(pool.loc[pool.class10 == withheld, "rec_id"])
    side = set(pool.loc[(pool.dev_fold != fold) & (pool.class10 != withheld), "rec_id"])
    if set(ids["fit"]) | set(ids["cal"]) | set(ids["eval"]) != side or (cls.loc[ids["fit"]] == withheld).any():
        raise ValueError(f"{withheld}/{fold}: the saved assignment does not partition the known training side")
    ids["context"] = NormContext(calibration=frozenset(ids["cal"]),
                                 evaluation=frozenset(ids["eval"]) | frozenset(ids["foldk_known"]) | frozenset(ids["foldk_withheld"]),
                                 withheld=frozenset(all_c))
    return ids


def three_way(pred, msp, maha, ae, tau_p: float, tau_d: float, tau_e: float,
              ordering: str = "classifier_first") -> tuple[np.ndarray, np.ndarray]:
    """Three-way decision per recording: "healthy", a known fault class name, or "unknown". Confident = MSP ≥ τ_p AND
    Mahalanobis d ≤ τ_d (ties accepted); AE healthy = error ≤ τ_e (ties healthy). Never falls back to a second class.

    classifier_first (PRIMARY): confident → healthy (top class normal) or that known fault; otherwise unknown, EXCEPT
    top class normal and AE healthy → healthy (the AE override).
    ae_first (comparison): AE healthy → healthy; else top class a fault AND confident → that fault; else unknown (this
    includes a top class of normal while the AE says abnormal).
    Returns (decision, override) where override marks classifier-first cases changed from unknown to healthy by the AE."""
    pred, msp, maha, ae = map(np.asarray, (pred, msp, maha, ae))
    conf = (msp >= tau_p) & (maha <= tau_d)
    ae_ok = ae <= tau_e
    is_normal = pred == "normal"
    if ordering == "classifier_first":
        out = np.where(conf, np.where(is_normal, "healthy", pred), "unknown").astype(object)
        override = ~conf & is_normal & ae_ok
        out[override] = "healthy"
        return out, override
    if ordering == "ae_first":
        out = np.where(ae_ok, "healthy", np.where(~is_normal & conf, pred, "unknown")).astype(object)
        return out, np.zeros(len(out), dtype=bool)
    raise ValueError(ordering)


def outcome_category(decision, true_class, withheld: bool) -> np.ndarray:
    """Outcome labels: withheld recordings → healthy | known_fault | unknown; known faults → healthy | correct |
    wrong_fault | unknown; normal recordings → healthy | fault | unknown."""
    d, t = np.asarray(decision, dtype=object), np.asarray(true_class, dtype=object)
    is_fault_decision = (d != "healthy") & (d != "unknown")
    if withheld:
        return np.where(d == "healthy", "healthy", np.where(is_fault_decision, "known_fault", "unknown"))
    normal = t == "normal"
    known = np.where(d == "healthy", "healthy",
                     np.where(d == "unknown", "unknown", np.where(d == t, "correct", "wrong_fault")))
    healthy_rec = np.where(d == "healthy", "healthy", np.where(d == "unknown", "unknown", "fault"))
    return np.where(normal, healthy_rec, known)
