"""Stage R8 (CLAUDE.md §7): Relative Mahalanobis vs the shared Mahalanobis (and kNN, reference) on the 81 saved LOCO runs.
Pure evaluation helpers plus loaders of saved scores and embeddings; no training, no forward pass, no reserved data. The
notebook 19_relative_mahalanobis calls these and only displays the results."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ml.diagnostics import gate_rejects, matched_u_threshold
from ml.loco_eval import ACCEPT, SEEDS
from ml.openset import (knn_kth_cosine, mahalanobis_min, recording_mean, relative_mahalanobis_min, weighted_auroc,
                        weighted_oscr)
from ml.paths import RunPaths
from ml.protocol import order_statistic_threshold
from ml.splits import FAULT_CLASSES

KNN_K = 10
ARMS = ("M", "K", "R")
ARM_NAMES = {"M": "shared-covariance Mahalanobis (current)", "K": "kNN, cosine, k = 10 (reference)",
             "R": "Relative Mahalanobis (new)"}
MAHA_REL_TOL = 1e-4   # R7's reproduction tolerance |Δ maha| ≤ 1e-4 × maha
BAR_POINTS = 0.05     # +5 percentage points of unknown recall (R4 / R5 bar)


def runs() -> list[tuple[int, str, int]]:
    """The 81 saved LOCO runs: (seed, withheld class, outer fold)."""
    return [(s, c, k) for s in SEEDS for c in FAULT_CLASSES for k in (0, 1, 2)]


def run_scores(seed: int, withheld: str, fold: int) -> tuple[pd.DataFrame, dict]:
    """The run's saved recording table (role, class10, top class, MSP, saved maha / knn) with the three arms recomputed
    from the saved window embeddings exactly as Stage 3C scored them: fit windows → the class / background fits, every
    other window scored, recording = mean of its windows. Returns (table, fitted parameters)."""
    rp = RunPaths("loco", "cnn6", seed, protocol="loco", withheld=withheld)
    z = np.load(rp.embeddings(fold))
    win = pd.read_parquet(rp.scores(f"fold{fold}_logits"), columns=["rec_id", "window_idx", "class10", "role"])
    if not (np.array_equal(win.rec_id.to_numpy().astype(str), z["rec_id"]) and np.array_equal(win.role.to_numpy().astype(str), z["role"])):
        raise ValueError(f"{seed}/{withheld}/{fold}: embeddings and window table are not aligned")
    emb, fit = z["embedding"], (z["role"] == "fit")
    labels = win.class10.to_numpy()[fit]
    ev_ids = z["rec_id"][~fit]
    m_w, m_info = mahalanobis_min(emb[fit], labels, emb[~fit])
    k_w = knn_kth_cosine(emb[fit], emb[~fit], KNN_K)
    r_w, r_info = relative_mahalanobis_min(emb[fit], labels, emb[~fit])
    rec = pd.read_parquet(rp.scores(f"fold{fold}_recording"))
    for arm, w in (("M", m_w), ("K", k_w), ("R", r_w)):
        rec[arm] = recording_mean(w, ev_ids).loc[rec.rec_id].to_numpy()
    info = {"n_fit_windows": int(fit.sum()), "n_scored_windows": int((~fit).sum()), "shrinkage_class": r_info["shrinkage_class"],
            "shrinkage_background": r_info["shrinkage_background"], "shrinkage_m": m_info["shrinkage"]}
    return rec, info


def reproduction(rec: pd.DataFrame) -> dict:
    """Arm M against the saved Stage 3C recording maha (relative), arm K against the saved knn (absolute)."""
    rel = np.abs(rec.M - rec.maha) / rec.maha
    return {"max_rel_diff_maha": float(rel.max()), "max_abs_diff_knn": float(np.abs(rec.K - rec.knn).max()),
            "ok": bool((rel <= MAHA_REL_TOL).all() and np.allclose(rec.K, rec.knn, rtol=0, atol=1e-6))}


def thresholds(rec: pd.DataFrame, arms=ARMS) -> dict:
    """τ_p and τ_d as Stage 3C / ml.loco_eval.decide set them (0.975 order statistic of the calibration recordings; MSP
    on its negation), and the calibrated τ_s of every arm the same way (the secondary result)."""
    cal = rec[rec.role == "cal"]
    out = {"tau_p": -order_statistic_threshold(-cal.msp.to_numpy(), ACCEPT)["threshold"]}
    for arm in arms:
        out[f"tau_{arm}"] = order_statistic_threshold(cal[arm].to_numpy(), ACCEPT)["threshold"]
    return out


def gate_ok(msp, score, tau_p: float, tau_s: float) -> np.ndarray:
    """The confidence gate: MSP ≥ τ_p AND score ≤ τ_s (ties within 1e-12 accepted, R0's rule)."""
    return ~gate_rejects(np.asarray(score, float), tau_s, -np.asarray(msp, float), -tau_p)


def rule_b(pred, conf) -> np.ndarray:
    """Rule B (the decisions of B_with_caution): confident → healthy (top class normal) or that fault; else unknown."""
    pred = np.asarray(pred, dtype=object)
    return np.where(conf, np.where(pred == "normal", "healthy", pred), "unknown").astype(object)


def matched_tau(rec: pd.DataFrame, arm: str, tau_p: float, n_reject: int) -> float | None:
    """τ_s at which the gate (τ_p fixed) rejects exactly `n_reject` foldk_known recordings (None if MSP alone rejects more)."""
    kn = rec[rec.role == "foldk_known"]
    return matched_u_threshold(kn[arm].to_numpy(), -kn.msp.to_numpy(), -tau_p, n_reject)


def outcomes(rec: pd.DataFrame, decision: np.ndarray) -> dict:
    """Counts on the run's withheld (foldk_withheld) and known (foldk_known) recordings."""
    d = pd.Series(decision, index=rec.index)
    wh, kn = rec.role == "foldk_withheld", rec.role == "foldk_known"
    fault = kn & (rec.class10 != "normal")
    is_fault_decision = ~d.isin(["healthy", "unknown"])
    return {"withheld_n": int(wh.sum()), "withheld_unknown": int((wh & (d == "unknown")).sum()),
            "withheld_healthy": int((wh & (d == "healthy")).sum()), "withheld_known_fault": int((wh & is_fault_decision).sum()),
            "known_n": int(kn.sum()), "known_unknown": int((kn & (d == "unknown")).sum()),
            "known_wrong_fault": int((fault & is_fault_decision & (d != rec.class10)).sum()),
            "known_correct_fault": int((fault & (d == rec.class10)).sum())}


def threshold_free(rec: pd.DataFrame, arm: str) -> dict:
    """AUROC and OSCR: foldk_withheld positives vs foldk_known negatives, weight 1, OSCR correctness = top class is true."""
    kn, un = (rec.role == "foldk_known").to_numpy(), (rec.role == "foldk_withheld").to_numpy()
    u = rec[arm].to_numpy(float)
    correct = (rec.pred == rec.class10).to_numpy()
    return {"auroc": weighted_auroc(u[kn], np.ones(kn.sum()), u[un], np.ones(un.sum())),
            "oscr": weighted_oscr(u[kn], correct[kn], np.ones(kn.sum()), u[un], np.ones(un.sum()))}


def evaluate_run(rec: pd.DataFrame, arms=ARMS) -> list[dict]:
    """Per arm: threshold-free metrics, the matched-count rule outcomes and the calibrated-threshold rule outcomes. `arms`
    must include "M" (the reference that sets the matched count); R10 passes ("M", "F")."""
    t = thresholds(rec, arms)
    m_dec = rule_b(rec.pred, gate_ok(rec.msp, rec.M, t["tau_p"], t["tau_M"]))
    n_reject = outcomes(rec, m_dec)["known_unknown"]
    rows = []
    for arm in arms:
        tau_m = t["tau_M"] if arm == "M" else matched_tau(rec, arm, t["tau_p"], n_reject)
        row = {"arm": arm, **threshold_free(rec, arm), "tau_p": t["tau_p"], "tau_matched": tau_m,
               "tau_calibrated": t[f"tau_{arm}"], "matched_target_known_unknown": n_reject}
        if tau_m is not None:
            row |= {f"matched_{k}": v for k, v in outcomes(rec, rule_b(rec.pred, gate_ok(rec.msp, rec[arm], t["tau_p"], tau_m))).items()}
        row |= {f"calibrated_{k}": v for k, v in
                outcomes(rec, rule_b(rec.pred, gate_ok(rec.msp, rec[arm], t["tau_p"], t[f"tau_{arm}"]))).items()}
        rows.append(row)
    return rows


def pooled(per_run: pd.DataFrame, mode: str) -> pd.DataFrame:
    """Per seed and arm, pooled over the 27 runs: unknown recall on withheld recordings, withheld called healthy, known
    called wrong fault, known-side rejections."""
    g = per_run.groupby(["seed", "arm"])
    out = pd.DataFrame({"unknown_recall": g[f"{mode}_withheld_unknown"].sum() / g[f"{mode}_withheld_n"].sum(),
                        "withheld_unknown": g[f"{mode}_withheld_unknown"].sum(), "withheld_n": g[f"{mode}_withheld_n"].sum(),
                        "withheld_healthy": g[f"{mode}_withheld_healthy"].sum(),
                        "known_wrong_fault": g[f"{mode}_known_wrong_fault"].sum(),
                        "known_rejected": g[f"{mode}_known_unknown"].sum(), "known_n": g[f"{mode}_known_n"].sum()})
    return out.reset_index()


def reading(matched: pd.DataFrame, arm: str = "R") -> dict:
    """The declared reading of `arm` (R8: R; R10: F) against M at the matched count: "helps, confirmed" if on every seed recall gain ≥ +5
    points and withheld-called-healthy does not rise; else "worse" if the mean recall change is negative; else "does not
    help"."""
    m = matched.set_index(["seed", "arm"])
    per_seed = {}
    for s in sorted(matched.seed.unique()):
        gain = float(m.loc[(s, arm), "unknown_recall"] - m.loc[(s, "M"), "unknown_recall"])
        dh = int(m.loc[(s, arm), "withheld_healthy"] - m.loc[(s, "M"), "withheld_healthy"])
        per_seed[int(s)] = {"recall_gain": gain, "healthy_change": dh, "bar_met": gain >= BAR_POINTS - 1e-12 and dh <= 0}
    mean_gain = float(np.mean([v["recall_gain"] for v in per_seed.values()]))
    verdict = ("helps, confirmed" if all(v["bar_met"] for v in per_seed.values())
               else "worse" if mean_gain < 0 else "does not help")
    return {"reading": verdict, "mean_recall_gain": mean_gain, "per_seed": per_seed}
