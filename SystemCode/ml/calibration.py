"""Calibration of the bundle thresholds on the deployment scoring path (Stage ML-4.2). The scores come from
ml.inference.score_recording — the same function diagnose() uses — and the formulas are the Stage ML-4 ones:
τ_p, τ_d = the 97.5 % order statistic over all calibration recordings (MSP on its negation; ties accepted);
τ_e, τ_b = the maximum over the calibration normals. Nothing is optimised against fault outcomes."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ml.protocol import order_statistic_threshold

ACCEPT = 0.975


def score_calibration_set(bundle, rec_ids) -> pd.DataFrame:
    """Scores of the calibration recordings from ml.inference.score_recording (CPU), full precision."""
    from ml.inference import load_recording, score_recording

    rows = []
    for r in rec_ids:
        x, fr = load_recording(r)
        s = score_recording(x, fr, bundle)
        rows.append({"rec_id": r, "top_class": s["top"], "msp": s["msp"], "maha": s["maha"], "ae": s["ae"], "b1s": s["b1s"]})
    return pd.DataFrame(rows).set_index("rec_id")


def thresholds_from_scores(scores: pd.DataFrame, is_normal: pd.Series, acceptance: float = ACCEPT) -> dict:
    """The four thresholds, their defining recordings and the order-statistic details."""
    tp = order_statistic_threshold(-scores.msp.to_numpy(), acceptance)
    td = order_statistic_threshold(scores.maha.to_numpy(), acceptance)
    n = scores[is_normal.loc[scores.index].to_numpy()]
    out = {"tau_p": float(-tp["threshold"]), "tau_d": float(td["threshold"]), "tau_e": float(n.ae.max()),
           "tau_b": float(n.b1s.max()), "acceptance_target": acceptance, "n_calibration": int(tp["n"]),
           "order_statistic_r": int(tp["r"]), "n_calibration_normals_tau_e": int(len(n)),
           "n_calibration_normals_tau_b": int(len(n))}
    out["defining_recordings"] = {"tau_p": list(scores.index[scores.msp == out["tau_p"]]),
                                  "tau_d": list(scores.index[scores.maha == out["tau_d"]]),
                                  "tau_e": list(n.index[n.ae == out["tau_e"]]), "tau_b": list(n.index[n.b1s == out["tau_b"]])}
    return out


def acceptance(scores: pd.DataFrame, t: dict) -> dict:
    """Marginal acceptance of the MSP and Mahalanobis checks and their measured joint acceptance (ties accepted)."""
    p, d = scores.msp >= t["tau_p"], scores.maha <= t["tau_d"]
    j = p & d
    return {"n": int(len(scores)), "msp": {"count": int(p.sum()), "rate": float(p.mean())},
            "mahalanobis": {"count": int(d.sum()), "rate": float(d.mean())},
            "joint": {"count": int(j.sum()), "rate": float(j.mean())}}
