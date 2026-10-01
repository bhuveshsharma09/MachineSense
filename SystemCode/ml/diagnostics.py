"""Stage R7: diagnostics on saved models (CLAUDE.md §7; no training). Pure functions: recording embeddings and logit-based
unknown scores aggregated over the 9 windows, the linear probe from embeddings to physics descriptors, and the gate with a
logit term plus the unchanged Mahalanobis term, including its threshold at a matched known-side rejection count."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.special import logsumexp
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold

from ml.rejection import TIE_EPS
from ml.speed_calibration import matched_threshold

PROBE_ALPHAS = np.logspace(-3, 4, 15)
DESCRIPTORS = ["f_under_radial_ratio_2x_1x", "f_over_radial_ratio_2x_1x", "f_under_ratio_axial_radial_1x",
               "f_over_ratio_axial_radial_1x", "f_under_axial_o1", "f_under_axial_o2", "f_under_radial_o1",
               "f_under_radial_o2", "f_under_tangential_o1", "f_under_tangential_o2", "f_over_axial_o1", "f_over_axial_o2",
               "f_over_radial_o1", "f_over_radial_o2", "f_over_tangential_o1", "f_over_tangential_o2"]


def recording_embeddings(emb: np.ndarray, rec_ids) -> pd.DataFrame:
    """Recording embedding = mean of its window embeddings (rows: recordings in first-seen order)."""
    return pd.DataFrame(np.asarray(emb, dtype=np.float64)).groupby(np.asarray(rec_ids), sort=False).mean()


def logit_scores(logits: np.ndarray, rec_ids) -> pd.DataFrame:
    """Per recording: u_maxlogit = −(mean over windows of the window's maximum logit) and u_energy = mean over windows of
    −logsumexp(logits). Both: higher = more unknown."""
    z = np.asarray(logits, dtype=np.float64)
    w = pd.DataFrame({"maxlogit": z.max(axis=1), "energy": -logsumexp(z, axis=1)})
    r = w.groupby(np.asarray(rec_ids), sort=False).mean()
    return pd.DataFrame({"u_maxlogit": -r.maxlogit, "u_energy": r.energy})


def fit_probe(z_fit: np.ndarray, y_fit: np.ndarray, seed: int) -> dict:
    """Ridge probe z → descriptors on the FIT recordings only: z and y standardised with the fit mean / SD, one α for all
    outputs chosen by 5-fold CV (shuffled with `seed`) on those recordings. Returns the fitted parameters."""
    z, y = np.asarray(z_fit, float), np.asarray(y_fit, float)
    zm, zs = z.mean(0), z.std(0, ddof=0)
    zs = np.where(zs > 0, zs, 1.0)
    ym, ys = y.mean(0), y.std(0, ddof=0)
    ys = np.where(ys > 0, ys, 1.0)
    model = RidgeCV(alphas=PROBE_ALPHAS, cv=KFold(5, shuffle=True, random_state=seed)).fit((z - zm) / zs, (y - ym) / ys)
    return {"model": model, "zm": zm, "zs": zs, "ym": ym, "ys": ys, "alpha": float(model.alpha_)}


def probe_scores(p: dict, z: np.ndarray, y: np.ndarray) -> dict:
    """R² per descriptor (against the evaluated set's own mean) and RMSE in fit-standardised units."""
    pred = p["model"].predict((np.asarray(z, float) - p["zm"]) / p["zs"])
    yt = (np.asarray(y, float) - p["ym"]) / p["ys"]
    r2 = np.array([r2_score(yt[:, j], pred[:, j]) for j in range(yt.shape[1])])
    rmse = np.sqrt(((pred - yt) ** 2).mean(axis=0))
    return {"r2": r2, "rmse": rmse, "r2_mean": float(r2.mean()), "rmse_mean": float(rmse.mean())}


def gate_rejects(u, tau_u: float, maha, tau_d: float, eps: float = TIE_EPS) -> np.ndarray:
    """True where the gate rejects: the logit term u > τ_u or Mahalanobis > τ_d (ties within eps accepted)."""
    u, m = np.asarray(u, float), np.asarray(maha, float)
    over = lambda a, t: (a > t) & ~(np.abs(a - t) <= eps)
    return over(u, tau_u) | over(m, tau_d)


def matched_u_threshold(u_known, maha_known, tau_d: float, n_reject: int) -> float | None:
    """The logit-term threshold at which the gate (τ_d fixed) rejects exactly `n_reject` known recordings; None when the
    Mahalanobis term alone already rejects more."""
    u, m = np.asarray(u_known, float), np.asarray(maha_known, float)
    by_maha = gate_rejects(np.full(len(m), -np.inf), np.inf, m, tau_d)
    need = n_reject - int(by_maha.sum())
    if need < 0:
        return None
    return matched_threshold(u[~by_maha], need)
