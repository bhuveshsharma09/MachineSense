"""Healthy-only detectors (Track B classical rungs). Pure functions; scores are oriented HIGHER = more anomalous."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

from ml.config import AnomalyConfig

RUNGS = ["B0_rules", "B1_mahalanobis", "B1_gmm", "B1s_speed_poly", "B1s_knn_speed", "B2_pca", "B3_ocsvm",
         "B3_iforest", "B3_lof"]
RECORDING_LEVEL = {"B1s_knn_speed"}


class Prep:
    """Fitted on healthy training windows only: log10 of columns positive there, then standard scaling."""

    def fit(self, X):
        self.log = (X > 0).all(axis=0)
        self.sc = StandardScaler().fit(self._t(X))
        return self

    def _t(self, X):
        Y = X.copy()
        Y[:, self.log] = np.log10(np.maximum(Y[:, self.log], 1e-12))
        return Y

    def __call__(self, X):
        return self.sc.transform(self._t(X))


class SpeedPoly:
    """B1s, the speed-aware polynomial healthy baseline, as a fitted object (so it can be saved in the bundle): Prep on the
    healthy training windows, a per-feature polynomial in log(speed) fitted by least squares, and the residual standard
    deviation. Window score = mean over features of |residual| / residual sd (higher = more anomalous)."""

    def __init__(self, degree: int):
        self.degree = degree

    def fit(self, Xtr: np.ndarray, speed_tr: np.ndarray) -> "SpeedPoly":
        self.prep = Prep().fit(Xtr)
        A = self.prep(Xtr)
        V = np.vander(np.log(speed_tr), self.degree + 1)
        self.coef, *_ = np.linalg.lstsq(V, A, rcond=None)
        self.resid_sd = (A - V @ self.coef).std(axis=0, ddof=self.degree + 1)
        return self

    def score(self, X: np.ndarray, speed: np.ndarray) -> np.ndarray:
        B = self.prep(X)
        pred = np.vander(np.log(speed), self.degree + 1) @ self.coef
        return (np.abs(B - pred) / np.maximum(self.resid_sd, 1e-9)).mean(axis=1)


def window_scores(rung: str, ac: AnomalyConfig, Xtr, str_, Xte, ste, cols):
    """Anomaly score per test window (higher = more anomalous) for window-level rungs."""
    prep = Prep().fit(Xtr)
    A, B = prep(Xtr), prep(Xte)
    if rung == "B0_rules":
        idx = [cols.index(c) for c in ac.b0_features if c in cols]
        med = np.median(Xtr[:, idx], axis=0)
        q = np.quantile(Xtr[:, idx], ac.b0_quantile, axis=0)
        return ((Xte[:, idx] - med) / np.maximum(q - med, 1e-12)).max(axis=1)
    if rung == "B1_mahalanobis":
        lw = LedoitWolf().fit(A)
        return lw.mahalanobis(B)
    if rung == "B1_gmm":
        g = GaussianMixture(ac.gmm_components, covariance_type=ac.gmm_covariance, random_state=ac.seed).fit(A)
        return -g.score_samples(B)
    if rung == "B1s_speed_poly":
        return SpeedPoly(ac.b1s_degree).fit(Xtr, str_).score(Xte, ste)
    if rung == "B2_pca":
        p = PCA(n_components=ac.pca_variance, svd_solver="full").fit(A)
        return ((B - p.inverse_transform(p.transform(B))) ** 2).mean(axis=1)
    if rung == "B3_ocsvm":
        return -OneClassSVM(nu=ac.ocsvm_nu, gamma="scale").fit(A).decision_function(B)
    if rung == "B3_iforest":
        return -IsolationForest(n_estimators=ac.iforest_trees, random_state=ac.seed).fit(A).score_samples(B)
    if rung == "B3_lof":
        return -LocalOutlierFactor(n_neighbors=ac.lof_neighbors, novelty=True).fit(A).score_samples(B)
    raise ValueError(rung)


def knn_speed_scores(ac, tr: pd.DataFrame, te: pd.DataFrame, cols):
    """Recording level: distance (standardised recording-mean features) to the k healthy training recordings nearest in
    measured speed; also returns each scored recording's median speed gap to those neighbours."""
    prep = Prep().fit(tr[cols].to_numpy())
    rtr = tr.groupby("rec_id").agg({**{c: "mean" for c in cols}, "fr_measured_hz": "first"})
    rte = te.groupby("rec_id").agg({**{c: "mean" for c in cols}, "fr_measured_hz": "first"})
    A, B = prep(rtr[cols].to_numpy()), prep(rte[cols].to_numpy())
    s_tr, s_te = rtr.fr_measured_hz.to_numpy(), rte.fr_measured_hz.to_numpy()
    scores, gaps = [], []
    for i in range(len(rte)):
        nn = np.argsort(np.abs(s_tr - s_te[i]))[:ac.b1s_knn]
        scores.append(np.linalg.norm(A[nn] - B[i], axis=1).mean())
        gaps.append(np.median(np.abs(s_tr[nn] - s_te[i])))
    return pd.Series(scores, index=rte.index), pd.Series(gaps, index=rte.index)


