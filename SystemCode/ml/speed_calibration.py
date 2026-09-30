"""Stage R1: speed-conditioned healthy scores (CLAUDE.md §7). s_std = (s − m(x)) / d(x), x = log(speed Hz), with m a
least-squares polynomial (degree ≤ 2) and d = exp(a + b x) a Laplace scale fitted by maximum likelihood on the residuals,
both fitted on a run's healthy FIT normals only; x is clipped to the fitted speed range (constant extrapolation). The
threshold on s_std is the existing maximum rule over the calibration normals, ties within 1e-12 accepted (R0).
Pure functions; no file I/O."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from ml.protocol import max_threshold
from ml.rejection import TIE_EPS


@dataclass(frozen=True)
class SpeedModel:
    """m(x) = polyval(m_coef, x); d(x) = exp(d_coef[0] + d_coef[1] x); x = log(speed Hz) clipped to [x_lo, x_hi]."""
    m_coef: tuple[float, ...]
    d_coef: tuple[float, float]
    x_lo: float
    x_hi: float
    n_fit: int

    def _x(self, speed_hz) -> np.ndarray:
        return np.clip(np.log(np.asarray(speed_hz, dtype=float)), self.x_lo, self.x_hi)

    def m(self, speed_hz) -> np.ndarray:
        return np.polyval(self.m_coef, self._x(speed_hz))

    def d(self, speed_hz) -> np.ndarray:
        return np.exp(self.d_coef[0] + self.d_coef[1] * self._x(speed_hz))

    def transform(self, score, speed_hz) -> np.ndarray:
        """The standardised score (dimensionless; higher = further above the healthy level expected at that speed)."""
        return (np.asarray(score, dtype=float) - self.m(speed_hz)) / self.d(speed_hz)


IDENTITY = SpeedModel(m_coef=(0.0,), d_coef=(0.0, 0.0), x_lo=-np.inf, x_hi=np.inf, n_fit=0)


def fit_scale(resid: np.ndarray, x: np.ndarray) -> tuple[float, float]:
    """Laplace-scale ML with a log link, d = exp(a + b x): minimise Σ [log d + |r| / d] (convex in (a, b))."""
    r = np.abs(np.asarray(resid, dtype=float))
    x = np.asarray(x, dtype=float)
    a0 = float(np.log(max(r.mean(), np.finfo(float).tiny)))

    def f(p):
        eta = p[0] + p[1] * x
        return float(np.sum(eta + r * np.exp(-eta)))

    def g(p):
        w = 1.0 - r * np.exp(-(p[0] + p[1] * x))
        return np.array([w.sum(), (w * x).sum()])

    res = minimize(f, np.array([a0, 0.0]), jac=g, method="BFGS", options={"gtol": 1e-10, "maxiter": 1000})
    return float(res.x[0]), float(res.x[1])


def fit_speed_model(score: np.ndarray, speed_hz: np.ndarray, m_degree: int = 2) -> SpeedModel:
    """Fit m (degree `m_degree`, ≤ 2) and d (degree 1, log link) on healthy FIT-normal scores and speeds only."""
    if m_degree not in (0, 1, 2):
        raise ValueError("m_degree is capped at 2 (CLAUDE.md §7, Stage R1)")
    s = np.asarray(score, dtype=float)
    x = np.log(np.asarray(speed_hz, dtype=float))
    if len(s) < m_degree + 3:
        raise ValueError(f"{len(s)} fit normals are too few for degree {m_degree} plus a degree-1 scale")
    m_coef = np.polyfit(x, s, m_degree)
    a, b = fit_scale(s - np.polyval(m_coef, x), x)
    return SpeedModel(tuple(float(c) for c in m_coef), (a, b), float(x.min()), float(x.max()), len(s))


def std_threshold(cal_std: np.ndarray) -> float:
    """τ_std = the maximum standardised score over the calibration normals (the existing rule)."""
    return max_threshold(np.asarray(cal_std, dtype=float))


def exceeds(score, threshold, eps: float = TIE_EPS) -> np.ndarray:
    """True where a score is above its threshold (a failed healthy check); within `eps` is a tie, accepted."""
    s, t = np.asarray(score, dtype=float), np.asarray(threshold, dtype=float)
    return (s > t) & ~(np.abs(s - t) <= eps)


def matched_threshold(healthy_scores: np.ndarray, n_flagged: int) -> float:
    """A threshold that flags exactly `n_flagged` of the given healthy scores (the n-th largest is the last accepted);
    used only to compare detection at a matched false-alarm count, never deployed."""
    h = np.sort(np.asarray(healthy_scores, dtype=float))[::-1]
    if n_flagged >= len(h):
        return -np.inf
    return float(h[n_flagged])


def fit_for_run(run: pd.DataFrame, score_col: str, m_degree: int = 2) -> SpeedModel:
    """The run's speed model from its rows with role "fit" and class normal ONLY (columns: role, class10, speed_hz and
    `score_col`); calibration and evaluation rows are never passed to the fit."""
    fit = run[(run.role == "fit") & (run.class10 == "normal")]
    return fit_speed_model(fit[score_col].to_numpy(), fit.speed_hz.to_numpy(), m_degree)


def standardise_run(run: pd.DataFrame, score_col: str, m_degree: int = 2) -> tuple[pd.Series, float, SpeedModel]:
    """s_std for every row of one run, and τ_std from the run's calibration normals (existing maximum rule)."""
    model = fit_for_run(run, score_col, m_degree)
    s = pd.Series(model.transform(run[score_col].to_numpy(), run.speed_hz.to_numpy()), index=run.index)
    cal = (run.role == "cal") & (run.class10 == "normal")
    return s, std_threshold(s[cal].to_numpy()), model
