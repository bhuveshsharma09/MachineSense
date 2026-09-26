"""Evaluation-protocol principles and helpers for the corrected Q1/Q3 classical evaluation (Stage 2D.1).

Principles (written 27 Sep 2026, before running; also CLAUDE.md §5 rule 13):
1. Consistent scoring. Within each experiment's outer evaluation, every evaluated recording is scored exactly once, by
   the model of its own outer dev fold (fitted without that fold). No averaging over fold models. Q3's seen-folder
   control may reuse a recording across different outer-fold runs; those counts are reported.
2. Calibration is separate from evaluation. A threshold is chosen on calibration recordings only, and every rate
   reported at that threshold comes from recordings not used to choose it or to fit the model.
3. Primary summaries are PER-FOLD threshold-free metrics (AUROC; OSCR for Q3), reported as mean and spread across folds.
   Raw scores from different fitted models have different scales, so any metric on pooled raw scores is supplementary.
4. The healthy sample is small (25 recordings, one folder). Every Q1 result states this.

Score directions are fixed before running and never changed after seeing results: every "anomaly"/"unknown" score used
with these helpers is oriented so that HIGHER = more anomalous / more unknown.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd


def speed_stratified_split(recs: pd.DataFrame, pattern: dict[str, int], seed: int, speed_col: str = "fr_measured_hz",
                           id_col: str = "rec_id") -> pd.Series:
    """Assign recordings to named parts by dealing them, sorted by speed, in blocks of sum(pattern) (e.g. {"fit": 2,
    "cal": 1}); within each block the labels are randomly permuted (fixed seed). A final partial block takes a random
    subset of the block's labels. Returns a Series part name per rec_id."""
    labels = np.array([name for name, n in pattern.items() for _ in range(n)], dtype=object)
    rng = np.random.default_rng(seed)
    order = recs.sort_values([speed_col, id_col])[id_col].to_numpy()
    out = []
    for start in range(0, len(order), len(labels)):
        block = order[start: start + len(labels)]
        out += list(rng.permutation(labels)[: len(block)])
    return pd.Series(out, index=order, name="part")


def max_threshold(cal_scores: np.ndarray) -> float:
    """Q1 declared exploratory threshold: the maximum calibration score. Alarm iff score > threshold (ties: no alarm).
    Under exchangeability and continuous scores, P(a new healthy score ≤ threshold) = n / (n + 1): a theoretical
    reference only, not a guarantee for speed-stratified data."""
    return float(np.max(cal_scores))


def order_statistic_threshold(cal_scores: np.ndarray, acceptance: float) -> dict:
    """Q3 threshold on an 'unknown' score (higher = more unknown) targeting `acceptance` of calibration recordings.
    Rule: r = ceil(acceptance × (n + 1)); if r ≤ n, threshold = the r-th smallest calibration score, else +inf (accept
    all). Accept as known iff score ≤ threshold (ties accepted). Returns the threshold, n, r, the nominal attainable level
    r / (n + 1) (or 1 when r > n) and the acceptance measured on the calibration scores themselves."""
    s = np.sort(np.asarray(cal_scores, dtype=float))
    n = s.size
    r = math.ceil(acceptance * (n + 1))
    thr = float(s[r - 1]) if r <= n else float("inf")
    return {"threshold": thr, "n": n, "r": r, "attainable_level": (r / (n + 1)) if r <= n else 1.0,
            "calibration_acceptance": float(np.mean(s <= thr))}
