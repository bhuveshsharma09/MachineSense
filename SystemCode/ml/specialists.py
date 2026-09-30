"""Stage R4: a bank of condition-specialist autoencoders for unfamiliar-fault detection (CLAUDE.md §7). Pure functions:
the per-run plan (who trains on what, and on what each is calibrated), score calibration per specialist, the combining
rule and the verdicts. No file I/O; training and scoring live in scripts/34_train_specialists.py.

Scores (declared): s_u(x) = log e_u(x) − log median{e_u(c) : c a calibration recording of class u}, leave-one-out for class
u's own calibration recordings. S(x) = min_u s_u(x); τ_S = the 97.5 % order statistic of S over all calibration recordings;
specialist u accepts x iff s_u(x) ≤ τ_S (ties within 1e-12 accepted); the accepting set is kept, never collapsed."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ml.openset import loco_run_ids
from ml.protocol import order_statistic_threshold
from ml.rejection import TIE_EPS

NORMAL = "normal"
ACCEPTANCE = 0.975
EVAL_ROLES = ("eval", "foldk_known", "foldk_withheld")


def run_plan(assignment: pd.DataFrame, pool: pd.DataFrame, withheld: str, fold: int) -> dict:
    """The run's specialists (normal + every fault class except `withheld`), each with its fit recordings (the run's fit
    recordings of that class; they are also its normalisation set) and its calibration recordings (the run's calibration
    recordings of that class). Fails loudly if the withheld class would reach any specialist's fit or calibration set."""
    ids = loco_run_ids(assignment, pool, withheld, fold)
    cls = pool.set_index("rec_id").class10
    known = sorted(set(pool.class10) - {withheld})
    spec = {u: {"fit": sorted(r for r in ids["fit"] if cls[r] == u), "cal": sorted(r for r in ids["cal"] if cls[r] == u)}
            for u in known}
    wh = set(pool.loc[pool.class10 == withheld, "rec_id"])
    if withheld in spec or any(wh & set(s["fit"] + s["cal"]) for s in spec.values()):
        raise ValueError(f"withheld class {withheld} reaches a specialist")
    if any(not s["fit"] or not s["cal"] for s in spec.values()):
        raise ValueError("a specialist has no fit or no calibration recordings")
    return {"withheld": withheld, "fold": fold, "specialists": spec, "ids": ids}


def log_ratio_scores(errors: pd.DataFrame, roles: pd.Series, classes: pd.Series) -> pd.DataFrame:
    """Comparable per-specialist scores. `errors`: one row per recording of ONE run, one column per specialist (recording
    reconstruction error, > 0); `roles`, `classes` aligned with its rows. Each column is divided by the median of that
    specialist's errors on its OWN class's calibration recordings (leave-one-out for those recordings), then logged."""
    if ((roles == "cal") & ~classes.isin(errors.columns)).any():
        raise ValueError("a calibration recording belongs to a class without a specialist (the withheld class?)")
    out = pd.DataFrame(index=errors.index, columns=errors.columns, dtype=float)
    for u in errors.columns:
        e = errors[u].to_numpy(dtype=float)
        own = np.flatnonzero(((roles == "cal") & (classes == u)).to_numpy())
        if len(own) < 2:
            raise ValueError(f"specialist {u}: fewer than 2 calibration recordings (leave-one-out impossible)")
        med = np.full(len(e), np.median(e[own]))
        for i in own:
            med[i] = np.median(e[own[own != i]])
        out[u] = np.log(e) - np.log(med)
    return out


def accepts(s, tau: float, eps: float = TIE_EPS) -> np.ndarray:
    """Specialist acceptance: s ≤ τ, a score within `eps` of τ is a tie (accepted)."""
    s = np.asarray(s, dtype=float)
    return (s <= tau) | (np.abs(s - tau) <= eps)


def calibrate(scores: pd.DataFrame, roles: pd.Series) -> dict:
    """τ_S from the calibration recordings only: the 97.5 % order statistic of S = min over specialists."""
    S = scores.min(axis=1)
    return order_statistic_threshold(S[roles == "cal"].to_numpy(dtype=float), ACCEPTANCE)


def accepting_sets(scores: pd.DataFrame, tau: float) -> pd.Series:
    """The set of accepting specialists per recording, as a sorted "+"-joined string ("" = none accepts)."""
    acc = pd.DataFrame(accepts(scores.to_numpy(dtype=float), tau), index=scores.index, columns=scores.columns)
    return acc.apply(lambda r: "+".join(c for c in sorted(scores.columns) if r[c]), axis=1)


def verdict(accepting: str) -> str:
    """unfamiliar (none accepts) / healthy (normal only) / known_fault (one fault) / ambiguous_fault (several faults) /
    ambiguous_healthy_fault (normal and at least one fault)."""
    a = [x for x in accepting.split("+") if x]
    if not a:
        return "unfamiliar"
    if a == [NORMAL]:
        return "healthy"
    if NORMAL in a:
        return "ambiguous_healthy_fault"
    return "known_fault" if len(a) == 1 else "ambiguous_fault"


def empirical_rank_scores(errors: pd.DataFrame, roles: pd.Series, classes: pd.Series) -> pd.DataFrame:
    """Secondary diagnostic: r_u(x) = #{own-class calibration errors ≤ x} / (n + 1) (ml.openset.empirical_rank convention,
    leave-one-out for the specialist's own calibration recordings); capped at n / (n + 1)."""
    out = pd.DataFrame(index=errors.index, columns=errors.columns, dtype=float)
    for u in errors.columns:
        e = errors[u].to_numpy(dtype=float)
        own = np.flatnonzero(((roles == "cal") & (classes == u)).to_numpy())
        ref = np.sort(e[own])
        r = np.searchsorted(ref, e, side="right") / (len(ref) + 1)
        for i in own:
            rr = np.sort(e[own[own != i]])
            r[i] = np.searchsorted(rr, e[i], side="right") / (len(rr) + 1)
        out[u] = r
    return out
