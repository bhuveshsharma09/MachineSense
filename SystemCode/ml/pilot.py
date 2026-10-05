"""Stage D7 pilot (CLAUDE.md §7): pure helpers for case selection and for the analysis of the blind scores. No backend
import (the notebook imports this); side effects live in scripts/d7_*.py and the notebook."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

SEED = 42
# (stratum, number of cases, predicate on a pool_verdicts row), in selection order (CLAUDE.md §7 D7 item 1)
STRATA = (
    ("S1_unknown", 4, lambda r: r.decision == "unknown"),
    ("S2_known_fault_caution_or_disagreeing", 4,
     lambda r: r.decision == "known_fault" and (bool(r.caution) or bool(r.variants_disagree))),
    ("S3_known_fault_clean", 3,
     lambda r: (r.decision == "known_fault" and not r.caution and not r.variants_disagree and bool(r.cnn_lr_agree)
                and not r.narrow and not r.acquisition_warning)),
    ("S4_healthy", 2, lambda r: r.decision == "healthy"),
    ("S5_acquisition_warning", 2, lambda r: bool(r.acquisition_warning)),
)
VERDICT_COLUMNS = ("rec_id", "class10", "decision", "fault_class", "caution", "variants_disagree", "cnn_lr_agree", "narrow",
                   "acquisition_warning", "warnings")


BOOL_COLUMNS = ("variants_disagree", "cnn_lr_agree", "narrow", "acquisition_warning")


def normalise(verdicts: pd.DataFrame) -> pd.DataFrame:
    """Booleans as booleans (a CSV gives "True" / "False" strings) and no missing caution."""
    v = verdicts.copy()
    for c in BOOL_COLUMNS:
        v[c] = v[c].map(lambda x: x if isinstance(x, (bool, np.bool_)) else str(x).strip().lower() == "true").astype(bool)
    v["caution"] = v["caution"].fillna("").astype(str)
    return v


class InsufficientStratum(RuntimeError):
    """A stratum has fewer eligible candidates than the protocol needs; `counts` says how many each one had."""

    def __init__(self, counts: dict):
        super().__init__(f"too few eligible cases: {counts}")
        self.counts = counts


def stratum_counts(verdicts: pd.DataFrame) -> dict:
    """Candidates per stratum before any is removed by an earlier stratum, and after (the selection order)."""
    v = normalise(verdicts)
    taken, out = set(), {}
    for name, n, pred in STRATA:
        mask = v.apply(pred, axis=1)
        cand = set(v.loc[mask, "rec_id"])
        out[name] = {"needed": n, "candidates": len(cand), "after_earlier_strata": len(cand - taken)}
        taken |= cand  # an upper bound on what earlier strata could remove; exact removals happen in select_cases
    return out


def select_cases(verdicts: pd.DataFrame, seed: int = SEED) -> pd.DataFrame:
    """15 cases: per stratum (in order) candidates sorted by rec_id, then a seeded choice without replacement from one
    generator; a case chosen earlier is not a candidate later. Raises InsufficientStratum with every stratum's counts."""
    v = normalise(verdicts).sort_values("rec_id").reset_index(drop=True)
    rng = np.random.default_rng(seed)
    chosen, rows, counts, short = set(), [], {}, False
    for name, n, pred in STRATA:
        cand = [r for r in v.loc[v.apply(pred, axis=1), "rec_id"] if r not in chosen]
        counts[name] = {"needed": n, "candidates": len(cand)}
        if len(cand) < n:
            short = True
            continue
        for rec in rng.choice(cand, size=n, replace=False):
            chosen.add(str(rec))
            rows.append({"stratum": name, "rec_id": str(rec)})
    if short:
        raise InsufficientStratum(counts)
    return pd.DataFrame(rows).assign(seed=seed)
