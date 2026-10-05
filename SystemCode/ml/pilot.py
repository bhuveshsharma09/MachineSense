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


# ------------------------------------------------------------------------------------------------ analysis (D7 item 1)
MEASURES = ("supported", "unsupported_repair", "useful_next_inspection", "unnecessary_refusal", "correct")


def wilson(k: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    """Wilson 95 % score interval for k successes in n (nan, nan for n = 0)."""
    if n == 0:
        return (math.nan, math.nan)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def mcnemar_exact(b: int, c: int) -> float:
    """Exact two-sided McNemar p-value on discordant pairs: min(1, 2 · P(X ≤ min(b, c))), X ~ Binomial(b + c, ½)."""
    n = b + c
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n)


def scored_table(scores: pd.DataFrame, key: dict) -> pd.DataFrame:
    """scores.csv joined with key.json: one row per item with arm, rec_id, q and each measure as True / False / NaN
    (blank = not scored or not applicable)."""
    k = pd.DataFrame.from_dict(key["items"], orient="index").rename_axis("item_id").reset_index()
    t = scores.drop(columns=[c for c in ("q",) if c in scores.columns]).merge(k, on="item_id", how="right", validate="1:1")
    for m in MEASURES:
        t[m] = t[m].map(lambda x: {"yes": True, "no": False}.get(str(x).strip().lower(), np.nan)) if m in t else np.nan
    return t


def per_arm(table: pd.DataFrame) -> pd.DataFrame:
    """Counts and proportions per arm and measure with Wilson 95 % intervals (n counts scored items only)."""
    rows = []
    for arm, g in table.groupby("arm"):
        for m in MEASURES:
            v = g[m].dropna()
            k, n = int(v.sum()), int(len(v))
            lo, hi = wilson(k, n)
            rows.append({"arm": arm, "measure": m, "k": k, "n": n, "p": k / n if n else math.nan, "lo": lo, "hi": hi})
    return pd.DataFrame(rows)


def paired(table: pd.DataFrame, first: str = "B", second: str = "C") -> pd.DataFrame:
    """Paired comparison on the same case and question: discordant pairs and the exact McNemar test, per measure."""
    a = table[table.arm == first].set_index(["rec_id", "q"])
    b = table[table.arm == second].set_index(["rec_id", "q"])
    rows = []
    for m in MEASURES:
        j = pd.concat([a[m].rename("x"), b[m].rename("y")], axis=1, join="inner").dropna()
        x, y = j.x.astype(bool), j.y.astype(bool)
        yes_first, yes_second = int((x & ~y).sum()), int((~x & y).sum())
        rows.append({"measure": m, "pairs": int(len(j)), f"yes_{first}_only": yes_first, f"yes_{second}_only": yes_second,
                     "both_yes": int((x & y).sum()), "both_no": int((~x & ~y).sum()),
                     "mcnemar_p": mcnemar_exact(yes_first, yes_second)})
    return pd.DataFrame(rows)


def citation_valid(response: dict, case_guidance_ids) -> bool | None:
    """Valid if lookup_guidance was called in that turn and every cited id is a section 04 entry id of the case; None
    when nothing is cited."""
    cited = [c["id"] for c in response.get("citations") or []]
    if not cited:
        return None
    called = any(a["tool"] == "lookup_guidance" for a in response.get("actions") or [])
    return called and set(cited) <= set(case_guidance_ids)


def automatic(runs: list[dict], cases: dict) -> dict:
    """Per agent arm, from the logs: fallback rate (Wilson) and reasons, revisions, latency, tool calls, the review's
    findings on what was shown (by check), citation validity."""
    out = {}
    for arm in ("B", "C"):
        rs = [r["response"] for r in runs if r["arm"] == arm]
        lines = [r for r in runs if r["arm"] == arm]
        if not rs:
            continue
        fb = [r for r in rs if r["source"] == "fallback"]
        lat = np.array([r["latency_ms"] / 1000 for r in rs])
        shown_failed = [r["review"]["failed"] for r in rs if r["source"] != "fallback"]
        by_check = {}
        for f in (r["review"]["failed"] for r in rs):
            for c in {x["check"] for x in f}:
                by_check[c] = by_check.get(c, 0) + 1
        valid = [citation_valid(r, cases[l["rec_id"]]["guidance_ids"]) for r, l in zip(rs, lines)]
        valid = [v for v in valid if v is not None]
        out[arm] = {"answers": len(rs), "fallbacks": len(fb), "fallback_rate": len(fb) / len(rs),
                    "fallback_rate_wilson": wilson(len(fb), len(rs)),
                    "fallback_reasons": pd.Series([r.get("reason") for r in fb], dtype=object).value_counts().to_dict(),
                    "revised": sum(r["review"]["status"] == "revised" for r in rs),
                    "latency_s": {"median": float(np.median(lat)), "q1": float(np.percentile(lat, 25)),
                                  "q3": float(np.percentile(lat, 75)), "max": float(lat.max())},
                    "tool_calls_mean": float(np.mean([r["tool_calls_used"] for r in rs])),
                    "shown_with_review_findings": sum(bool(f) for f in shown_failed),
                    "answers_with_finding_by_check": by_check,
                    "citations": {"answers_citing": len(valid), "valid": int(sum(valid))}}
    return out
