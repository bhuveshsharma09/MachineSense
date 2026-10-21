"""Stage R9 (CLAUDE.md §7): per-class (class-conditional) reject thresholds vs today's shared τ_p, τ_d on the 81 saved
LOCO runs. Pure helpers on saved recording scores; no training, no forward pass, no reserved data. Notebook
20_classwise_thresholds calls these and only displays the results."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ml.rmd_eval import BAR_POINTS, outcomes, rule_b

ARMS = ("G", "P", "S")
ARM_NAMES = {"G": "global thresholds (current)", "P": "per-class thresholds", "S": "shrunk per-class (50/50 with global)"}
MIN_CAL = 3          # a class with fewer calibration recordings uses the global thresholds (declared)
SHRINK = 0.5         # fixed shrink weight (declared, not tuned)
Q_GRID = np.round(np.arange(0.5, 1.0 + 1e-9, 0.0005), 4)
SECONDARY_Q = 0.975
TIE_EPS = 1e-12
BISECT_STEPS = 60


def quantiles(values, q) -> np.ndarray:
    """numpy linear quantile(s) of `values` at level(s) q."""
    return np.quantile(np.asarray(values, dtype=float), q, method="linear")


def thresholds(cal: pd.DataFrame, q, arm: str, classes) -> dict:
    """Per class c of `classes`: (τ_p, τ_d) at level(s) q for arm "P" (per-class) or "S" (shrunk), from the run's
    calibration recordings by TRUE class; a class with < MIN_CAL calibration recordings uses the global quantiles. Also
    returns the global pair and the set of fallback classes. With an array q every threshold is an array over q."""
    g_p, g_d = quantiles(cal.msp, 1 - np.asarray(q)), quantiles(cal.maha, q)
    out, fallback = {}, set()
    for c in classes:
        cc = cal[cal.class10 == c]
        if len(cc) < MIN_CAL:
            fallback.add(c)
            out[c] = (g_p, g_d)
            continue
        p, d = quantiles(cc.msp, 1 - np.asarray(q)), quantiles(cc.maha, q)
        out[c] = (p, d) if arm == "P" else (SHRINK * p + (1 - SHRINK) * g_p, SHRINK * d + (1 - SHRINK) * g_d)
    if arm not in ("P", "S"):
        raise ValueError(arm)
    return {"per_class": out, "global": (g_p, g_d), "fallback": fallback}


def gate(rec: pd.DataFrame, table: dict) -> np.ndarray:
    """conf per recording: MSP ≥ τ_p and maha ≤ τ_d of its PREDICTED class (ties within 1e-12 accepted). Scalar thresholds
    give a vector over recordings; array (grid) thresholds give a matrix recordings × q."""
    tp = np.stack([np.atleast_1d(table["per_class"][c][0]) for c in rec.pred])
    td = np.stack([np.atleast_1d(table["per_class"][c][1]) for c in rec.pred])
    msp, maha = rec.msp.to_numpy(float)[:, None], rec.maha.to_numpy(float)[:, None]
    ok = ((msp >= tp) | (np.abs(msp - tp) <= TIE_EPS)) & ((maha <= td) | (np.abs(maha - td) <= TIE_EPS))
    return ok[:, 0] if ok.shape[1] == 1 and np.ndim(table["global"][0]) == 0 else ok


def decisions(rec: pd.DataFrame, cal: pd.DataFrame, q: float, arm: str, classes) -> np.ndarray:
    """Rule B decisions of an arm at one level q."""
    return rule_b(rec.pred, gate(rec, thresholds(cal, q, arm, classes)))


def known_rejections(rec: pd.DataFrame, cal: pd.DataFrame, q, arm: str, classes) -> np.ndarray:
    """Number of foldk_known recordings the arm rejects ("unknown" under rule B = gate fails), at level(s) q."""
    kn = rec[rec.role == "foldk_known"]
    ok = gate(kn, thresholds(cal, q, arm, classes))
    return (~ok).sum(axis=0)


def matched_q(rec: pd.DataFrame, cal: pd.DataFrame, arm: str, classes, target: int) -> dict:
    """The level q whose foldk_known rejection count is closest to `target` (item 7 / 14): grid on [0.5, 1.0] brackets the
    boundary count ≥ target | count < target, bisection locates it; exact → the largest q with that count; a jump over the
    target → the closer side (tie → fewer rejections); out of range → the end, with the difference recorded."""
    counts = known_rejections(rec, cal, Q_GRID, arm, classes)  # non-increasing in q
    count = lambda q: int(known_rejections(rec, cal, float(q), arm, classes))
    if counts[-1] > target:
        return {"q": 1.0, "count": int(counts[-1]), "diff": int(counts[-1] - target), "note": "q = 1.0 still rejects more"}
    if counts[0] < target:
        return {"q": float(Q_GRID[0]), "count": int(counts[0]), "diff": int(counts[0] - target), "note": "q = 0.5 rejects fewer"}
    i = int(np.flatnonzero(counts >= target)[-1])  # last grid q with count ≥ target
    lo, hi = float(Q_GRID[i]), float(Q_GRID[min(i + 1, len(Q_GRID) - 1)])
    if lo == hi:
        return {"q": lo, "count": count(lo), "diff": count(lo) - target, "note": "upper end"}
    for _ in range(BISECT_STEPS):  # invariant: count(lo) ≥ target > count(hi)
        mid = 0.5 * (lo + hi)
        if count(mid) >= target:
            lo = mid
        else:
            hi = mid
    c_lo, c_hi = count(lo), count(hi)
    if c_lo == target:
        return {"q": lo, "count": c_lo, "diff": 0, "note": "exact"}
    pick_hi = abs(c_hi - target) <= abs(c_lo - target)  # tie → fewer rejections (hi)
    q, c = (hi, c_hi) if pick_hi else (lo, c_lo)
    return {"q": q, "count": c, "diff": c - target, "note": "count jumps over the target"}


def eval_outcomes(rec: pd.DataFrame, decision: np.ndarray) -> dict:
    """Descriptive counts on the seen-folder evaluation recordings (role eval): rejected and wrong fault."""
    d = pd.Series(decision, index=rec.index)
    ev = rec.role == "eval"
    fault = ev & (rec.class10 != "normal")
    return {"eval_n": int(ev.sum()), "eval_unknown": int((ev & (d == "unknown")).sum()),
            "eval_wrong_fault": int((fault & ~d.isin(["healthy", "unknown"]) & (d != rec.class10)).sum())}


def evaluate_run(rec: pd.DataFrame, g_decision: np.ndarray) -> list[dict]:
    """G (today's decisions, given), and P and S at the matched q and at q = 0.975: outcomes, q, counts, fallbacks."""
    cal = rec[rec.role == "cal"]
    classes = sorted(set(rec.pred) | set(cal.class10))
    g_out = outcomes(rec, g_decision)
    target = g_out["known_unknown"]
    rows = [{"arm": "G", "target": target, "q": None, "count": target, "diff": 0, "note": "reference",
             **{f"matched_{k}": v for k, v in g_out.items()}, **{f"secondary_{k}": v for k, v in g_out.items()},
             **{f"matched_{k}": v for k, v in eval_outcomes(rec, g_decision).items()}}]
    for arm in ("P", "S"):
        m = matched_q(rec, cal, arm, classes, target)
        d_m = decisions(rec, cal, m["q"], arm, classes)
        d_s = decisions(rec, cal, SECONDARY_Q, arm, classes)
        fb = thresholds(cal, SECONDARY_Q, arm, classes)["fallback"]
        rows.append({"arm": arm, "target": target, **m, "fallback_classes": ";".join(sorted(fb)),
                     **{f"matched_{k}": v for k, v in outcomes(rec, d_m).items()},
                     **{f"matched_{k}": v for k, v in eval_outcomes(rec, d_m).items()},
                     **{f"secondary_{k}": v for k, v in outcomes(rec, d_s).items()}})
    return rows


def withheld_recall_by_class(per_run: pd.DataFrame, mode: str) -> pd.DataFrame:
    """Unknown recall per withheld class (pooled over its folds) per seed and arm."""
    g = per_run.groupby(["seed", "withheld", "arm"])
    return (g[f"{mode}_withheld_unknown"].sum() / g[f"{mode}_withheld_n"].sum()).unstack("arm")


def reading(matched: pd.DataFrame, arm: str, ref: str = "G") -> dict:
    """The declared reading of `arm` against `ref` at the matched count: "helps, confirmed" if on every seed the recall gain
    is ≥ +5 points and withheld-called-healthy does not rise; else "worse" if the mean recall change is negative; else
    "does not help"."""
    m = matched.set_index(["seed", "arm"])
    per_seed = {}
    for s in sorted(matched.seed.unique()):
        gain = float(m.loc[(s, arm), "unknown_recall"] - m.loc[(s, ref), "unknown_recall"])
        dh = int(m.loc[(s, arm), "withheld_healthy"] - m.loc[(s, ref), "withheld_healthy"])
        per_seed[int(s)] = {"recall_gain": gain, "healthy_change": dh, "bar_met": gain >= BAR_POINTS - 1e-12 and dh <= 0}
    mean_gain = float(np.mean([v["recall_gain"] for v in per_seed.values()]))
    verdict = ("helps, confirmed" if all(v["bar_met"] for v in per_seed.values())
               else "worse" if mean_gain < 0 else "does not help")
    return {"arm": arm, "reading": verdict, "mean_recall_gain": mean_gain, "per_seed": per_seed}


def calibration_counts(rec_all: pd.DataFrame) -> pd.DataFrame:
    """Calibration recordings per class per run (rows: withheld, outer_fold, class10)."""
    cal = rec_all[rec_all.role == "cal"]
    return cal.groupby(["seed", "withheld", "outer_fold", "class10"]).size().rename("n_cal").reset_index()


def tau_d_spread(rec: pd.DataFrame, q: float = SECONDARY_Q) -> dict:
    """Spread of the per-class τ_d^c(q) across the run's classes that have ≥ MIN_CAL calibration recordings."""
    cal = rec[rec.role == "cal"]
    t = {c: float(quantiles(g.maha, q)) for c, g in cal.groupby("class10") if len(g) >= MIN_CAL}
    v = np.array(list(t.values()))
    return {"tau_d_min": float(v.min()), "tau_d_max": float(v.max()), "tau_d_ratio_max_min": float(v.max() / v.min()),
            "tau_d_global": float(quantiles(cal.maha, q)), "n_classes": len(v)}
