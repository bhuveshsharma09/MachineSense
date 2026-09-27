"""Deep-LOCO evaluation helpers shared by the Stage 3C.1 and 3D analyses (scripts/09b_loco_analysis.py,
scripts/09c_loco_complementary.py): recompute the 3C thresholds and rule decisions from saved recording scores, and
tabulate outcome groups in instances and distinct recordings. No torch."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ml.openset import outcome_category, three_way
from ml.paths import RunPaths
from ml.protocol import order_statistic_threshold
from ml.splits import FAULT_CLASSES

ACCEPT = 0.975
SEEDS = (42, 43, 44)
CATS = {"withheld": ["healthy", "known_fault", "unknown"],
        "known": ["healthy", "correct", "wrong_fault", "unknown"],
        "normal": ["healthy", "fault", "unknown"]}
# outcome groups: (name, role, subset, category set)
GROUPS = [("withheld", "foldk_withheld", None, "withheld"),
          ("known_a_seen_folder", "eval", "fault", "known"),
          ("known_b_unseen_foldk", "foldk_known", "fault", "known"),
          ("normal_foldk", "foldk_known", "normal", "normal"),
          ("normal_seen_folder", "eval", "normal", "normal")]
STD_27 = "std over the 27 class × fold runs (mixes between-class and between-fold variation; not seed uncertainty)"
STD_3 = "std over the 3 folds of the per-fold macro value"

def recordings(seed: int) -> pd.DataFrame:
    parts = [pd.read_parquet(RunPaths("loco", "cnn6", seed, protocol="loco", withheld=c).scores(f"fold{k}_recording"))
             for c in FAULT_CLASSES for k in (0, 1, 2)]
    return pd.concat(parts, ignore_index=True)


def outcomes(d: np.ndarray, rec: pd.DataFrame) -> np.ndarray:
    w = (rec.role == "foldk_withheld").to_numpy()
    oc = np.empty(len(rec), dtype=object)
    oc[w] = outcome_category(d[w], rec.class10.to_numpy()[w], True)
    oc[~w] = outcome_category(d[~w], rec.class10.to_numpy()[~w], False)
    return oc


def decide(rec: pd.DataFrame) -> pd.DataFrame:
    """The 3C thresholds per run (τ_p, τ_d: 97.5 % marginal calibration targets, order statistic, ties accepted; τ_e: max
    calibration-normal AE error) and the decisions of variants A (3C primary), B (no AE override), C (strict healthy,
    post hoc), plus AE-first (3C comparison)."""
    parts = []
    for (c, k), g in rec.groupby(["withheld", "outer_fold"]):
        cal = g[g.role == "cal"]
        tp = -order_statistic_threshold(-cal.msp.to_numpy(), ACCEPT)["threshold"]
        td = order_statistic_threshold(cal.maha.to_numpy(), ACCEPT)["threshold"]
        te = float(cal[cal.class10 == "normal"].ae.max())
        g = g.assign(tau_p=tp, tau_d=td, tau_e=te, n_cal_normals=int((cal.class10 == "normal").sum()))
        g["confident"] = (g.msp >= tp) & (g.maha <= td)
        for v, o in (("A", "classifier_first"), ("B", "classifier_first_no_override"), ("C", "strict_healthy"),
                     ("AEfirst", "ae_first")):
            d, ov = three_way(g.pred, g.msp, g.maha, g.ae, tp, td, te, o)
            g[f"decision_{v}"] = d
            g[f"outcome_{v}"] = outcomes(d, g)
            if v == "A":
                g["override_A"] = ov
        parts.append(g)
    return pd.concat(parts, ignore_index=True)


def select(dec: pd.DataFrame, role: str, subset: str | None) -> pd.DataFrame:
    g = dec[dec.role == role]
    if subset == "fault":
        g = g[g.class10 != "normal"]
    elif subset == "normal":
        g = g[g.class10 == "normal"]
    return g


def group_table(dec: pd.DataFrame, col: str) -> dict:
    """Every 3C outcome group, in instances and distinct recordings: pooled rates and counts, per class (known faults:
    per true class; withheld: per withheld class), the macro, the std over the 27 runs and the std over the 3 folds of
    the per-fold macro."""
    out = {}
    for name, role, subset, cs in GROUPS:
        g = select(dec, role, subset)
        cats = CATS[cs]
        per_key = "withheld" if name == "withheld" else "class10"
        pooled = {c: {"instances": int((g[col] == c).sum()), "rate": float((g[col] == c).mean()),
                      "distinct_recordings": int(g.loc[g[col] == c, "rec_id"].nunique())} for c in cats}
        per_class = {cl: {c: float((h[col] == c).mean()) for c in cats} | {"instances": len(h), "distinct": int(h.rec_id.nunique())}
                     for cl, h in g.groupby(per_key)}
        # per-run value: withheld / normal → the run's rate; known faults → the run's macro over known fault classes
        run_vals = []
        for (c, k), h in g.groupby(["withheld", "outer_fold"]):
            if cs == "known":
                m = pd.DataFrame({cl: {x: float((hh[col] == x).mean()) for x in cats} for cl, hh in h.groupby("class10")}).T
                run_vals.append({"withheld": c, "fold": k, **{x: float(m[x].mean()) for x in cats}})
            else:
                run_vals.append({"withheld": c, "fold": k, **{x: float((h[col] == x).mean()) for x in cats}})
        rv = pd.DataFrame(run_vals)
        fold_macro = rv.groupby("fold")[cats].mean()
        entry = {"instances": len(g), "distinct_recordings": int(g.rec_id.nunique()), "pooled": pooled,
                 "per_class": per_class, STD_27: {x: float(rv[x].std()) for x in cats},
                 "per_fold_macro": fold_macro.to_dict(orient="index"), STD_3: {x: float(fold_macro[x].std()) for x in cats}}
        if cs == "known":
            pc = pd.DataFrame(per_class).T
            entry["macro_over_known_fault_classes"] = {x: float(pc[x].mean()) for x in cats}
        if name == "withheld":
            pc = pd.DataFrame(per_class).T
            entry["macro_over_withheld_classes"] = {x: float(pc[x].mean()) for x in cats}
        out[name] = entry
    return out


def lr_decisions() -> pd.DataFrame:
    return pd.read_parquet(RunPaths("loco", "logreg", 42, protocol="loco").scores("decisions"))


def run_metrics(cnn: pd.DataFrame, lr: pd.DataFrame, weights: pd.Series | None = None) -> pd.DataFrame:
    """Per run: CNN Mahalanobis and LR −MSP AUROC and OSCR, withheld vs fold-k known; OSCR correctness = each method's own
    top class. Optional recording weights (bootstrap multiplicities); runs lacking either group are dropped."""
    from ml.openset import weighted_auroc, weighted_oscr

    rows = []
    for (c, k), g in cnn.groupby(["withheld", "outer_fold"]):
        h = lr[(lr.withheld == c) & (lr.outer_fold == k)]
        r = {"withheld": c, "outer_fold": int(k)}
        for name, df, u in (("cnn_maha", g, g.maha.to_numpy()), ("lr_msp", h, -h.msp.to_numpy())):
            kn = (df.role == "foldk_known").to_numpy()
            un = (df.role == "foldk_withheld").to_numpy()
            w = np.ones(len(df)) if weights is None else weights.reindex(df.rec_id).fillna(0).to_numpy()
            if w[kn].sum() == 0 or w[un].sum() == 0:
                r = None
                break
            correct = (df.pred == df.class10).to_numpy()
            r[f"{name}_auroc"] = weighted_auroc(u[kn], w[kn], u[un], w[un])
            r[f"{name}_oscr"] = weighted_oscr(u[kn], correct[kn], w[kn], u[un], w[un])
        if r is not None:
            rows.append(r)
    return pd.DataFrame(rows)


def macro(df: pd.DataFrame, col: str) -> float:
    """Macro over withheld classes of the mean over available folds."""
    return float(df.groupby("withheld")[col].mean().mean())


