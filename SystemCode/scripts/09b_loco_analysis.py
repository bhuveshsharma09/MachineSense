"""Stage 3C.1: analysis and reporting on the saved deep-LOCO scores (protocol declared in CLAUDE.md §7 before running).
Torch-free and LightGBM-free: reads saved recording scores and decisions only.

    python scripts/09b_loco_analysis.py counts       # item 1: outcome tables in instances and distinct recordings
    python scripts/09b_loco_analysis.py confusion    # item 2: withheld → assigned class (CNN rule, LR/MSP)
    python scripts/09b_loco_analysis.py variants     # item 3: healthy-rule variants A / B / C
    python scripts/09b_loco_analysis.py compare      # item 4: CNN rule vs LR (paired bootstrap, operating points)
    python scripts/09b_loco_analysis.py seeds        # item 5: seeds 42 (recomputed), 43, 44

Nothing here selects a model, a variant or an application threshold. Development results on one partition; reserved
tests sealed. Outputs: results/loco/analysis/*.json|csv.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from ml.config import load_config
from ml.openset import outcome_category, three_way
from ml.paths import DEFAULT_CONFIG, SYSTEM_CODE_DIR, RunPaths, results_dir
from ml.protocol import order_statistic_threshold
from ml.splits import FAULT_CLASSES, split_hash
from ml.utils import git_sha

OUT = results_dir() / "loco" / "analysis"
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
TAG = "Development results on one partition; reserved tests sealed."


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


def provenance() -> dict:
    meta = pd.read_csv(SYSTEM_CODE_DIR / "metadata.csv")
    return {"split_hash": split_hash(meta), "git_sha": git_sha(), "note": TAG,
            "instances_vs_distinct": "a recording evaluated in several withheld-class runs counts once per run as an "
                                     "instance and once as a distinct recording"}


def run_counts() -> None:
    """Item 1: the 3C outcome tables (seed 42) recomputed with instances, distinct recordings and relabelled spreads."""
    dec = decide(recordings(42))
    saved = pd.read_parquet(RunPaths("loco", "cnn6ae6", 42, protocol="loco").scores("decisions"))
    key = ["withheld", "outer_fold", "rec_id"]
    chk = dec.merge(saved[key + ["decision_classifier_first", "decision_ae_first"]], on=key)
    same = bool((chk.decision_A == chk.decision_classifier_first).all() and (chk.decision_AEfirst == chk.decision_ae_first).all()
                and len(chk) == len(saved) == len(dec))
    lr = lr_decisions()
    out = {"provenance": provenance(), "recomputed_equals_saved_3C_decisions": same,
           "cnn_primary_A": group_table(dec, "outcome_A"), "cnn_ae_first": group_table(dec, "outcome_AEfirst"),
           "lr_msp": group_table(lr, "outcome"),
           "labels": {"std_27": STD_27, "std_3": STD_3}}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "item1_outcomes_counts.json").write_text(json.dumps(out, indent=1))
    rows = []
    for method, t in (("CNN rule A (primary)", out["cnn_primary_A"]), ("CNN AE-first", out["cnn_ae_first"]),
                      ("LR/MSP", out["lr_msp"])):
        for name, e in t.items():
            for cat, p in e["pooled"].items():
                rows.append({"method": method, "group": name, "outcome": cat, "instances": p["instances"],
                             "of_instances": e["instances"], "rate": p["rate"], "distinct_recordings": p["distinct_recordings"],
                             "of_distinct": e["distinct_recordings"], "std_27_runs": e[STD_27][cat], "std_3_folds_macro": e[STD_3][cat]})
    pd.DataFrame(rows).to_csv(OUT / "item1_outcomes_counts.csv", index=False)
    print("recomputed seed-42 decisions equal the saved 3C decisions:", same)
    t = pd.DataFrame(rows)
    print(t[t.method == "CNN rule A (primary)"].round(3).to_string(index=False))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("task", choices=["counts"])
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    a = ap.parse_args()
    load_config(a.config)
    {"counts": run_counts}[a.task]()
    return 0


if __name__ == "__main__":
    sys.exit(main())
