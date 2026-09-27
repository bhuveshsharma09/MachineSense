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


def expected_checks(cnn: pd.DataFrame, dec: pd.DataFrame) -> list[dict]:
    """The counts Bhuvesh read from cnn6ae6_s42_loco_decisions.parquet (Stage 3C.1 brief), reproduced."""
    def n(c, d):
        return int(cnn.loc[c].get(d, 0)) if c in cnn.index else 0
    hm = dec[(dec.role == "foldk_withheld") & (dec.withheld == "horizontal_misalignment") & (dec.decision_A == "healthy")]
    sev = hm.severity.round(2).value_counts().to_dict()
    checks = [("imbalance → vertical_misalignment", n("imbalance", "vertical_misalignment"), 139),
              ("imbalance → unknown", n("imbalance", "unknown"), 5),
              ("vertical_misalignment → imbalance", n("vertical_misalignment", "imbalance"), 109),
              ("vertical_misalignment → unknown", n("vertical_misalignment", "unknown"), 12),
              ("vertical_misalignment → healthy", n("vertical_misalignment", "healthy"), 4),
              ("horizontal_misalignment → healthy", n("horizontal_misalignment", "healthy"), 22),
              ("horizontal_misalignment → healthy at 0.5 mm", int(sev.get(0.5, 0)), 10),
              ("horizontal_misalignment → healthy at 1.0 mm", int(sev.get(1.0, 0)), 7),
              ("horizontal_misalignment → healthy at 1.5 mm", int(sev.get(1.5, 0)), 5),
              ("horizontal_misalignment → healthy via the AE override", int(hm.override_A.sum()), 7),
              ("horizontal_misalignment → unknown", n("horizontal_misalignment", "unknown"), 51),
              ("overhang_ball → underhang_outer", n("overhang_ball", "underhang_outer"), 11),
              ("underhang_ball → overhang_outer", n("underhang_ball", "overhang_outer"), 11)]
    return [{"check": c, "reproduced": int(a), "expected": b, "match": int(a) == b} for c, a, b in checks]


def exploratory_distances(seed: int = 42) -> dict:
    """EXPLORATORY, descriptive only (no causal claims). In the imbalance-withheld runs: (i) Mahalanobis distance of the
    withheld imbalance recordings to EACH known-class centroid (same centroids and shared within-class Ledoit–Wolf
    covariance as 3C; per window, then the mean over the recording's windows); (ii) distance of the fold-k known bearing
    recordings to the centroid of the withheld imbalance windows of that run, by added imbalance mass."""
    from sklearn.covariance import LedoitWolf

    out = {"to_known_centroids": [], "bearing_to_withheld_imbalance": [], "sanity_min_equals_saved": []}
    for k in (0, 1, 2):
        rp = RunPaths("loco", "cnn6", seed, protocol="loco", withheld="imbalance")
        w = pd.read_parquet(rp.scores(f"fold{k}_logits"))
        z = np.load(rp.embeddings(k))
        emb = z["embedding"].astype(np.float64)
        assert (z["rec_id"] == w.rec_id.to_numpy()).all()
        fit = (w.role == "fit").to_numpy()
        classes = sorted(set(w.class10[fit]))
        mu = np.stack([emb[fit & (w.class10 == c).to_numpy()].mean(axis=0) for c in classes])
        lab = w.class10[fit].to_numpy()
        prec = LedoitWolf().fit(emb[fit] - mu[np.searchsorted(classes, lab)]).precision_

        def dist(x, m):
            return np.sqrt(np.maximum(np.einsum("ij,jk,ik->i", x - m, prec, x - m), 0.0))

        wh = (w.role == "foldk_withheld").to_numpy()
        per_class = pd.DataFrame({c: dist(emb[wh], mu[i]) for i, c in enumerate(classes)}).assign(rec_id=w.rec_id[wh].to_numpy())
        rec_d = per_class.groupby("rec_id").mean()
        saved = pd.read_parquet(rp.scores(f"fold{k}_recording")).set_index("rec_id").maha
        win_min = per_class[classes].min(axis=1).groupby(per_class.rec_id).mean()
        out["sanity_min_equals_saved"].append(float(np.max(np.abs(win_min - saved.loc[win_min.index]))))
        for c in classes:
            out["to_known_centroids"].append({"fold": k, "known_class": c, "median": float(rec_d[c].median()),
                                              "mean": float(rec_d[c].mean()),
                                              "nearest_share": float((rec_d[classes].idxmin(axis=1) == c).mean()),
                                              "n_recordings": len(rec_d)})
        centre = emb[wh].mean(axis=0)
        kb = ((w.role == "foldk_known") & w.class10.str.contains("hang_")).to_numpy()
        d_b = pd.DataFrame({"d": dist(emb[kb], centre), "rec_id": w.rec_id[kb].to_numpy(),
                            "class10": w.class10[kb].to_numpy()}).groupby(["rec_id", "class10"]).d.mean().reset_index()
        meta = pd.read_csv(SYSTEM_CODE_DIR / "metadata.csv").set_index("rec_id")
        d_b["added_mass_g"] = meta.loc[d_b.rec_id, "severity"].to_numpy()
        for (c, m), g in d_b.groupby(["class10", "added_mass_g"]):
            out["bearing_to_withheld_imbalance"].append({"fold": k, "class10": c, "added_mass_g": float(m),
                                                         "median": float(g.d.median()), "n_recordings": len(g)})
    return out


def run_confusion() -> None:
    """Item 2: withheld class → final decision (unknown / healthy / each known class), CNN rule A and LR/MSP, in instances
    and distinct recordings; the counts from the brief reproduced; the post-hoc subset; exploratory distances."""
    dec = decide(recordings(42))
    lr = lr_decisions()
    tabs = {}
    for name, df, col in (("cnn_rule_A", dec, "decision_A"), ("lr_msp", lr, "decision")):
        w = df[df.role == "foldk_withheld"]
        inst = pd.crosstab(w.withheld, w[col])
        dist = w.groupby(["withheld", col]).rec_id.nunique().unstack(fill_value=0)
        tabs[name] = {"instances": inst, "distinct": dist}
        inst.to_csv(OUT / f"item2_withheld_to_assigned_{name}_instances.csv")
        dist.to_csv(OUT / f"item2_withheld_to_assigned_{name}_distinct.csv")
    checks = expected_checks(tabs["cnn_rule_A"]["instances"], dec)
    w = dec[(dec.role == "foldk_withheld") & ~dec.withheld.isin(["imbalance", "vertical_misalignment"])]
    subset = {"label": "POST HOC failure analysis, not a headline result: withheld classes other than imbalance and "
                       "vertical misalignment",
              "instances": len(w), "distinct": int(w.rec_id.nunique()),
              **{k: int((w.outcome_A == k).sum()) for k in CATS["withheld"]}}
    expl = exploratory_distances()
    out = {"provenance": provenance(), "wording": "these tables show what the model confuses, not the physical cause",
           "cnn_rule_A": {k: v.to_dict(orient="index") for k, v in tabs["cnn_rule_A"].items()},
           "lr_msp": {k: v.to_dict(orient="index") for k, v in tabs["lr_msp"].items()},
           "expected_checks": checks, "post_hoc_subset": subset,
           "exploratory_distances": {"label": "EXPLORATORY and descriptive only; no causal claims", **expl}}
    (OUT / "item2_withheld_confusion.json").write_text(json.dumps(out, indent=1))
    pd.DataFrame(expl["to_known_centroids"]).to_csv(OUT / "item2_imbalance_to_known_centroids.csv", index=False)
    pd.DataFrame(expl["bearing_to_withheld_imbalance"]).to_csv(OUT / "item2_bearing_to_withheld_imbalance.csv", index=False)
    for name in tabs:
        print(f"== {name} (instances; distinct recordings are identical for withheld rows)")
        print(tabs[name]["instances"].to_string())
    print(pd.DataFrame(checks).to_string(index=False))
    print(subset)
    print("sanity: max |recomputed min distance − saved| per fold:", expl["sanity_min_equals_saved"])
    tk = pd.DataFrame(expl["to_known_centroids"])
    print(tk.groupby("known_class")[["median", "nearest_share"]].mean().sort_values("median").round(2).to_string())
    bt = pd.DataFrame(expl["bearing_to_withheld_imbalance"])
    print(bt.pivot_table(index="class10", columns="added_mass_g", values="median").round(1).to_string())


VARIANT_LABEL = {"A": "A: 3C primary (classifier-first with the AE override)",
                 "B": "B: A without the AE override",
                 "C": "C: strict healthy (identified post hoc during review on 27 Sep; exploratory)"}


def variant_deltas(dec: pd.DataFrame, v: str) -> dict:
    """Relative to A: withheld-fault → healthy instances removed, normal → unknown instances added (fold k and seen
    folder), and any known fault → healthy change. Counts only; no ratio."""
    out = {}
    w = dec[dec.role == "foldk_withheld"]
    out["withheld_to_healthy_removed"] = int(((w.decision_A == "healthy") & (w[f"decision_{v}"] != "healthy")).sum())
    out["withheld_to_healthy_added"] = int(((w.decision_A != "healthy") & (w[f"decision_{v}"] == "healthy")).sum())
    for name, role in (("normal_foldk", "foldk_known"), ("normal_seen_folder", "eval")):
        n = dec[(dec.role == role) & (dec.class10 == "normal")]
        added = (n[f"decision_{v}"] == "unknown") & (n.decision_A != "unknown")
        out[f"{name}_to_unknown_added"] = int(added.sum())
        out[f"{name}_to_unknown_added_distinct"] = int(n.loc[added, "rec_id"].nunique())
    kf = dec[(dec.role != "foldk_withheld") & (dec.role != "cal") & (dec.class10 != "normal")]
    out["known_fault_to_healthy_removed"] = int(((kf.decision_A == "healthy") & (kf[f"decision_{v}"] != "healthy")).sum())
    return out


def run_variants(seed: int = 42) -> dict:
    """Item 3: healthy-rule variants A / B / C from saved scores (thresholds exactly as in 3C)."""
    dec = decide(recordings(seed))
    res = {"provenance": provenance(), "seed": seed, "labels": VARIANT_LABEL,
           "definitions": {"both_classifier_checks_pass": "MSP ≥ τ_p AND Mahalanobis ≤ τ_d",
                           "ae_healthy": "AE error ≤ τ_e", "thresholds": "exactly as in 3C"},
           "tau_e_limitation": f"τ_e rests on {int(dec.n_cal_normals.min())}–{int(dec.n_cal_normals.max())} calibration "
                               "normals per run",
           "normal_instances_note": "the normal instances repeat 25 distinct healthy recordings",
           "variants": {v: group_table(dec, f"outcome_{v}") for v in ("A", "B", "C")},
           "relative_to_A": {v: variant_deltas(dec, v) for v in ("B", "C")},
           "app_rule": "App healthy rule pending Bhuvesh's decision after 3C.1; not inferred from pooled experimental counts"}
    return res


def run_variants_report() -> None:
    res = run_variants(42)
    (OUT / "item3_healthy_rule_variants.json").write_text(json.dumps(res, indent=1))
    rows = []
    for v, t in res["variants"].items():
        for g, e in t.items():
            for cat, p in e["pooled"].items():
                rows.append({"variant": v, "group": g, "outcome": cat, "instances": p["instances"], "of": e["instances"],
                             "distinct": p["distinct_recordings"], "of_distinct": e["distinct_recordings"],
                             "rate": p["rate"]})
    t = pd.DataFrame(rows)
    t.to_csv(OUT / "item3_healthy_rule_variants.csv", index=False)
    piv = t.pivot_table(index=["group", "outcome"], columns="variant", values="instances").astype(int)
    print(piv.to_string())
    print(json.dumps(res["relative_to_A"], indent=1))
    exp = {("withheld", "healthy"): (28, 19, 8), ("normal_foldk", "healthy"): (192, 184, 161),
           ("normal_seen_folder", "healthy"): (68, 66, 55)}
    for key, e in exp.items():
        got = tuple(int(piv.loc[key, v]) for v in "ABC")
        print(key, "reproduced", got, "expected", e, "match" if got == e else "MISMATCH")
    print(res["tau_e_limitation"])


RETRO = ("RETROSPECTIVE and EXPLORATORY: thresholds matched on evaluation data; they must not be deployed or used for "
         "model selection")


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


def lr_outcomes(lr: pd.DataFrame, accept: np.ndarray) -> pd.DataFrame:
    d = np.where(accept, np.where(lr.pred == "normal", "healthy", lr.pred), "unknown")
    return lr.assign(decision_x=d, outcome_x=outcomes(d, lr))


def run_compare() -> None:
    """Item 4: CNN rule vs LR on the saved scores (seed 42)."""
    from ml.config import load_config as _lc
    from ml.evaluation import mixed_bootstrap_draws

    cfg = _lc(DEFAULT_CONFIG)
    dec = decide(recordings(42))
    lr = lr_decisions()
    cnn_eval = dec[dec.role.isin(["foldk_known", "foldk_withheld"])]
    lr_eval = lr[lr.role.isin(["foldk_known", "foldk_withheld"])]
    per_run = run_metrics(cnn_eval, lr_eval)
    for m in ("auroc", "oscr"):
        per_run[f"diff_{m}"] = per_run[f"cnn_maha_{m}"] - per_run[f"lr_msp_{m}"]
    per_run.to_csv(OUT / "item4_per_run_threshold_free.csv", index=False)
    point = {m: {"cnn_maha_macro": macro(per_run, f"cnn_maha_{m}"), "lr_msp_macro": macro(per_run, f"lr_msp_{m}"),
                 "diff_macro": macro(per_run, f"diff_{m}"), "diff_mean_over_runs": float(per_run[f"diff_{m}"].mean()),
                 "diff_std_over_runs": float(per_run[f"diff_{m}"].std()), "runs_cnn_higher": int((per_run[f"diff_{m}"] > 0).sum()),
                 "runs": len(per_run)} for m in ("auroc", "oscr")}
    # paired bootstrap: one mixed-unit draw over the whole pool applied to all 27 runs (recording multiplicities)
    pool = pd.read_csv(results_dir() / cfg.development.folds_file).sort_values("rec_id").reset_index(drop=True)
    draws = mixed_bootstrap_draws(pool.class10.to_numpy(), pool.folder.to_numpy(), cfg.baselines.n_bootstrap, cfg.baselines.seed)
    boot, dropped = {m: [] for m in ("auroc", "oscr")}, []
    for idx in draws:
        wts = pd.Series(np.bincount(idx, minlength=len(pool)).astype(float), index=pool.rec_id)
        pr = run_metrics(cnn_eval, lr_eval, wts)
        dropped.append(27 - len(pr))
        for m in ("auroc", "oscr"):
            boot[m].append(macro(pr.assign(d=pr[f"cnn_maha_{m}"] - pr[f"lr_msp_{m}"]), "d"))
    for m in ("auroc", "oscr"):
        b = np.array(boot[m])
        point[m]["paired_bootstrap_ci95"] = [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]
        point[m]["p_cnn_higher"] = float(np.mean(b > 0))
    point["bootstrap"] = {"n": len(draws), "runs_dropped_per_draw_mean": float(np.mean(dropped)),
                          "runs_dropped_per_draw_max": int(np.max(dropped)),
                          "scheme": "stratified mixed-unit over the whole development pool (fault folders within class, "
                                    "normal recordings individually); one draw weights every run's recordings by their "
                                    "multiplicity, so recordings reused across withheld-class runs are resampled jointly"}
    # operating point matched on CALIBRATION: LR acceptance count = CNN joint-acceptance count per run
    op_rows, parts = [], []
    for (c, k), g in dec.groupby(["withheld", "outer_fold"]):
        cal = g[g.role == "cal"]
        h = lr[(lr.withheld == c) & (lr.outer_fold == k)]
        hc = h[h.role == "cal"]
        assert set(hc.rec_id) == set(cal.rec_id)
        m, n = int(cal.confident.sum()), len(cal)
        u = np.sort(-hc.msp.to_numpy())
        if m == 0:
            thr, unattainable = -np.inf, True
        else:
            thr, unattainable = float(u[m - 1]), False
        achieved = float(np.mean(-hc.msp.to_numpy() <= thr))
        op_rows.append({"withheld": c, "outer_fold": int(k), "n_cal": n, "cnn_joint_accepted": m, "target": m / n,
                        "lr_achieved": achieved, "ties_at_threshold": int(np.sum(u == thr)) if m else 0,
                        "unattainable": unattainable, "lr_threshold_neg_msp": thr})
        parts.append(lr_outcomes(h, (-h.msp.to_numpy()) <= thr))
    op = pd.DataFrame(op_rows)
    op.to_csv(OUT / "item4_calibration_matched_thresholds.csv", index=False)
    lr_cal = pd.concat(parts, ignore_index=True)
    # retrospective diagnostic: LR false-unknown count on (a), then (b), matched to the CNN rule's per run
    retro = {}
    for label, role in (("a_seen_folder", "eval"), ("b_unseen_foldk", "foldk_known")):
        parts, rows = [], []
        for (c, k), g in dec.groupby(["withheld", "outer_fold"]):
            h = lr[(lr.withheld == c) & (lr.outer_fold == k)]
            kf = g[(g.role == role) & (g.class10 != "normal")]
            target = int((kf.decision_A == "unknown").sum())
            hk = h[(h.role == role) & (h.class10 != "normal")]
            ud = np.sort(-hk.msp.to_numpy())[::-1]
            thr = -np.inf if target >= len(ud) else float(ud[target])
            got = int(np.sum(-hk.msp.to_numpy() > thr))
            rows.append({"withheld": c, "outer_fold": int(k), "cnn_false_unknowns": target, "lr_false_unknowns": got,
                         "ties": int(np.sum(ud == thr))})
            parts.append(lr_outcomes(h, (-h.msp.to_numpy()) <= thr))
        retro[label] = {"label": RETRO, "matching": pd.DataFrame(rows).to_dict(orient="records"),
                        "outcomes": group_table(pd.concat(parts, ignore_index=True), "outcome_x")}
    out = {"provenance": provenance(),
           "wording": "an interval excluding zero supports a specific metric difference under this protocol, not an "
                      "overall winner",
           "threshold_free": point,
           "calibration_matched_operating_point": {
               "rule": "per run, LR threshold = the m-th smallest LR −MSP of the calibration recordings, m = number of "
                       "calibration recordings passing both CNN checks; accept iff ≤ (ties accepted)",
               "summary": {"target_mean": float(op.target.mean()), "lr_achieved_mean": float(op.lr_achieved.mean()),
                           "runs_with_ties": int((op.ties_at_threshold > 1).sum()), "unattainable": int(op.unattainable.sum())},
               "cnn_rule_A": group_table(dec, "outcome_A"), "lr_matched": group_table(lr_cal, "outcome_x")},
           "retrospective_diagnostic": retro}
    (OUT / "item4_cnn_vs_lr.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(point, indent=1))
    print(out["calibration_matched_operating_point"]["summary"])
    for name, t in (("CNN A", out["calibration_matched_operating_point"]["cnn_rule_A"]),
                    ("LR cal-matched", out["calibration_matched_operating_point"]["lr_matched"]),
                    ("LR retro (a)", retro["a_seen_folder"]["outcomes"]), ("LR retro (b)", retro["b_unseen_foldk"]["outcomes"])):
        print(name, {g: {k: v["instances"] for k, v in e["pooled"].items()} for g, e in t.items()})


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("task", choices=["counts", "confusion", "variants", "compare"])
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    a = ap.parse_args()
    load_config(a.config)
    OUT.mkdir(parents=True, exist_ok=True)
    {"counts": run_counts, "confusion": run_confusion, "variants": run_variants_report,
     "compare": run_compare}[a.task]()
    return 0


if __name__ == "__main__":
    sys.exit(main())
