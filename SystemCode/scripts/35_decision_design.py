"""Stage R5: decision design for the specialist bank (Q3, development data, seed 42). No training: Step 0 (the floor),
E1 (gated composition) and E2 (level correction) re-score Stage R4's saved per-recording table. Protocol: CLAUDE.md §7
(Stage R5), declared before running.

    python scripts/35_decision_design.py                     # seed 42 (the original run; original output paths)
    python scripts/35_decision_design.py --seed 43           # Stage R5 addendum: R4 and E1 only, seed-suffixed outputs
    python scripts/35_decision_design.py seeds               # the three-seed table and reading (r5_seeds.json)

Outputs: results/research/r5_decision_design.json, r5_per_variant.csv (one row per variant × run), r5_per_recording.csv
(one row per recording × run, with every variant's score and verdicts). Exploratory further analysis of development data;
the reserved partitions stay sealed and F1 has already been run, so nothing here is an independent claim.
"""
from __future__ import annotations

import importlib.util
import json
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from ml.arbitrator import decide
from ml.config import load_config
from ml.openset import oscr
from ml.paths import DEFAULT_CONFIG, SYSTEM_CODE_DIR, results_dir
from ml.protocol import order_statistic_threshold
from ml.q1 import mild_mask
from ml.rejection import TIE_EPS, clopper_pearson, snap_ties
from ml.specialists import (ACCEPTANCE, NORMAL, accepting_sets, db_to_ln_power, gated_verdict, healthy_gates,
                            level_corrected, verdict, within_class_slope)
from ml.speed_calibration import exceeds, matched_threshold
from ml.splits import split_hash
from ml.utils import git_sha

OUT = results_dir() / "research"
KEY = ["withheld", "outer_fold", "rec_id"]
VARIANTS = {  # name: (composition, level correction)
    "R4": ("bank", None), "E1": ("gated", None),
    "E2a-bank": ("bank", "beta1"), "E2b-bank": ("bank", "betafit"),
    "E1+E2a": ("gated", "beta1"), "E1+E2b": ("gated", "betafit")}
BASELINE = {"bank": "M0", "gated": "M1"}
N_BOOT, BOOT_SEED = 2000, 42
WORDING = ("Exploratory further analysis of Stage R4's development results, seed 42 only; the reserved partitions stay "
           "sealed and F1 has already been run, so nothing here is an independent claim.")


def r4_module(seed: int):
    """scripts/34_train_specialists.py with its seed set (its path and saved-score plumbing is the single source)."""
    spec = importlib.util.spec_from_file_location("s34", SYSTEM_CODE_DIR / "scripts" / "34_train_specialists.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    m.set_seed(seed)
    return m


def load(seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame, object]:
    cfg = load_config(DEFAULT_CONFIG)
    meta = pd.read_csv(SYSTEM_CODE_DIR / "metadata.csv")
    m34 = r4_module(seed)
    t = pd.read_csv(m34.out_file("r4_ambiguity", "csv"), keep_default_na=False, na_values=["", "nan", "NaN"])
    t["accepting"] = t.accepting.astype(str).replace("nan", "")
    dec = m34.saved_scores()[KEY + ["pred", "msp", "ae"]]
    b1s = pd.read_parquet(results_dir() / "scores" / "b1s_nomic_loco_q3fit_scores.parquet")[KEY + ["b1s", "tau_b"]]
    thr = m34.thresholds().reset_index()[["withheld", "outer_fold", "tau_p", "tau_e"]]
    df = t.merge(dec, on=KEY, validate="1:1").merge(b1s, on=KEY, validate="1:1").merge(thr, on=["withheld", "outer_fold"])
    assert len(df) == len(t) == 11548 and (df.cnn_top == df.pred).all()
    df = snap_ties(df)  # R0 tie rule on msp, maha, ae, b1s against their thresholds
    df["gates_pass"] = [healthy_gates(r.ae, r.b1s, {"tau_p": r.tau_p, "tau_d": r.tau_d, "tau_e": r.tau_e, "tau_b": r.tau_b})
                        for r in df.itertuples()]
    df["fault_family"] = meta.set_index("rec_id").fault_family.loc[df.rec_id].to_numpy()
    df["L"] = db_to_ln_power(df.level)
    return df, meta, cfg


def baselines(df: pd.DataFrame) -> pd.DataFrame:
    """M0 (R4's Mahalanobis-only) and M1 (Mahalanobis in the gated composition)."""
    rej = df.maha > df.tau_d
    df["M0"] = np.where(rej, "unknown", np.where(df.pred == NORMAL, "healthy", "known_fault"))
    df["M1"] = np.where(df.gates_pass, "healthy", np.where(rej | (df.pred == NORMAL), "unknown", "known_fault"))
    return df


def step0(df: pd.DataFrame) -> dict:
    """The floor: withheld instances passing the healthy gates; and the full rules' healthy counts for context."""
    un = df[df.role == "foldk_withheld"]
    rules = {}
    for v in ("A", "B", "C", "D", "B_with_caution"):
        out = [decide(v, r.pred, float(r.msp), float(r.maha), float(r.ae), float(r.b1s),
                      {"tau_p": r.tau_p, "tau_d": r.tau_d, "tau_e": r.tau_e, "tau_b": r.tau_b}) for r in un.itertuples()]
        rules[v] = {"healthy": int(sum(o["decision"] == "healthy" for o in out)),
                    "healthy_with_caution": int(sum(o["decision"] == "healthy" and o["caution"] is not None for o in out))}
    k = int(un.gates_pass.sum())
    return {"withheld_instances": len(un), "distinct_recordings": int(un.rec_id.nunique()),
            "floor_gates_pass": k, "floor_rate": k / len(un), "floor_ci95": list(clopper_pearson(k, len(un))),
            "floor_by_class": un.groupby("withheld").gates_pass.agg(["sum", "size"]).astype(int).to_dict("index"),
            "ae_gate_pass": int((un.ae <= un.tau_e).sum()), "b1s_gate_pass": int((un.b1s <= un.tau_b).sum()),
            "full_rules_healthy": rules,
            "known_side_gates_pass": {"instances": int(df[df.role == "foldk_known"].gates_pass.sum()),
                                      "of": int((df.role == "foldk_known").sum())}}


def run_variant(g: pd.DataFrame, name: str, comp: str, corr: str | None) -> tuple[dict, pd.DataFrame]:
    """One variant on one run: scores, τ (all calibration recordings; gated also known-fault calibration only), accepting
    sets, verdicts at τ and at the matched known-side rejection count."""
    specs = [c for c in g.columns if c.startswith("s_") and g[c].notna().all()]
    if comp == "gated":
        specs = [c for c in specs if c != f"s_{NORMAL}"]
    s = g[specs].set_axis([c[2:] for c in specs], axis=1)
    beta = 0.0
    if corr == "beta1":
        beta = 1.0
    elif corr == "betafit":
        cal = g[g.role == "cal"]
        own = [cal.at[i, f"s_{c}"] for i, c in zip(cal.index, cal.class10)]
        beta = within_class_slope(own, cal.L, cal.class10)
    if beta:
        s = level_corrected(s, g.L, beta)
    S = s.min(axis=1)
    cal_mask = (g.role == "cal").to_numpy()
    tau = float(order_statistic_threshold(S[cal_mask].to_numpy(), ACCEPTANCE)["threshold"])
    tau_fault = float(order_statistic_threshold(S[cal_mask & (g.class10 != NORMAL).to_numpy()].to_numpy(),
                                                ACCEPTANCE)["threshold"]) if comp == "gated" else np.nan

    def verdicts(t):
        A = accepting_sets(s, t)
        return (A.map(verdict) if comp == "bank" else
                pd.Series([gated_verdict(gp, a) for gp, a in zip(g.gates_pass, A)], index=g.index)), A

    v_cal, A_cal = verdicts(tau)
    base = g[BASELINE[comp]]
    kn = (g.role == "foldk_known").to_numpy()
    un = (g.role == "foldk_withheld").to_numpy()
    n_b = int((base[kn] == "unknown").sum())
    elig = kn & (~g.gates_pass.to_numpy() if comp == "gated" else np.ones(len(g), bool))
    feasible = n_b <= int(elig.sum())
    t_m = matched_threshold(S[elig].to_numpy(), min(n_b, int(elig.sum())))
    v_m, _ = verdicts(t_m)
    v_fault = verdicts(tau_fault)[0] if comp == "gated" else None
    y = np.r_[np.zeros(kn.sum()), np.ones(un.sum())]
    corr_known = (g.pred[kn] == g.class10[kn]).to_numpy()
    row = {"variant": name, "composition": comp, "correction": corr or "none", "baseline": BASELINE[comp],
           "withheld": g.withheld.iat[0], "outer_fold": int(g.outer_fold.iat[0]), "beta": beta, "tau": tau,
           "n_known": int(kn.sum()), "n_unknown": int(un.sum()),
           "baseline_known_rejected": n_b, "matched_feasible": bool(feasible),
           "known_rejected_matched": int((v_m[kn] == "unfamiliar").sum()),
           "unknown_recall_matched": int((v_m[un] == "unfamiliar").sum()),
           "baseline_unknown_recall": int((base[un] == "unknown").sum()),
           "known_rejected_calibrated": int((v_cal[kn] == "unfamiliar").sum()),
           "unknown_recall_calibrated": int((v_cal[un] == "unfamiliar").sum()),
           "healthy_matched": int((v_m[un] == "healthy").sum()), "healthy_calibrated": int((v_cal[un] == "healthy").sum()),
           "ambiguous_healthy_matched": int((v_m[un] == "ambiguous_healthy_fault").sum()),
           "baseline_healthy": int((base[un] == "healthy").sum()),
           "auroc": roc_auc_score(y, np.r_[S[kn], S[un]]),
           "oscr": oscr(S[kn].to_numpy(), corr_known, S[un].to_numpy()),
           "maha_auroc": roc_auc_score(y, np.r_[g.maha[kn], g.maha[un]]),
           "maha_oscr": oscr(g.maha[kn].to_numpy(), corr_known, g.maha[un].to_numpy())}
    if comp == "gated":
        row.update({"tau_fault_cal": tau_fault, "known_rejected_fault_cal": int((v_fault[kn] == "unfamiliar").sum()),
                    "unknown_recall_fault_cal": int((v_fault[un] == "unfamiliar").sum())})
    rec = pd.DataFrame({f"{name}__S": S, f"{name}__verdict_cal": v_cal, f"{name}__verdict_matched": v_m,
                        f"{name}__accepting_cal": A_cal}, index=g.index)
    return row, rec


def known_side(P: pd.DataFrame, name: str, base: str) -> dict:
    """Known-side costs at the matched threshold (not part of the bar): normals called healthy / a fault / unfamiliar, and
    known faults named correctly (single accepting specialist = own class; baseline: CNN-6 top class = own class), named as
    an ambiguous set containing their own class, named wrongly, or called unfamiliar / unknown."""
    kn = P[P.role == "foldk_known"]
    v, A = kn[f"{name}__verdict_matched"], None
    nor, flt = kn.class10 == NORMAL, kn.class10 != NORMAL

    def share(mask, cond):
        return float(cond[mask].mean())
    b = kn[base]
    b_named_ok = (b == "known_fault") & (kn.pred == kn.class10)
    out = {"normals": {"n": int(nor.sum()),
                       "variant": {k: share(nor, v == k) for k in ("healthy", "known_fault", "ambiguous_fault",
                                                                    "ambiguous_healthy_fault", "unfamiliar")},
                       "baseline": {k: share(nor, b == k) for k in ("healthy", "known_fault", "unknown")}},
           "known_faults": {"n": int(flt.sum()),
                            "baseline_named_correctly": share(flt, b_named_ok),
                            "baseline_named_wrongly": share(flt, (b == "known_fault") & ~b_named_ok),
                            "baseline_healthy": share(flt, b == "healthy"), "baseline_unknown": share(flt, b == "unknown")}}
    acc = P.loc[kn.index, f"{name}__accepting_cal"].astype(str)  # composition of the accepting set at the calibrated τ
    single_ok = (v == "known_fault") & (acc == kn.class10)
    amb_ok = v.str.startswith("ambiguous") & pd.Series([c in a.split("+") for c, a in zip(kn.class10, acc)], index=kn.index)
    out["known_faults"].update({"variant_named_correctly_single": share(flt, single_ok),
                                "variant_ambiguous_containing_own": share(flt, amb_ok),
                                "variant_healthy": share(flt, v == "healthy"),
                                "variant_unfamiliar": share(flt, v == "unfamiliar"),
                                "note": "variant naming uses the accepting set at the calibrated threshold, verdicts at the "
                                        "matched one"})
    return out


def bootstrap_diff(un: pd.DataFrame, a: str, b: str) -> dict:
    """Class-stratified bootstrap over the withheld recordings (thresholds fixed) of recall(a) − recall(b)."""
    rng = np.random.default_rng(BOOT_SEED)
    d = (un[a].to_numpy(float) - un[b].to_numpy(float))
    groups = [np.flatnonzero((un.withheld == c).to_numpy()) for c in sorted(un.withheld.unique())]
    draws = [d[np.concatenate([rng.choice(ix, len(ix)) for ix in groups])].mean() for _ in range(N_BOOT)]
    return {"diff": float(d.mean()), "ci95": [float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))]}


def compute(seed: int = 42, variants=None) -> dict:
    """Step 0 and the declared variants for one seed, through the unchanged R5 functions (no file written)."""
    variants = {k: VARIANTS[k] for k in (variants or VARIANTS)}
    df, meta, cfg = load(seed)
    df = baselines(df)
    s0 = step0(df)
    print(f"STEP 0 floor: {s0['floor_gates_pass']} of {s0['withheld_instances']} withheld pass the healthy gates "
          f"({s0['floor_rate']:.3f}); AE gate {s0['ae_gate_pass']}, B1s gate {s0['b1s_gate_pass']}; full rules healthy "
          f"{ {k: v['healthy'] for k, v in s0['full_rules_healthy'].items()} }", flush=True)
    rows, recs = [], []
    for name, (comp, corr) in variants.items():
        parts = []
        for _, g in df.groupby(["withheld", "outer_fold"]):
            r, rc = run_variant(g, name, comp, corr)
            rows.append(r)
            parts.append(rc)
        recs.append(pd.concat(parts))
    V = pd.DataFrame(rows)
    P = pd.concat([df] + [r.loc[df.index] for r in recs], axis=1)
    # recording-level matched flags for the bootstrap
    un = P[P.role == "foldk_withheld"].copy()
    un["M0_unknown"] = un.M0 == "unknown"
    un["M1_unknown"] = un.M1 == "unknown"
    mild = mild_mask(un.set_index("rec_id").rename(columns={"class10": "class10"}), cfg.anomaly)
    summary = {}
    for name, (comp, corr) in variants.items():
        v = V[V.variant == name]
        base = BASELINE[comp]
        n_u = int(v.n_unknown.sum())
        rec_m, rec_b = int(v.unknown_recall_matched.sum()), int(v.baseline_unknown_recall.sum())
        un[f"{name}_um"] = un[f"{name}__verdict_matched"] == "unfamiliar"
        fold = v.groupby("outer_fold")[["auroc", "maha_auroc", "oscr", "maha_oscr"]].mean()
        per_class = v.groupby("withheld").apply(lambda x: pd.Series({
            "recall_matched": x.unknown_recall_matched.sum() / x.n_unknown.sum(),
            "baseline_recall": x.baseline_unknown_recall.sum() / x.n_unknown.sum(),
            "recall_calibrated": x.unknown_recall_calibrated.sum() / x.n_unknown.sum(),
            "healthy_matched": int(x.healthy_matched.sum()), "baseline_healthy": int(x.baseline_healthy.sum()),
            "auroc": x.auroc.mean(), "maha_auroc": x.maha_auroc.mean(), "n": int(x.n_unknown.sum())}),
            include_groups=False)
        mild_res = {lab: {"n": int(m.sum()),
                          "recall_matched": float(un.set_index("rec_id").loc[m[m].index, f"{name}__verdict_matched"].eq("unfamiliar").mean()),
                          "baseline_recall": float(un.set_index("rec_id").loc[m[m].index, base].eq("unknown").mean())}
                    for lab, m in mild.items()}
        sev = un.groupby(["class10", "severity"]).apply(lambda x: pd.Series({
            "n": len(x), "recall_matched": float((x[f"{name}__verdict_matched"] == "unfamiliar").mean()),
            "baseline_recall": float((x[base] == "unknown").mean()),
            "healthy_matched": int((x[f"{name}__verdict_matched"] == "healthy").sum())}), include_groups=False).reset_index()
        healthy_m, healthy_b = int(v.healthy_matched.sum()), int(v.baseline_healthy.sum())
        diff = rec_m / n_u - rec_b / n_u
        crit = {"recall_gain_ge_5pp": diff >= 0.05, "healthy_not_higher": healthy_m <= healthy_b}
        reading = "helps" if all(crit.values()) else ("does not help" if (diff <= 0 or not crit["healthy_not_higher"])
                                                      else "too thin to tell")
        summary[name] = {
            "composition": comp, "correction": corr or "none", "baseline": base,
            "matched": {"variant_recall": rec_m / n_u, "baseline_recall": rec_b / n_u, "diff": diff,
                        "variant_ci95": list(clopper_pearson(rec_m, n_u)), "baseline_ci95": list(clopper_pearson(rec_b, n_u)),
                        "paired_bootstrap": bootstrap_diff(un, f"{name}_um", f"{base}_unknown"),
                        "infeasible_runs": v[~v.matched_feasible][["withheld", "outer_fold"]].to_dict("records"),
                        "per_fold": {int(k): {"variant": x.unknown_recall_matched.sum() / x.n_unknown.sum(),
                                              "baseline": x.baseline_unknown_recall.sum() / x.n_unknown.sum()}
                                     for k, x in v.groupby("outer_fold")}},
            "calibrated": {"variant_recall": int(v.unknown_recall_calibrated.sum()) / n_u,
                           "variant_known_rejection": int(v.known_rejected_calibrated.sum()) / int(v.n_known.sum()),
                           "baseline_known_rejection": int(v.baseline_known_rejected.sum()) / int(v.n_known.sum()),
                           "withheld_healthy": int(v.healthy_calibrated.sum())},
            "withheld_healthy": {"variant_matched": healthy_m, "baseline": healthy_b,
                                 "variant_ambiguous_healthy_fault_matched": int(v.ambiguous_healthy_matched.sum())},
            "auroc_oscr_per_fold": fold.round(6).to_dict("index"),
            "per_class": per_class.round(4).to_dict("index"), "mildest": mild_res, "severity": sev.round(4).to_dict("records"),
            "beta": {"mean": float(v.beta.mean()), "min": float(v.beta.min()), "max": float(v.beta.max())},
            "criteria": crit, "reading": reading}
        summary[name]["known_side"] = known_side(P, name, base)
        if comp == "gated":
            summary[name]["sensitivity_tau_known_fault_cal_only"] = {
                "variant_recall": int(v.unknown_recall_fault_cal.sum()) / n_u,
                "variant_known_rejection": int(v.known_rejected_fault_cal.sum()) / int(v.n_known.sum())}
        print(f"{name:9s} vs {base}: matched recall {rec_m / n_u:.3f} vs {rec_b / n_u:.3f} (Δ {diff:+.3f}, "
              f"{summary[name]['matched']['paired_bootstrap']['ci95']}); healthy {healthy_m} vs {healthy_b}; "
              f"AUROC folds {fold.auroc.round(3).tolist()} vs {fold.maha_auroc.round(3).tolist()} → {reading}", flush=True)
    V.insert(0, "seed", seed)
    return {"summary": summary, "V": V, "P": P, "s0": s0, "meta": meta, "cfg": cfg}


SEED_VARIANTS = ("R4", "E1")  # the addendum: E2 is not rerun


def run(seed: int) -> int:
    res = compute(seed, None if seed == 42 else SEED_VARIANTS)
    summary, V, P, s0, meta, cfg = (res[k] for k in ("summary", "V", "P", "s0", "meta", "cfg"))
    m34 = r4_module(seed)
    lat = m34.latency_and_size(cfg)
    sfx = "" if seed == 42 else f"_s{seed}"
    out = {"stage": "R5" if seed == 42 else "R5 addendum (seeds 43 / 44)", "wording": WORDING, "seed": seed,
           "split_hash": split_hash(meta), "git_sha": git_sha(),
           "step0_floor": s0, "variants": summary,
           "latency": {"note": "every variant uses 9 autoencoders per recording (bank: 9 specialists; gated: the AE-6 for "
                               "the healthy gate + 8 fault specialists) besides CNN-6 and B1s; measured as in R4",
                       **lat},
           "e3": "deferred (Bhuvesh, 1 Oct): ~80 min for seed 42; declared in an addendum after these results",
           "e4": "not started: runs only if a variant passes the bar"}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"r5_decision_design{sfx}.json").write_text(json.dumps(out, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
    V.to_csv(OUT / f"r5_per_variant{sfx}.csv", index=False)
    keep = ["withheld", "outer_fold", "rec_id", "role", "class10", "severity", "speed_hz", "level", "L", "fault_family",
            "gates_pass", "ae", "tau_e", "b1s", "tau_b", "maha", "tau_d", "pred", "M0", "M1"]
    P[keep + [c for c in P.columns if "__" in c]].to_csv(OUT / f"r5_per_recording{sfx}.csv", index=False)
    print("latency:", lat["cpu_ms_per_recording"])
    return 0


def seeds() -> int:
    """The addendum's three-seed table and reading: E1 vs M1 and R4 vs M0 per seed from the per-seed result files;
    r5_per_variant.csv rewritten with every seed (seed column)."""
    res = {s: json.loads((OUT / ("r5_decision_design.json" if s == 42 else f"r5_decision_design_s{s}.json")).read_text())
           for s in (42, 43, 44)}
    table = {}
    for s, r in res.items():
        for name in ("E1", "R4"):
            v = r["variants"][name]
            table[f"{name}_s{s}"] = {"seed": s, "variant": name, "baseline": v["baseline"],
                                     "matched_recall": v["matched"]["variant_recall"], "baseline_recall": v["matched"]["baseline_recall"],
                                     "gain": v["matched"]["diff"], "gain_ci95_sampling": v["matched"]["paired_bootstrap"]["ci95"],
                                     "withheld_healthy": v["withheld_healthy"]["variant_matched"],
                                     "baseline_healthy": v["withheld_healthy"]["baseline"],
                                     "calibrated_recall": v["calibrated"]["variant_recall"],
                                     "calibrated_known_rejection": v["calibrated"]["variant_known_rejection"],
                                     "baseline_known_rejection": v["calibrated"]["baseline_known_rejection"],
                                     "per_fold": v["matched"]["per_fold"], "reading_single_seed": v["reading"]}
    e1 = [table[f"E1_s{s}"] for s in (42, 43, 44)]
    gains = np.array([t["gain"] for t in e1])
    fails = [{"seed": t["seed"], "condition": c} for t in e1 for c, bad in
             (("matched recall below M1 + 5 points", t["gain"] < 0.05), ("withheld called healthy above M1", t["withheld_healthy"] > t["baseline_healthy"]))
             if bad]
    reading = "helps, confirmed on three seeds" if not fails else "seed-dependent"
    out = {"addendum": "Stage R5 — E1 confirmed on seeds 43 and 44 (1 Oct 2026)", "table": table,
           "e1_gain": {"per_seed": dict(zip((42, 43, 44), gains.tolist())), "mean": float(gains.mean()),
                       "worst": float(gains.min()), "seed_sd": float(gains.std(ddof=1)),
                       "note": "seed spread (across seeds) and sampling uncertainty (per-seed bootstrap) are reported apart"},
           "failures": fails, "reading": reading}
    (OUT / "r5_seeds.json").write_text(json.dumps(out, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
    v42 = pd.read_csv(OUT / "r5_per_variant.csv")
    v42 = v42[v42.seed == 42] if "seed" in v42 else v42.assign(seed=42)
    allv = pd.concat([v42] + [pd.read_csv(OUT / f"r5_per_variant_s{s}.csv") for s in (43, 44)], ignore_index=True)
    allv = allv[["seed"] + [c for c in allv.columns if c != "seed"]]
    allv.to_csv(OUT / "r5_per_variant.csv", index=False)
    print(json.dumps({k: (v["gain"], v["withheld_healthy"], v["baseline_healthy"]) for k, v in table.items()}, indent=1))
    print("E1 gain per seed:", out["e1_gain"]["per_seed"], "→", reading, fails)
    return 0


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("task", nargs="?", default="run", choices=["run", "seeds"])
    ap.add_argument("--seed", type=int, default=42, choices=(42, 43, 44))
    a = ap.parse_args()
    return seeds() if a.task == "seeds" else run(a.seed)


if __name__ == "__main__":
    sys.exit(main())
