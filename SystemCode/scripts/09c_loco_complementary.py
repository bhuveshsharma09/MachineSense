"""Stage 3D: complementary evidence for Q3 from saved scores (CPU only; no neural-network training; no torch, no LightGBM).
Protocol declared in CLAUDE.md §7 before running; candidates, formulas and criteria are fixed; nothing is selected.

    python scripts/09c_loco_complementary.py e1         # E1 deep ensemble over seeds 42/43/44
    python scripts/09c_loco_complementary.py e2         # E2 LR + CNN combined unfamiliarity (per seed)
    python scripts/09c_loco_complementary.py e3         # E3 B1s healthy veto, D and D′ (per seed)
    python scripts/09c_loco_complementary.py evaluate   # tables vs rules A/B/C, paired threshold-free, criteria

Exploratory development results on one partition; the candidates were chosen after seeing the 3C.1 results.
Outputs: results/loco/complementary/*.json|csv; per-recording decisions results/scores/<name>_decisions.parquet.
"""
from __future__ import annotations

import argparse
import json
import sys
from itertools import combinations

import numpy as np
import pandas as pd

from ml.loco_eval import ACCEPT, SEEDS, decide, lr_decisions, outcomes, recordings
from ml.openset import empirical_rank, js_divergence, recording_probs
from ml.paths import SYSTEM_CODE_DIR, RunPaths, results_dir
from ml.protocol import order_statistic_threshold
from ml.splits import FAULT_CLASSES, split_hash
from ml.utils import git_sha

OUT = results_dir() / "loco" / "complementary"
ROLES = ("cal", "eval", "foldk_known", "foldk_withheld")
WORDING = ("Exploratory development results on one partition. The candidates were chosen after seeing 3C.1 results. Any "
           "final claim uses the existing reserved-recording protocol (the loco_* and other reserved-test columns in "
           "metadata.csv), opened once at the end. All nine fault classes have informed development decisions. Reserved "
           "recordings test those decisions; they do not establish performance on entirely new fault families.")


def provenance() -> dict:
    meta = pd.read_csv(SYSTEM_CODE_DIR / "metadata.csv")
    return {"split_hash": split_hash(meta), "git_sha": git_sha(), "wording": WORDING}


def threshold(cal_scores: np.ndarray) -> tuple[float, bool]:
    """Order-statistic threshold (higher = more unfamiliar; accept iff ≤, ties accepted) and whether it is attainable."""
    t = order_statistic_threshold(np.asarray(cal_scores, dtype=np.float64), ACCEPT)
    return float(t["threshold"]), bool(np.isfinite(t["threshold"]))


def rule_decisions(conf: np.ndarray, pred: np.ndarray, healthy_ok: np.ndarray | None = None) -> np.ndarray:
    """Confident → healthy (label normal, and `healthy_ok` if given) or that fault; otherwise unknown. No override."""
    normal = pred == "normal"
    ok = np.ones(len(pred), bool) if healthy_ok is None else healthy_ok
    return np.where(conf & ~normal, pred, np.where(conf & normal & ok, "healthy", "unknown")).astype(object)


def decisions_path(name: str, seed: int | None):
    """results/scores/<name>_s<seed>_loco_decisions.parquet; the ensemble (seed None) names all three seeds."""
    if seed is None:
        return results_dir() / "scores" / f"{name}_s{'-'.join(map(str, SEEDS))}_loco_decisions.parquet"
    return RunPaths("loco3d", name, seed, protocol="loco").scores("decisions")


def save(name: str, seed: int | None, df: pd.DataFrame) -> None:
    p = decisions_path(name, seed)
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(p, index=False)


def load(name: str, seed: int | None) -> pd.DataFrame:
    return pd.read_parquet(decisions_path(name, seed))


def run_e1() -> None:
    """E1: one ensemble predictor per run over seeds 42/43/44."""
    recs = {s: recordings(s) for s in SEEDS}
    parts, thr_rows = [], []
    for c in FAULT_CLASSES:
        for k in (0, 1, 2):
            probs, ranks, base = {}, {}, None
            for s in SEEDS:
                w = pd.read_parquet(RunPaths("loco", "cnn6", s, protocol="loco", withheld=c).scores(f"fold{k}_logits"))
                w = w[w.role.isin(ROLES)]
                classes = [x.removeprefix("logit_") for x in w.columns if x.startswith("logit_")]
                probs[s] = recording_probs(w[[f"logit_{x}" for x in classes]].to_numpy(), w.rec_id, classes)
                r = recs[s][(recs[s].withheld == c) & (recs[s].outer_fold == k)].set_index("rec_id")
                ranks[s] = pd.Series(empirical_rank(r.maha, r.maha[r.role == "cal"]), index=r.index)
                base = r if base is None else base
            classes = sorted(probs[SEEDS[0]].columns)
            idx = base.index
            P = {s: probs[s].loc[idx, classes].to_numpy() for s in SEEDS}
            ens = np.mean([P[s] for s in SEEDS], axis=0)
            g = base[["role", "class10", "folder", "severity", "dev_fold"]].copy()
            g["ens_msp"] = ens.max(axis=1)
            g["pred"] = np.asarray(classes)[ens.argmax(axis=1)]
            g["ens_maha_rank"] = np.mean([ranks[s].loc[idx].to_numpy() for s in SEEDS], axis=0)
            g["disagreement"] = np.mean([js_divergence(P[a], P[b]) for a, b in combinations(SEEDS, 2)], axis=0)
            cal = g[g.role == "cal"]
            tp, ok_p = threshold(-cal.ens_msp.to_numpy())
            td, ok_d = threshold(cal.ens_maha_rank.to_numpy())
            conf = (g.ens_msp.to_numpy() >= -tp) & (g.ens_maha_rank.to_numpy() <= td)
            g["confident"] = conf
            g["decision"] = rule_decisions(conf, g.pred.to_numpy())
            g = g.reset_index().assign(withheld=c, outer_fold=k)
            g["outcome"] = outcomes(g.decision.to_numpy(), g)
            parts.append(g)
            thr_rows.append({"withheld": c, "outer_fold": k, "tau_p_ens": -tp, "tau_d_ens_rank": td, "n_cal": len(cal),
                             "attainable_p": ok_p, "attainable_d": ok_d,
                             "cal_joint_acceptance": float(conf[(g.role == "cal").to_numpy()].mean())})
    dec = pd.concat(parts, ignore_index=True)
    save("e1_ensemble", None, dec)
    OUT.mkdir(parents=True, exist_ok=True)
    thr = pd.DataFrame(thr_rows)
    thr.to_csv(OUT / "e1_thresholds.csv", index=False)
    w = dec[dec.role == "foldk_withheld"]
    print("unattainable targets:", int((~thr.attainable_p).sum() + (~thr.attainable_d).sum()),
          "| cal joint acceptance mean", round(thr.cal_joint_acceptance.mean(), 3))
    print("E1 withheld:", w.outcome.value_counts().to_dict())
    print("median disagreement by role:", dec.groupby("role").disagreement.median().round(4).to_dict())


def run_e2() -> None:
    """E2 per seed: s = max(r(1 − LR MSP), r(CNN Mahalanobis)) against the run's calibration recordings; τ_s by the
    order-statistic rule; the CNN MSP check (τ_p) unchanged from 3C; no AE override."""
    lr = lr_decisions().set_index(["withheld", "outer_fold", "rec_id"])
    thr_rows, acc_rows = [], []
    for seed in SEEDS:
        dec = decide(recordings(seed))
        parts = []
        for (c, k), g in dec.groupby(["withheld", "outer_fold"]):
            g = g.copy()
            lr_msp = lr.loc[[(c, k, r) for r in g.rec_id], "msp"].to_numpy()
            cal = (g.role == "cal").to_numpy()
            u_lr = 1.0 - lr_msp
            g["r_lr"] = empirical_rank(u_lr, u_lr[cal])
            g["r_cnn_maha"] = empirical_rank(g.maha, g.maha[cal])
            g["s"] = np.maximum(g.r_lr, g.r_cnn_maha)
            ts, ok = threshold(g.s[cal].to_numpy())
            msp_ok = (g.msp >= g.tau_p).to_numpy()
            conf = msp_ok & (g.s <= ts).to_numpy()
            g["confident_e2"] = conf
            g["decision_e2"] = rule_decisions(conf, g.pred.to_numpy())
            g["outcome_e2"] = outcomes(g.decision_e2.to_numpy(), g)
            parts.append(g)
            thr_rows.append({"seed": seed, "withheld": c, "outer_fold": k, "tau_s": ts, "attainable": ok,
                             "n_cal": int(cal.sum())})
            for grp, role in (("calibration", "cal"), ("seen_folder_eval", "eval"), ("unseen_foldk", "foldk_known")):
                m = (g.role == role).to_numpy()
                acc_rows.append({"seed": seed, "withheld": c, "outer_fold": k, "group": grp, "n": int(m.sum()),
                                 "msp_acceptance": float(msp_ok[m].mean()), "s_acceptance": float((g.s <= ts).to_numpy()[m].mean()),
                                 "joint_acceptance": float(conf[m].mean())})
        d = pd.concat(parts, ignore_index=True)
        save("e2_lr_cnn_rank", seed, d)
        w = d[d.role == "foldk_withheld"]
        print(f"seed {seed}: E2 withheld", w.outcome_e2.value_counts().to_dict(), "| rule B", w.outcome_B.value_counts().to_dict())
    thr = pd.DataFrame(thr_rows)
    thr.to_csv(OUT / "e2_thresholds.csv", index=False)
    acc = pd.DataFrame(acc_rows)
    acc.to_csv(OUT / "e2_joint_acceptance_per_run.csv", index=False)
    print("unattainable:", int((~thr.attainable).sum()))
    print(acc.groupby(["seed", "group"])[["msp_acceptance", "s_acceptance", "joint_acceptance"]].mean().round(3).to_string())


B1S = "B1s_speed_poly"


def b1s_scores() -> pd.DataFrame:
    """B1s refitted per run on exactly that run's AE fit normals (vibration-only capped R1 features); recording scores for
    the calibration, seen-folder and fold-k recordings; τ_b = maximum calibration-normal score. Seed-independent."""
    from ml.cache import load_features
    from ml.config import load_config
    from ml.features import feature_columns
    from ml.openset import loco_run_ids
    from ml.paths import DEFAULT_CONFIG, data_dir
    from ml.q1 import classical_scores
    from ml.splits import development_pool

    cfg = load_config(DEFAULT_CONFIG)
    meta = pd.read_csv(SYSTEM_CODE_DIR / "metadata.csv")
    dc = cfg.development
    pool = development_pool(meta, dc.allowed_closed, dc.reserved_columns).merge(
        pd.read_csv(results_dir() / dc.folds_file)[["rec_id", "dev_fold"]], on="rec_id")
    path = max((data_dir() / "features").glob(f"{cfg.baselines.feature_table}_*.parquet"),
               key=lambda p: p.stat().st_mtime)
    df = load_features(path, meta, cfg.preprocess).merge(pool[["rec_id", "folder", "dev_fold"]], on="rec_id")
    df = df.sort_values(["rec_id", "window_idx"]).reset_index(drop=True)
    cols = [c for c in feature_columns(df) if not c.startswith("f_mic_")]
    assignment = pd.read_csv(results_dir() / "q1q3" / "q3_fit_cal_eval_assignment.csv")
    normals = set(pool.loc[pool.class10 == "normal", "rec_id"])
    rows = []
    for c in FAULT_CLASSES:
        for k in (0, 1, 2):
            ids = loco_run_ids(assignment, pool, c, k)
            fit_n = sorted(set(ids["fit"]) & normals)
            saved = json.loads(RunPaths("loco", "ae6", 42, protocol="loco", withheld=c).norm_stats(k).read_text())["fit_ids"]
            assert fit_n == sorted(saved), f"{c}/{k}: not the AE's fit normals"
            role_of = {r: role for role in ROLES for r in ids[role]}
            target = df[df.rec_id.isin(role_of)]
            sc = classical_scores(B1S, cfg.anomaly, df[df.rec_id.isin(fit_n)], target, cols)
            t = pd.DataFrame({"rec_id": sc.index, "b1s": sc.to_numpy(), "withheld": c, "outer_fold": k,
                              "is_normal": sc.index.isin(normals)})
            t["role"] = t.rec_id.map(role_of)
            cal_n = t[(t.role == "cal") & t.rec_id.isin(normals)]
            t["tau_b"] = float(cal_n.b1s.max())
            t["n_cal_normals"] = len(cal_n)
            t["n_fit_normals"] = len(fit_n)
            rows.append(t)
    return pd.concat(rows, ignore_index=True)


def run_e3() -> None:
    """E3 per seed: D (rule B's checks + B1s healthy veto) and D′ (D + AE ≤ τ_e for healthy)."""
    b = b1s_scores()
    b.to_parquet(results_dir() / "scores" / "b1s_nomic_loco_q3fit_scores.parquet", index=False)
    key = ["withheld", "outer_fold", "rec_id"]
    fa_rows = []
    for (c, k), g in b.groupby(["withheld", "outer_fold"]):
        for grp, role in (("normal_foldk", "foldk_known"), ("normal_seen_folder", "eval")):
            n = g[(g.role == role) & g.is_normal]
            fa_rows.append({"withheld": c, "outer_fold": k, "group": grp, "n": len(n), "false_alarms": int((n.b1s > n.tau_b).sum()),
                            "n_cal_normals": int(g.n_cal_normals.iloc[0]), "n_fit_normals": int(g.n_fit_normals.iloc[0])})
    fa = pd.DataFrame(fa_rows)
    fa.to_csv(OUT / "e3_b1s_false_alarms_per_run.csv", index=False)
    for seed in SEEDS:
        d = decide(recordings(seed)).merge(b[key + ["b1s", "tau_b"]], on=key, how="left")
        assert d.b1s.notna().all()
        b1s_ok = (d.b1s <= d.tau_b).to_numpy()
        ae_ok = (d.ae <= d.tau_e).to_numpy()
        conf, pred = d.confident.to_numpy(), d.pred.to_numpy()
        d["decision_D"] = rule_decisions(conf, pred, b1s_ok)
        d["decision_Dp"] = rule_decisions(conf, pred, b1s_ok & ae_ok)
        for v in ("D", "Dp"):
            d[f"outcome_{v}"] = outcomes(d[f"decision_{v}"].to_numpy(), d)
        save("e3_b1s_veto", seed, d)
        w = d[d.role == "foldk_withheld"]
        print(f"seed {seed}: withheld D", w.outcome_D.value_counts().to_dict(), "D′", w.outcome_Dp.value_counts().to_dict(),
              "B", w.outcome_B.value_counts().to_dict())
    print(fa.groupby("group")[["false_alarms", "n"]].sum().to_string())
    print("calibration normals per run:", sorted(fa.n_cal_normals.unique()), "fit normals:", sorted(fa.n_fit_normals.unique()))


def summary(dec: pd.DataFrame, col: str) -> dict:
    """The 3C.1 outcome groups for one decision column: withheld unknown / known fault / healthy (instances = distinct
    recordings; also per withheld class); known faults (a), (b): unknown and wrong fault (instances, distinct, macro rate
    over known fault classes, pooled rate); normals fold k and seen folder: healthy / unknown (instances, distinct)."""
    from ml.loco_eval import group_table

    t = group_table(dec, col)
    out = {"withheld": {k: t["withheld"]["pooled"][k]["instances"] for k in ("unknown", "known_fault", "healthy")},
           "withheld_per_class": {c: {k: int(round(v[k] * v["instances"])) for k in ("unknown", "known_fault", "healthy")}
                                  for c, v in t["withheld"]["per_class"].items()}}
    for g in ("known_a_seen_folder", "known_b_unseen_foldk"):
        e = t[g]
        out[g] = {k: {"instances": e["pooled"][k]["instances"], "distinct": e["pooled"][k]["distinct_recordings"],
                      "macro_rate": e["macro_over_known_fault_classes"][k], "pooled_rate": e["pooled"][k]["rate"]}
                  for k in ("unknown", "wrong_fault")} | {"n_instances": e["instances"], "n_distinct": e["distinct_recordings"]}
    for g in ("normal_foldk", "normal_seen_folder"):
        e = t[g]
        out[g] = {k: {"instances": e["pooled"][k]["instances"], "distinct": e["pooled"][k]["distinct_recordings"]}
                  for k in ("healthy", "unknown")} | {"n_instances": e["instances"], "n_distinct": e["distinct_recordings"]}
    return out


def flat(sm: dict) -> dict:
    """Key numbers of a summary for tables and seed means."""
    return {"withheld_unknown": sm["withheld"]["unknown"], "withheld_known_fault": sm["withheld"]["known_fault"],
            "withheld_healthy": sm["withheld"]["healthy"],
            "a_unknown_inst": sm["known_a_seen_folder"]["unknown"]["instances"],
            "a_unknown_macro": sm["known_a_seen_folder"]["unknown"]["macro_rate"],
            "a_wrong_inst": sm["known_a_seen_folder"]["wrong_fault"]["instances"],
            "b_unknown_inst": sm["known_b_unseen_foldk"]["unknown"]["instances"],
            "b_unknown_macro": sm["known_b_unseen_foldk"]["unknown"]["macro_rate"],
            "b_wrong_inst": sm["known_b_unseen_foldk"]["wrong_fault"]["instances"],
            "normal_foldk_healthy": sm["normal_foldk"]["healthy"]["instances"],
            "normal_foldk_unknown": sm["normal_foldk"]["unknown"]["instances"],
            "normal_seen_healthy": sm["normal_seen_folder"]["healthy"]["instances"],
            "normal_seen_unknown": sm["normal_seen_folder"]["unknown"]["instances"]}


def criteria(m: dict, b: dict) -> dict:
    """Pre-declared pass/fail against rule B (macro false-unknown rates, 2-point margin); no selection."""
    p1 = (m["withheld_known_fault"] < b["withheld_known_fault"] and m["a_unknown_macro"] <= b["a_unknown_macro"] + 0.02
          and m["b_unknown_macro"] <= b["b_unknown_macro"] + 0.02)
    return {"problem1_pass": bool(p1),
            "problem1_detail": {"withheld_known_fault": [m["withheld_known_fault"], b["withheld_known_fault"]],
                                "a_unknown_macro_minus_B_pp": 100 * (m["a_unknown_macro"] - b["a_unknown_macro"]),
                                "b_unknown_macro_minus_B_pp": 100 * (m["b_unknown_macro"] - b["b_unknown_macro"])},
            "problem2_pass": bool(m["withheld_healthy"] < b["withheld_healthy"]),
            "problem2_detail": {"withheld_healthy": [m["withheld_healthy"], b["withheld_healthy"]],
                                "normal_to_unknown_change_foldk": m["normal_foldk_unknown"] - b["normal_foldk_unknown"],
                                "normal_to_unknown_change_seen": m["normal_seen_unknown"] - b["normal_seen_unknown"]}}


def threshold_free_paired(n_boot: int, boot_seed: int) -> dict:
    """Per run AUROC / OSCR (withheld vs fold-k known) for −ens_msp, ens_maha_rank (E1) and s (E2), paired against CNN
    Mahalanobis (seed-matched; E1 against each seed), with the 3C.1 bootstrap: one stratified mixed-unit draw over the
    development pool applied to all 27 runs (recording multiplicities); runs without withheld or known instances in a
    draw are dropped from that draw; macro = mean over classes of the mean over folds."""
    from ml.config import load_config
    from ml.evaluation import mixed_bootstrap_draws
    from ml.openset import weighted_auroc, weighted_oscr
    from ml.paths import DEFAULT_CONFIG

    cfg = load_config(DEFAULT_CONFIG)
    pool = pd.read_csv(results_dir() / cfg.development.folds_file).sort_values("rec_id").reset_index(drop=True)
    pos = pd.Series(np.arange(len(pool)), index=pool.rec_id)
    e1 = load("e1_ensemble", None)
    sets = {"e1_ens_msp": (e1, lambda d: -d.ens_msp.to_numpy()), "e1_ens_maha_rank": (e1, lambda d: d.ens_maha_rank.to_numpy())}
    for seed in SEEDS:
        e2 = load("e2_lr_cnn_rank", seed)
        sets[f"cnn_maha_s{seed}"] = (e2, lambda d: d.maha.to_numpy())
        sets[f"e2_s_s{seed}"] = (e2, lambda d: d.s.to_numpy())
    runs = {}
    for name, (df, f) in sets.items():
        ev = df[df.role.isin(["foldk_known", "foldk_withheld"])]
        runs[name] = {}
        for (c, k), g in ev.groupby(["withheld", "outer_fold"]):
            un = (g.role == "foldk_withheld").to_numpy()
            runs[name][(c, k)] = (f(g), un, (g.pred == g.class10).to_numpy(), pos.loc[g.rec_id].to_numpy())

    def metrics(name, counts):
        out = {}
        for key, (u, un, corr, ix) in runs[name].items():
            w = np.ones(len(u)) if counts is None else counts[ix].astype(float)
            if w[~un].sum() == 0 or w[un].sum() == 0:
                continue
            out[key] = (weighted_auroc(u[~un], w[~un], u[un], w[un]), weighted_oscr(u[~un], corr[~un], w[~un], u[un], w[un]))
        return out

    def macro_diff(ma, mb, j):
        d = pd.DataFrame([{"c": key[0], "v": ma[key][j] - mb[key][j]} for key in ma if key in mb])
        return float(d.groupby("c").v.mean().mean())

    pairs = [(e, f"cnn_maha_s{s_}") for e in ("e1_ens_msp", "e1_ens_maha_rank") for s_ in SEEDS] + \
            [(f"e2_s_s{s_}", f"cnn_maha_s{s_}") for s_ in SEEDS]
    base = {n: metrics(n, None) for n in runs}
    per_run = pd.DataFrame([{"score": n, "withheld": key[0], "outer_fold": key[1], "auroc": v[0], "oscr": v[1]}
                            for n, m in base.items() for key, v in m.items()])
    draws = mixed_bootstrap_draws(pool.class10.to_numpy(), pool.folder.to_numpy(), n_boot, boot_seed)
    boot = {(a, b, j): [] for a, b in pairs for j in (0, 1)}
    dropped = []
    for idx in draws:
        counts = np.bincount(idx, minlength=len(pool))
        mm = {n: metrics(n, counts) for n in runs}
        dropped.append(27 - len(mm["cnn_maha_s42"]))
        for a, b in pairs:
            for j in (0, 1):
                boot[(a, b, j)].append(macro_diff(mm[a], mm[b], j))
    res = {}
    for a, b in pairs:
        for j, m in ((0, "auroc"), (1, "oscr")):
            bb = np.array(boot[(a, b, j)])
            d_runs = np.array([base[a][key][j] - base[b][key][j] for key in base[a]])
            res[f"{a}_minus_{b}_{m}"] = {"macro_a": float(per_run[per_run.score == a].groupby("withheld")[m].mean().mean()),
                                         "macro_b": float(per_run[per_run.score == b].groupby("withheld")[m].mean().mean()),
                                         "diff_macro": macro_diff(base[a], base[b], j),
                                         "diff_mean_over_runs": float(d_runs.mean()), "diff_std_over_runs": float(d_runs.std(ddof=1)),
                                         "runs_a_higher": int((d_runs > 0).sum()),
                                         "ci95": [float(np.percentile(bb, 2.5)), float(np.percentile(bb, 97.5))]}
    return {"per_run": per_run, "comparisons": res,
            "bootstrap": {"n": n_boot, "runs_dropped_per_draw_mean": float(np.mean(dropped)),
                          "runs_dropped_per_draw_max": int(np.max(dropped))},
            "wording": "an interval excluding zero supports a specific metric difference under this protocol, not an "
                       "overall winner"}


def run_evaluate() -> None:
    from ml.config import load_config
    from ml.paths import DEFAULT_CONFIG

    cfg = load_config(DEFAULT_CONFIG)
    rows, sums = [], {}
    e1 = load("e1_ensemble", None)
    sums["E1"] = summary(e1, "outcome")
    rows.append({"method": "E1 (ensemble, once)", "seed": "42-43-44", **flat(sums["E1"])})
    per_seed = {}
    for seed in SEEDS:
        d3 = load("e3_b1s_veto", seed)
        d2 = load("e2_lr_cnn_rank", seed)
        for name, df, col in (("A", d3, "outcome_A"), ("B", d3, "outcome_B"), ("C", d3, "outcome_C"),
                              ("E2", d2, "outcome_e2"), ("D", d3, "outcome_D"), ("D′", d3, "outcome_Dp")):
            sm = summary(df, col)
            sums[f"{name}_s{seed}"] = sm
            per_seed[(name, seed)] = flat(sm)
            rows.append({"method": name, "seed": seed, **flat(sm)})
        caution = d3[(d3.decision_B == "healthy") & (d3.ae > d3.tau_e)]
        rows[-1]["_B_caution_flags"] = None
        sums[f"B_with_caution_s{seed}"] = {"caution_flags_instances": len(caution),
                                           "by_role": caution.groupby("role").size().to_dict(),
                                           "note": "B_with_caution makes identical decisions to B; count only"}
    tab = pd.DataFrame(rows).drop(columns=["_B_caution_flags"], errors="ignore")
    mean_rows = tab[tab.seed != "42-43-44"].groupby("method").mean(numeric_only=True).reset_index().assign(seed="mean")
    tab = pd.concat([tab, mean_rows], ignore_index=True)
    tab.to_csv(OUT / "evaluation_table.csv", index=False)
    crit = {}
    for name in ("E2", "D", "D′"):
        for seed in SEEDS:
            crit[f"{name}_s{seed}"] = criteria(per_seed[(name, seed)], per_seed[("B", seed)])
        m = mean_rows.set_index("method")
        crit[f"{name}_seed_mean"] = criteria(m.loc[name].to_dict(), m.loc["B"].to_dict())
    fl = flat(sums["E1"])
    crit["E1_vs_seed_mean_B"] = criteria(fl, mean_rows.set_index("method").loc["B"].to_dict())
    for seed in SEEDS:
        crit[f"E1_vs_B_s{seed}"] = criteria(fl, per_seed[("B", seed)])
    tf = threshold_free_paired(cfg.baselines.n_bootstrap, cfg.baselines.seed)
    tf["per_run"].to_csv(OUT / "threshold_free_per_run.csv", index=False)
    out = {"provenance": provenance(), "summaries": sums, "criteria": crit,
           "criteria_definition": "Problem 1: fewer withheld instances passed as known than rule B, with the known-fault "
                                  "false-unknown macro rates on (a) and (b) each at most 2 points above rule B's. Problem 2: "
                                  "fewer withheld instances called healthy than rule B, with the change in normal instances "
                                  "sent to unknown. Pass/fail only; no selection, no ratio or cost conclusion.",
           "threshold_free": {k: v for k, v in tf.items() if k != "per_run"}}
    (OUT / "evaluation.json").write_text(json.dumps(out, indent=1, default=float))
    pd.set_option("display.width", 250)
    show = ["method", "seed", "withheld_unknown", "withheld_known_fault", "withheld_healthy", "a_unknown_macro",
            "b_unknown_macro", "a_wrong_inst", "b_wrong_inst", "normal_foldk_healthy", "normal_seen_healthy"]
    print(tab[show].round(3).to_string(index=False))
    for k, v in crit.items():
        print(f"{k:22s} P1 {'PASS' if v['problem1_pass'] else 'fail'} {v['problem1_detail']} | P2 "
              f"{'PASS' if v['problem2_pass'] else 'fail'} {v['problem2_detail']}")
    for k, v in tf["comparisons"].items():
        print(f"{k:45s} {v['diff_macro']:+.3f} [{v['ci95'][0]:+.3f}, {v['ci95'][1]:+.3f}] runs a higher {v['runs_a_higher']}/27")
    print(tf["bootstrap"])
    print({k: v["caution_flags_instances"] for k, v in sums.items() if k.startswith("B_with_caution")})


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("task", choices=["e1", "e2", "e3", "evaluate"])
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    {"e1": run_e1, "e2": run_e2, "e3": run_e3, "evaluate": run_evaluate}[a.task]()
    return 0


if __name__ == "__main__":
    sys.exit(main())
