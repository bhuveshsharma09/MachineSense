"""Stage D1: nested learned unfamiliarity combiner (CLAUDE.md §7, declared before any training). Development data only;
F1 stays unrun; reserved tests sealed. Never imports LightGBM.

    caffeinate -i env MACHINESENSE_DEVICE=mps python scripts/31_d1_combiner.py inner      # GPU: 8 inner CNN-6 + LR per run
    python scripts/31_d1_combiner.py combine                                              # CPU: features, combiner, rule D1
    python scripts/31_d1_combiner.py evaluate                                             # CPU: tables, bootstrap, criteria

Per 3C run (withheld w, outer fold k, seed 42): F is split by folder into H1 / H2 (normal-folder exception: normal
recordings split by recording); for each fault class u ≠ w an inner CNN-6 and an inner LR are trained on H1 minus u and
score the H2 known rows (excluding u; target 0) and every u recording in F (target 1). Outer AE-6 / B1s supply those two
features for every row. Features are rank-normalised against the scoring model's held-out known reference (inner: its H2
known rows; outer: C). One L2 logistic combiner per run, fitted on the inner rows only; τ_s on C; rule D1.
Outputs: results/scores/d1_inner_w-<w>_fold<k>.parquet (inner rows; gitignored), results/scores/d1_decisions.parquet,
results/loco/d1/*.json|csv.
"""
from __future__ import annotations

import json
import sys
import time

import numpy as np
import pandas as pd

from ml.config import load_config
from ml.loco_eval import decide, group_table, lr_decisions, outcomes, recordings
from ml.openset import empirical_rank, loco_run_ids, mahalanobis_min, recording_mean, recording_probs, weighted_auroc, weighted_oscr
from ml.paths import DEFAULT_CONFIG, SYSTEM_CODE_DIR, RunPaths, artifacts_dir, data_dir, results_dir
from ml.protocol import order_statistic_threshold
from ml.splits import FAULT_CLASSES, development_pool, split_hash
from ml.utils import git_sha

SEED = 42
VIB = [0, 1, 2, 3, 4, 5]
ACCEPT = 0.975
OUT = results_dir() / "loco" / "d1"
SCORES = results_dir() / "scores"
ASSIGNMENT = results_dir() / "q1q3" / "q3_fit_cal_eval_assignment.csv"
FEATURES = ["cnn_1msp", "cnn_maha", "lr_1msp", "ae", "b1s", "cnn_gap", "cnn_lr_differ", "speed"]
CONTINUOUS = [f for f in FEATURES if f != "cnn_lr_differ"]
GUARD_HOURS = 3.0
WORDING = ("Exploratory development results, seed 42 only; the candidate was designed after seeing the 3D and ML-5 results. "
           "A fail means this combiner did not improve these results, not that the representation is the proven limit. "
           "Nothing enters F1 unless Bhuvesh amends §12 before F1.")


def setup():
    cfg = load_config(DEFAULT_CONFIG)
    meta = pd.read_csv(SYSTEM_CODE_DIR / "metadata.csv")
    dc = cfg.development
    pool = development_pool(meta, dc.allowed_closed, dc.reserved_columns).merge(
        pd.read_csv(results_dir() / dc.folds_file)[["rec_id", "dev_fold"]], on="rec_id")
    return cfg, meta, pool, pd.read_csv(ASSIGNMENT)


def halves(F: pd.DataFrame, seed: int = SEED) -> tuple[list[str], list[str]]:
    """Folder-grouped halves of the fit set: per class, folders in a seeded random order assigned alternately; normal (a
    single folder) is split by recording the same way (the normal-folder exception)."""
    rng = np.random.default_rng(seed)
    h1, h2 = [], []
    for cls, g in sorted(F.groupby("class10")):
        units = sorted(g.rec_id) if cls == "normal" else sorted(g.folder.unique())
        units = [units[i] for i in rng.permutation(len(units))]
        for j, unit in enumerate(units):
            ids = [unit] if cls == "normal" else sorted(g.rec_id[g.folder == unit])
            (h1 if j % 2 == 0 else h2).extend(ids)
    return sorted(h1), sorted(h2)


def prob_rows(proba: np.ndarray, rec_ids, classes, floor: float) -> pd.DataFrame:
    """Recording probabilities from window probabilities (clip, mean log-probability, renormalise; the 3A rule)."""
    lp = pd.DataFrame(np.log(np.clip(proba, floor, 1.0)), columns=list(classes)).assign(rec_id=list(rec_ids))
    lp = lp.groupby("rec_id", sort=False).mean()
    p = np.exp(lp.to_numpy())
    return pd.DataFrame(p / p.sum(axis=1, keepdims=True), index=lp.index, columns=list(classes))


def summarise_probs(P: pd.DataFrame) -> pd.DataFrame:
    a = np.sort(P.to_numpy(), axis=1)
    return pd.DataFrame({"msp": a[:, -1], "gap": a[:, -1] - a[:, -2], "top": P.columns[P.to_numpy().argmax(axis=1)]},
                        index=P.index)


def inner_path(c: str, k: int):
    return SCORES / f"d1_inner_w-{c}_fold{k}.parquet"


# ----------------------------------------------------------------------------------------------------------- inner
def run_inner() -> None:
    import torch

    from ml.cache import load_features
    from ml.classical import logreg_pipeline
    from ml.config import AeConfig
    from ml.datasets import NormContext, dev_window_rows, load_representation, norm_stats
    from ml.features import feature_columns
    from ml.labels import CLASS10
    from ml.models.ae import build_ae
    from ml.train import gpu_job, release_device_memory, score_windows, train_model
    from ml.train_ae import window_channel_errors
    from ml.utils import get_device, seed_everything

    cfg, meta, pool, assignment = setup()
    device = get_device()
    cc = cfg.cnn.model_copy(update={"seed": SEED})
    floor = cfg.baselines.prob_floor
    arr, index = load_representation(data_dir() / "representations", cfg, meta)
    rows = dev_window_rows(index, pool)
    path = max((data_dir() / "features").glob(f"{cfg.baselines.feature_table}_*.parquet"), key=lambda p: p.stat().st_mtime)
    df = load_features(path, meta, cfg.preprocess).merge(pool[["rec_id"]], on="rec_id").sort_values(["rec_id", "window_idx"])
    cols = feature_columns(df)
    info = pool.set_index("rec_id")
    todo = [(c, k) for c in FAULT_CLASSES for k in (0, 1, 2) if not inner_path(c, k).exists()]
    print(f"{len(todo)} runs to do", flush=True)
    if not todo:
        return
    OUT.mkdir(parents=True, exist_ok=True)
    timing = json.loads((OUT / "inner_timing.json").read_text()) if (OUT / "inner_timing.json").exists() else {"runs": {}}
    with gpu_job(device, artifacts_dir(), cc.mps_memory_fraction):
        pos = rows.row.to_numpy()
        order = np.argsort(pos)
        x = np.empty((len(pos), *arr.shape[1:]), dtype=np.float16)
        x[order] = arr[pos[order]]
        cache = torch.from_numpy(x).to(device)  # development-pool windows only
        del x
        for c, k in todo:
            t_run = time.perf_counter()
            ids = loco_run_ids(assignment, pool, c, k)
            F = pool[pool.rec_id.isin(ids["fit"])]
            assert c not in set(F.class10)
            h1, h2 = halves(F)
            parts = []
            gpu_s = 0.0
            for u in [x_ for x_ in FAULT_CLASSES if x_ != c]:
                train_ids = sorted(set(h1) - set(F.rec_id[F.class10 == u]))
                known_rows = sorted(set(h2) - set(F.rec_id[F.class10 == u]))
                unfam = sorted(F.rec_id[F.class10 == u])
                classes = [x_ for x_ in CLASS10 if x_ not in (c, u)]
                present = set(info.class10.loc[train_ids])
                assert present == set(classes), f"{c}/{k}/{u}: H1 lacks {set(classes) - present}"
                y = torch.from_numpy(rows.class10.map({x_: i for i, x_ in enumerate(classes)}).fillna(0).to_numpy(np.int64)).to(device)
                st = norm_stats(arr, index, meta, fit_ids=train_ids,
                                context=NormContext(evaluation=frozenset(known_rows) | frozenset(unfam)))
                mean, std = st["mean"][VIB], st["std"][VIB]
                tr_rows = np.flatnonzero(rows.rec_id.isin(train_ids).to_numpy())
                sc_ids = known_rows + unfam
                sc_rows = np.flatnonzero(rows.rec_id.isin(sc_ids).to_numpy())
                seed_everything(SEED)
                t0 = time.perf_counter()
                cnn, _ = train_model(cache, y, tr_rows, VIB, mean, std, len(classes), cc, device)
                _, emb_tr = score_windows(cnn, cache, tr_rows, VIB, mean, std, cc, device)
                logits, emb = score_windows(cnn, cache, sc_rows, VIB, mean, std, cc, device)
                if device.type == "mps":
                    torch.mps.synchronize()
                gpu_s += time.perf_counter() - t0
                del cnn
                release_device_memory(device)
                sc_rec = rows.rec_id.to_numpy()[sc_rows]
                maha_w, _ = mahalanobis_min(emb_tr.astype(np.float64), rows.class10.to_numpy()[tr_rows], emb.astype(np.float64))
                pc = summarise_probs(recording_probs(logits, sc_rec, classes, floor))
                maha = recording_mean(maha_w, sc_rec)
                # inner LR on the same H1-minus-u windows (scikit-learn only)
                dtr, dsc = df[df.rec_id.isin(train_ids)], df[df.rec_id.isin(sc_ids)]
                lr = logreg_pipeline(cfg.baselines).fit(dtr[cols].to_numpy(), dtr.class10.to_numpy())
                pl = summarise_probs(prob_rows(lr.predict_proba(dsc[cols].to_numpy()), dsc.rec_id, lr.classes_, floor))
                r = pd.DataFrame({"rec_id": sc_ids}).set_index("rec_id")
                r["cnn_1msp"] = 1 - pc.msp.loc[r.index]
                r["cnn_gap"] = pc.gap.loc[r.index]
                r["cnn_top"] = pc.top.loc[r.index]
                r["cnn_maha"] = maha.loc[r.index]
                r["lr_1msp"] = 1 - pl.msp.loc[r.index]
                r["lr_top"] = pl.top.loc[r.index]
                r["target"] = [0] * len(known_rows) + [1] * len(unfam)
                r["half"] = ["H2" if x_ in set(h2) else "H1" for x_ in r.index]
                parts.append(r.reset_index().assign(u=u, withheld=c, outer_fold=k, class10=info.class10.loc[sc_ids].to_numpy(),
                                                    speed=info.fr_measured_hz.loc[sc_ids].to_numpy()))
            inner = pd.concat(parts, ignore_index=True)
            # outer AE-6 of the run on every F recording (the AE's own fit normals are in-sample; declared)
            ra = RunPaths("loco", "ae6", SEED, protocol="loco", withheld=c)
            na = json.loads(ra.norm_stats(k).read_text())
            ae = build_ae("frame", len(VIB), cfg.ae)
            ae.load_state_dict(torch.load(ra.weights(k), map_location=device, weights_only=True))
            ae.to(device)
            f_rows = np.flatnonzero(rows.rec_id.isin(F.rec_id).to_numpy())
            err = window_channel_errors(ae, cache, f_rows, VIB, np.asarray(na["mean"]), np.asarray(na["std"]), "frame",
                                        AeConfig(**cfg.ae.model_dump()), device).mean(axis=1)
            ae_rec = recording_mean(err, rows.rec_id.to_numpy()[f_rows])
            inner["ae"] = ae_rec.loc[inner.rec_id].to_numpy()
            inner.to_parquet(inner_path(c, k), index=False)
            del ae
            release_device_memory(device)
            run_s = time.perf_counter() - t_run
            timing["runs"][f"{c}/{k}"] = {"gpu_seconds": gpu_s, "run_seconds": run_s, "h1": len(h1), "h2": len(h2)}
            done = len(timing["runs"])
            est_h = run_s * 27 / 3600
            timing["estimate_hours_after_first_run"] = timing.get("estimate_hours_after_first_run", est_h)
            (OUT / "inner_timing.json").write_text(json.dumps(timing, indent=1))
            print(f"  {c}/{k}: H1 {len(h1)} / H2 {len(h2)} rec, {len(inner)} inner rows, GPU {gpu_s:.0f} s, run {run_s:.0f} s "
                  f"({done}/27)", flush=True)
            if done == 1 and est_h > GUARD_HOURS:
                print(f"TIME GUARD: estimated {est_h:.1f} h > {GUARD_HOURS} h; stopping after the first run.")
                raise SystemExit(3)


# ----------------------------------------------------------------------------------------------------------- combine
def b1s_rows(cfg, meta, pool, assignment) -> pd.DataFrame:
    """B1s refitted per run on the run's AE fit normals exactly as Stage 3D (vibration-only capped R1), scored on every
    F recording and every evaluation role."""
    from ml.cache import load_features
    from ml.features import feature_columns
    from ml.q1 import classical_scores

    path = max((data_dir() / "features").glob(f"{cfg.baselines.feature_table}_*.parquet"), key=lambda p: p.stat().st_mtime)
    df = load_features(path, meta, cfg.preprocess).merge(pool[["rec_id", "folder", "dev_fold"]], on="rec_id")
    df = df.sort_values(["rec_id", "window_idx"]).reset_index(drop=True)
    cols = [c for c in feature_columns(df) if not c.startswith("f_mic_")]
    normals = set(pool.loc[pool.class10 == "normal", "rec_id"])
    out = []
    for c in FAULT_CLASSES:
        for k in (0, 1, 2):
            ids = loco_run_ids(assignment, pool, c, k)
            fit_n = sorted(set(ids["fit"]) & normals)
            saved = json.loads(RunPaths("loco", "ae6", SEED, protocol="loco", withheld=c).norm_stats(k).read_text())["fit_ids"]
            assert fit_n == sorted(saved)
            target = df[df.rec_id.isin(set(ids["fit"]) | set(ids["cal"]) | set(ids["eval"]) | set(ids["foldk_known"])
                                       | set(ids["foldk_withheld"]))]
            sc = classical_scores("B1s_speed_poly", cfg.anomaly, df[df.rec_id.isin(fit_n)], target, cols)
            out.append(pd.DataFrame({"rec_id": sc.index, "b1s": sc.to_numpy(), "withheld": c, "outer_fold": k}))
    return pd.concat(out, ignore_index=True)


def rank_features(rows: pd.DataFrame, ref: pd.DataFrame) -> pd.DataFrame:
    X = pd.DataFrame(index=rows.index)
    for f in CONTINUOUS:
        X[f] = empirical_rank(rows[f].to_numpy(), ref[f].to_numpy())
    X["cnn_lr_differ"] = rows.cnn_lr_differ.astype(float).to_numpy()
    return X[FEATURES]


def run_combine() -> None:
    from sklearn.linear_model import LogisticRegression

    cfg, meta, pool, assignment = setup()
    floor = cfg.baselines.prob_floor
    info = pool.set_index("rec_id")
    dec = decide(recordings(SEED))
    lr = lr_decisions().set_index(["withheld", "outer_fold", "rec_id"])
    b1 = b1s_rows(cfg, meta, pool, assignment).set_index(["withheld", "outer_fold", "rec_id"]).b1s
    e3 = pd.read_parquet(SCORES / "e3_b1s_veto_s42_loco_decisions.parquet").set_index(["withheld", "outer_fold", "rec_id"])
    parts, coefs, thr_rows = [], [], []
    for (c, k), g in dec.groupby(["withheld", "outer_fold"]):
        # ---- inner rows → combiner
        inner = pd.read_parquet(inner_path(c, k))
        inner["b1s"] = b1.loc[[(c, k, r) for r in inner.rec_id]].to_numpy()
        inner["cnn_lr_differ"] = (inner.cnn_top != inner.lr_top).astype(int)
        Xs, ys = [], []
        for u, h in inner.groupby("u"):
            ref = h[h.target == 0]
            Xs.append(rank_features(h, ref))
            ys.append(h.target.to_numpy())
        X, y = pd.concat(Xs), np.concatenate(ys)
        model = LogisticRegression(C=1.0).fit(X.to_numpy(), y)
        coefs.append({"withheld": c, "outer_fold": int(k), **dict(zip(FEATURES, model.coef_[0])),
                      "intercept": float(model.intercept_[0]), "n_iter": int(model.n_iter_[0]),
                      "n_inner_known": int((y == 0).sum()), "n_inner_unfamiliar": int((y == 1).sum())})
        # ---- outer rows (cal, eval, fold-k) with the 3C CNN, the 2D.1 LR, the run's AE and B1s
        g = g.copy()
        w = pd.read_parquet(RunPaths("loco", "cnn6", SEED, protocol="loco", withheld=c).scores(f"fold{k}_logits"))
        w = w[w.role != "fit"]
        classes = [x.removeprefix("logit_") for x in w.columns if x.startswith("logit_")]
        pc = summarise_probs(recording_probs(w[[f"logit_{x}" for x in classes]].to_numpy(), w.rec_id, classes, floor))
        key = [(c, k, r) for r in g.rec_id]
        g["cnn_1msp"] = 1 - g.msp.to_numpy()
        assert np.allclose(pc.msp.loc[g.rec_id].to_numpy(), g.msp.to_numpy())
        g["cnn_gap"] = pc.gap.loc[g.rec_id].to_numpy()
        g["cnn_maha"] = g.maha.to_numpy()
        g["lr_1msp"] = 1 - lr.loc[key, "msp"].to_numpy()
        g["lr_top"] = lr.loc[key, "pred"].to_numpy()
        g["cnn_lr_differ"] = (g.pred.to_numpy() != g.lr_top.to_numpy()).astype(int)
        g["b1s"] = b1.loc[key].to_numpy()
        assert np.allclose(g.b1s.to_numpy(), e3.loc[key, "b1s"].to_numpy()), "B1s does not reproduce Stage 3D"
        g["tau_b"] = e3.loc[key, "tau_b"].to_numpy()
        g["speed"] = info.fr_measured_hz.loc[g.rec_id].to_numpy()
        cal = g[g.role == "cal"]
        Xo = rank_features(g, cal)
        g["s"] = model.predict_proba(Xo.to_numpy())[:, 1]
        t = order_statistic_threshold(g.s[g.role == "cal"].to_numpy(), ACCEPT)
        ts = float(t["threshold"])
        ok = (g.s <= ts).to_numpy()
        normal = (g.pred == "normal").to_numpy()
        d = np.where(ok & ~normal, g.pred, np.where(ok & normal, "healthy", "unknown")).astype(object)
        g["decision_D1"] = d
        g["outcome_D1"] = outcomes(d, g)
        g["tau_s"] = ts
        hv = d == "healthy"
        g["caution_D1"] = np.where(hv & (g.ae > g.tau_e) & (g.b1s > g.tau_b), "both",
                                   np.where(hv & (g.ae > g.tau_e), "ae_disagrees", np.where(hv & (g.b1s > g.tau_b), "b1s_disagrees", None)))
        hb = (g.decision_B == "healthy").to_numpy()
        g["caution_B"] = np.where(hb & (g.ae > g.tau_e) & (g.b1s > g.tau_b), "both",
                                  np.where(hb & (g.ae > g.tau_e), "ae_disagrees", np.where(hb & (g.b1s > g.tau_b), "b1s_disagrees", None)))
        thr_rows.append({"withheld": c, "outer_fold": int(k), "tau_s": ts, "n_cal": int(t["n"]), "r": int(t["r"]),
                         "cal_acceptance": float(ok[(g.role == "cal").to_numpy()].mean())})
        parts.append(g)
    out = pd.concat(parts, ignore_index=True)
    out.to_parquet(SCORES / "d1_decisions.parquet", index=False)
    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(coefs).to_csv(OUT / "coefficients_per_run.csv", index=False)
    pd.DataFrame(thr_rows).to_csv(OUT / "tau_s_per_run.csv", index=False)
    print(pd.DataFrame(coefs)[["withheld", "outer_fold", *FEATURES, "n_iter"]].round(2).to_string())
    print(pd.DataFrame(thr_rows).describe().round(3).to_string())


# ----------------------------------------------------------------------------------------------------------- evaluate
def per_run(dec: pd.DataFrame, weights: pd.Series | None = None) -> pd.DataFrame:
    rows = []
    for (c, k), g in dec.groupby(["withheld", "outer_fold"]):
        kn = (g.role == "foldk_known").to_numpy()
        un = (g.role == "foldk_withheld").to_numpy()
        w = np.ones(len(g)) if weights is None else weights.reindex(g.rec_id).fillna(0).to_numpy()
        if w[kn].sum() == 0 or w[un].sum() == 0:
            continue
        correct = (g.pred == g.class10).to_numpy()
        r = {"withheld": c, "outer_fold": int(k)}
        for name, u in (("combiner", g.s.to_numpy()), ("cnn_maha", g.maha.to_numpy())):
            r[f"{name}_auroc"] = weighted_auroc(u[kn], w[kn], u[un], w[un])
            r[f"{name}_oscr"] = weighted_oscr(u[kn], correct[kn], w[kn], u[un], w[un])
        rows.append(r)
    return pd.DataFrame(rows)


def macro(df: pd.DataFrame, col: str) -> float:
    return float(df.groupby("withheld")[col].mean().mean())


def run_evaluate() -> None:
    from ml.evaluation import mixed_bootstrap_draws

    cfg, meta, pool, _ = setup()
    dec = pd.read_parquet(SCORES / "d1_decisions.parquet")
    ev = dec[dec.role != "cal"]
    pr = per_run(ev)
    for m in ("auroc", "oscr"):
        pr[f"diff_{m}"] = pr[f"combiner_{m}"] - pr[f"cnn_maha_{m}"]
    pr.to_csv(OUT / "threshold_free_per_run.csv", index=False)
    point = {m: {"combiner_macro": macro(pr, f"combiner_{m}"), "cnn_maha_macro": macro(pr, f"cnn_maha_{m}"),
                 "diff_macro": macro(pr, f"diff_{m}"), "runs_combiner_higher": int((pr[f"diff_{m}"] > 0).sum()), "runs": len(pr),
                 "per_class": pr.groupby("withheld")[[f"combiner_{m}", f"cnn_maha_{m}"]].mean().round(4).to_dict(orient="index")}
             for m in ("auroc", "oscr")}
    folds = pd.read_csv(results_dir() / cfg.development.folds_file).sort_values("rec_id").reset_index(drop=True)
    draws = mixed_bootstrap_draws(folds.class10.to_numpy(), folds.folder.to_numpy(), cfg.baselines.n_bootstrap, cfg.baselines.seed)
    boot, dropped = {m: [] for m in ("auroc", "oscr")}, []
    for idx in draws:
        wts = pd.Series(np.bincount(idx, minlength=len(folds)).astype(float), index=folds.rec_id)
        b = per_run(ev, wts)
        dropped.append(27 - len(b))
        for m in ("auroc", "oscr"):
            boot[m].append(macro(b.assign(d=b[f"combiner_{m}"] - b[f"cnn_maha_{m}"]), "d"))
    for m in ("auroc", "oscr"):
        a = np.array(boot[m])
        point[m]["paired_bootstrap_ci95_diff"] = [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))]
        point[m]["p_combiner_higher"] = float(np.mean(a > 0))
    point["bootstrap"] = {"n": len(draws), "runs_dropped_per_draw_mean": float(np.mean(dropped)),
                          "runs_dropped_per_draw_max": int(np.max(dropped)),
                          "scheme": "Stage 3C.1: one stratified mixed-unit draw over the whole development pool weights every "
                                    "run's recordings; runs lacking fold-k known or withheld recordings in a draw are dropped "
                                    "(so draws average different run sets; caveat of 3C.1)"}
    tables = {"D1": group_table(ev, "outcome_D1"), "B (= B_with_caution decisions)": group_table(ev, "outcome_B")}
    cautions = {name: {"instances": int(ev[col].notna().sum()), "by_role": ev[ev[col].notna()].groupby("role").size().to_dict(),
                       "by_kind": ev[col].value_counts().to_dict()} for name, col in (("D1", "caution_D1"), ("B_with_caution", "caution_B"))}
    wd, wb = tables["D1"]["withheld"], tables["B (= B_with_caution decisions)"]["withheld"]
    u_known = (wd["pooled"]["known_fault"]["instances"], wb["pooled"]["known_fault"]["instances"])
    fu = {g: (tables["D1"][g]["pooled"]["unknown"]["rate"], tables["B (= B_with_caution decisions)"][g]["pooled"]["unknown"]["rate"])
          for g in ("known_a_seen_folder", "known_b_unseen_foldk")}
    crit1 = point["auroc"]["diff_macro"] > 0 and point["auroc"]["paired_bootstrap_ci95_diff"][0] > 0
    crit2 = u_known[0] < u_known[1] and all(d <= b + 0.02 for d, b in fu.values())
    criteria = {"i_macro_auroc_combiner_gt_maha_and_ci_lower_gt_0": {"pass": bool(crit1), "diff_macro": point["auroc"]["diff_macro"],
                                                                   "ci95": point["auroc"]["paired_bootstrap_ci95_diff"]},
                "ii_fewer_u_passed_as_known_and_false_unknown_within_2_points": {
                    "pass": bool(crit2), "u_passed_as_known_D1_vs_B": list(u_known),
                    "false_unknown_rate_D1_vs_B": {g: list(v) for g, v in fu.items()}},
                "overall_pass": bool(crit1 and crit2)}
    coef = pd.read_csv(OUT / "coefficients_per_run.csv")
    coef_summary = {"mean_over_runs": coef[FEATURES].mean().round(3).to_dict(), "std_over_runs": coef[FEATURES].std().round(3).to_dict(),
                    "runs_positive": {f: int((coef[f] > 0).sum()) for f in FEATURES},
                    "per_withheld_class_mean": coef.groupby("withheld")[FEATURES].mean().round(3).to_dict(orient="index"),
                    "n_iter_max": int(coef.n_iter.max())}
    timing = json.loads((OUT / "inner_timing.json").read_text())
    out = {"wording": WORDING, "provenance": {"split_hash": split_hash(meta), "git_sha": git_sha(), "seed": SEED,
                                              "features": FEATURES, "dropped_feature": "signal-quality warning flag (§3)"},
           "threshold_free": point, "outcomes": tables, "cautions": cautions, "criteria": criteria,
           "coefficients": coef_summary,
           "gpu": {"total_gpu_seconds": sum(r["gpu_seconds"] for r in timing["runs"].values()),
                   "total_run_seconds": sum(r["run_seconds"] for r in timing["runs"].values()),
                   "estimate_hours_after_first_run": timing.get("estimate_hours_after_first_run")}}
    (OUT / "evaluation.json").write_text(json.dumps(out, indent=1, default=float))
    print(json.dumps({"threshold_free": {m: {k: v for k, v in point[m].items() if k != "per_class"} for m in ("auroc", "oscr")},
                      "criteria": criteria, "gpu": out["gpu"]}, indent=1, default=float))


def main() -> int:
    task = sys.argv[1] if len(sys.argv) > 1 else ""
    if task not in ("inner", "combine", "evaluate"):
        print(__doc__)
        return 2
    {"inner": run_inner, "combine": run_combine, "evaluate": run_evaluate}[task]()
    return 0


if __name__ == "__main__":
    sys.exit(main())
