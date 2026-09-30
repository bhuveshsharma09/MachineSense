"""Stage R4: condition-specialist autoencoder bank for unfamiliar-fault detection (Q3, development data, seed 42).
Never imports LightGBM. Protocol: CLAUDE.md §7 (Stage R4), declared before any training.

    caffeinate -i env MACHINESENSE_DEVICE=mps python scripts/34_train_specialists.py train     # 27 runs; resumable
    python scripts/34_train_specialists.py evaluate                                           # tables vs Mahalanobis

Per run (withheld w, outer fold k; the saved Stage 3C assignment): eight fault specialists (every fault class except w)
are trained with the AE-6 recipe on the run's fit recordings of their class only, normalised on exactly those recordings;
the normal specialist is the saved Stage 3C AE-6 of the run. Every calibration and evaluation recording of the run is
scored by all nine. Outputs: results/scores/specialists_w-<w>_s42_loco_fold<k>_errors.parquet (recording errors per
specialist), results/specialists/w-<w>/fold<k>_<class>_norm.json and fold<k>_train.json, weights
artifacts/specialists/w-<w>_s42/fold<k>/<class>.pt (state_dict; gitignored); evaluation in results/research/r4_*.
"""
from __future__ import annotations

import argparse
import json
import sys
import time

import numpy as np
import pandas as pd

from ml.config import load_config
from ml.openset import loco_run_ids, oscr, recording_mean
from ml.paths import DEFAULT_CONFIG, SYSTEM_CODE_DIR, RunPaths, artifacts_dir, data_dir, results_dir
from ml.splits import FAULT_CLASSES, development_pool, split_hash
from ml.utils import git_sha

SEED = 42
VIB = [0, 1, 2, 3, 4, 5]
ASSIGNMENT = results_dir() / "q1q3" / "q3_fit_cal_eval_assignment.csv"
ROLES = ("cal", "eval", "foldk_known", "foldk_withheld")
OUT = results_dir() / "research"
SPEC_RES = results_dir() / "specialists"
GUARD_MIN = 120.0
AE_REL_TOL = 1e-5
SPEED_BANDS = [(0, 20, "<20 Hz"), (20, 30, "20-30 Hz"), (30, 99, ">=30 Hz")]
WORDING = ("Exploratory development results, seed 42 only (27 LOCO runs of Stage 3C); the design and its reading were "
           "declared before training. Every fault class has informed earlier development decisions; reserved tests stay "
           "sealed.")


def errors_path(c: str, k: int):
    return results_dir() / "scores" / f"specialists_w-{c}_s{SEED}_loco_fold{k}_errors.parquet"


def weights_path(c: str, k: int, u: str):
    return artifacts_dir() / "specialists" / f"w-{c}_s{SEED}" / f"fold{k}" / f"{u}.pt"


def setup():
    cfg = load_config(DEFAULT_CONFIG)
    meta = pd.read_csv(SYSTEM_CODE_DIR / "metadata.csv")
    dc = cfg.development
    pool = development_pool(meta, dc.allowed_closed, dc.reserved_columns).merge(
        pd.read_csv(results_dir() / dc.folds_file)[["rec_id", "dev_fold"]], on="rec_id")
    return cfg, meta, pool, pd.read_csv(ASSIGNMENT)


# ----------------------------------------------------------------------------------------------------------- train
def run_train() -> None:
    import torch

    from ml.config import AeConfig
    from ml.datasets import dev_window_rows, load_representation, norm_stats
    from ml.models.ae import build_ae
    from ml.specialists import NORMAL, run_plan
    from ml.train import gpu_job, release_device_memory
    from ml.train_ae import train_ae, window_channel_errors
    from ml.utils import get_device, seed_everything

    cfg, meta, pool, assignment = setup()
    ac = AeConfig(**cfg.ae.model_dump())
    device = get_device()
    arr, index = load_representation(data_dir() / "representations", cfg, meta)
    rows = dev_window_rows(index, pool)
    rec_of = rows.rec_id.to_numpy()
    saved_ae = pd.read_parquet(results_dir() / "scores" / "cnn6ae6_s42_loco_decisions.parquet").set_index(
        ["withheld", "outer_fold", "rec_id"]).ae
    tau_e = pd.read_csv(results_dir() / "loco" / "decision_thresholds.csv").set_index(["withheld", "outer_fold"]).tau_e
    todo = [(c, k) for c in FAULT_CLASSES for k in (0, 1, 2) if not errors_path(c, k).exists()]
    print(f"{len(todo)} runs to do", flush=True)
    if not todo:
        return
    t_start = time.perf_counter()
    with gpu_job(device, artifacts_dir(), ac.mps_memory_fraction):
        pos = rows.row.to_numpy()
        order = np.argsort(pos)
        x = np.empty((len(pos), *arr.shape[1:]), dtype=np.float16)
        x[order] = arr[pos[order]]
        cache = torch.from_numpy(x).to(device)  # development-pool windows only
        del x
        for n_done, (c, k) in enumerate(todo):
            t_run = time.perf_counter()
            plan = run_plan(assignment, pool, c, k)
            ids = plan["ids"]
            role_of = {r: role for role in ROLES for r in ids[role]}
            scored = sorted(role_of)
            sel = np.flatnonzero(np.isin(rec_of, scored))
            errs, info = {}, {}
            (SPEC_RES / f"w-{c}").mkdir(parents=True, exist_ok=True)
            for u, sp in plan["specialists"].items():
                if u == NORMAL:  # the saved Stage 3C AE-6 of this run (declared)
                    ra = RunPaths("loco", "ae6", SEED, protocol="loco", withheld=c)
                    na = json.loads(ra.norm_stats(k).read_text())
                    assert sorted(na["fit_ids"]) == sp["fit"], f"{c}/{k}: 3C AE-6 fit normals differ from the plan"
                    mean, std = np.asarray(na["mean"]), np.asarray(na["std"])
                    model = build_ae("frame", len(VIB), ac)
                    model.load_state_dict(torch.load(ra.weights(k), map_location=device, weights_only=True))
                    model.to(device)
                    secs = 0.0
                else:
                    st = norm_stats(arr, index, meta, fit_ids=sp["fit"], context=ids["context"])
                    mean, std = st["mean"][VIB], st["std"][VIB]
                    (SPEC_RES / f"w-{c}" / f"fold{k}_{u}_norm.json").write_text(json.dumps(
                        {"withheld": c, "fold": k, "specialist": u, "fit_ids": sp["fit"], "n_windows": st["n_windows"],
                         "mean": mean.tolist(), "std": std.tolist()}))
                    fit_rows = np.flatnonzero(np.isin(rec_of, sp["fit"]))
                    seed_everything(SEED)
                    t0 = time.perf_counter()
                    model, hist = train_ae(cache, fit_rows, VIB, mean, std, "frame", ac, device)
                    if device.type == "mps":
                        torch.mps.synchronize()
                    secs = time.perf_counter() - t0
                    weights_path(c, k, u).parent.mkdir(parents=True, exist_ok=True)
                    torch.save(model.state_dict(), weights_path(c, k, u))
                e = window_channel_errors(model, cache, sel, VIB, mean, std, "frame", ac, device).mean(axis=1)
                errs[u] = recording_mean(e, rec_of[sel]).loc[scored]
                info[u] = {"n_fit_recordings": len(sp["fit"]), "n_cal_recordings": len(sp["cal"]), "train_seconds": secs,
                           "reused_3c_ae6": u == NORMAL}
                del model
            ref = saved_ae.loc[(c, k)].loc[scored]
            diff = float(np.max(np.abs(errs[NORMAL].to_numpy() - ref.to_numpy())))
            ok = diff <= AE_REL_TOL * float(tau_e.loc[(c, k)])
            df = pd.DataFrame({f"e_{u}": v.to_numpy() for u, v in errs.items()}, index=scored).rename_axis("rec_id").reset_index()
            df["role"] = df.rec_id.map(role_of)
            df["class10"] = pool.set_index("rec_id").class10.loc[df.rec_id].to_numpy()
            df["withheld"], df["outer_fold"] = c, k
            df.to_parquet(errors_path(c, k), index=False)
            run_s = time.perf_counter() - t_run
            (SPEC_RES / f"w-{c}" / f"fold{k}_train.json").write_text(json.dumps(
                {"withheld": c, "fold": k, "specialists": info, "run_seconds": run_s, "normal_reproduction_max_abs_diff": diff,
                 "normal_reproduced": bool(ok), "device": str(device), "git_sha": git_sha(), "split_hash": split_hash(meta)},
                indent=1))
            print(f"  {c}/{k}: {len(plan['specialists'])} specialists, train {sum(v['train_seconds'] for v in info.values()):.0f} s, "
                  f"run {run_s:.0f} s; normal reproduction max |diff| {diff:.1e} {'ok' if ok else 'MISMATCH'}", flush=True)
            if not ok:
                raise SystemExit(f"{c}/{k}: the reused AE-6 does not reproduce its saved errors; stopping")
            release_device_memory(device)
            if n_done == 0:
                proj = (time.perf_counter() - t_start) * len(todo) / 60
                print(f"  TIME ESTIMATE after the first run: {proj:.0f} min for {len(todo)} runs (guard {GUARD_MIN:.0f})", flush=True)
                OUT.mkdir(parents=True, exist_ok=True)
                (OUT / "r4_time_estimate.json").write_text(json.dumps({"first_run_seconds": time.perf_counter() - t_start,
                                                                       "runs": len(todo), "projected_minutes": proj,
                                                                       "guard_minutes": GUARD_MIN}, indent=1))
                if proj > GUARD_MIN:
                    raise SystemExit(f"TIME GUARD: projected {proj:.0f} min > {GUARD_MIN:.0f} min; stopping (§10)")
    print(f"done in {(time.perf_counter() - t_start) / 60:.1f} min")


# ----------------------------------------------------------------------------------------------------------- evaluate
def level_per_recording(cfg, meta, pool) -> pd.Series:
    """Mean cached log-spectral level (vibration channels, before normalisation) per development recording."""
    from ml.datasets import dev_window_rows, load_representation

    arr, index = load_representation(data_dir() / "representations", cfg, meta)
    rows = dev_window_rows(index, pool)
    pos = rows.row.to_numpy()
    order = np.argsort(pos)
    lv = np.empty(len(pos))
    for i in range(0, len(pos), 512):
        p = pos[order[i:i + 512]]
        lv[order[i:i + 512]] = np.asarray(arr[p][:, VIB], dtype=np.float64).mean(axis=(1, 2, 3))
    return pd.Series(lv, index=rows.rec_id.to_numpy()).groupby(level=0).mean()


def latency_and_size(cfg) -> dict:
    """CPU ms per recording (9 windows from the cache): the 9-specialist bank vs CNN-6 + Mahalanobis (shared)."""
    import torch

    from ml.config import AeConfig
    from ml.models.ae import build_ae
    from ml.models.cnn import ScratchCNN, n_parameters
    from ml.train import normalise
    from ml.train_ae import normalise_any

    ac = AeConfig(**cfg.ae.model_dump())
    c, k = "imbalance", 0
    aes = []
    for u in sorted(set(json.loads((SPEC_RES / f"w-{c}" / f"fold{k}_train.json").read_text())["specialists"])):
        m = build_ae("frame", len(VIB), ac)
        p = weights_path(c, k, u) if u != "normal" else RunPaths("loco", "ae6", SEED, protocol="loco", withheld=c).weights(k)
        m.load_state_dict(torch.load(p, map_location="cpu", weights_only=True))
        aes.append(m.eval())
    cnn = ScratchCNN(len(VIB), 9, cfg.cnn)
    cnn.load_state_dict(torch.load(RunPaths("loco", "cnn6", SEED, protocol="loco", withheld=c).weights(k), map_location="cpu",
                                   weights_only=True))
    cnn.eval()
    prec, cents = torch.eye(cfg.cnn.embedding_dim), torch.randn(9, cfg.cnn.embedding_dim)
    x = torch.randn(9, len(VIB), 128, 25)
    mu, sd = torch.zeros(len(VIB)), torch.ones(len(VIB))
    lat = {}
    default_threads = torch.get_num_threads()
    for threads in (1, default_threads):  # CPU timing varies with the thread setting; both are reported
        torch.set_num_threads(threads)
        t_bank, t_maha = [], []
        with torch.no_grad():
            for _ in range(51):
                t0 = time.perf_counter()
                for m in aes:  # frame level: 9 windows × 25 frames per specialist
                    f = normalise_any(x, mu, sd).permute(0, 3, 1, 2).reshape(-1, len(VIB), 128)
                    ((m(f) - f) ** 2).mean()
                t_bank.append((time.perf_counter() - t0) * 1000)
                t0 = time.perf_counter()
                _, z = cnn(normalise(x, mu, sd))
                d = z[:, None, :] - cents[None]
                torch.einsum("wci,ij,wcj->wc", d, prec, d).min(dim=1)
                t_maha.append((time.perf_counter() - t0) * 1000)
        lat[f"{threads}_threads"] = {"bank_9_specialists": float(np.median(t_bank[1:])),
                                     "cnn6_plus_mahalanobis": float(np.median(t_maha[1:]))}
    torch.set_num_threads(default_threads)
    return {"cpu_ms_per_recording": lat,
            "parameters": {"bank_9_specialists": int(sum(sum(p.numel() for p in m.parameters()) for m in aes)),
                           "one_specialist": int(sum(p.numel() for p in aes[0].parameters())),
                           "cnn6_loco": int(n_parameters(cnn))},
            "note": "the Mahalanobis check needs the CNN-6 forward pass (which the system runs anyway for Q2); the bank "
                    "is additional to it in a deployment"}


def run_evaluate() -> None:
    from sklearn.metrics import roc_auc_score

    from ml.specialists import NORMAL, accepting_sets, calibrate, empirical_rank_scores, log_ratio_scores, verdict
    from ml.speed_calibration import matched_threshold
    from ml.rejection import TIE_EPS

    cfg, meta, pool, assignment = setup()
    info = pool.set_index("rec_id")
    dec = pd.read_parquet(results_dir() / "scores" / "cnn6ae6_s42_loco_decisions.parquet")
    thr = pd.read_csv(results_dir() / "loco" / "decision_thresholds.csv").set_index(["withheld", "outer_fold"])
    level = level_per_recording(cfg, meta, pool)
    runs, inst, gen_rows, trains = [], [], [], []
    for c in FAULT_CLASSES:
        for k in (0, 1, 2):
            e = pd.read_parquet(errors_path(c, k)).set_index("rec_id")
            specs = [col[2:] for col in e.columns if col.startswith("e_")]
            assert c not in specs and len(specs) == 9
            E = e[[f"e_{u}" for u in specs]].set_axis(specs, axis=1)
            s = log_ratio_scores(E, e.role, e.class10)
            cal = calibrate(s, e.role)
            tau = float(cal["threshold"])
            S = s.min(axis=1)
            A = accepting_sets(s, tau)
            d = dec[(dec.withheld == c) & (dec.outer_fold == k)].set_index("rec_id").loc[e.index]
            tau_d, tau_p = float(thr.loc[(c, k), "tau_d"]), float(thr.loc[(c, k), "tau_p"])
            maha_rej = (d.maha > tau_d) & ~((d.maha - tau_d).abs() <= TIE_EPS)
            maha_only = np.where(maha_rej, "unknown", np.where(d.pred == NORMAL, "healthy", "known_fault"))
            t = pd.DataFrame({"withheld": c, "outer_fold": k, "rec_id": e.index, "role": e.role, "class10": e.class10,
                              "S": S.to_numpy(), "best": s.idxmin(axis=1).to_numpy(), "accepting": A.to_numpy(),
                              "n_accepting": A.map(lambda a: len([x for x in a.split("+") if x])).to_numpy(),
                              "bank_verdict": A.map(verdict).to_numpy(), "tau_S": tau, "maha": d.maha.to_numpy(),
                              "tau_d": tau_d, "maha_rejects": maha_rej.to_numpy(), "maha_only_verdict": maha_only,
                              "rule_B": d.decision_classifier_first.to_numpy(), "cnn_top": d.pred.to_numpy(),
                              "speed_hz": info.fr_measured_hz.loc[e.index].to_numpy(),
                              "severity": info.severity.loc[e.index].to_numpy(), "level": level.loc[e.index].to_numpy()})
            for u in specs:
                t[f"s_{u}"] = s[u].to_numpy()
            inst.append(t)
            kn, un = t[t.role == "foldk_known"], t[t.role == "foldk_withheld"]
            y = np.r_[np.zeros(len(kn)), np.ones(len(un))]
            corr = (kn.cnn_top == kn.class10).to_numpy()
            n_rej = int(kn.maha_rejects.sum())
            mt = matched_threshold(kn.S.to_numpy(), n_rej)
            bank_rej_known = (kn.S > tau) & ~((kn.S - tau).abs() <= TIE_EPS)
            runs.append({"withheld": c, "outer_fold": k, "n_known": len(kn), "n_unknown": len(un),
                         "auroc_bank": roc_auc_score(y, np.r_[kn.S, un.S]), "auroc_maha": roc_auc_score(y, np.r_[kn.maha, un.maha]),
                         "oscr_bank": oscr(kn.S.to_numpy(), corr, un.S.to_numpy()),
                         "oscr_maha": oscr(kn.maha.to_numpy(), corr, un.maha.to_numpy()),
                         "tau_S": tau, "tau_S_attainable_level": cal["attainable_level"], "n_cal": int((e.role == "cal").sum()),
                         "maha_known_rejected": n_rej, "maha_unknown_recall": float(un.maha_rejects.mean()),
                         "bank_matched_unknown_recall": float(((un.S > mt) & ~((un.S - mt).abs() <= TIE_EPS)).mean()),
                         "bank_known_rejected_calibrated": int(bank_rej_known.sum()),
                         "bank_unknown_recall_calibrated": float((un.bank_verdict == "unfamiliar").mean()),
                         "cal_ambiguous_share": float((t[t.role == "cal"].n_accepting >= 2).mean())})
            # generalisation: each specialist's acceptance of the withheld class vs of other foreign known classes
            for u in specs:
                acc = t[f"s_{u}"] <= tau + TIE_EPS
                foreign = kn[kn.class10 != u]
                gen_rows.append({"withheld": c, "outer_fold": k, "specialist": u,
                                 "withheld_accept_rate": float(acc[un.index].mean()) if len(un) else np.nan,
                                 "withheld_median_s": float(un[f"s_{u}"].median()) if len(un) else np.nan,
                                 "foreign_known_accept_rate": float(acc[foreign.index].mean()),
                                 "own_known_accept_rate": float(acc[kn[kn.class10 == u].index].mean()) if (kn.class10 == u).any() else np.nan})
            er = empirical_rank_scores(E, e.role, e.class10)
            t["empirical_rank_min"] = er.min(axis=1).to_numpy()
            t["empirical_rank_normal_cap"] = int(((e.role == "cal") & (e.class10 == NORMAL)).sum() - 1)
            trains.append(json.loads((SPEC_RES / f"w-{c}" / f"fold{k}_train.json").read_text()))
    R = pd.DataFrame(runs)
    I = pd.concat(inst, ignore_index=True)
    G = pd.DataFrame(gen_rows)
    # ---- primary: per-fold means, per-class means
    fold = R.groupby("outer_fold")[["auroc_bank", "auroc_maha", "oscr_bank", "oscr_maha"]].mean()
    delta = (fold.auroc_bank - fold.auroc_maha)
    per_class = R.groupby("withheld")[["auroc_bank", "auroc_maha", "oscr_bank", "oscr_maha"]].mean()
    class_all3 = R.assign(better=R.auroc_bank > R.auroc_maha).groupby("withheld").better.sum()
    un_all = I[I.role == "foldk_withheld"]
    pooled_matched = float((R.bank_matched_unknown_recall * R.n_unknown).sum() / R.n_unknown.sum())
    pooled_maha = float((R.maha_unknown_recall * R.n_unknown).sum() / R.n_unknown.sum())
    crit = {"a_all_folds_positive": bool((delta > 0).all()), "b_mean_delta_ge_0.02": bool(delta.mean() >= 0.02),
            "c_matched_recall_gain_ge_5pp": bool(pooled_matched - pooled_maha >= 0.05),
            "d_classes_better_ge_5_of_9": bool(int((per_class.auroc_bank > per_class.auroc_maha).sum()) >= 5)}
    reading = ("helps" if all(crit.values()) else
               "does not help" if ((delta <= 0).all() or delta.mean() <= -0.02) else "too thin to tell")
    # ---- unknowns: healthy vs known fault
    def counts(col, frame):
        return frame[col].value_counts().to_dict()
    unk = {"bank": counts("bank_verdict", un_all), "mahalanobis_only": counts("maha_only_verdict", un_all),
           "rule_B_3C": {("known_fault" if v not in ("healthy", "unknown") else v): n
                         for v, n in un_all.rule_B.value_counts().items()}}
    unk["rule_B_3C"] = un_all.rule_B.map(lambda v: v if v in ("healthy", "unknown") else "known_fault").value_counts().to_dict()
    unk["distinct_recordings"] = int(un_all.rec_id.nunique())
    # ---- ambiguity
    amb = I.assign(ambiguous=I.n_accepting >= 2)
    amb_by_role = amb.groupby("role").agg(instances=("rec_id", "size"), ambiguous=("ambiguous", "mean"),
                                          mean_n_accepting=("n_accepting", "mean")).to_dict("index")
    amb_by_class = amb[amb.role == "foldk_known"].groupby("class10").ambiguous.mean().round(4).to_dict()
    top_sets = amb[amb.ambiguous].accepting.value_counts().head(12).to_dict()
    amb_speed = amb.assign(band=pd.cut(amb.speed_hz, [0, 20, 30, 99], right=False, labels=[b[2] for b in SPEED_BANDS])) \
        .groupby("band", observed=True).ambiguous.mean().round(4).to_dict()
    # ---- speed and severity
    I["band"] = pd.cut(I.speed_hz, [0, 20, 30, 99], right=False, labels=[b[2] for b in SPEED_BANDS])
    kn_all = I[I.role == "foldk_known"]
    sp = {"unknown_recall_bank_calibrated": un_all.assign(band=I.band).groupby("band", observed=True).bank_verdict
          .apply(lambda v: float((v == "unfamiliar").mean())).to_dict(),
          "unknown_recall_maha_calibrated": un_all.assign(band=I.band).groupby("band", observed=True).maha_rejects.mean().to_dict(),
          "known_rejection_bank_calibrated": kn_all.groupby("band", observed=True).bank_verdict
          .apply(lambda v: float((v == "unfamiliar").mean())).to_dict(),
          "known_rejection_maha_calibrated": kn_all.groupby("band", observed=True).maha_rejects.mean().to_dict()}
    sev = un_all.groupby(["class10", "severity"]).agg(n=("rec_id", "size"),
                                                       bank=("bank_verdict", lambda v: float((v == "unfamiliar").mean())),
                                                       maha=("maha_rejects", "mean")).reset_index()
    # ---- generalisation check
    gsum = G.groupby("specialist")[["withheld_accept_rate", "foreign_known_accept_rate", "own_known_accept_rate",
                                     "withheld_median_s"]].mean().round(4)
    best_on_unknown = un_all.groupby("withheld").best.value_counts(normalize=True).round(3)
    normal_accepts_unknown = float(un_all.accepting.str.split("+").map(lambda a: NORMAL in a).mean())
    lvl_corr = float(un_all[["S", "level"]].corr(method="spearman").iloc[0, 1])
    lvl_corr_known = float(kn_all[["S", "level"]].corr(method="spearman").iloc[0, 1])
    # ---- cost
    train_s = float(sum(v["train_seconds"] for t in trains for v in t["specialists"].values()))
    n_fit = pd.DataFrame([{"withheld": t["withheld"], "fold": t["fold"], "specialist": u, **v}
                          for t in trains for u, v in t["specialists"].items()])
    cost = {"specialist_training_seconds_total": train_s, "specialists_trained": int((~n_fit.reused_3c_ae6).sum()),
            "normal_reused_from_3c": int(n_fit.reused_3c_ae6.sum()),
            "cnn6_loco_training_seconds_total_3c": 958.9, "ae6_loco_training_seconds_total_3c": 191.5,
            **latency_and_size(cfg)}
    time_est = json.loads((OUT / "r4_time_estimate.json").read_text()) if (OUT / "r4_time_estimate.json").exists() else None
    out = {"stage": "R4", "wording": WORDING, "seed": SEED, "split_hash": split_hash(meta), "git_sha": git_sha(),
           "calibration": "s_u = log(e_u / median of specialist u's own calibration errors), leave-one-out for its own "
                          "calibration recordings; S = min_u s_u; tau_S = 97.5 % order statistic over all calibration recordings",
           "primary_per_fold": fold.round(6).to_dict("index"), "delta_auroc_per_fold": delta.round(6).to_dict(),
           "mean_delta_auroc": float(delta.mean()), "fold_spread": {c2: [float(fold[c2].min()), float(fold[c2].max())] for c2 in fold},
           "per_withheld_class": per_class.round(6).to_dict("index"), "runs_bank_better_per_class": class_all3.to_dict(),
           "matched_unknown_recall": {"bank": pooled_matched, "mahalanobis": pooled_maha,
                                      "per_fold": R.groupby("outer_fold").apply(lambda g: {
                                          "bank": float((g.bank_matched_unknown_recall * g.n_unknown).sum() / g.n_unknown.sum()),
                                          "mahalanobis": float((g.maha_unknown_recall * g.n_unknown).sum() / g.n_unknown.sum())},
                                          include_groups=False).to_dict()},
           "calibrated": {"bank_known_rejection": float(R.bank_known_rejected_calibrated.sum() / R.n_known.sum()),
                          "maha_known_rejection": float(R.maha_known_rejected.sum() / R.n_known.sum()),
                          "bank_unknown_recall": float((un_all.bank_verdict == "unfamiliar").mean()),
                          "maha_unknown_recall": float(un_all.maha_rejects.mean())},
           "unknown_outcomes": unk, "criteria": crit, "reading": reading,
           "ambiguity": {"by_role": amb_by_role, "known_side_by_class": amb_by_class, "top_sets": top_sets,
                         "by_speed_band": amb_speed},
           "speed": sp, "severity_unknown_recall": sev.to_dict("records"),
           "generalisation": {"per_specialist_mean": gsum.to_dict("index"),
                              "best_specialist_share_on_withheld": {f"{a}|{b}": v for (a, b), v in best_on_unknown.items()},
                              "withheld_accepted_by_normal_specialist": normal_accepts_unknown,
                              "spearman_S_vs_input_level_withheld": lvl_corr, "spearman_S_vs_input_level_known": lvl_corr_known},
           "training_set_sizes": n_fit.groupby("specialist")[["n_fit_recordings", "n_cal_recordings"]].agg(["min", "max"])
           .pipe(lambda x: {u: {f"{a}_{b}": int(x.loc[u, (a, b)]) for a, b in x.columns} for u in x.index}),
           "normal_reproduction_max_abs_diff": float(max(t["normal_reproduction_max_abs_diff"] for t in trains)),
           "cost": cost, "time_estimate": time_est}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "r4_specialists.json").write_text(json.dumps(out, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
    R.to_csv(OUT / "r4_per_run.csv", index=False)
    I[["withheld", "outer_fold", "rec_id", "role", "class10", "severity", "speed_hz", "level", "S", "tau_S", "best",
       "accepting", "n_accepting", "bank_verdict", "maha", "tau_d", "maha_rejects", "maha_only_verdict", "rule_B",
       "cnn_top", "empirical_rank_min", "empirical_rank_normal_cap"] + [c2 for c2 in I.columns if c2.startswith("s_")]] \
        .to_csv(OUT / "r4_ambiguity.csv", index=False)
    G.to_csv(OUT / "r4_generalisation.csv", index=False)
    print(fold.round(4)); print("Δ AUROC per fold:", delta.round(4).to_dict(), "mean", round(float(delta.mean()), 4))
    print(per_class.round(4))
    print("matched unknown recall: bank", round(pooled_matched, 4), "maha", round(pooled_maha, 4))
    print("calibrated:", out["calibrated"])
    print("unknown outcomes:", unk)
    print("criteria:", crit, "→", reading)
    print("ambiguity:", amb_by_role)
    print("generalisation:\n", gsum)
    print("normal accepts withheld:", normal_accepts_unknown, "| S vs level (withheld / known):", lvl_corr, lvl_corr_known)
    print("cost:", cost)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("task", choices=["train", "evaluate"])
    a = ap.parse_args()
    run_train() if a.task == "train" else run_evaluate()
    return 0


if __name__ == "__main__":
    sys.exit(main())
