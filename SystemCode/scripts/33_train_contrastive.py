"""Stage R3: supervised contrastive pretraining of the CNN-6 encoder, then a frozen-encoder classifier head (Q2,
development data only). Never imports LightGBM. Protocol: CLAUDE.md §7 (Stage R3), declared before any training.

    caffeinate -i env MACHINESENSE_DEVICE=mps python scripts/33_train_contrastive.py train --seed 42   # first (time guard)
    caffeinate -i env MACHINESENSE_DEVICE=mps python scripts/33_train_contrastive.py train --seed 43
    caffeinate -i env MACHINESENSE_DEVICE=mps python scripts/33_train_contrastive.py train --seed 44
    python scripts/33_train_contrastive.py evaluate      # tables vs CNN-6 (same folds, same seeds); CPU latency

Follows scripts/06_train_cnn.py (Stage 3A/3B) with the objective replaced: development pool, results/dev_folds.csv (K = 3),
normalisation from ml.datasets.fold_norm_stats on each fold's fit recordings, fixed epochs, every pool recording scored
once by its own fold's model. Stage A: SupCon (ml.contrastive), no same-recording positives. Stage B: the CNN-6 head on
the frozen encoder, trained by ml.train.train_model with the cnn: settings. Outputs (ml.paths.RunPaths, family
"contrastive", variant "supcon6"): results/scores/supcon6_s<seed>_dev_logits.parquet / _embeddings.npz,
artifacts/contrastive/<name>/fold<k>.pt (ScratchCNN state_dict; same architecture as CNN-6) and fold<k>_proj.pt,
results/contrastive/<name>/fold<k>_norm.json, results/contrastive/<name>_train.json; evaluation in results/research/r3_*.
Scope: trained on all ten classes → Q2 only; not valid for LOCO / Q3.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time

import numpy as np
import pandas as pd

from ml.config import load_config
from ml.contrastive import load_contrastive_config
from ml.labels import CLASS6, CLASS10
from ml.paths import DEFAULT_CONFIG, SYSTEM_CODE_DIR, RunPaths, artifacts_dir, results_dir
from ml.splits import split_hash
from ml.utils import git_sha

spec = importlib.util.spec_from_file_location("s06", SYSTEM_CODE_DIR / "scripts" / "06_train_cnn.py")
S3A = importlib.util.module_from_spec(spec)
spec.loader.exec_module(S3A)   # Stage 3A helpers: data, outputs, checks, metrics, latency (used unchanged)

CCFG = SYSTEM_CODE_DIR / "configs" / "contrastive.yaml"
SEEDS = (42, 43, 44)
VIB = [0, 1, 2, 3, 4, 5]
OUT = results_dir() / "research"
PAIRS = {"imbalance_vs_vertical_misalignment": ("imbalance", "vertical_misalignment"),
         "horizontal_vs_vertical_misalignment": ("horizontal_misalignment", "vertical_misalignment")}
SPEED_BANDS = [(0, 20, "<20 Hz"), (20, 30, "20-30 Hz"), (30, 99, ">=30 Hz")]
WORDING = ("Exploratory development results (folder-grouped K = 3 folds, seeds 42/43/44); the development pool runs below "
           "40 Hz and excludes the largest severity of each family, so these folds cannot measure the development-to-final "
           "gap. CNN-6 misclassifies 8 / 11 / 4 of 804 recordings on these folds, so any difference rests on a handful of "
           "recordings.")


def paths(seed: int) -> RunPaths:
    return RunPaths("contrastive", "supcon6", seed)


class TimeGuard(Exception):
    pass


def run_train(seed: int) -> None:
    import mlflow
    import torch

    from ml.contrastive import FrozenEncoderClassifier, fit_recordings, steps_per_epoch, train_supcon
    from ml.datasets import fold_fit_context, fold_norm_stats
    from ml.models.cnn import ScratchCNN
    from ml.train import gpu_job, release_device_memory, score_windows, train_model
    from ml.utils import get_device, mlflow_run, seed_everything

    cfg = load_config(DEFAULT_CONFIG)
    kc = load_contrastive_config(CCFG).model_copy(update={"seed": seed})
    cc = cfg.cnn.model_copy(update={"seed": seed, "epochs": kc.head_epochs})
    device = get_device()
    rp = paths(seed)
    meta, pool, arr, index, rows, cache_key = S3A.load_all(cfg)
    assert (rows.index == np.arange(len(rows))).all()
    speed = pool.set_index("rec_id").fr_measured_hz
    cls_idx = {c: i for i, c in enumerate(CLASS10)}
    prov = {"stage": "R3", "variant": "supcon6", "split_hash": split_hash(meta), "cache_key": cache_key,
            "git_sha": git_sha(), "seed": seed, "channels": VIB}
    n = len(rows)
    all_logits, all_emb = np.zeros((n, len(CLASS10)), np.float32), np.zeros((n, cc.embedding_dim), np.float32)
    run_ids = np.empty(n, dtype=object)
    folds_info = []
    for p in (rp.weights(0), rp.norm_stats(0), rp.result("train")):
        p.parent.mkdir(parents=True, exist_ok=True)
    t_start = time.perf_counter()
    with gpu_job(device, artifacts_dir(), kc.mps_memory_fraction), mlflow_run(cfg, run_name=rp.run_name()) as parent:
        mlflow.set_tags({"stage": "R3", "scope": S3A.TAG, **rp.tags(), "split_hash": prov["split_hash"],
                         "cache_key": cache_key, "protocol": "dev folder-grouped K=3; scored once by own fold model"})
        cache, y = S3A.resident(arr, rows, device)
        for k in sorted(pool.dev_fold.unique()):
            fit_ids, _ = fold_fit_context(pool, k)
            st = fold_norm_stats(arr, index, meta, pool, k)
            mean, std = st["mean"][VIB], st["std"][VIB]
            tr_idx = np.flatnonzero((rows.dev_fold != k).to_numpy())
            ev_idx = np.flatnonzero((rows.dev_fold == k).to_numpy())
            assert set(rows.rec_id.iloc[tr_idx]) == set(fit_ids) and not set(fit_ids) & set(rows.rec_id.iloc[ev_idx])
            fr = fit_recordings(rows, fit_ids, cls_idx, speed)
            spe = steps_per_epoch(len(tr_idx), kc.batch_size)
            rp.norm_stats(k).write_text(json.dumps({"fit_ids_sha256": S3A.ids_hash(fit_ids), "n_fit_recordings": len(fit_ids),
                                                     "n_windows": st["n_windows"], "channels": VIB,
                                                     "mean": mean.tolist(), "std": std.tolist()}, indent=1))
            run_id = rp.run_name(k)
            with mlflow.start_run(run_name=run_id, nested=True, experiment_id=parent.info.experiment_id):
                mlflow.set_tags({"stage": "R3", "scope": S3A.TAG, **rp.tags(k), "fit_ids_sha256": S3A.ids_hash(fit_ids),
                                 "split_hash": prov["split_hash"], "cache_key": cache_key, "git_sha": prov["git_sha"]})
                seed_everything(seed)
                net = ScratchCNN(len(VIB), len(CLASS10), cc).to(device)
                t0 = time.perf_counter()
                net, proj, hist_a, pair_stats = train_supcon(
                    net, cache, fr, VIB, mean, std, kc, device, spe,
                    on_epoch=lambda ep, r: mlflow.log_metric("supcon_loss", r["supcon_loss"], step=ep))
                if device.type == "mps":
                    torch.mps.synchronize()
                t_a = time.perf_counter() - t0
                assert pair_stats["same_recording_positives"] == 0
                seed_everything(seed)
                wrapper = FrozenEncoderClassifier(net)
                t0 = time.perf_counter()
                wrapper, hist_b = train_model(cache, y, tr_idx, VIB, mean, std, len(CLASS10), cc, device, model=wrapper,
                                              param_groups=[{"params": list(net.head.parameters())}],
                                              on_epoch=lambda ep, r: mlflow.log_metrics(
                                                  {"head_train_loss": r["train_loss"], "head_train_acc": r["train_acc"]}, step=ep))
                if device.type == "mps":
                    torch.mps.synchronize()
                t_b = time.perf_counter() - t0
                logits, emb = score_windows(net, cache, ev_idx, VIB, mean, std, cc, device)
                all_logits[ev_idx], all_emb[ev_idx], run_ids[ev_idx] = logits, emb, run_id
                torch.save(net.state_dict(), rp.weights(k))
                torch.save(proj.state_dict(), rp.weights(k).with_name(f"fold{k}_proj.pt"))
                rec = S3A.recording_predictions(S3A.softmax(logits), rows.iloc[ev_idx], cfg.baselines.prob_floor)
                fm = S3A.fold_metrics(rec)
                mlflow.log_metrics({**fm, "stage_a_seconds": t_a, "stage_b_seconds": t_b,
                                    **{f"pairs_{a}": v for a, v in pair_stats.items() if isinstance(v, (int, float)) and v is not None}})
                print(f"  {run_id}: stage A {t_a:.0f} s (final SupCon {hist_a[-1]['supcon_loss']:.3f}), stage B {t_b:.0f} s "
                      f"(loss {hist_b[-1]['train_loss']:.4f}); class10 macro-F1 {fm['class10_macro_f1']:.4f}; pairs "
                      f"diff-folder {pair_stats['frac_different_folders']:.3f} diff-speed {pair_stats['frac_different_speeds']:.3f}",
                      flush=True)
                folds_info.append({"fold": int(k), "run_id": run_id, "n_train_windows": len(tr_idx),
                                   "n_train_recordings": len(fit_ids), "fit_ids_sha256": S3A.ids_hash(fit_ids),
                                   "steps_per_epoch": spe, "stage_a_seconds": t_a, "stage_b_seconds": t_b,
                                   "pair_stats": pair_stats, "history_a": hist_a, "history_b": hist_b, "metrics": fm})
                net.to(device)
            release_device_memory(device)
            if seed == SEEDS[0] and k == 0:
                proj_min = (time.perf_counter() - t_start) * 3 * len(SEEDS) / 60
                print(f"  TIME ESTIMATE after the first fold: {proj_min:.0f} min for 3 seeds × 3 folds", flush=True)
                (OUT / "r3_time_estimate.json").parent.mkdir(parents=True, exist_ok=True)
                (OUT / "r3_time_estimate.json").write_text(json.dumps({"first_fold_seconds": time.perf_counter() - t_start,
                                                                       "projected_total_minutes": proj_min,
                                                                       "guard_minutes": kc.time_guard_minutes}, indent=1))
                if proj_min > kc.time_guard_minutes:
                    raise TimeGuard(f"projected {proj_min:.0f} min > {kc.time_guard_minutes} min; stopping (§10)")
        p1, p2 = S3A.write_scores(rp.scores("logits"), rp.embeddings(), rows, np.arange(n), all_logits, all_emb,
                                  run_ids.astype(str), prov)
        check = S3A.check_outputs(p1, p2, n)
        out = {"provenance": {**prov, "scope": S3A.TAG, "device": str(device), "contrastive_config": kc.model_dump(),
                              "head_config": cc.model_dump(mode="json"), "mlflow_run_id": parent.info.run_id},
               "folds": folds_info, "outputs_check": check}
        rp.result("train").write_text(json.dumps(out, indent=1))
    print(f"{rp.name}: done in {(time.perf_counter() - t_start) / 60:.1f} min")


# ----------------------------------------------------------------------------------------------------------- evaluate
def cnn6_path(seed: int):
    p = RunPaths("cnn", "cnn6", seed).scores("logits")
    return p if p.exists() else S3A.legacy_seed42("cnn6")[0]


def cnn6_weights(seed: int, k: int):
    p = RunPaths("cnn", "cnn6", seed).weights(k)
    return p if p.exists() else SYSTEM_CODE_DIR / "artifacts" / "cnn" / "cnn6_dev" / f"fold{k}.pt"


def per_class_recall(r: pd.DataFrame) -> dict:
    from sklearn.metrics import recall_score
    return dict(zip(CLASS10, recall_score(r.class10, r.pred10, labels=CLASS10, average=None, zero_division=0).round(4).tolist()))


def pair_errors(r: pd.DataFrame) -> dict:
    return {name: {f"{a}->{b}": int(((r.class10 == a) & (r.pred10 == b)).sum()),
                   f"{b}->{a}": int(((r.class10 == b) & (r.pred10 == a)).sum())} for name, (a, b) in PAIRS.items()}


def latency(seed: int, cfg) -> dict:
    """Median CPU ms per recording (9 windows from the cache, the 3A method) for CNN-6 and the contrastive model, same
    fold-0 recordings, same process."""
    import torch

    from ml.models.cnn import ScratchCNN

    meta, pool, arr, index, rows, _ = S3A.load_all(cfg)
    ev_idx = np.flatnonzero((rows.dev_fold == 0).to_numpy())
    cpu_rows = np.asarray(arr[np.sort(rows.row.to_numpy()[ev_idx])[:9 * 50]])
    out = {}
    ns = json.loads(paths(seed).norm_stats(0).read_text())  # fold 0's fit-set statistics (identical for both models)
    for name, w in (("cnn6", cnn6_weights(seed, 0)), ("supcon6", paths(seed).weights(0))):
        m = ScratchCNN(len(VIB), len(CLASS10), cfg.cnn)
        m.load_state_dict(torch.load(w, map_location="cpu", weights_only=True))
        out[name] = S3A.cpu_latency_ms(m, cpu_rows, VIB, np.asarray(ns["mean"]), np.asarray(ns["std"]), cfg.cnn)
    out["n_parameters_equal"] = True
    return out


def run_evaluate() -> None:
    from ml.evaluation import metrics_mixed, mixed_bootstrap_draws, paired_mixed

    cfg = load_config(DEFAULT_CONFIG)
    bc = cfg.baselines
    meta = pd.read_csv(SYSTEM_CODE_DIR / "metadata.csv")
    pool = S3A.dev_pool(cfg, meta)
    order = sorted(pool.rec_id)
    speed = pool.set_index("rec_id").fr_measured_hz.loc[order]
    band = pd.cut(speed, [b[0] for b in SPEED_BANDS] + [SPEED_BANDS[-1][1]], right=False, labels=[b[2] for b in SPEED_BANDS])
    recs = {}
    for s in SEEDS:
        recs[("cnn6", s)] = S3A.cnn_recordings(cnn6_path(s), pool, bc.prob_floor).loc[order]
        recs[("supcon6", s)] = S3A.cnn_recordings(paths(s).scores("logits"), pool, bc.prob_floor).loc[order]
    base = recs[("cnn6", 42)]
    draws = mixed_bootstrap_draws(base.class10.to_numpy(), base.folder.to_numpy(), bc.n_bootstrap, bc.seed)
    per_fold, conf_rows, models = [], [], {}
    for (m, s), r in recs.items():
        for k, g in r.groupby("dev_fold"):
            per_fold.append({"model": m, "seed": s, "fold": int(k), **S3A.fold_metrics(g)})
        mm10 = metrics_mixed(r.class10, r.pred10, CLASS10, draws, r.folder)
        mm6 = metrics_mixed(r.true6, r.pred6, CLASS6, draws, r.folder)
        for i, t in enumerate(CLASS10):
            for j, p in enumerate(CLASS10):
                if mm10["confusion"][i][j]:
                    conf_rows.append({"model": m, "seed": s, "true": t, "pred": p, "recordings": mm10["confusion"][i][j]})
        models[f"{m}_s{s}"] = {
            "class10": {k2: mm10[k2] for k2 in ("macro_f1", "macro_f1_ci", "balanced_accuracy", "balanced_accuracy_ci",
                                                "per_class_f1")},
            "class6": {k2: mm6[k2] for k2 in ("macro_f1", "macro_f1_ci", "balanced_accuracy", "balanced_accuracy_ci")},
            "per_class_recall": per_class_recall(r), "errors": int((r.class10 != r.pred10).sum()),
            "pair_errors": pair_errors(r),
            "per_folder_accuracy": {f: v["accuracy"] for f, v in mm10["per_folder_accuracy"].items()},
            "per_speed_band_accuracy": {str(b): {"accuracy": float((r.class10 == r.pred10)[band == b].mean()),
                                                 "n": int((band == b).sum())} for b in band.cat.categories}}
    pf = pd.DataFrame(per_fold)
    fold_mean = pf.groupby(["model", "seed"])[["class10_macro_f1", "class10_balanced_accuracy", "class6_macro_f1",
                                               "class6_balanced_accuracy"]].mean()
    fold_spread = pf.groupby(["model", "seed"])[["class10_macro_f1", "class6_macro_f1"]].agg(["min", "max"])
    delta = {s: float(fold_mean.loc[("supcon6", s), "class10_macro_f1"] - fold_mean.loc[("cnn6", s), "class10_macro_f1"])
             for s in SEEDS}
    delta6 = {s: float(fold_mean.loc[("supcon6", s), "class6_macro_f1"] - fold_mean.loc[("cnn6", s), "class6_macro_f1"])
              for s in SEEDS}
    folds_ge = {s: int((pf[(pf.model == "supcon6") & (pf.seed == s)].sort_values("fold").class10_macro_f1.to_numpy()
                        >= pf[(pf.model == "cnn6") & (pf.seed == s)].sort_values("fold").class10_macro_f1.to_numpy()).sum())
                for s in SEEDS}
    mean_d, mean_d6 = float(np.mean(list(delta.values()))), float(np.mean(list(delta6.values())))
    crit = {"a_all_seeds_positive": all(d > 0 for d in delta.values()), "b_mean_delta_ge_0.003": mean_d >= 0.003,
            "c_ge_in_2_of_3_folds_every_seed": all(v >= 2 for v in folds_ge.values()),
            "d_class6_mean_delta_ge_-0.003": mean_d6 >= -0.003}
    reading = ("helps" if all(crit.values()) else
               "does not help" if (all(d <= 0 for d in delta.values()) or mean_d <= -0.003) else "too thin to tell")
    paired = {s: {"class10": paired_mixed(base.class10, recs[("supcon6", s)].pred10, recs[("cnn6", s)].pred10, CLASS10, draws),
                  "class6": paired_mixed(base.true6, recs[("supcon6", s)].pred6, recs[("cnn6", s)].pred6, CLASS6, draws)}
              for s in SEEDS}
    pair_read = {}
    for name, (a, b) in PAIRS.items():
        tot = {m: [sum(models[f"{m}_s{s}"]["pair_errors"][name].values()) for s in SEEDS] for m in ("cnn6", "supcon6")}
        better = all(x < y for x, y in zip(tot["supcon6"], tot["cnn6"]))
        pair_read[name] = {"cnn6_per_seed": tot["cnn6"], "supcon6_per_seed": tot["supcon6"],
                           "separates_better": better,
                           "note": "CNN-6 has no confusions here, so only a worsening can be observed" if sum(tot["cnn6"]) == 0 else ""}
    train = {s: json.loads(paths(s).result("train").read_text()) for s in SEEDS}
    cost = {"train_seconds": {s: {"stage_a": sum(f["stage_a_seconds"] for f in train[s]["folds"]),
                                  "stage_b": sum(f["stage_b_seconds"] for f in train[s]["folds"])} for s in SEEDS},
            "cnn6_train_seconds": {s: sum(f["train_seconds"] for f in json.loads(
                (RunPaths("cnn", "cnn6", s).result("train") if s != 42 else results_dir() / "cnn" / "cnn6_dev_train.json")
                .read_text())["folds"]) for s in SEEDS},
            "cpu_latency_ms_per_recording": latency(42, cfg)}
    pairs_stats = {f"s{s}_fold{f['fold']}": f["pair_stats"] for s in SEEDS for f in train[s]["folds"]}
    seed_spread = {m: {v: {"values": [float(fold_mean.loc[(m, s), v]) for s in SEEDS],
                           "mean": float(np.mean([fold_mean.loc[(m, s), v] for s in SEEDS])),
                           "std_over_seeds": float(np.std([fold_mean.loc[(m, s), v] for s in SEEDS], ddof=1))}
                       for v in ("class10_macro_f1", "class10_balanced_accuracy", "class6_macro_f1", "class6_balanced_accuracy")}
                   for m in ("cnn6", "supcon6")}
    out = {"stage": "R3", "wording": WORDING, "split_hash": split_hash(meta), "git_sha": git_sha(),
           "n_recordings": len(order), "n_bootstrap": bc.n_bootstrap, "bootstrap_seed": bc.seed,
           "uncertainty": "seed spread (std over seeds 42/43/44 of the fold-mean metric) and sampling uncertainty (paired "
                          "mixed-unit bootstrap per seed) are different sources, reported separately, never merged",
           "fold_mean_delta_class10_macro_f1": delta, "fold_mean_delta_class6_macro_f1": delta6,
           "folds_where_supcon_ge_cnn6": folds_ge, "mean_delta_class10": mean_d, "mean_delta_class6": mean_d6,
           "criteria": crit, "reading": reading, "pair_reading": pair_read, "seed_spread": seed_spread,
           "fold_spread": {f"{m}_s{s}": {c: [float(fold_spread.loc[(m, s), (c, "min")]), float(fold_spread.loc[(m, s), (c, "max")])]
                                         for c in ("class10_macro_f1", "class6_macro_f1")} for m, s in fold_spread.index},
           "paired_bootstrap": paired, "models": models, "pair_statistics": pairs_stats, "cost": cost,
           "optional_q3_step": "not run" if reading != "helps" else "pending"}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "r3_contrastive.json").write_text(json.dumps(out, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
    pf.to_csv(OUT / "r3_per_fold.csv", index=False)
    pd.DataFrame(conf_rows).to_csv(OUT / "r3_confusion.csv", index=False)
    print(pf.pivot_table(index=["seed", "fold"], columns="model", values="class10_macro_f1").round(4))
    print("Δ class10 per seed:", {s: round(d, 4) for s, d in delta.items()}, "| class6:", {s: round(d, 4) for s, d in delta6.items()})
    print("errors:", {k2: v["errors"] for k2, v in models.items()})
    print("criteria:", crit, "→", reading)
    print("pairs:", pair_read)
    print("cost:", cost)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("task", choices=["train", "evaluate"])
    ap.add_argument("--seed", type=int, default=42, choices=SEEDS)
    a = ap.parse_args()
    if a.task == "train":
        try:
            run_train(a.seed)
        except TimeGuard as e:
            print(f"TIME GUARD: {e}")
            return 3
    else:
        run_evaluate()
    return 0


if __name__ == "__main__":
    sys.exit(main())
