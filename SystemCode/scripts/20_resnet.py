"""ResNet-18 transfer learning on the order spectrogram (Stage ML-5, development experiment). Never imports LightGBM.

    caffeinate -i env MACHINESENSE_DEVICE=mps python scripts/20_resnet.py train --variant r18ft --seed 42   # B1; first
    caffeinate -i env MACHINESENSE_DEVICE=mps python scripts/20_resnet.py train --variant r18probe --seed 42
    caffeinate -i env MACHINESENSE_DEVICE=mps python scripts/20_resnet.py loco                              # B2 (6 runs)
    python scripts/20_resnet.py evaluate        # B1 tables vs CNN-6
    python scripts/20_resnet.py loco-eval       # B2 tables vs CNN-6 (3C, seed 42)
    python scripts/20_resnet.py cost            # B3

Protocol: CLAUDE.md §7 (Stage ML-5), declared before running. B1 follows scripts/06_train_cnn.py (Stage 3A) exactly,
with the model replaced: dev pool, results/dev_folds.csv (K = 3), norm_stats in development mode on each fold's training
recordings, fixed epochs, every pool recording scored once by its own fold's model. B2 follows scripts/09_loco_cnn.py
(Stage 3C) on the saved Q3 assignment, for withheld imbalance and vertical misalignment only, without the AE.
Outputs: results/scores/<variant>_s<seed>_dev_logits.parquet / _embeddings.npz (512-d), artifacts/resnet/<name>/fold<k>.pt,
results/resnet/<name>/fold<k>_norm.json, results/resnet/<name>_train.json; LOCO: results/scores/r18ft_w-<c>_s42_loco_fold<k>_*,
artifacts/loco/<name>/fold<k>.pt; tables in results/resnet/.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from ml.config import Config, load_config
from ml.datasets import norm_stats
from ml.labels import CLASS6, CLASS10
from ml.models.resnet import (ResNet18Spectrogram, load_resnet_config, n_parameters, param_groups, verify_weights,
                              weights_path)
from ml.openset import loco_run_ids, mahalanobis_min, mahalanobis_min_per_class, oscr, recording_mean, recording_msp
from ml.paths import DEFAULT_CONFIG, SYSTEM_CODE_DIR, RunPaths, artifacts_dir, results_dir
from ml.splits import split_hash
from ml.utils import git_sha


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(name.replace(".py", ""), SYSTEM_CODE_DIR / "scripts" / name)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


S3A = _load_script("06_train_cnn.py")   # Stage 3A helpers: data, outputs, checks, metrics (used unchanged)
OUT = results_dir() / "resnet"
WITHHELD = ("imbalance", "vertical_misalignment")
ASSIGNMENT = results_dir() / "q1q3" / "q3_fit_cal_eval_assignment.csv"
LOCO_ROLES = ("cal", "eval", "foldk_known", "foldk_withheld")
VIB = [0, 1, 2, 3, 4, 5]


class TimeGuard(Exception):
    pass


def estimate_guard(rc, variant: str, seed: int, epoch_s: float) -> dict:
    """After epoch 0 of fold 0: projected B1 seed-42 total (both variants, 3 folds) in minutes."""
    this = epoch_s * rc.epochs * 3 / 60
    if variant == "r18ft":
        total = 2 * this
        basis = "2 × R18-ft projection (the probe costs no more than the fine-tune)"
    else:
        ft = json.loads(RunPaths("resnet", "r18ft", seed).result("train").read_text())
        total = sum(f["train_seconds"] for f in ft["folds"]) / 60 + this
        basis = "measured R18-ft total + R18-probe projection"
    return {"variant": variant, "seed": seed, "first_epoch_seconds": epoch_s, "projected_variant_minutes": this,
            "projected_b1_seed42_minutes": total, "basis": basis, "limit_minutes": rc.time_guard_minutes}


# ----------------------------------------------------------------------------------------------------------- B1
def run_train(cfg: Config, variant: str, seed: int) -> None:
    import mlflow
    import torch

    from ml.train import gpu_job, release_device_memory, score_windows, train_model
    from ml.utils import get_device, mlflow_run, seed_everything

    rc, device = load_resnet_config().model_copy(update={"seed": seed}), get_device()
    sha = verify_weights(rc)
    rp = RunPaths("resnet", variant, seed)
    meta, pool, arr, index, rows, cache_key = S3A.load_all(cfg)
    ch = list(rc.vibration_channels)
    prov = {"variant": variant, "split_hash": split_hash(meta), "cache_key": cache_key, "git_sha": git_sha(),
            "seed": seed, "channels": ch, "weights": rc.weights, "weights_sha256": sha}
    folds = sorted(pool.dev_fold.unique())
    n = len(rows)
    all_logits, all_emb = np.zeros((n, len(CLASS10)), np.float32), np.zeros((n, 512), np.float32)
    run_ids = np.empty(n, dtype=object)
    fold_info, guard = [], None
    for p in (rp.weights(0), rp.norm_stats(0), rp.result("train")):
        p.parent.mkdir(parents=True, exist_ok=True)
    with gpu_job(device, artifacts_dir(), rc.mps_memory_fraction), mlflow_run(cfg, run_name=rp.run_name()) as parent:
        mlflow.set_tags({"stage": "ML-5", "scope": S3A.TAG, **rp.tags(), "split_hash": prov["split_hash"],
                         "cache_key": cache_key, "weights_sha256": sha})
        cache, y = S3A.resident(arr, rows, device)
        for k in folds:
            fit_ids, ctx = S3A.fold_fit_context(pool, k)
            st = norm_stats(arr, index, meta, fit_ids=fit_ids, context=ctx)
            mean, std = st["mean"][ch], st["std"][ch]
            tr_idx = np.flatnonzero((rows.dev_fold != k).to_numpy())
            ev_idx = np.flatnonzero((rows.dev_fold == k).to_numpy())
            assert set(rows.rec_id.iloc[tr_idx]) == set(fit_ids)
            run_id = rp.run_name(k)
            rp.norm_stats(k).write_text(json.dumps({"fit_ids_sha256": S3A.ids_hash(fit_ids), "n_fit_recordings": len(fit_ids),
                                                     "n_windows": st["n_windows"], "channels": ch,
                                                     "mean": mean.tolist(), "std": std.tolist()}, indent=1))

            def on_epoch(ep, r, k=k):
                nonlocal guard
                mlflow.log_metrics({"train_loss": r["train_loss"], "train_acc": r["train_acc"]}, step=ep)
                print(f"    fold {k} epoch {ep}: loss {r['train_loss']:.4f} acc {r['train_acc']:.3f} {r['seconds']:.0f} s",
                      flush=True)
                if k == folds[0] and ep == 0 and seed == 42:
                    guard = estimate_guard(rc, variant, seed, r["seconds"])
                    print(f"  time guard: {json.dumps(guard)}", flush=True)
                    if guard["projected_b1_seed42_minutes"] > rc.time_guard_minutes:
                        raise TimeGuard(json.dumps(guard))

            with mlflow.start_run(run_name=run_id, nested=True, experiment_id=parent.info.experiment_id):
                mlflow.set_tags({"stage": "ML-5", "scope": S3A.TAG, **rp.tags(k), "fit_ids_sha256": S3A.ids_hash(fit_ids),
                                 "split_hash": prov["split_hash"], "cache_key": cache_key, "git_sha": prov["git_sha"]})
                seed_everything(seed)
                model = ResNet18Spectrogram(len(ch), len(CLASS10), rc, variant).to(device)
                t0 = time.perf_counter()
                try:
                    model, hist = train_model(cache, y, tr_idx, ch, mean, std, len(CLASS10), rc, device, on_epoch=on_epoch,
                                              model=model, param_groups=param_groups(model, rc))
                except TimeGuard as e:
                    OUT.mkdir(parents=True, exist_ok=True)
                    (OUT / f"time_guard_{variant}_s{seed}.json").write_text(json.dumps(json.loads(str(e)), indent=1))
                    print(f"TIME GUARD: projected B1 exceeds {rc.time_guard_minutes} min; stopping. {e}")
                    raise SystemExit(3)
                if device.type == "mps":
                    torch.mps.synchronize()
                train_s = time.perf_counter() - t0
                logits, emb = score_windows(model, cache, ev_idx, ch, mean, std, rc, device)
                all_logits[ev_idx], all_emb[ev_idx], run_ids[ev_idx] = logits, emb, run_id
                torch.save(model.state_dict(), rp.weights(k))
                cpu_rows = np.asarray(arr[np.sort(rows.row.to_numpy()[ev_idx])[:9 * 50]])
                ms = S3A.cpu_latency_ms(model, cpu_rows, ch, mean, std, rc)
                rec = S3A.recording_predictions(S3A.softmax(logits), rows.iloc[ev_idx], cfg.baselines.prob_floor)
                fm = S3A.fold_metrics(rec)
                mlflow.log_metrics({**fm, "train_seconds": train_s, "cpu_ms_per_recording": ms})
                print(f"  {run_id}: {len(tr_idx)} train windows, {train_s:.0f} s, final loss {hist[-1]['train_loss']:.4f}, "
                      f"class10 macro-F1 {fm['class10_macro_f1']:.3f}, class6 {fm['class6_macro_f1']:.3f}, "
                      f"CPU {ms:.1f} ms/recording", flush=True)
                fold_info.append({"fold": int(k), "run_id": run_id, "n_train_windows": len(tr_idx),
                                  "n_train_recordings": len(fit_ids), "fit_ids_sha256": S3A.ids_hash(fit_ids),
                                  "n_scored_windows": len(ev_idx), "train_seconds": train_s, "cpu_ms_per_recording": ms,
                                  "history": hist, "metrics": fm})
                model.to(device)
            del model
            release_device_memory(device)
        p1, p2 = S3A.write_scores(rp.scores("logits"), rp.embeddings(), rows, np.arange(n), all_logits, all_emb,
                                  run_ids.astype(str), prov)
        check = S3A.check_outputs(p1, p2, n)
        rec = S3A.recording_predictions(S3A.softmax(all_logits), rows, cfg.baselines.prob_floor)
        assert len(rec) == len(pool)
        pooled = S3A.fold_metrics(rec)
        mlflow.log_metrics({f"pooled_{k2}": v for k2, v in pooled.items()})
        total = sum(f["train_seconds"] for f in fold_info)
        mlflow.log_metric("train_seconds_total", total)
        out = {"provenance": {**prov, "scope": S3A.TAG, "device": str(device), "config": rc.model_dump(mode="json"),
                              "n_parameters": n_parameters(ResNet18Spectrogram(len(ch), len(CLASS10), rc, variant,
                                                                               pretrained=False)),
                              "weights_file": str(weights_path(rc).name), "mlflow_run_id": parent.info.run_id},
               "time_guard": guard, "folds": fold_info, "train_seconds_total": total, "pooled_point_metrics": pooled,
               "outputs_check": check}
        rp.result("train").write_text(json.dumps(out, indent=1))
        mlflow.log_artifact(str(rp.result("train")))
    print(f"{rp.name}: {total / 60:.1f} min; pooled class10 macro-F1 {pooled['class10_macro_f1']:.3f}, "
          f"class6 {pooled['class6_macro_f1']:.3f}")


def _cnn6_logits(seed: int) -> Path:
    p = RunPaths("cnn", "cnn6", seed).scores("logits")
    return S3A.legacy_seed42("cnn6")[0] if seed == 42 and not p.exists() else p


def run_evaluate(cfg: Config) -> None:
    """B1: per available R18 run, recording-level class10 / class6 macro-F1 and balanced accuracy with the 3A mixed-unit
    bootstrap, paired R18 − CNN-6 (same seed) over identical draws, per-class F1, confusion matrices, the imbalance /
    vertical-misalignment confusions and horizontal misalignment at 0.5 mm; across seeds for R18-ft if 43 / 44 exist."""
    from sklearn.metrics import confusion_matrix

    from ml.evaluation import metrics_mixed, mixed_bootstrap_draws, paired_mixed

    bc = cfg.baselines
    meta = pd.read_csv(SYSTEM_CODE_DIR / "metadata.csv")
    pool = S3A.dev_pool(cfg, meta)
    order = sorted(pool.rec_id)
    sev = pool.set_index("rec_id").severity
    runs = [(v, s) for v in ("r18ft", "r18probe") for s in (42, 43, 44) if RunPaths("resnet", v, s).scores("logits").exists()]
    seeds = sorted({s for _, s in runs})
    recs = {f"cnn6_s{s}": S3A.cnn_recordings(_cnn6_logits(s), pool, bc.prob_floor).loc[order] for s in seeds}
    recs.update({f"{v}_s{s}": S3A.cnn_recordings(RunPaths("resnet", v, s).scores("logits"), pool, bc.prob_floor).loc[order]
                 for v, s in runs})
    base = recs[f"cnn6_s{seeds[0]}"]
    draws = mixed_bootstrap_draws(base.class10.to_numpy(), base.folder.to_numpy(), bc.n_bootstrap, bc.seed)
    models, paired, conf, pairs = {}, {}, {}, {}
    for name, r in recs.items():
        m10 = metrics_mixed(r.class10, r.pred10, CLASS10, draws, r.folder)
        m6 = metrics_mixed(r.true6, r.pred6, CLASS6, draws, r.folder)
        models[name] = {view: {k: m[k] for k in ("macro_f1", "macro_f1_ci", "balanced_accuracy", "balanced_accuracy_ci",
                                                 "per_class_f1")} for view, m in (("class10", m10), ("class6", m6))}
        models[name]["per_fold"] = {int(k): S3A.fold_metrics(g) for k, g in r.groupby("dev_fold")}
        cm = confusion_matrix(r.class10, r.pred10, labels=CLASS10)
        conf[name] = pd.DataFrame(cm, index=CLASS10, columns=CLASS10)
        i, v = CLASS10.index("imbalance"), CLASS10.index("vertical_misalignment")
        hm05 = r[(r.class10 == "horizontal_misalignment") & np.isclose(sev.loc[r.index], 0.5)]
        pairs[name] = {"imbalance_called_vertical_misalignment": int(cm[i, v]), "n_imbalance": int(cm[i].sum()),
                       "vertical_misalignment_called_imbalance": int(cm[v, i]), "n_vertical_misalignment": int(cm[v].sum()),
                       "horizontal_misalignment_0p5mm_correct": int((hm05.pred10 == hm05.class10).sum()),
                       "n_horizontal_misalignment_0p5mm": len(hm05),
                       "horizontal_misalignment_0p5mm_predictions": hm05.pred10.value_counts().to_dict()}
    for v, s in runs:
        a, b = recs[f"{v}_s{s}"], recs[f"cnn6_s{s}"]
        paired[f"{v}_s{s}_minus_cnn6_s{s}"] = {
            "class10": paired_mixed(base.class10, a.pred10, b.pred10, CLASS10, draws),
            "class6": paired_mixed(base.true6, a.pred6, b.pred6, CLASS6, draws)}
    across = {}
    ft_seeds = [s for v, s in runs if v == "r18ft"]
    if len(ft_seeds) > 1:
        for view in ("class10", "class6"):
            for name in ("r18ft", "cnn6"):
                vals = [models[f"{name}_s{s}"][view]["macro_f1"] for s in ft_seeds]
                across[f"{name}_{view}_macro_f1"] = {"seeds": ft_seeds, "values": vals, "mean": float(np.mean(vals)),
                                                      "std": float(np.std(vals, ddof=1))}
            vals = [paired[f"r18ft_s{s}_minus_cnn6_s{s}"][view]["diff"] for s in ft_seeds]
            across[f"r18ft_minus_cnn6_{view}_diff"] = {"seeds": ft_seeds, "values": vals, "mean": float(np.mean(vals)),
                                                        "std": float(np.std(vals, ddof=1)),
                                                        "same_sign_all_seeds": len({np.sign(x) for x in vals}) == 1}
    OUT.mkdir(parents=True, exist_ok=True)
    for name, cm in conf.items():
        cm.to_csv(OUT / f"b1_confusion_{name}.csv")
    out = {"provenance": {"split_hash": split_hash(meta), "git_sha": git_sha(), "scope": S3A.TAG, "k": 3,
                          "n_recordings": len(order), "n_bootstrap": bc.n_bootstrap, "bootstrap_seed": bc.seed,
                          "bootstrap": "stratified mixed-unit as Stage 3A; identical draws for every model; intervals do "
                                       "not include training-run variation (seed spread reported separately)",
                          "cnn6_sources": {s: str(_cnn6_logits(s).relative_to(SYSTEM_CODE_DIR)) for s in seeds}},
           "models": models, "paired": paired, "pairs": pairs, "across_seeds": across}
    (OUT / "b1_eval.json").write_text(json.dumps(out, indent=1, default=float))
    for name in models:
        c10, c6 = models[name]["class10"], models[name]["class6"]
        print(f"{name:14s} class10 {c10['macro_f1']:.3f} [{c10['macro_f1_ci'][0]:.3f}, {c10['macro_f1_ci'][1]:.3f}] "
              f"| class6 {c6['macro_f1']:.3f} [{c6['macro_f1_ci'][0]:.3f}, {c6['macro_f1_ci'][1]:.3f}] | "
              f"imb→vm {pairs[name]['imbalance_called_vertical_misalignment']}/{pairs[name]['n_imbalance']} "
              f"vm→imb {pairs[name]['vertical_misalignment_called_imbalance']}/{pairs[name]['n_vertical_misalignment']} "
              f"hm0.5 {pairs[name]['horizontal_misalignment_0p5mm_correct']}/{pairs[name]['n_horizontal_misalignment_0p5mm']}")
    for name, d in paired.items():
        print(f"{name:26s} class10 {d['class10']['diff']:+.3f} [{d['class10']['ci'][0]:+.3f}, {d['class10']['ci'][1]:+.3f}] "
              f"| class6 {d['class6']['diff']:+.3f} [{d['class6']['ci'][0]:+.3f}, {d['class6']['ci'][1]:+.3f}]")
    for k, v in across.items():
        print(f"  {k:32s} {v['mean']:+.4f} ± {v['std']:.4f}")


# ----------------------------------------------------------------------------------------------------------- B2
def loco_paths(c: str, seed: int = 42) -> RunPaths:
    return RunPaths("loco", "r18ft", seed, protocol="loco", withheld=c)


def run_loco(cfg: Config, seed: int = 42) -> None:
    import mlflow
    import torch

    from ml.datasets import dev_window_rows, load_representation
    from ml.paths import data_dir
    from ml.train import gpu_job, release_device_memory, score_windows, train_model
    from ml.utils import get_device, mlflow_run, seed_everything

    rc, device = load_resnet_config().model_copy(update={"seed": seed}), get_device()
    sha = verify_weights(rc)
    meta = pd.read_csv(SYSTEM_CODE_DIR / "metadata.csv")
    pool = S3A.dev_pool(cfg, meta)
    assignment = pd.read_csv(ASSIGNMENT)
    arr, index = load_representation(data_dir() / "representations", cfg, meta)
    rows = dev_window_rows(index, pool)
    sh = split_hash(meta)
    todo = [(c, k) for c in WITHHELD for k in (0, 1, 2) if not loco_paths(c, seed).scores(f"fold{k}_recording").exists()]
    print(f"{len(todo)} LOCO runs to train")
    if not todo:
        return
    with gpu_job(device, artifacts_dir(), rc.mps_memory_fraction), mlflow_run(cfg, run_name=f"loco_r18ft_s{seed}") as parent:
        mlflow.set_tags({"stage": "ML-5", "question": "Q3 (targeted)", "family": "loco", "seed": str(seed),
                         "split_hash": sh, "weights_sha256": sha})
        pos = rows.row.to_numpy()
        order = np.argsort(pos)
        x = np.empty((len(pos), *arr.shape[1:]), dtype=np.float16)
        x[order] = arr[pos[order]]
        cache = torch.from_numpy(x).to(device)  # dev-pool windows only
        del x
        for c, k in todo:
            rp = loco_paths(c, seed)
            for p in (rp.weights(k), rp.norm_stats(k), rp.scores("x")):
                p.parent.mkdir(parents=True, exist_ok=True)
            ids = loco_run_ids(assignment, pool, c, k)
            known = [x_ for x_ in CLASS10 if x_ != c]
            y = torch.from_numpy(rows.class10.map({x_: i for i, x_ in enumerate(known)}).fillna(0).to_numpy(np.int64)).to(device)
            st = norm_stats(arr, index, meta, fit_ids=ids["fit"], context=ids["context"])
            mean, std = st["mean"][VIB], st["std"][VIB]
            rp.norm_stats(k).write_text(json.dumps({"fit_ids": ids["fit"], "n_windows": st["n_windows"],
                                                    "mean": mean.tolist(), "std": std.tolist()}))
            role_of = {r: role for role in ("fit", *LOCO_ROLES) for r in ids[role]}
            sel = rows.rec_id.map(role_of)
            fit_rows = np.flatnonzero((sel == "fit").to_numpy())
            score_rows = np.flatnonzero(sel.notna().to_numpy())
            assert c not in set(rows.class10.iloc[fit_rows])
            run_id = rp.run_name(k)
            with mlflow.start_run(run_name=run_id, nested=True, experiment_id=parent.info.experiment_id):
                mlflow.set_tags({"stage": "ML-5", **rp.tags(k), "split_hash": sh, "git_sha": git_sha()})
                seed_everything(seed)
                model = ResNet18Spectrogram(len(VIB), len(known), rc, "r18ft").to(device)
                t0 = time.perf_counter()
                model, hist = train_model(cache, y, fit_rows, VIB, mean, std, len(known), rc, device,
                                          on_epoch=lambda ep, r: mlflow.log_metrics(
                                              {"train_loss": r["train_loss"], "train_acc": r["train_acc"]}, step=ep),
                                          model=model, param_groups=param_groups(model, rc))
                if device.type == "mps":
                    torch.mps.synchronize()
                t_train = time.perf_counter() - t0
                logits, emb = score_windows(model, cache, score_rows, VIB, mean, std, rc, device)
                torch.save(model.state_dict(), rp.weights(k))
                w = rows.iloc[score_rows]
                wt = pd.DataFrame({"rec_id": w.rec_id.to_numpy(), "window_idx": w.window_idx.to_numpy(),
                                   "dev_fold": w.dev_fold.to_numpy(), "class10": w.class10.to_numpy(),
                                   "role": sel.iloc[score_rows].to_numpy(), "withheld": c, "outer_fold": k,
                                   "variant": "r18ft", "seed": seed, "run_id": run_id})
                for j, name in enumerate(known):
                    wt[f"logit_{name}"] = logits[:, j]
                wt.to_parquet(rp.scores(f"fold{k}_logits"), index=False)
                np.savez_compressed(rp.embeddings(k), embedding=emb, rec_id=wt.rec_id.to_numpy().astype(str),
                                    window_idx=wt.window_idx.to_numpy(), role=wt.role.to_numpy().astype(str),
                                    withheld=np.array(c), outer_fold=np.array(k), run_id=np.array(run_id))
                rec = loco_recording_scores(wt, emb, known, cfg.baselines.prob_floor)
                info = pool.set_index("rec_id").loc[rec.rec_id]
                rec = rec.assign(class10=info.class10.to_numpy(), folder=info.folder.to_numpy(),
                                 severity=info.severity.to_numpy(), withheld=c, outer_fold=k, seed=seed, run_id=run_id)
                rec.to_parquet(rp.scores(f"fold{k}_recording"), index=False)
                rp.fold_result(k, "train").write_text(json.dumps(
                    {"withheld": c, "fold": k, "known_classes": known, "n": {r_: len(ids[r_]) for r_ in ("fit", *LOCO_ROLES)},
                     "n_fit_windows": int(len(fit_rows)), "train_seconds": t_train, "history": hist,
                     "split_hash": sh, "git_sha": git_sha()}, indent=1))
                kn, un = rec[rec.role == "foldk_known"], rec[rec.role == "foldk_withheld"]
                auc = roc_auc_score(np.r_[np.zeros(len(kn)), np.ones(len(un))], np.r_[-kn.msp, -un.msp])
                mlflow.log_metrics({"train_seconds": t_train, "auroc_msp": auc})
                print(f"  {run_id}: fit {len(ids['fit'])} rec ({len(fit_rows)} windows), {t_train:.0f} s; MSP AUROC {auc:.3f}",
                      flush=True)
            del model
            release_device_memory(device)


def loco_recording_scores(wt: pd.DataFrame, emb: np.ndarray, known: list[str], floor: float) -> pd.DataFrame:
    """Recording scores of one LOCO run from its window table (every role) and embeddings: MSP and predicted class
    (mean log-probability over windows), Mahalanobis with per-class and with shared Ledoit–Wolf covariances (fit
    windows only; min over known classes per window; mean over windows)."""
    is_fit = (wt.role == "fit").to_numpy()
    ev = ~is_fit
    ev_ids = wt.rec_id[ev].to_numpy()
    logits = wt[[f"logit_{c}" for c in known]].to_numpy()
    rec = recording_msp(logits[ev], ev_ids, known, floor)
    pc, _ = mahalanobis_min_per_class(emb[is_fit], wt.class10[is_fit].to_numpy(), emb[ev])
    shd, _ = mahalanobis_min(emb[is_fit], wt.class10[is_fit].to_numpy(), emb[ev])
    rec["maha_pc"] = recording_mean(pc, ev_ids).loc[rec.index]
    rec["maha_shared"] = recording_mean(shd, ev_ids).loc[rec.index]
    role = wt[ev].groupby("rec_id").role.first()
    return rec.assign(role=role.loc[rec.index].to_numpy()).rename_axis("rec_id").reset_index()


def cnn6_loco_recordings(c: str, k: int, floor: float, pool: pd.DataFrame) -> pd.DataFrame:
    """CNN-6 (Stage 3C, seed 42) on the same run: MSP and shared Mahalanobis recomputed from its saved window logits and
    128-d embeddings with the same function as above (the shared variant must equal the saved 3C `maha`), plus the
    per-class variant."""
    rp = RunPaths("loco", "cnn6", 42, protocol="loco", withheld=c)
    wt = pd.read_parquet(rp.scores(f"fold{k}_logits"))
    z = np.load(rp.embeddings(k))
    assert (z["rec_id"] == wt.rec_id.to_numpy()).all() and (z["window_idx"] == wt.window_idx.to_numpy()).all()
    known = [x for x in CLASS10 if x != c]
    rec = loco_recording_scores(wt, z["embedding"], known, floor)
    saved = pd.read_parquet(rp.scores(f"fold{k}_recording")).set_index("rec_id")
    r = rec.set_index("rec_id")
    assert np.allclose(r.maha_shared, saved.maha.loc[r.index], rtol=1e-9), "shared Mahalanobis does not reproduce 3C"
    assert np.allclose(r.msp, saved.msp.loc[r.index], rtol=1e-9), "MSP does not reproduce 3C"
    info = pool.set_index("rec_id").loc[rec.rec_id]
    return rec.assign(class10=info.class10.to_numpy(), withheld=c, outer_fold=k)


def run_loco_eval(cfg: Config) -> None:
    meta = pd.read_csv(SYSTEM_CODE_DIR / "metadata.csv")
    pool = S3A.dev_pool(cfg, meta)
    floor = cfg.baselines.prob_floor
    rows, assigned = [], []
    for c in WITHHELD:
        for k in (0, 1, 2):
            models = {"r18ft": pd.read_parquet(loco_paths(c).scores(f"fold{k}_recording")),
                      "cnn6": cnn6_loco_recordings(c, k, floor, pool)}
            for m, g in models.items():
                kn, un = g[g.role == "foldk_known"], g[g.role == "foldk_withheld"]
                correct = (kn.pred == kn.class10).to_numpy()
                for s in ("msp", "maha_pc", "maha_shared"):
                    u_kn = (-kn.msp if s == "msp" else kn[s]).to_numpy()
                    u_un = (-un.msp if s == "msp" else un[s]).to_numpy()
                    rows.append({"model": m, "withheld": c, "outer_fold": k, "score": s,
                                 "auroc": float(roc_auc_score(np.r_[np.zeros(len(u_kn)), np.ones(len(u_un))], np.r_[u_kn, u_un])),
                                 "oscr": oscr(u_kn, correct, u_un), "n_known": len(kn), "n_withheld": len(un),
                                 "closed_set_accuracy_known": float(correct.mean())})
                for cls, n in un.pred.value_counts().items():
                    assigned.append({"model": m, "withheld": c, "outer_fold": k, "assigned": cls, "n": int(n)})
    pf = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    pf.to_csv(OUT / "b2_per_fold_auroc_oscr.csv", index=False)
    wide = pf.pivot_table(index=["withheld", "outer_fold", "score"], columns="model", values=["auroc", "oscr"])
    reading = {}
    for c in WITHHELD:
        r = {}
        for s in ("msp", "maha_pc", "maha_shared"):
            d = pf[(pf.withheld == c) & (pf.score == s)].pivot(index="outer_fold", columns="model", values="auroc")
            r[s] = {"folds_r18ft_higher": int((d.r18ft > d.cnn6).sum()), "r18ft": d.r18ft.round(4).tolist(),
                    "cnn6": d.cnn6.round(4).tolist()}
        r["helps (declared rule: per-class Mahalanobis or MSP higher in ≥ 2 of 3 folds)"] = bool(
            r["maha_pc"]["folds_r18ft_higher"] >= 2 or r["msp"]["folds_r18ft_higher"] >= 2)
        reading[c] = r
    at = pd.DataFrame(assigned).pivot_table(index=["withheld", "model"], columns="assigned", values="n", aggfunc="sum",
                                            fill_value=0)
    at.to_csv(OUT / "b2_withheld_assigned.csv")
    out = {"provenance": {"split_hash": split_hash(meta), "git_sha": git_sha(), "seed": 42,
                          "assignment": str(ASSIGNMENT.relative_to(SYSTEM_CODE_DIR)),
                          "cnn6": "Stage 3C seed-42 LOCO runs; MSP and shared Mahalanobis recomputed from its saved window "
                                  "outputs and checked equal to the saved 3C scores; per-class Mahalanobis added",
                          "mahalanobis": {"maha_pc": "per-class Ledoit–Wolf (brief's primary)",
                                          "maha_shared": "one shared within-class Ledoit–Wolf (Stage 3C)"},
                          "reading": "exploratory; 6 runs cannot establish a general Q3 improvement"},
           "reading": reading, "per_fold": rows,
           "fold_mean": pf.groupby(["model", "withheld", "score"])[["auroc", "oscr"]].mean().reset_index().to_dict("records")}
    (OUT / "b2_eval.json").write_text(json.dumps(out, indent=1, default=float))
    print(wide.round(3).to_string())
    print(at.to_string())
    print(json.dumps(reading, indent=1))


# ----------------------------------------------------------------------------------------------------------- B3
def run_cost(cfg: Config) -> None:
    from ml.models.cnn import ScratchCNN
    from ml.models.cnn import n_parameters as cnn_params

    out = {"runs": {}}
    for v in ("r18ft", "r18probe"):
        for s in (42, 43, 44):
            p = RunPaths("resnet", v, s).result("train")
            if p.exists():
                t = json.loads(p.read_text())
                out["runs"][f"{v}_s{s}"] = {"train_seconds_per_fold": [f["train_seconds"] for f in t["folds"]],
                                            "train_minutes_total": t["train_seconds_total"] / 60,
                                            "cpu_ms_per_recording_per_fold": [f["cpu_ms_per_recording"] for f in t["folds"]],
                                            "n_parameters": t["provenance"]["n_parameters"]}
    loco = [json.loads(loco_paths(c).fold_result(k, "train").read_text()) for c in WITHHELD for k in (0, 1, 2)
            if loco_paths(c).fold_result(k, "train").exists()]
    out["loco_r18ft_train_seconds"] = [t["train_seconds"] for t in loco]
    cnn = json.loads((results_dir() / "cnn" / "cnn6_dev_train.json").read_text()) \
        if (results_dir() / "cnn" / "cnn6_dev_train.json").exists() else None
    cc = cfg.cnn
    out["cnn6"] = {"n_parameters": cnn_params(ScratchCNN(6, len(CLASS10), cc)),
                   **({"train_seconds_per_fold": [f["train_seconds"] for f in cnn["folds"]],
                       "cpu_ms_per_recording_per_fold": [f["cpu_ms_per_recording"] for f in cnn["folds"]]} if cnn else {})}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "b3_cost.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("task", choices=["train", "evaluate", "loco", "loco-eval", "cost"])
    ap.add_argument("--variant", choices=["r18ft", "r18probe"], default="r18ft")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    a = ap.parse_args()
    cfg = load_config(a.config)
    {"train": lambda: run_train(cfg, a.variant, a.seed), "evaluate": lambda: run_evaluate(cfg),
     "loco": lambda: run_loco(cfg), "loco-eval": lambda: run_loco_eval(cfg), "cost": lambda: run_cost(cfg)}[a.task]()
    return 0


if __name__ == "__main__":
    sys.exit(main())
