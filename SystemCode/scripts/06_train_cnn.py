"""Scratch 2D CNN, early fusion, on cache C: Q2 development evaluation (Stage 3A). Never imports LightGBM.

    caffeinate -i env MACHINESENSE_DEVICE=mps python scripts/06_train_cnn.py smoke
    caffeinate -i env MACHINESENSE_DEVICE=mps python scripts/06_train_cnn.py train --variant cnn7 --seed 43
    python scripts/06_train_cnn.py evaluate          # Stage 3A, seed 42 (original paths)
    python scripts/06_train_cnn.py seeds             # Stage 3B seed replication (42, 43, 44)

Protocol (CLAUDE.md §5 rules 12–13, §7): development pool and results/dev_folds.csv only (K = 3). For each outer fold k
the model is trained on the windows of the other folds for a FIXED number of epochs (no early stopping, nothing chosen
on fold k), normalised by norm_stats in DEVELOPMENT mode with fit_ids = those training recordings, and scores fold k
only: every pool recording is scored exactly once, by its own fold's model. One seed.

SCOPE: trained on all ten classes → outputs are Q2-only and NOT valid for LOCO / Q3 (every output is tagged).

Outputs (Stage 3B path scheme, ml.paths.RunPaths, name = <variant>_s<seed>_dev): results/scores/<name>_logits.parquet
(per-window logits) and <name>_embeddings.npz (128-d z), each row with rec_id, window_idx, dev_fold, run_id;
artifacts/cnn/<name>/fold<k>.pt (state_dict); results/cnn/<name>/fold<k>_norm.json; results/cnn/<name>_train.json;
MLflow runs <name> and <name>_fold<k>. Stage 3A's seed-42 outputs keep their original paths (<variant>_dev…, read by
`evaluate` and, for seed 42, by `seeds`); results/cnn/dev_eval.json (evaluate), results/cnn/seed_replication.json (seeds).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from ml.cache import load_features
from ml.classical import logreg_pipeline
from ml.config import Config, load_config
from ml.datasets import NormContext, dev_window_rows, fold_fit_context, load_representation, norm_stats
from ml.evaluation import metrics_mixed, mixed_bootstrap_draws, paired_mixed, recording_predictions
from ml.features import feature_columns
from ml.labels import CLASS6, CLASS10
from ml.paths import DEFAULT_CONFIG, SYSTEM_CODE_DIR, RunPaths, data_dir, results_dir
from ml.splits import development_pool, split_hash
from ml.utils import git_sha

TAG = "Q2-only; trained on all 10 classes; not valid for LOCO"
OUT = results_dir() / "cnn"
SCORES = results_dir() / "scores"
ART = SYSTEM_CODE_DIR / "artifacts" / "cnn"
VARIANTS = {"cnn7": None, "cnn6": "vibration"}  # channel sets: all 7 / cfg.cnn.vibration_channels


# ----------------------------------------------------------------------------------------------------------- data
def dev_pool(cfg: Config, meta: pd.DataFrame) -> pd.DataFrame:
    dc = cfg.development
    folds = pd.read_csv(results_dir() / dc.folds_file)[["rec_id", "dev_fold"]]
    pool = development_pool(meta, dc.allowed_closed, dc.reserved_columns).merge(folds, on="rec_id")
    assert len(pool) == len(folds), "dev_folds.csv does not match the development pool"
    return pool


def channels_of(variant: str, cfg: Config) -> list[int]:
    return list(range(7)) if VARIANTS[variant] is None else list(cfg.cnn.vibration_channels)


def ids_hash(ids) -> str:
    return hashlib.sha256("\n".join(sorted(ids)).encode()).hexdigest()


def load_all(cfg: Config):
    meta = pd.read_csv(SYSTEM_CODE_DIR / "metadata.csv")
    pool = dev_pool(cfg, meta)
    arr, index = load_representation(data_dir() / "representations", cfg, meta)
    rows = dev_window_rows(index, pool)
    cache_key = json.loads(next((data_dir() / "representations").glob(
        f"{cfg.representation.leading_candidate}_*.json")).read_text())["cache_key"]
    return meta, pool, arr, index, rows, cache_key


def resident(arr: np.ndarray, rows: pd.DataFrame, device):
    """Dev-pool windows as ONE float16 tensor on the device (rows ordered like `rows`), and int64 labels."""
    import torch

    pos = rows.row.to_numpy()
    order = np.argsort(pos)  # read the memory map in file order, then restore the `rows` order
    x = np.empty((len(pos), *arr.shape[1:]), dtype=np.float16)
    x[order] = arr[pos[order]]
    y = rows.class10.map({c: i for i, c in enumerate(CLASS10)}).to_numpy()
    return torch.from_numpy(x).to(device), torch.from_numpy(y.astype(np.int64)).to(device)


# ----------------------------------------------------------------------------------------------------------- outputs
def legacy_seed42(variant: str) -> tuple[Path, Path]:
    """Stage 3A seed-42 score and embedding paths (before the RunPaths scheme); never rewritten."""
    return SCORES / f"{variant}_dev.parquet", SCORES / f"{variant}_dev_embeddings.npz"


def write_scores(p1: Path, p2: Path, rows: pd.DataFrame, idx: np.ndarray, logits: np.ndarray, emb: np.ndarray,
                 run_ids: np.ndarray, prov: dict) -> tuple[Path, Path]:
    import pyarrow as pa
    import pyarrow.parquet as pq

    sel = rows.iloc[idx]
    df = pd.DataFrame({"rec_id": sel.rec_id.to_numpy(), "window_idx": sel.window_idx.to_numpy(),
                       "dev_fold": sel.dev_fold.to_numpy(), "run_id": run_ids, "y_true": sel.class10.to_numpy()})
    for j, c in enumerate(CLASS10):
        df[f"logit_{c}"] = logits[:, j]
    table = pa.Table.from_pandas(df, preserve_index=False)
    md = {**(table.schema.metadata or {}),
          **{f"machinesense.{k}".encode(): str(v).encode() for k, v in {**prov, "scope": TAG}.items()}}
    p1.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table.replace_schema_metadata(md), p1)
    np.savez_compressed(p2, embedding=emb.astype(np.float32), rec_id=df.rec_id.to_numpy().astype(str),
                        window_idx=df.window_idx.to_numpy(), dev_fold=df.dev_fold.to_numpy(),
                        run_id=run_ids.astype(str), scope=np.array(TAG))
    return p1, p2


def check_outputs(p_scores: Path, p_emb: Path, n_expected: int) -> dict:
    """Re-read saved outputs: row counts, required columns, scope tag, finite values, one row per scored window."""
    import pyarrow.parquet as pq

    t = pq.read_table(p_scores)
    df = t.to_pandas()
    tag = t.schema.metadata[b"machinesense.scope"].decode()
    z = np.load(p_emb)
    ok = {"rows": len(df), "expected": n_expected, "scope_tag": tag,
          "columns_ok": {"rec_id", "window_idx", "dev_fold", "run_id"} <= set(df.columns),
          "unique_windows": not df.duplicated(["rec_id", "window_idx"]).any(),
          "finite": bool(np.isfinite(df.filter(like="logit_").to_numpy()).all() and np.isfinite(z["embedding"]).all()),
          "embedding_shape": list(z["embedding"].shape),
          "embedding_rows_match": bool((z["rec_id"] == df.rec_id.to_numpy()).all()
                                       and (z["window_idx"] == df.window_idx.to_numpy()).all())}
    if not (ok["rows"] == n_expected and ok["columns_ok"] and ok["unique_windows"] and ok["finite"]
            and ok["embedding_rows_match"] and tag == TAG):
        raise RuntimeError(f"saved outputs failed the check: {ok}")
    return ok


def cpu_latency_ms(model, cache_cpu_rows: np.ndarray, channels: list[int], mean, std, cfg, n_rec: int = 50) -> float:
    """Median CPU milliseconds to score one recording (9 windows, one batch) from its cached representation."""
    import torch

    from ml.train import normalise

    m = model.to("cpu").eval()
    mu, sd = torch.tensor(mean, dtype=torch.float32), torch.tensor(std, dtype=torch.float32)
    times = []
    with torch.no_grad():
        for r in range(min(n_rec, len(cache_cpu_rows) // 9)):
            x16 = torch.from_numpy(cache_cpu_rows[r * 9:(r + 1) * 9][:, channels])
            t0 = time.perf_counter()
            m(normalise(x16.to(torch.float32), mu, sd))
            times.append((time.perf_counter() - t0) * 1000)
    return float(np.median(times[1:] if len(times) > 1 else times))


def fold_metrics(rec: pd.DataFrame) -> dict:
    from sklearn.metrics import balanced_accuracy_score, f1_score

    p10 = [c for c in CLASS10 if c in set(rec.class10)]
    p6 = [c for c in CLASS6 if c in set(rec.true6)]
    return {"class10_macro_f1": float(f1_score(rec.class10, rec.pred10, labels=p10, average="macro", zero_division=0)),
            "class10_balanced_accuracy": float(balanced_accuracy_score(rec.class10, rec.pred10)),
            "class6_macro_f1": float(f1_score(rec.true6, rec.pred6, labels=p6, average="macro", zero_division=0)),
            "class6_balanced_accuracy": float(balanced_accuracy_score(rec.true6, rec.pred6)), "n_recordings": len(rec)}


def softmax(logits: np.ndarray) -> np.ndarray:
    z = logits.astype(np.float64) - logits.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


# ----------------------------------------------------------------------------------------------------------- tasks
def run_smoke(cfg: Config) -> None:
    """One fold (0), smoke_epochs epochs, a class-stratified subset of that fold's training recordings: checks that the
    loss decreases, measures MPS step time and memory, projects the full training time, and checks the saved outputs."""
    import torch

    from ml.train import gpu_job, score_windows, train_model
    from ml.utils import get_device, seed_everything

    cc, device = cfg.cnn, get_device()
    meta, pool, arr, index, rows, cache_key = load_all(cfg)
    k = 0
    rng = np.random.default_rng(cc.seed)
    train_pool = pool[pool.dev_fold != k]
    sub_ids = sorted(r for _, g in train_pool.groupby("class10")
                     for r in rng.choice(g.rec_id.to_numpy(), max(1, round(cc.smoke_fraction * len(g))), replace=False))
    st = norm_stats(arr, index, meta, fit_ids=sub_ids, context=NormContext(
        evaluation=frozenset(pool.loc[pool.dev_fold == k, "rec_id"])))
    tr_idx = np.flatnonzero(rows.rec_id.isin(sub_ids).to_numpy())
    ev_idx = np.flatnonzero((rows.dev_fold == k).to_numpy())
    full_train_windows = int((rows.dev_fold != k).sum())
    with gpu_job(device, ART, cc.mps_memory_fraction):
        seed_everything(cc.seed)
        t0 = time.perf_counter()
        cache, y = resident(arr, rows, device)
        load_s = time.perf_counter() - t0
        mem = {}
        model, hist = train_model(cache, y, tr_idx, list(range(7)), st["mean"], st["std"], len(CLASS10), cc, device,
                                  epochs=cc.smoke_epochs,
                                  on_epoch=lambda ep, r: print(f"  epoch {ep}: loss {r['train_loss']:.4f} acc "
                                                               f"{r['train_acc']:.3f} step {r['step_seconds'] * 1000:.1f} ms"))
        if device.type == "mps":
            torch.mps.synchronize()
            mem = {"current_allocated_mb": torch.mps.current_allocated_memory() / 2**20,
                   "driver_allocated_mb": torch.mps.driver_allocated_memory() / 2**20,
                   "recommended_max_mb": torch.mps.recommended_max_memory() / 2**20}
        t0 = time.perf_counter()
        logits, emb = score_windows(model, cache, ev_idx, list(range(7)), st["mean"], st["std"], cc, device)
        score_s = time.perf_counter() - t0
    step_s = float(np.median([h["step_seconds"] for h in hist[1:]] or [hist[0]["step_seconds"]]))
    steps_full = int(np.ceil(full_train_windows / cc.batch_size))
    per_fold_s = steps_full * cc.epochs * step_s
    prov = {"run": "smoke", "split_hash": split_hash(meta), "cache_key": cache_key, "git_sha": git_sha()}
    rp = RunPaths("cnn", "cnn7", cc.seed, protocol="smoke")
    p1, p2 = write_scores(rp.scores("logits"), rp.embeddings(), rows, ev_idx, logits, emb,
                          np.full(len(ev_idx), rp.run_name(0)), prov)
    check = check_outputs(p1, p2, len(ev_idx))
    rp.weights(0).parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), rp.weights(0))
    sd = torch.load(rp.weights(0), weights_only=True)
    check["state_dict_tensors"] = len(sd)
    out = {"device": str(device), "fold": k, "epochs": cc.smoke_epochs, "train_recordings": len(sub_ids),
           "train_windows": len(tr_idx), "resident_cache_mb": cache.numel() * 2 / 2**20, "resident_load_s": load_s,
           "history": hist, "loss_decreased": hist[-1]["train_loss"] < hist[0]["train_loss"],
           "median_step_ms": step_s * 1000, "first_epoch_step_ms": hist[0]["step_seconds"] * 1000,
           "memory": mem, "scoring_s": score_s, "scored_windows": len(ev_idx),
           "projection": {"full_train_windows_fold0": full_train_windows, "steps_per_epoch": steps_full,
                          "epochs": cc.epochs, "train_minutes_per_fold": per_fold_s / 60,
                          "train_minutes_per_variant_3_folds": 3 * per_fold_s / 60,
                          "train_minutes_both_variants": 6 * per_fold_s / 60},
           "outputs_check": check, **prov}
    OUT.mkdir(parents=True, exist_ok=True)
    rp.result("train").write_text(json.dumps(out, indent=1))
    print(json.dumps({k2: out[k2] for k2 in ("loss_decreased", "median_step_ms", "memory", "projection",
                                             "outputs_check")}, indent=1))


def run_train(cfg: Config, variant: str, seed: int) -> None:
    import mlflow
    import torch

    from ml.train import gpu_job, release_device_memory, score_windows, train_model
    from ml.utils import get_device, mlflow_run, seed_everything

    from ml.models.cnn import n_parameters

    cc, device = cfg.cnn.model_copy(update={"seed": seed}), get_device()
    rp = RunPaths("cnn", variant, seed)
    meta, pool, arr, index, rows, cache_key = load_all(cfg)
    ch = channels_of(variant, cfg)
    prov = {"variant": variant, "split_hash": split_hash(meta), "cache_key": cache_key, "git_sha": git_sha(),
            "seed": cc.seed, "channels": ch}
    folds = sorted(pool.dev_fold.unique())
    n = len(rows)
    all_logits, all_emb = np.zeros((n, len(CLASS10)), np.float32), np.zeros((n, cc.embedding_dim), np.float32)
    run_ids = np.empty(n, dtype=object)
    fold_info = []
    for p in (rp.weights(0), rp.norm_stats(0), rp.result("train")):
        p.parent.mkdir(parents=True, exist_ok=True)
    with gpu_job(device, ART, cc.mps_memory_fraction), mlflow_run(cfg, run_name=rp.run_name()) as parent:
        mlflow.set_tags({"stage": "3B", "scope": TAG, **rp.tags(), "split_hash": prov["split_hash"],
                         "cache_key": cache_key, "protocol": "dev folder-grouped K=3; scored once by own fold model"})
        cache, y = resident(arr, rows, device)
        for k in folds:
            fit_ids, _ = fold_fit_context(pool, k)
            st = norm_stats(arr, index, meta, fit_ids=fit_ids, context=fold_fit_context(pool, k)[1])
            mean, std = st["mean"][ch], st["std"][ch]
            tr_idx = np.flatnonzero((rows.dev_fold != k).to_numpy())
            ev_idx = np.flatnonzero((rows.dev_fold == k).to_numpy())
            assert set(rows.rec_id.iloc[tr_idx]) == set(fit_ids)
            run_id = rp.run_name(k)
            rp.norm_stats(k).write_text(json.dumps({"fit_ids_sha256": ids_hash(fit_ids), "n_fit_recordings": len(fit_ids),
                                                     "n_windows": st["n_windows"], "channels": ch,
                                                     "mean": mean.tolist(), "std": std.tolist()}, indent=1))
            with mlflow.start_run(run_name=run_id, nested=True, experiment_id=parent.info.experiment_id):
                mlflow.set_tags({"stage": "3B", "scope": TAG, **rp.tags(k),
                                 "fit_ids_sha256": ids_hash(fit_ids), "split_hash": prov["split_hash"],
                                 "cache_key": cache_key, "git_sha": prov["git_sha"]})
                seed_everything(cc.seed)
                t0 = time.perf_counter()
                model, hist = train_model(cache, y, tr_idx, ch, mean, std, len(CLASS10), cc, device,
                                          on_epoch=lambda ep, r: mlflow.log_metrics(
                                              {"train_loss": r["train_loss"], "train_acc": r["train_acc"]}, step=ep))
                if device.type == "mps":
                    torch.mps.synchronize()
                train_s = time.perf_counter() - t0
                logits, emb = score_windows(model, cache, ev_idx, ch, mean, std, cc, device)
                all_logits[ev_idx], all_emb[ev_idx], run_ids[ev_idx] = logits, emb, run_id
                torch.save(model.state_dict(), rp.weights(k))
                cpu_rows = np.asarray(arr[np.sort(rows.row.to_numpy()[ev_idx])[:9 * 50]])
                ms = cpu_latency_ms(model, cpu_rows, ch, mean, std, cc)
                rec = recording_predictions(softmax(logits), rows.iloc[ev_idx], cfg.baselines.prob_floor)
                fm = fold_metrics(rec)
                mlflow.log_metrics({**fm, "train_seconds": train_s, "cpu_ms_per_recording": ms})
                print(f"  {run_id}: {len(tr_idx)} train windows, {train_s:.0f} s, final loss "
                      f"{hist[-1]['train_loss']:.4f}, class10 macro-F1 {fm['class10_macro_f1']:.3f}, "
                      f"class6 {fm['class6_macro_f1']:.3f}, CPU {ms:.1f} ms/recording")
                fold_info.append({"fold": int(k), "run_id": run_id, "n_train_windows": len(tr_idx),
                                  "n_train_recordings": len(fit_ids), "fit_ids_sha256": ids_hash(fit_ids),
                                  "n_scored_windows": len(ev_idx), "train_seconds": train_s,
                                  "cpu_ms_per_recording": ms, "norm_mean": mean.tolist(), "norm_std": std.tolist(),
                                  "history": hist, "metrics": fm})
                model.to(device)
            release_device_memory(device)
        p1, p2 = write_scores(rp.scores("logits"), rp.embeddings(), rows, np.arange(n), all_logits, all_emb,
                              run_ids.astype(str), prov)
        check = check_outputs(p1, p2, n)
        rec = recording_predictions(softmax(all_logits), rows, cfg.baselines.prob_floor)
        assert len(rec) == len(pool)
        pooled = fold_metrics(rec)
        mlflow.log_metrics({f"pooled_{k2}": v for k2, v in pooled.items()})
        mlflow.log_metric("train_seconds_total", sum(f["train_seconds"] for f in fold_info))
        from ml.models.cnn import ScratchCNN
        out = {"provenance": {**prov, "scope": TAG, "device": str(device), "config": cc.model_dump(mode="json"),
                              "n_parameters": n_parameters(ScratchCNN(len(ch), len(CLASS10), cc)),
                              "mlflow_run_id": parent.info.run_id},
               "folds": fold_info, "pooled_point_metrics": pooled, "outputs_check": check}
        OUT.mkdir(parents=True, exist_ok=True)
        rp.result("train").write_text(json.dumps(out, indent=1))
        mlflow.log_artifact(str(rp.result("train")))
    print(f"{rp.name}: pooled class10 macro-F1 {pooled['class10_macro_f1']:.3f}, class6 {pooled['class6_macro_f1']:.3f}")


def lr_oof(cfg: Config, meta: pd.DataFrame, pool: pd.DataFrame) -> tuple[np.ndarray, pd.DataFrame]:
    """The classical reference refitted on the same folds: logistic regression on the capped R1 table (scikit-learn
    only). Out-of-fold window probabilities, rows ordered by rec_id then window_idx."""
    bc = cfg.baselines
    path = max((data_dir() / "features").glob(f"{bc.feature_table}_*.parquet"), key=lambda p: p.stat().st_mtime)
    df = load_features(path, meta, cfg.preprocess).merge(pool[["rec_id", "folder", "dev_fold"]], on="rec_id")
    df = df.sort_values(["rec_id", "window_idx"]).reset_index(drop=True)
    cols = feature_columns(df)
    proba = np.zeros((len(df), len(CLASS10)))
    for k in sorted(df.dev_fold.unique()):
        tr, te = (df.dev_fold != k).to_numpy(), (df.dev_fold == k).to_numpy()
        m = logreg_pipeline(bc).fit(df.loc[tr, cols].to_numpy(), df.class10[tr].to_numpy())
        p = m.predict_proba(df.loc[te, cols].to_numpy())
        proba[te] = p[:, [list(m.classes_).index(c) for c in CLASS10]]
    return proba, df


def cnn_recordings(path: Path, pool: pd.DataFrame, floor: float) -> pd.DataFrame:
    """Recording-level predictions (and dev_fold) from a saved per-window logits parquet; checks every pool recording is
    scored exactly once."""
    s = pd.read_parquet(path).sort_values(["rec_id", "window_idx"]).reset_index(drop=True)
    s = s.merge(pool[["rec_id", "folder"]], on="rec_id").rename(columns={"y_true": "class10"})
    assert s.rec_id.nunique() == len(pool) and not s.duplicated(["rec_id", "window_idx"]).any()
    rec = recording_predictions(softmax(s[[f"logit_{c}" for c in CLASS10]].to_numpy()), s, floor)
    rec["dev_fold"] = s.groupby("rec_id", sort=False).dev_fold.first().loc[rec.index]
    return rec


def lr_recordings(cfg: Config, meta: pd.DataFrame, pool: pd.DataFrame) -> pd.DataFrame:
    lp, ldf = lr_oof(cfg, meta, pool)
    rec = recording_predictions(lp, ldf, cfg.baselines.prob_floor)
    rec["dev_fold"] = ldf.groupby("rec_id", sort=False).dev_fold.first().loc[rec.index]
    return rec


def run_evaluate(cfg: Config) -> None:
    """Recording-level metrics with mixed-unit intervals for CNN-7, CNN-6 and the refitted logistic regression; paired
    comparisons over identical draws. No torch needed."""
    import mlflow

    from ml.utils import mlflow_run

    bc = cfg.baselines
    meta = pd.read_csv(SYSTEM_CODE_DIR / "metadata.csv")
    pool = dev_pool(cfg, meta)
    recs = {v: cnn_recordings(legacy_seed42(v)[0], pool, bc.prob_floor) for v in VARIANTS}
    recs["logreg"] = lr_recordings(cfg, meta, pool)
    order = sorted(pool.rec_id)
    recs = {m: r.loc[order] for m, r in recs.items()}
    base = recs["cnn7"]
    draws = mixed_bootstrap_draws(base.class10.to_numpy(), base.folder.to_numpy(), bc.n_bootstrap, bc.seed)
    res = {}
    for m, r in recs.items():
        res[m] = {"class10": metrics_mixed(r.class10, r.pred10, CLASS10, draws, r.folder),
                  "class6": metrics_mixed(r.true6, r.pred6, CLASS6, draws, r.folder),
                  "per_fold": {int(k): fold_metrics(g) for k, g in r.groupby("dev_fold")}}
    paired = {}
    for a, b in (("cnn7", "logreg"), ("cnn7", "cnn6"), ("cnn6", "logreg")):
        paired[f"{a}_minus_{b}"] = {
            "class10": paired_mixed(base.class10, recs[a].pred10, recs[b].pred10, CLASS10, draws),
            "class6": paired_mixed(base.true6, recs[a].pred6, recs[b].pred6, CLASS6, draws)}
    folder_acc = pd.DataFrame({m: {f: v["accuracy"] for f, v in res[m]["class10"]["per_folder_accuracy"].items()}
                               for m in recs})
    folder_acc["n"] = pd.Series({f: v["n"] for f, v in res["cnn7"]["class10"]["per_folder_accuracy"].items()})
    OUT.mkdir(parents=True, exist_ok=True)
    folder_acc.sort_values("cnn7").to_csv(OUT / "dev_per_folder_accuracy.csv")
    out = {"provenance": {"split_hash": split_hash(meta), "git_sha": git_sha(), "scope": TAG, "k": 3,
                          "n_recordings": len(order), "n_bootstrap": bc.n_bootstrap, "bootstrap_seed": bc.seed,
                          "bootstrap": "stratified mixed-unit (fault folders within class; normal recordings "
                                       "individually); identical draws for every model; one CNN training seed, so the "
                                       "intervals do not include run-to-run training variation",
                          "logreg": f"refitted on the same folds ({bc.feature_table}, baselines.logreg)"},
           "models": res, "paired": paired}
    (OUT / "dev_eval.json").write_text(json.dumps(out, indent=1))
    with mlflow_run(cfg, run_name="cnn_dev_eval"):
        mlflow.set_tags({"stage": "3A", "scope": TAG, "split_hash": out["provenance"]["split_hash"]})
        for m in res:
            for v in ("class10", "class6"):
                mlflow.log_metrics({f"{m}_{v}_macro_f1": res[m][v]["macro_f1"],
                                    f"{m}_{v}_balanced_accuracy": res[m][v]["balanced_accuracy"]})
        for name, d in paired.items():
            mlflow.log_metrics({f"{name}_{v}_diff": d[v]["diff"] for v in ("class10", "class6")})
        mlflow.log_artifact(str(OUT / "dev_eval.json"))
    for m in res:
        c10, c6 = res[m]["class10"], res[m]["class6"]
        print(f"{m:7s} class10 macro-F1 {c10['macro_f1']:.3f} [{c10['macro_f1_ci'][0]:.3f}, {c10['macro_f1_ci'][1]:.3f}] "
              f"bal.acc {c10['balanced_accuracy']:.3f} | class6 {c6['macro_f1']:.3f} "
              f"[{c6['macro_f1_ci'][0]:.3f}, {c6['macro_f1_ci'][1]:.3f}]")
    for name, d in paired.items():
        print(f"{name:18s} class10 {d['class10']['diff']:+.3f} [{d['class10']['ci'][0]:+.3f}, {d['class10']['ci'][1]:+.3f}] "
              f"P(>0) {d['class10']['p_positive']:.2f} | class6 {d['class6']['diff']:+.3f} "
              f"[{d['class6']['ci'][0]:+.3f}, {d['class6']['ci'][1]:+.3f}]")
    print(folder_acc.sort_values("cnn7").head(8).round(3).to_string())


SEEDS = (42, 43, 44)


def run_seeds(cfg: Config) -> None:
    """Seed replication (Stage 3B): per seed, pooled class10/class6 macro-F1 of CNN-7 and CNN-6 with mixed-unit
    intervals and paired differences (CNN-7 − CNN-6, each CNN − logistic regression) over the SAME draws; then mean ±
    std across seeds. Seed variation and bootstrap intervals are reported separately, never merged."""
    bc = cfg.baselines
    meta = pd.read_csv(SYSTEM_CODE_DIR / "metadata.csv")
    pool = dev_pool(cfg, meta)
    order = sorted(pool.rec_id)
    lr = lr_recordings(cfg, meta, pool).loc[order]
    draws = mixed_bootstrap_draws(lr.class10.to_numpy(), lr.folder.to_numpy(), bc.n_bootstrap, bc.seed)
    per_seed, sources = {}, {}
    for seed in SEEDS:
        recs = {}
        for v in VARIANTS:
            p = RunPaths("cnn", v, seed).scores("logits")
            if seed == 42 and not p.exists():
                p = legacy_seed42(v)[0]
            sources[f"{v}_s{seed}"] = str(p.relative_to(SYSTEM_CODE_DIR))
            recs[v] = cnn_recordings(p, pool, bc.prob_floor).loc[order]
        m = {v: {"class10": metrics_mixed(r.class10, r.pred10, CLASS10, draws, r.folder),
                 "class6": metrics_mixed(r.true6, r.pred6, CLASS6, draws, r.folder)} for v, r in recs.items()}
        recs["logreg"] = lr
        paired = {f"{a}_minus_{b}": {
            "class10": paired_mixed(lr.class10, recs[a].pred10, recs[b].pred10, CLASS10, draws),
            "class6": paired_mixed(lr.true6, recs[a].pred6, recs[b].pred6, CLASS6, draws)}
            for a, b in (("cnn7", "cnn6"), ("cnn7", "logreg"), ("cnn6", "logreg"))}
        per_seed[seed] = {"models": {v: {view: {k: m[v][view][k] for k in ("macro_f1", "macro_f1_ci", "balanced_accuracy",
                                                                            "balanced_accuracy_ci", "per_class_f1")}
                                         for view in ("class10", "class6")} for v in m},
                          "paired": paired,
                          "per_folder_accuracy": {v: {f: d["accuracy"] for f, d in m[v]["class10"]["per_folder_accuracy"].items()}
                                                  for v in m}}
    across = {}
    for v in VARIANTS:
        for view in ("class10", "class6"):
            vals = [per_seed[s]["models"][v][view]["macro_f1"] for s in SEEDS]
            across[f"{v}_{view}_macro_f1"] = {"values": vals, "mean": float(np.mean(vals)), "std": float(np.std(vals, ddof=1))}
    for name in per_seed[SEEDS[0]]["paired"]:
        for view in ("class10", "class6"):
            vals = [per_seed[s]["paired"][name][view]["diff"] for s in SEEDS]
            across[f"{name}_{view}_diff"] = {"values": vals, "mean": float(np.mean(vals)), "std": float(np.std(vals, ddof=1)),
                                             "same_sign_all_seeds": bool(len({np.sign(x) for x in vals}) == 1)}
    out = {"provenance": {"split_hash": split_hash(meta), "git_sha": git_sha(), "scope": TAG, "seeds": list(SEEDS),
                          "sources": sources, "n_bootstrap": bc.n_bootstrap, "bootstrap_seed": bc.seed,
                          "uncertainty": "bootstrap intervals (per seed) describe the evaluated recordings and folders; "
                                         "the std across seeds describes CNN training-run variation; they are different "
                                         "sources and are reported separately, never merged",
                          "logreg": "refitted on the same folds (deterministic; identical for every seed)"},
           "per_seed": {str(k): v for k, v in per_seed.items()}, "across_seeds": across}
    (OUT / "seed_replication.json").write_text(json.dumps(out, indent=1))
    for seed in SEEDS:
        ps = per_seed[seed]
        line = " | ".join(f"{v} {ps['models'][v]['class10']['macro_f1']:.3f}/{ps['models'][v]['class6']['macro_f1']:.3f}"
                          for v in VARIANTS)
        d = ps["paired"]
        print(f"seed {seed}: {line} | " + " | ".join(
            f"{n} {d[n]['class10']['diff']:+.3f} [{d[n]['class10']['ci'][0]:+.3f}, {d[n]['class10']['ci'][1]:+.3f}]"
            for n in d))
    for k, v in across.items():
        print(f"  {k:32s} {v['mean']:+.4f} ± {v['std']:.4f}" + (f"  same sign: {v['same_sign_all_seeds']}"
                                                                 if "same_sign_all_seeds" in v else ""))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("task", choices=["smoke", "train", "evaluate", "seeds"])
    ap.add_argument("--variant", choices=list(VARIANTS), default="cnn7")
    ap.add_argument("--seed", type=int, default=None, help="training seed (default: cnn.seed in the config)")
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    a = ap.parse_args()
    cfg = load_config(a.config)
    seed = cfg.cnn.seed if a.seed is None else a.seed
    {"smoke": lambda: run_smoke(cfg), "train": lambda: run_train(cfg, a.variant, seed),
     "evaluate": lambda: run_evaluate(cfg), "seeds": lambda: run_seeds(cfg)}[a.task]()
    return 0


if __name__ == "__main__":
    sys.exit(main())
