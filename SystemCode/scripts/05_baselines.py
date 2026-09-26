"""Closed-set classical baselines on the R1 physics features (Stage 2B). LightGBM lives here; this file never imports
torch (CLAUDE.md §2).

    python scripts/05_baselines.py closed   # A0 + A1 rungs, sensitivity, envelope diagnostic, permutation importance
    python scripts/05_baselines.py leaky    # deliberate leakage illustration: window-level vs recording-grouped CV
    python scripts/05_baselines.py dev-inventory  # development pool (Stage 2B.1): recordings by class and folder proxy
    python scripts/05_baselines.py dev-folds      # folder-grouped development folds → results/dev_folds.csv
    python scripts/05_baselines.py dev            # out-of-fold evaluation of the classical models on the dev pool
    python scripts/05_baselines.py dev-compare    # LightGBM comparisons (a)–(d) under the same folds
    python scripts/05_baselines.py dev-protocol   # evaluation-protocol comparison (for the deck; not a leakage estimate)
    python scripts/05_baselines.py dev-lr-capped  # logistic regression vs LightGBM on the capped table, same dev folds

Train on split_closed train windows; evaluate on split_closed VAL. The test split stays sealed: its rows are dropped
right after loading. Outputs: results/baselines/*.json|csv, results/scores/<run_name>.parquet (train/val windows only),
MLflow runs in sqlite:///mlruns.db.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold, KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import QuantileTransformer
from sklearn.svm import SVC

from ml.cache import load_features, read_cache_metadata
from ml.classical import logreg_pipeline
from ml.config import BaselineConfig, Config, PreprocessConfig, load_config
from ml.evaluation import (CLASS10_TO_6, aggregate_recordings, block_permutation_importance, metrics,
                           metrics_mixed, mixed_bootstrap_draws, paired_bootstrap, paired_mixed, recording_predictions,
                           to_class6)
from ml.features import ACCEL_CHANNELS, feature_columns
from ml.labels import CLASS6, CLASS10
from ml.paths import CONFIGS_DIR, DEFAULT_CONFIG, SYSTEM_CODE_DIR, data_dir, results_dir
from ml.splits import assign_dev_folds, development_pool, split_hash
from ml.utils import git_sha

OUT = results_dir() / "baselines"
SCORES = results_dir() / "scores"


# ----------------------------------------------------------------------------- data
def load_table(name: str, pre: PreprocessConfig, meta: pd.DataFrame) -> tuple[Path, pd.DataFrame]:
    """Validated feature table (current labels and splits joined), TRAIN and VAL rows only, ordered by recording
    and window."""
    path = max((data_dir() / "features").glob(f"{name}_*.parquet"), key=lambda p: p.stat().st_mtime)
    df = load_features(path, meta, pre)
    df = df[df.split_closed.isin(["train", "val"])].sort_values(["rec_id", "window_idx"]).reset_index(drop=True)
    assert not (df.split_closed == "test").any()
    return path, df


def xy(df: pd.DataFrame, cols: list[str]) -> tuple[np.ndarray, np.ndarray]:
    return df[cols].to_numpy(np.float64), df.class10.map({c: i for i, c in enumerate(CLASS10)}).to_numpy()


# ----------------------------------------------------------------------------- models
def quantile(bc: BaselineConfig) -> QuantileTransformer:
    return QuantileTransformer(n_quantiles=bc.quantile_n, output_distribution="normal", random_state=bc.seed)


def build(name: str, bc: BaselineConfig, groups: np.ndarray, X: np.ndarray, y: np.ndarray):
    cw = bc.class_weight
    if name == "majority":
        return DummyClassifier(strategy="most_frequent")
    if name == "stratified":
        return DummyClassifier(strategy="stratified", random_state=bc.seed)
    if name == "lgbm":
        return lgb.LGBMClassifier(**bc.lgbm, class_weight=cw, random_state=bc.seed, n_jobs=-1, verbose=-1)
    if name == "rf":
        return RandomForestClassifier(**bc.rf, class_weight=cw, random_state=bc.seed, n_jobs=-1)
    if name == "svm":
        # No SVC(probability=True): its internal random folds would mix windows of one recording. Calibrate on
        # recording-grouped folds instead; the scaler is refitted inside each fold (it is part of the pipeline).
        pipe = Pipeline([("scale", quantile(bc)),
                         ("svc", SVC(kernel="rbf", **bc.svm, class_weight=cw, random_state=bc.seed))])
        folds = list(GroupKFold(n_splits=bc.svm_calibration["n_group_folds"]).split(X, y, groups))
        return CalibratedClassifierCV(pipe, method=bc.svm_calibration["method"], cv=folds)
    if name == "knn":
        return Pipeline([("scale", quantile(bc)), ("knn", KNeighborsClassifier(**bc.knn, n_jobs=-1))])
    if name == "logreg":
        return logreg_pipeline(bc)
    raise ValueError(name)


def proba10(model, X: np.ndarray) -> np.ndarray:
    """predict_proba with columns in CLASS10 order (classes a model never saw get 0)."""
    p = model.predict_proba(X)
    out = np.zeros((len(X), len(CLASS10)))
    out[:, np.asarray(model.classes_, dtype=int)] = p
    return out


# ----------------------------------------------------------------------------- evaluation
def evaluate(proba: np.ndarray, df: pd.DataFrame, bc: BaselineConfig) -> tuple[dict, pd.Series]:
    """Recording-level class10 and class6 metrics (bootstrap over recordings) and window-level macro-F1."""
    y_win = df.class10.to_numpy()
    present = [c for c in CLASS10 if c in set(y_win)]
    win_pred = np.asarray(CLASS10)[proba.argmax(axis=1)]
    agg = aggregate_recordings(proba, df.rec_id.to_numpy(), bc.prob_floor)
    y_rec = df.groupby("rec_id").class10.first().loc[agg.index]
    pred10 = agg.idxmax(axis=1)
    pred6 = to_class6(agg).idxmax(axis=1)
    out = {
        "class10": metrics(y_rec, pred10, CLASS10, bc.n_bootstrap, bc.seed),
        "class6": metrics(y_rec.map(CLASS10_TO_6), pred6, CLASS6, bc.n_bootstrap, bc.seed),
        "window_macro_f1": float(f1_score(y_win, win_pred, labels=present, average="macro", zero_division=0)),
    }
    return out, pred10


def save_scores(run: str, df: pd.DataFrame, proba: np.ndarray, extra: dict) -> Path:
    """Per-window scores for TRAIN/VAL rows only, with split hash and provenance in the parquet metadata."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    assert set(df.split_closed) <= {"train", "val"}
    s = pd.DataFrame({"rec_id": df.rec_id.to_numpy(), "window_idx": df.window_idx.to_numpy(),
                      "split": df.split_closed.to_numpy(), "y_true": df.class10.to_numpy()})
    for j, c in enumerate(CLASS10):
        s[f"score_{c}"] = proba[:, j]
    table = pa.Table.from_pandas(s, preserve_index=False)
    meta = {**(table.schema.metadata or {}), **{f"machinesense.{k}".encode(): str(v).encode() for k, v in extra.items()}}
    SCORES.mkdir(parents=True, exist_ok=True)
    path = SCORES / f"{run}.parquet"
    pq.write_table(table.replace_schema_metadata(meta), path)
    return path


def log_mlflow(cfg: Config, run: str, res: dict, extra_tags: dict, train_s: float, ms_per_rec: float) -> None:
    import mlflow

    from ml.utils import mlflow_run

    with mlflow_run(cfg, run_name=run, experiment="closed_baselines"):
        mlflow.set_tags(extra_tags)
        mlflow.log_metrics({
            "val_rec_macro_f1_class10": res["class10"]["macro_f1"],
            "val_rec_macro_f1_class10_ci_lo": res["class10"]["macro_f1_ci"][0],
            "val_rec_macro_f1_class10_ci_hi": res["class10"]["macro_f1_ci"][1],
            "val_rec_macro_f1_class6": res["class6"]["macro_f1"],
            "val_rec_balanced_acc_class10": res["class10"]["balanced_accuracy"],
            "val_window_macro_f1": res["window_macro_f1"],
            "train_seconds": train_s, "ms_per_recording": ms_per_rec})


def families(cols: list[str]) -> dict[str, list[int]]:
    """Channel families for permutation importance: each accelerometer channel's features, each bearing's
    axial/radial ratio, each envelope channel, and the microphone."""
    fam: dict[str, list[int]] = {}
    for i, c in enumerate(cols):
        if c.startswith("f_env_"):
            key = "envelope " + c[len("f_env_"):].rsplit("_", 1)[0].split("_")[0] + "_radial"
        elif c.startswith("f_mic_"):
            key = "microphone"
        elif "_ratio_axial_radial_" in c:
            key = c.split("_")[1] + " axial/radial ratio"
        else:
            key = next(ch for ch in ACCEL_CHANNELS if c.startswith(f"f_{ch}_"))
        fam.setdefault(key, []).append(i)
    return fam


# ----------------------------------------------------------------------------- tasks
def run_closed(cfg: Config) -> None:
    bc, pre = cfg.baselines, cfg.preprocess
    meta = pd.read_csv(SYSTEM_CODE_DIR / "metadata.csv")
    path, df = load_table(bc.feature_table, pre, meta)
    key = read_cache_metadata(path)["cache_key"]
    cols = feature_columns(df)
    tr, va = df[df.split_closed == "train"], df[df.split_closed == "val"]
    Xtr, ytr = xy(tr, cols)
    Xva, _ = xy(va, cols)
    groups = tr.rec_id.to_numpy()
    n_val_rec = va.rec_id.nunique()
    common = {"split_hash": split_hash(meta), "cache_key": key, "feature_table": path.name, "git_sha": git_sha()}
    print(f"{path.name}: train {tr.rec_id.nunique()} recordings / {len(tr)} windows, val {n_val_rec} / {len(va)}, "
          f"{len(cols)} features")

    results, val_preds, val_proba, train_proba = {}, {}, {}, {}
    for name in ["majority", "stratified", "lgbm", "rf", "svm", "knn", "logreg"]:
        t = time.perf_counter()
        model = build(name, bc, groups, Xtr, ytr).fit(Xtr, ytr)
        train_s = time.perf_counter() - t
        t = time.perf_counter()
        pva = proba10(model, Xva)
        ms = (time.perf_counter() - t) * 1000 / n_val_rec
        ptr = proba10(model, Xtr)
        res, pred = evaluate(pva, va, bc)
        res.update(train_seconds=train_s, ms_per_recording=ms)
        results[name], val_preds[name], val_proba[name], train_proba[name] = res, pred, pva, ptr
        run = f"closed_{bc.feature_table}_{name}"
        save_scores(run, pd.concat([tr, va]), np.vstack([ptr, pva]), {**common, "run_name": run})
        log_mlflow(cfg, run, res, {**common, "rung": "A0" if name in ("majority", "stratified") else "A1"}, train_s, ms)
        print(f"  {name:10s} class10 macro-F1 {res['class10']['macro_f1']:.3f} "
              f"[{res['class10']['macro_f1_ci'][0]:.3f}, {res['class10']['macro_f1_ci'][1]:.3f}]  "
              f"class6 {res['class6']['macro_f1']:.3f}  window {res['window_macro_f1']:.3f}  ({train_s:.0f} s)")

    # Soft-vote ensemble with fixed equal weights over the A1 models' window probabilities.
    pens = np.mean([val_proba[m] for m in bc.ensemble], axis=0)
    res, pred = evaluate(pens, va, bc)
    results["ensemble"], val_preds["ensemble"] = res, pred
    run = f"closed_{bc.feature_table}_ensemble"
    save_scores(run, pd.concat([tr, va]), np.vstack([np.mean([train_proba[m] for m in bc.ensemble], axis=0), pens]),
                {**common, "run_name": run})
    log_mlflow(cfg, run, res, {**common, "rung": "A1"}, sum(results[m]["train_seconds"] for m in bc.ensemble),
               sum(results[m]["ms_per_recording"] for m in bc.ensemble))
    print(f"  {'ensemble':10s} class10 macro-F1 {res['class10']['macro_f1']:.3f} "
          f"[{res['class10']['macro_f1_ci'][0]:.3f}, {res['class10']['macro_f1_ci'][1]:.3f}]  "
          f"class6 {res['class6']['macro_f1']:.3f}  window {res['window_macro_f1']:.3f}")
    y_rec = va.groupby("rec_id").class10.first().loc[val_preds["lgbm"].index]

    # Sensitivity: primary vs capped table (LightGBM, same hyperparameters).
    cap_pre = load_config(CONFIGS_DIR / bc.sensitivity_config).preprocess
    cpath, cdf = load_table(bc.sensitivity_table, cap_pre, meta)
    ccols = feature_columns(cdf)
    ctr, cva = cdf[cdf.split_closed == "train"], cdf[cdf.split_closed == "val"]
    cmodel = build("lgbm", bc, None, None, None).fit(*xy(ctr, ccols))
    cres, cpred = evaluate(proba10(cmodel, xy(cva, ccols)[0]), cva, bc)
    sens = {"table": cpath.name, "capped": cres, "primary": results["lgbm"],
            "paired_primary_minus_capped": paired_bootstrap(y_rec, val_preds["lgbm"].loc[y_rec.index],
                                                            cpred.loc[y_rec.index], CLASS10, bc.n_bootstrap, bc.seed)}
    print(f"  sensitivity: capped class10 {cres['class10']['macro_f1']:.3f}; primary − capped "
          f"{sens['paired_primary_minus_capped']['diff']:+.3f} {sens['paired_primary_minus_capped']['ci']}")

    # Diagnostic: LightGBM without the envelope features.
    ncols = [c for c in cols if not c.startswith("f_env_")]
    nmodel = build("lgbm", bc, None, None, None).fit(*xy(tr, ncols))
    nres, npred = evaluate(proba10(nmodel, xy(va, ncols)[0]), va, bc)
    env = {"without_envelope": nres, "n_features_without": len(ncols),
           "paired_with_minus_without": paired_bootstrap(y_rec, val_preds["lgbm"].loc[y_rec.index],
                                                         npred.loc[y_rec.index], CLASS10, bc.n_bootstrap, bc.seed)}
    print(f"  envelope diagnostic: without class10 {nres['class10']['macro_f1']:.3f}; with − without "
          f"{env['paired_with_minus_without']['diff']:+.3f} {env['paired_with_minus_without']['ci']}")

    # Permutation importance (LightGBM, validation, recording level, whole 9-window blocks).
    lgbm = build("lgbm", bc, None, None, None).fit(Xtr, ytr)
    y_rec_all = va.groupby("rec_id").class10.first()
    pp = lambda X: proba10(lgbm, X)
    feat_imp = block_permutation_importance(Xva, va.rec_id.to_numpy(), y_rec_all, pp,
                                            {c: [i] for i, c in enumerate(cols)}, bc.n_permutation_repeats, bc.seed,
                                            bc.prob_floor)
    fam_imp = block_permutation_importance(Xva, va.rec_id.to_numpy(), y_rec_all, pp, families(cols),
                                           bc.n_permutation_repeats, bc.seed, bc.prob_floor)
    OUT.mkdir(parents=True, exist_ok=True)
    feat_imp.to_csv(OUT / "permutation_importance_features.csv", index=False)
    fam_imp.to_csv(OUT / "permutation_importance_families.csv", index=False)
    print("  top features:", feat_imp.head(5)[["group", "importance"]].round(3).to_dict("records"))

    summary = {"provenance": {**common, "n_train_recordings": int(tr.rec_id.nunique()),
                              "n_val_recordings": int(n_val_rec), "n_features": len(cols),
                              "config": cfg.baselines.model_dump(mode="json")},
               "models": results, "sensitivity": sens, "envelope_diagnostic": env}
    (OUT / "closed_val.json").write_text(json.dumps(summary, indent=1))
    print(f"→ {(OUT / 'closed_val.json').relative_to(SYSTEM_CODE_DIR)}")


def run_leaky(cfg: Config) -> None:
    """Deliberate leakage illustration (reported separately): LightGBM, train+val windows, (a) random 5-fold split at
    WINDOW level vs (b) the recording-grouped cv_fold. Window-level macro-F1 per fold. LightGBM needs no scaler, so
    there is no other learned preprocessing to refit; signal cleaning is fixed and reused (conditional on it)."""
    bc, pre = cfg.baselines, cfg.preprocess
    meta = pd.read_csv(SYSTEM_CODE_DIR / "metadata.csv")
    path, df = load_table(bc.feature_table, pre, meta)
    cols = feature_columns(df)
    X, y = xy(df, cols)
    labels = [c for c in CLASS10 if c in set(df.class10)]
    folds = {
        "window_random": list(KFold(bc.leaky_n_folds, shuffle=True, random_state=bc.seed).split(X)),
        "recording_grouped": [(np.flatnonzero(df.cv_fold.to_numpy() != k), np.flatnonzero(df.cv_fold.to_numpy() == k))
                              for k in sorted(df.cv_fold.unique())],
    }
    out = {"provenance": {"feature_table": path.name, "split_hash": split_hash(meta), "git_sha": git_sha(),
                          "n_windows": len(df), "n_recordings": int(df.rec_id.nunique()),
                          "note": "deliberate leakage illustration; conditional on the fixed preprocessing"}}
    for name, splits in folds.items():
        scores = []
        for itr, ite in splits:
            m = build("lgbm", bc, None, None, None).fit(X[itr], y[itr])
            pred = np.asarray(CLASS10)[proba10(m, X[ite]).argmax(axis=1)]
            scores.append(float(f1_score(df.class10.to_numpy()[ite], pred, labels=labels, average="macro",
                                         zero_division=0)))
        # How many test-fold windows had a sibling window of the same recording in training?
        shared = float(np.mean([np.isin(df.rec_id.to_numpy()[ite], df.rec_id.to_numpy()[itr]).mean()
                                for itr, ite in splits]))
        out[name] = {"fold_macro_f1": scores, "mean": float(np.mean(scores)), "std": float(np.std(scores)),
                     "share_of_test_windows_with_their_recording_in_training": shared}
        print(f"  {name:18s} window macro-F1 {np.mean(scores):.3f} ± {np.std(scores):.3f}  "
              f"(test windows whose recording is also in training: {shared:.0%})")
    out["gap"] = out["window_random"]["mean"] - out["recording_grouped"]["mean"]
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "leaky_vs_honest.json").write_text(json.dumps(out, indent=1))
    print(f"  gap {out['gap']:+.3f} → {(OUT / 'leaky_vs_honest.json').relative_to(SYSTEM_CODE_DIR)}")


def run_dev_inventory(cfg: Config) -> None:
    """Inventory of the development pool: recordings per class and folder proxy, and folders per class."""
    dc = cfg.development
    meta = pd.read_csv(SYSTEM_CODE_DIR / "metadata.csv")
    pool = development_pool(meta, dc.allowed_closed, dc.reserved_columns)
    inv = (pool.groupby(["class10", "folder"]).agg(recordings=("rec_id", "size"),
                                                   speed_min_hz=("fr_measured_hz", "min"),
                                                   speed_max_hz=("fr_measured_hz", "max")).reset_index())
    inv["class10"] = pd.Categorical(inv.class10, CLASS10, ordered=True)
    inv = inv.sort_values(["class10", "folder"])
    OUT.mkdir(parents=True, exist_ok=True)
    inv.to_csv(OUT / "dev_pool_inventory.csv", index=False)
    per = pool.groupby("class10").folder.nunique().reindex(CLASS10)
    k = int(per.drop("normal").min())
    print(f"pool: {len(pool)} of {len(meta)} recordings, {pool.fr_measured_hz.min():.1f}–{pool.fr_measured_hz.max():.1f} Hz")
    print(f"folders per class: {per.to_dict()}; K = min over fault classes = {k}")
    print(f"→ {(OUT / 'dev_pool_inventory.csv').relative_to(SYSTEM_CODE_DIR)}")


def dev_pool_with_folds(cfg: Config, meta: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    dc = cfg.development
    pool = development_pool(meta, dc.allowed_closed, dc.reserved_columns)
    fold, k = assign_dev_folds(pool, dc.seed, dc.min_k)
    return pool.assign(dev_fold=fold), k


def run_dev_folds(cfg: Config) -> None:
    """Write the development fold assignment (existing split columns are not touched)."""
    meta = pd.read_csv(SYSTEM_CODE_DIR / "metadata.csv")
    pool, k = dev_pool_with_folds(cfg, meta)
    out = results_dir() / cfg.development.folds_file
    pool[["rec_id", "class10", "folder", "dev_fold"]].sort_values(["dev_fold", "class10", "folder", "rec_id"]).to_csv(
        out, index=False)
    tab = pd.crosstab(pool.class10, pool.dev_fold).reindex(CLASS10)
    print(f"K = {k}; recordings per class and fold:\n{tab.to_string()}")
    print(f"folders per fold: {pool[pool.class10 != 'normal'].groupby('dev_fold').folder.nunique().to_dict()}")
    print(f"→ {out.relative_to(SYSTEM_CODE_DIR)}")


DEV_MODELS = ("lgbm", "rf", "logreg", "svm", "knn")


def load_dev_table(cfg: Config, table: str, pre: PreprocessConfig, meta: pd.DataFrame) -> tuple[Path, pd.DataFrame, int]:
    """Development-pool windows of a feature table with their dev_fold, ordered by recording and window."""
    pool, k = dev_pool_with_folds(cfg, meta)
    path, df = load_table(table, pre, meta)
    df = df.merge(pool[["rec_id", "folder", "dev_fold"]], on="rec_id", how="inner")
    assert df.rec_id.nunique() == len(pool)
    return path, df.sort_values(["rec_id", "window_idx"]).reset_index(drop=True), k


def oof_proba(name: str, bc: BaselineConfig, df: pd.DataFrame, cols: list[str], k: int,
              keep_models: bool = False) -> tuple[np.ndarray, list]:
    """Out-of-fold window probabilities: for each dev fold, fit on the other folds (scalers and calibration inside the
    model, so refitted per fold) and predict the held-out fold. Every pool window is predicted exactly once."""
    X, y = xy(df, cols)
    out = np.zeros((len(df), len(CLASS10)))
    models = []
    for j in range(k):
        tr, te = (df.dev_fold != j).to_numpy(), (df.dev_fold == j).to_numpy()
        m = build(name, bc, df.rec_id.to_numpy()[tr], X[tr], y[tr]).fit(X[tr], y[tr])
        out[te] = proba10(m, X[te])
        if keep_models:
            models.append(m)
    return out, models


def dev_recording_eval(proba: np.ndarray, df: pd.DataFrame, bc: BaselineConfig, draws) -> tuple[dict, pd.DataFrame]:
    """Recording-level class10/class6 metrics with mixed-unit intervals; also returns per-recording predictions."""
    rec = recording_predictions(proba, df, bc.prob_floor)
    win_pred = np.asarray(CLASS10)[proba.argmax(axis=1)]
    present = [c for c in CLASS10 if c in set(df.class10)]
    res = {"class10": metrics_mixed(rec.class10, rec.pred10, CLASS10, draws, rec.folder),
           "class6": metrics_mixed(rec.true6, rec.pred6, CLASS6, draws, rec.folder),
           "window_macro_f1": float(f1_score(df.class10, win_pred, labels=present, average="macro", zero_division=0))}
    return res, rec


def dev_draws(rec: pd.DataFrame, bc: BaselineConfig):
    return mixed_bootstrap_draws(rec.class10.to_numpy(), rec.folder.to_numpy(), bc.n_bootstrap, bc.seed)


def run_dev(cfg: Config) -> None:
    """Folder-grouped out-of-fold evaluation of the classical models on the development pool (Stage 2B.1)."""
    bc, pre = cfg.baselines, cfg.preprocess
    meta = pd.read_csv(SYSTEM_CODE_DIR / "metadata.csv")
    path, df, k = load_dev_table(cfg, bc.feature_table, pre, meta)
    cols = feature_columns(df)
    rec0 = df.groupby("rec_id", sort=False)[["class10", "folder"]].first()
    draws = dev_draws(rec0, bc)
    folders_per_class = rec0.groupby("class10").folder.nunique().reindex(CLASS10).to_dict()
    common = {"split_hash": split_hash(meta), "cache_key": read_cache_metadata(path)["cache_key"],
              "feature_table": path.name, "git_sha": git_sha(), "protocol": f"dev folder-grouped K={k}"}
    print(f"{path.name}: dev pool {df.rec_id.nunique()} recordings / {len(df)} windows, K = {k}; "
          f"folder proxies per class {folders_per_class}")
    results, probas = {}, {}
    for name in DEV_MODELS:
        t = time.perf_counter()
        proba, _ = oof_proba(name, bc, df, cols, k)
        secs = time.perf_counter() - t
        res, rec = dev_recording_eval(proba, df, bc, draws)
        res["seconds_total"] = secs
        results[name], probas[name] = res, proba
        print(f"  {name:8s} class10 macro-F1 {res['class10']['macro_f1']:.3f} [{res['class10']['macro_f1_ci'][0]:.3f}, "
              f"{res['class10']['macro_f1_ci'][1]:.3f}]  class6 {res['class6']['macro_f1']:.3f} "
              f"[{res['class6']['macro_f1_ci'][0]:.3f}, {res['class6']['macro_f1_ci'][1]:.3f}]  "
              f"window {res['window_macro_f1']:.3f}  ({secs:.0f} s)")
    pens = np.mean([probas[m] for m in DEV_MODELS], axis=0)
    res, _ = dev_recording_eval(pens, df, bc, draws)
    results["ensemble"] = res
    print(f"  ensemble class10 macro-F1 {res['class10']['macro_f1']:.3f} [{res['class10']['macro_f1_ci'][0]:.3f}, "
          f"{res['class10']['macro_f1_ci'][1]:.3f}]  class6 {res['class6']['macro_f1']:.3f}")
    for name, proba in {**probas, "ensemble": pens}.items():
        s_df = pd.DataFrame({"rec_id": df.rec_id, "window_idx": df.window_idx, "split": "dev_fold_" + df.dev_fold.astype(str),
                             "y_true": df.class10})
        for j, c in enumerate(CLASS10):
            s_df[f"score_{c}"] = proba[:, j]
        import pyarrow as pa
        import pyarrow.parquet as pq
        table = pa.Table.from_pandas(s_df, preserve_index=False)
        meta_b = {**(table.schema.metadata or {}),
                  **{f"machinesense.{kk}".encode(): str(v).encode() for kk, v in {**common, "run_name": f"dev_{name}"}.items()}}
        SCORES.mkdir(parents=True, exist_ok=True)
        pq.write_table(table.replace_schema_metadata(meta_b), SCORES / f"dev_{bc.feature_table}_{name}.parquet")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "dev_eval.json").write_text(json.dumps(
        {"provenance": {**common, "k": k, "n_recordings": int(df.rec_id.nunique()), "n_windows": len(df),
                        "folder_proxies_per_class": folders_per_class,
                        "bootstrap": "stratified mixed-unit: fault folder proxies resampled within class with all their "
                                     "recordings; normal recordings resampled individually (one folder: between-session "
                                     "uncertainty for normal cannot be estimated); conditional on the fitted folds"},
         "models": results}, indent=1))
    print(f"→ {(OUT / 'dev_eval.json').relative_to(SYSTEM_CODE_DIR)}")


def run_dev_compare(cfg: Config) -> None:
    """LightGBM comparisons under the folder-grouped dev folds, paired over identical mixed-unit draws:
    (a) primary vs capped at 10 kHz, (b) with vs without envelope features, (c) vibration only vs microphone only vs
    all, (d) family permutation importance within each held-out fold. A drop shows dependence on the removed
    information, not that it is an artefact."""
    bc, pre = cfg.baselines, cfg.preprocess
    meta = pd.read_csv(SYSTEM_CODE_DIR / "metadata.csv")
    path, df, k = load_dev_table(cfg, bc.feature_table, pre, meta)
    cols = feature_columns(df)
    rec0 = df.groupby("rec_id", sort=False)[["class10", "folder"]].first()
    draws = dev_draws(rec0, bc)

    def run(frame: pd.DataFrame, use: list[str], keep: bool = False):
        proba, models = oof_proba("lgbm", bc, frame, use, k, keep_models=keep)
        res, rec = dev_recording_eval(proba, frame, bc, draws)
        return res, rec.loc[rec0.index], proba, models

    base, rec_base, proba_base, models = run(df, cols, keep=True)
    out = {"provenance": {"feature_table": path.name, "split_hash": split_hash(meta), "git_sha": git_sha(), "k": k,
                          "folder_proxies_per_class": rec0.groupby("class10").folder.nunique().reindex(CLASS10).to_dict(),
                          "rule": "a drop shows dependence on the removed information, not that it is an artefact"},
           "all_features": base}
    pair = lambda a, b: paired_mixed(rec0.class10, a.pred10, b.pred10, CLASS10, draws)

    # (a) primary vs capped
    cap_pre = load_config(CONFIGS_DIR / bc.sensitivity_config).preprocess
    cpath, cdf, _ = load_dev_table(cfg, bc.sensitivity_table, cap_pre, meta)
    assert (cdf[["rec_id", "window_idx", "dev_fold"]].to_numpy() == df[["rec_id", "window_idx", "dev_fold"]].to_numpy()).all()
    cres, crec, _, _ = run(cdf, feature_columns(cdf))
    out["a_capped"] = {"table": cpath.name, "capped": cres, "primary_minus_capped": pair(rec_base, crec)}
    # (b) without envelope
    nres, nrec, _, _ = run(df, [c for c in cols if not c.startswith("f_env_")])
    out["b_without_envelope"] = {"without": nres, "with_minus_without": pair(rec_base, nrec)}
    # (c) vibration only / microphone only
    vib = [c for c in cols if not c.startswith("f_mic_")]
    mic = [c for c in cols if c.startswith("f_mic_")]
    vres, vrec, _, _ = run(df, vib)
    mres, mrec, _, _ = run(df, mic)
    out["c_modalities"] = {"n_vibration": len(vib), "n_microphone": len(mic), "vibration_only": vres,
                           "microphone_only": mres, "all_minus_vibration": pair(rec_base, vrec),
                           "all_minus_microphone": pair(rec_base, mrec), "vibration_minus_microphone": pair(vrec, mrec)}
    for key, r in [("a capped", cres), ("b no envelope", nres), ("c vibration", vres), ("c microphone", mres)]:
        print(f"  {key:14s} class10 {r['class10']['macro_f1']:.3f} [{r['class10']['macro_f1_ci'][0]:.3f}, "
              f"{r['class10']['macro_f1_ci'][1]:.3f}]  (all features {base['class10']['macro_f1']:.3f})")

    # (d) family permutation importance: within each held-out fold, shuffle whole recording blocks (all columns of a
    # family together) across that fold's held-out recordings; pooled out-of-fold recording-level macro-F1 drop.
    X, _ = xy(df, cols)
    fams = families(cols)
    rng = np.random.default_rng(bc.seed)
    base_f1 = base["class10"]["macro_f1"]
    y_rec = rec0.class10
    rows = []
    for fam, fcols in fams.items():
        drops = []
        for _ in range(bc.n_permutation_repeats):
            proba = proba_base.copy()
            for j in range(k):
                te = np.flatnonzero((df.dev_fold == j).to_numpy())
                ids = df.rec_id.to_numpy()[te]
                n_rec = len(pd.unique(ids))
                X3 = X[te].reshape(n_rec, -1, X.shape[1])
                perm = rng.permutation(n_rec)
                Xp = X3.copy()
                Xp[:, :, fcols] = X3[perm][:, :, fcols]
                proba[te] = proba10(models[j], Xp.reshape(-1, X.shape[1]))
            agg = aggregate_recordings(proba, df.rec_id.to_numpy(), bc.prob_floor)
            drops.append(base_f1 - f1_score(y_rec.loc[agg.index], agg.idxmax(axis=1), average="macro", zero_division=0))
        rows.append({"family": fam, "n_columns": len(fcols), "importance": float(np.mean(drops)), "std": float(np.std(drops))})
    fam_imp = pd.DataFrame(rows).sort_values("importance", ascending=False)
    fam_imp.to_csv(OUT / "dev_family_importance.csv", index=False)
    out["d_family_importance"] = fam_imp.to_dict("records")
    (OUT / "dev_compare.json").write_text(json.dumps(out, indent=1))
    for key, d in [("a primary−capped", out["a_capped"]["primary_minus_capped"]),
                   ("b with−without env", out["b_without_envelope"]["with_minus_without"]),
                   ("c all−vibration", out["c_modalities"]["all_minus_vibration"]),
                   ("c all−microphone", out["c_modalities"]["all_minus_microphone"])]:
        print(f"  {key:20s} {d['diff']:+.3f} [{d['ci'][0]:+.3f}, {d['ci'][1]:+.3f}]  P(>0) {d['p_positive']:.2f}")
    print(fam_imp.round(3).to_string(index=False))
    print(f"→ {(OUT / 'dev_compare.json').relative_to(SYSTEM_CODE_DIR)}")


def run_dev_protocol(cfg: Config) -> None:
    """Evaluation-protocol comparison within the development pool (LightGBM, K folds): window-random,
    recording-grouped and folder-grouped. Training sizes are matched exactly: in every protocol, fold j holds out the
    same number of windows of each class (window-random) or recordings of each class (recording-grouped) as the
    folder-grouped fold j; only WHICH windows/recordings are held out differs. A protocol comparison, not a pure
    leakage estimate."""
    bc, pre = cfg.baselines, cfg.preprocess
    meta = pd.read_csv(SYSTEM_CODE_DIR / "metadata.csv")
    path, df, k = load_dev_table(cfg, bc.feature_table, pre, meta)
    cols = feature_columns(df)
    X, y = xy(df, cols)
    rng = np.random.default_rng(bc.seed)
    rec = df.groupby("rec_id", sort=False)[["class10", "dev_fold"]].first()

    # recording-grouped: per class, a random permutation of its recordings split with the folder-grouped fold counts
    rec_fold = pd.Series(-1, index=rec.index)
    for _, g in rec.groupby("class10"):
        counts = g.dev_fold.value_counts().reindex(range(k), fill_value=0).to_numpy()
        perm = g.index[rng.permutation(len(g))]
        rec_fold[perm] = np.repeat(np.arange(k), counts)
    # window-random: per class, a random permutation of its windows split with the folder-grouped window counts
    win_fold = np.full(len(df), -1)
    for _, idx in df.groupby("class10").groups.items():
        idx = np.asarray(idx)
        counts = df.dev_fold.to_numpy()[idx]
        counts = np.bincount(counts, minlength=k)
        win_fold[rng.permutation(idx)] = np.repeat(np.arange(k), counts)
    assignments = {"window_random": win_fold, "recording_grouped": df.rec_id.map(rec_fold).to_numpy(),
                   "folder_grouped": df.dev_fold.to_numpy()}
    labels = [c for c in CLASS10 if c in set(df.class10)]
    out = {"provenance": {"feature_table": path.name, "split_hash": split_hash(meta), "git_sha": git_sha(), "k": k,
                          "matching": "fold j of every protocol holds out the same number of windows (window-random) or "
                                      "recordings (recording-grouped) of each class as folder-grouped fold j",
                          "label": "evaluation-protocol comparison, not a pure leakage estimate"}}
    for name, fold in assignments.items():
        per_fold, proba = [], np.zeros((len(df), len(CLASS10)))
        n_train = []
        for j in range(k):
            tr, te = fold != j, fold == j
            m = build("lgbm", bc, None, None, None).fit(X[tr], y[tr])
            proba[te] = proba10(m, X[te])
            pred = np.asarray(CLASS10)[proba[te].argmax(axis=1)]
            per_fold.append(float(f1_score(df.class10.to_numpy()[te], pred, labels=labels, average="macro", zero_division=0)))
            n_train.append(int(tr.sum()))
        shared = float(np.mean([np.isin(df.rec_id.to_numpy()[fold == j], df.rec_id.to_numpy()[fold != j]).mean()
                                for j in range(k)]))
        entry = {"window_macro_f1_per_fold": per_fold, "window_macro_f1_mean": float(np.mean(per_fold)),
                 "training_windows_per_fold": n_train,
                 "share_of_held_out_windows_whose_recording_is_in_training": shared}
        if name != "window_random":  # recording-level only where a recording's windows stay together
            agg = aggregate_recordings(proba, df.rec_id.to_numpy(), bc.prob_floor)
            yr = rec.class10.loc[agg.index]
            entry["recording_macro_f1"] = float(f1_score(yr, agg.idxmax(axis=1), labels=labels, average="macro",
                                                         zero_division=0))
        out[name] = entry
        print(f"  {name:18s} window macro-F1 {np.mean(per_fold):.3f} (folds {np.round(per_fold, 3).tolist()}), "
              f"train windows {n_train}, held-out windows with their recording in training {shared:.0%}"
              + (f", recording macro-F1 {entry['recording_macro_f1']:.3f}" if "recording_macro_f1" in entry else ""))
    (OUT / "dev_protocol.json").write_text(json.dumps(out, indent=1))
    print(f"→ {(OUT / 'dev_protocol.json').relative_to(SYSTEM_CODE_DIR)}")


def run_dev_lr_capped(cfg: Config) -> None:
    """Capped logistic-regression reference (Stage 2C-1): logistic regression and LightGBM on the r1cap10 table, the
    existing 2B.1 dev folds, no retuning; paired comparison over identical mixed-unit draws."""
    bc, pre = cfg.baselines, cfg.preprocess
    meta = pd.read_csv(SYSTEM_CODE_DIR / "metadata.csv")
    path, df, k = load_dev_table(cfg, "r1cap10", pre, meta)
    cols = feature_columns(df)
    rec0 = df.groupby("rec_id", sort=False)[["class10", "folder"]].first()
    draws = dev_draws(rec0, bc)
    res, recs = {}, {}
    for name in ("logreg", "lgbm"):
        t = time.perf_counter()
        proba, _ = oof_proba(name, bc, df, cols, k)
        res[name], rec = dev_recording_eval(proba, df, bc, draws)
        res[name]["seconds_total"] = time.perf_counter() - t
        recs[name] = rec.loc[rec0.index]
        print(f"  {name:7s} capped class10 {res[name]['class10']['macro_f1']:.3f} "
              f"[{res[name]['class10']['macro_f1_ci'][0]:.3f}, {res[name]['class10']['macro_f1_ci'][1]:.3f}]  class6 "
              f"{res[name]['class6']['macro_f1']:.3f} [{res[name]['class6']['macro_f1_ci'][0]:.3f}, "
              f"{res[name]['class6']['macro_f1_ci'][1]:.3f}]")
    d10 = paired_mixed(rec0.class10, recs["logreg"].pred10, recs["lgbm"].pred10, CLASS10, draws)
    d6 = paired_mixed(rec0.class10.map(CLASS10_TO_6), recs["logreg"].pred6, recs["lgbm"].pred6, CLASS6, draws)
    out = {"provenance": {"feature_table": path.name, "split_hash": split_hash(meta), "git_sha": git_sha(), "k": k,
                          "folder_proxies_per_class": rec0.groupby("class10").folder.nunique().reindex(CLASS10).to_dict()},
           "models": res, "logreg_minus_lgbm_class10": d10, "logreg_minus_lgbm_class6": d6}
    (OUT / "dev_capped_lr.json").write_text(json.dumps(out, indent=1))
    print(f"  logreg − lgbm class10 {d10['diff']:+.3f} [{d10['ci'][0]:+.3f}, {d10['ci'][1]:+.3f}] P(>0) {d10['p_positive']:.2f}; "
          f"class6 {d6['diff']:+.3f} [{d6['ci'][0]:+.3f}, {d6['ci'][1]:+.3f}]")
    print(f"→ {(OUT / 'dev_capped_lr.json').relative_to(SYSTEM_CODE_DIR)}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("task", choices=["closed", "leaky", "dev-inventory", "dev-folds", "dev", "dev-compare",
                                         "dev-protocol", "dev-lr-capped"])
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = ap.parse_args()
    cfg = load_config(args.config)
    {"closed": run_closed, "leaky": run_leaky, "dev-inventory": run_dev_inventory,
     "dev-folds": run_dev_folds, "dev": run_dev, "dev-compare": run_dev_compare,
     "dev-protocol": run_dev_protocol, "dev-lr-capped": run_dev_lr_capped}[args.task](cfg)
    return 0


if __name__ == "__main__":
    sys.exit(main())
