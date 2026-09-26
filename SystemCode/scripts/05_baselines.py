"""Closed-set classical baselines on the R1 physics features (Stage 2B). LightGBM lives here; this file never imports
torch (CLAUDE.md §2).

    python scripts/05_baselines.py closed   # A0 + A1 rungs, sensitivity, envelope diagnostic, permutation importance
    python scripts/05_baselines.py leaky    # deliberate leakage illustration: window-level vs recording-grouped CV
    python scripts/05_baselines.py dev-inventory  # development pool (Stage 2B.1): recordings by class and folder proxy

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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold, KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import QuantileTransformer
from sklearn.svm import SVC

from ml.cache import load_features, read_cache_metadata
from ml.config import BaselineConfig, Config, PreprocessConfig, load_config
from ml.evaluation import (CLASS10_TO_6, aggregate_recordings, block_permutation_importance, metrics,
                           paired_bootstrap, to_class6)
from ml.features import ACCEL_CHANNELS, feature_columns
from ml.labels import CLASS6, CLASS10
from ml.paths import CONFIGS_DIR, DEFAULT_CONFIG, SYSTEM_CODE_DIR, data_dir, results_dir
from ml.splits import development_pool, split_hash
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
        return Pipeline([("scale", quantile(bc)),
                         ("lr", LogisticRegression(**bc.logreg, class_weight=cw, random_state=bc.seed))])
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


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("task", choices=["closed", "leaky", "dev-inventory"])
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = ap.parse_args()
    cfg = load_config(args.config)
    {"closed": run_closed, "leaky": run_leaky, "dev-inventory": run_dev_inventory}[args.task](cfg)
    return 0


if __name__ == "__main__":
    sys.exit(main())
