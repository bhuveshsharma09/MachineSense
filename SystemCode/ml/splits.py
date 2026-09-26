"""Split assignment, by recording (never by window). Every split lives as a column of metadata.csv."""
from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd

from ml.labels import CLASS10

FAULT_CLASSES: tuple[str, ...] = tuple(c for c in CLASS10 if c != "normal")  # normal is never withheld
CLOSED_FRACTIONS = {"train": 0.70, "val": 0.15, "test": 0.15}
N_FOLDS = 5
SPEED_BLOCK_HZ = 40.0  # split_speed_block: train/val below, test at or above (measured speed)
# split_severity: the largest severity of each fault family is held out for test.
SEVERITY_TEST = {"imbalance": 35.0, "horizontal_misalignment": 2.0, "vertical_misalignment": 1.90, "bearing": 35.0}


def _quota(n: int, fractions: list[float]) -> np.ndarray:
    """Integer counts summing to n, closest to n × fractions (largest remainder)."""
    raw = np.asarray(fractions) * n
    counts = np.floor(raw).astype(int)
    for j in np.argsort(-(raw - counts), kind="stable")[: n - counts.sum()]:
        counts[j] += 1
    return counts


def _spread(n: int, fractions: list[float]) -> np.ndarray:
    """Label indices for n ordered items: exact quota counts, spread evenly along the order (error diffusion)."""
    targets = _quota(n, fractions)
    assigned = np.zeros(len(fractions))
    out = np.empty(n, dtype=int)
    for i in range(n):
        j = int(np.argmax(targets * (i + 1) / n - assigned))
        out[i] = j
        assigned[j] += 1
    return out


def speed_tercile(df: pd.DataFrame) -> pd.Series:
    """0/1/2 tercile of measured speed within each class10."""
    return df.groupby("class10")["fr_measured_hz"].transform(
        lambda s: pd.qcut(s.rank(method="first"), 3, labels=False)).astype(int)


def stratified_assign(df: pd.DataFrame, labels: list, fractions: list[float], rng: np.random.Generator) -> pd.Series:
    """Per class10: order by speed tercile (random within tercile), then spread labels with exact class quotas."""
    out = pd.Series(index=df.index, dtype=object)
    for _, g in df.groupby("class10", sort=True):
        order = g.assign(_r=rng.random(len(g))).sort_values(["speed_tercile", "_r"]).index
        out[order] = np.asarray(labels, dtype=object)[_spread(len(order), fractions)]
    return out


def is_severity_test(df: pd.DataFrame) -> pd.Series:
    """True for the held-out largest severity of each fault family (35 g, 2.0 mm, 1.90 mm, bearing 35 g added mass)."""
    key = np.where(df["fault_family"] == "bearing", "bearing", df["class10"])
    target = pd.Series(key, index=df.index).map(SEVERITY_TEST)
    return np.isclose(df["severity"], target.astype(float)) & target.notna()


def assign_splits(df: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Add speed_tercile and every split column. Deterministic for a given seed and metadata."""
    df = df.sort_values("rec_id").reset_index(drop=True).copy()
    rng = np.random.default_rng(seed)
    df["speed_tercile"] = speed_tercile(df)

    df["split_closed"] = stratified_assign(df, list(CLOSED_FRACTIONS), list(CLOSED_FRACTIONS.values()), rng)

    for cls in FAULT_CLASSES:
        df[f"loco_{cls}"] = np.where(df["class10"] == cls, "withheld", df["split_closed"])

    # Val reuses split_closed's stratified val draw; everything else on the training side is train.
    closed_val = df["split_closed"] == "val"
    fast = df["fr_measured_hz"] >= SPEED_BLOCK_HZ
    df["split_speed_block"] = np.where(fast, "test", np.where(closed_val, "val", "train"))

    sev_test = is_severity_test(df)
    normal = df["class10"] == "normal"
    df["split_severity"] = np.where(normal, df["split_closed"],
                                    np.where(sev_test, "test", np.where(closed_val, "val", "train")))

    df["cv_fold"] = -1
    trval = df["split_closed"] != "test"
    df.loc[trval, "cv_fold"] = stratified_assign(df[trval], list(range(N_FOLDS)), [1 / N_FOLDS] * N_FOLDS,
                                                 rng).astype(int)
    df["cv_fold"] = df["cv_fold"].astype(int)
    return df


SPLIT_COLUMNS_PREFIXES = ("split_", "loco_", "cv_fold")


def split_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.startswith(SPLIT_COLUMNS_PREFIXES)]


def split_hash(df: pd.DataFrame) -> str:
    """sha256 of rec_id + every split column, so a run can prove which frozen splits it used."""
    cols = ["rec_id", *sorted(split_columns(df))]
    payload = df[cols].sort_values("rec_id").to_csv(index=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def folder_proxy(df: pd.DataFrame) -> pd.Series:
    """Folder = class10 × severity (e.g. 'imbalance/20g', 'normal/-'): a PROXY for an acquisition session."""
    unit = df["fault_family"].map({"misalignment": "mm"}).fillna("g")
    sev = df["severity"].map(lambda v: f"{v:g}")
    return np.where(df["class10"] == "normal", "normal/-", df["class10"] + "/" + sev + unit)


def development_pool(meta: pd.DataFrame, allowed_closed=("train", "val"),
                     reserved_columns=("split_severity", "split_speed_block")) -> pd.DataFrame:
    """Recordings allowed in the development evaluation: split_closed in allowed_closed and no reserved column equal
    to "test". Adds a `folder` column (the session proxy)."""
    keep = meta["split_closed"].isin(allowed_closed)
    for col in reserved_columns:
        keep &= meta[col] != "test"
    out = meta[keep].copy()
    out["folder"] = folder_proxy(out)
    return out
