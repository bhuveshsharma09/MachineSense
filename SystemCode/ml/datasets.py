"""Representation cache access and per-track normalisation statistics. No torch here (loaders come later)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from ml.cache import CacheMismatchError, check_split_hash, representation_cache_key, sources_from_metadata
from ml.config import Config

N_CHANNELS = 7  # six accelerometer order-spectrograms + the time-aligned microphone log-Mel


def representation_paths(directory: Path, candidate: str, key: str) -> tuple[Path, Path, Path]:
    stem = Path(directory) / f"{candidate}_{key[:12]}"
    return stem.with_suffix(".npy"), stem.with_suffix(".index.parquet"), stem.with_suffix(".json")


def load_representation(directory: Path, cfg: Config, meta: pd.DataFrame, mmap: bool = True):
    """(array N × 7 × 128 × T float16, index DataFrame rec_id/window_idx) for the frozen candidate, after recomputing the
    cache key from the current config and the recordings' checksums and speeds. Raises CacheMismatchError otherwise."""
    rc, pre = cfg.representation, cfg.preprocess
    if not rc.frozen:
        raise RuntimeError("no representation is frozen")
    matches = sorted(Path(directory).glob(f"{rc.leading_candidate}_*.json"), key=lambda p: p.stat().st_mtime)
    if not matches:
        raise FileNotFoundError(f"no cache for {rc.leading_candidate} in {directory}")
    info = json.loads(matches[-1].read_text())
    index = pd.read_parquet(matches[-1].with_suffix("").with_suffix(".index.parquet"))
    key = representation_cache_key(pre, sources_from_metadata(meta, index.rec_id.unique()), rc.leading_candidate,
                                   rc.version)
    if info.get("cache_key") != key:
        raise CacheMismatchError(f"{matches[-1].name}: representation cache key does not match the current config, "
                                 "sources or version; rebuild it")
    check_split_hash(info.get("split_hash"), meta, matches[-1].name)
    arr = np.load(matches[-1].with_suffix("").with_suffix(".npy"), mmap_mode="r" if mmap else None)
    if arr.shape[0] != len(index):
        raise CacheMismatchError("array rows do not match the index")
    return arr, index


def track_train_mask(index: pd.DataFrame, meta: pd.DataFrame, track: str) -> np.ndarray:
    """Rows (windows) of the index that belong to `track`'s TRAINING side. Tracks: "closed" (split_closed), a LOCO
    column name ("loco_<class>"), "speed_block", "severity"."""
    col = {"closed": "split_closed", "speed_block": "split_speed_block", "severity": "split_severity"}.get(track, track)
    if col not in meta.columns:
        raise ValueError(f"unknown track {track!r}")
    train_ids = set(meta.loc[meta[col] == "train", "rec_id"])
    return index.rec_id.isin(train_ids).to_numpy()


def norm_stats(track: str, X: np.ndarray, index: pd.DataFrame, meta: pd.DataFrame, chunk: int = 512) -> dict:
    """Per-channel mean and standard deviation over the TRAINING windows of `track` only (computed at training time,
    never stored in the cache). Accumulated in float64 over row chunks."""
    rows = np.flatnonzero(track_train_mask(index, meta, track))
    if rows.size == 0:
        raise ValueError(f"track {track!r} has no training windows")
    n_ch = X.shape[1]
    s, ss, n = np.zeros(n_ch), np.zeros(n_ch), 0
    for i in range(0, rows.size, chunk):
        block = np.asarray(X[rows[i:i + chunk]], dtype=np.float64)
        s += block.sum(axis=(0, 2, 3))
        ss += (block**2).sum(axis=(0, 2, 3))
        n += block.shape[0] * block.shape[2] * block.shape[3]
    mean = s / n
    return {"track": track, "mean": mean, "std": np.sqrt(np.maximum(ss / n - mean**2, 0.0)), "n_windows": int(rows.size)}
