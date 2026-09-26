"""Representation cache access and per-track normalisation statistics. No torch here (loaders come later)."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from ml.cache import CacheMismatchError, check_split_hash, representation_cache_key, sources_from_metadata
from ml.config import Config
from ml.splits import development_pool

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


@dataclass(frozen=True)
class NormContext:
    """The recordings of one development run that must NOT inform normalisation: its calibration recordings, its
    evaluation recordings (seen-folder and held-out) and the LOCO-withheld recordings. Empty sets are allowed but
    must be passed explicitly."""
    calibration: frozenset[str] = field(default_factory=frozenset)
    evaluation: frozenset[str] = field(default_factory=frozenset)
    withheld: frozenset[str] = field(default_factory=frozenset)

    def __post_init__(self):
        for name in ("calibration", "evaluation", "withheld"):
            object.__setattr__(self, name, frozenset(getattr(self, name)))


def development_fit_mask(index: pd.DataFrame, meta: pd.DataFrame, fit_ids, context: NormContext,
                         allowed_closed=("train", "val"),
                         reserved_columns=("split_severity", "split_speed_block")) -> np.ndarray:
    """Rows of the index belonging to `fit_ids`, after checking the ID set. Raises ValueError unless every ID is a
    known recording in the development pool (ml.splits.development_pool) and none is in the context's calibration,
    evaluation or withheld sets."""
    fit = frozenset(fit_ids)
    if not fit:
        raise ValueError("fit_ids is empty")
    unknown = fit - set(meta.rec_id)
    if unknown:
        raise ValueError(f"{len(unknown)} fit IDs are not in the metadata, e.g. {sorted(unknown)[:3]}")
    outside = fit - set(development_pool(meta, allowed_closed, reserved_columns).rec_id)
    if outside:
        raise ValueError(f"{len(outside)} fit IDs are outside the development pool, e.g. {sorted(outside)[:3]}")
    for name in ("calibration", "evaluation", "withheld"):
        clash = fit & getattr(context, name)
        if clash:
            raise ValueError(f"{len(clash)} fit IDs are {name} recordings of this run, e.g. {sorted(clash)[:3]}")
    missing = fit - set(index.rec_id)
    if missing:
        raise ValueError(f"{len(missing)} fit IDs have no windows in the index, e.g. {sorted(missing)[:3]}")
    return index.rec_id.isin(fit).to_numpy()


def norm_stats(X: np.ndarray, index: pd.DataFrame, meta: pd.DataFrame, *, track: str | None = None,
               fit_ids=None, context: NormContext | None = None, chunk: int = 512) -> dict:
    """Per-channel mean and standard deviation over the chosen windows only (computed at training time, never stored in
    the cache). Accumulated in float64 over row chunks; only the chosen rows of X are read. Two modes, exactly one:

    - Named-track mode, `track=...` ("closed", "loco_<class>", "speed_block", "severity"): the TRAINING side of that
      frozen split column (unchanged behaviour). NOTE: "closed" uses split_closed == "train", which includes
      recordings OUTSIDE the development pool (the severity-test and speed-block-test recordings); it is valid only
      for closed-track runs, never for development (dev-fold) runs.
    - Development mode, `fit_ids=..., context=NormContext(...)`: exactly the given recordings. Fails loudly unless
      every ID is in the development pool and none is one of the run's calibration, evaluation or withheld
      recordings (development_fit_mask).
    """
    if (track is None) == (fit_ids is None):
        raise ValueError("pass exactly one of track= (named-track mode) or fit_ids= (development mode)")
    if track is not None:
        if context is not None:
            raise ValueError("context is only used in development mode")
        rows = np.flatnonzero(track_train_mask(index, meta, track))
        mode = {"track": track}
    else:
        if context is None:
            raise ValueError("development mode needs context=NormContext(calibration=, evaluation=, withheld=)")
        rows = np.flatnonzero(development_fit_mask(index, meta, fit_ids, context))
        mode = {"track": None, "n_recordings": int(len(frozenset(fit_ids)))}
    if rows.size == 0:
        raise ValueError(f"no windows selected ({mode})")
    n_ch = X.shape[1]
    s, ss, n = np.zeros(n_ch), np.zeros(n_ch), 0
    for i in range(0, rows.size, chunk):
        block = np.asarray(X[rows[i:i + chunk]], dtype=np.float64)
        s += block.sum(axis=(0, 2, 3))
        ss += (block**2).sum(axis=(0, 2, 3))
        n += block.shape[0] * block.shape[2] * block.shape[3]
    mean = s / n
    return {**mode, "mean": mean, "std": np.sqrt(np.maximum(ss / n - mean**2, 0.0)), "n_windows": int(rows.size)}


def dev_window_rows(index: pd.DataFrame, pool: pd.DataFrame) -> pd.DataFrame:
    """Cache rows of the development pool: one row per window, ordered by rec_id then window_idx, with the cache row
    position (`row`), class10, folder and dev_fold. `pool` needs rec_id, class10, folder and dev_fold."""
    rows = index.reset_index(drop=True).rename_axis("row").reset_index()
    out = rows.merge(pool[["rec_id", "class10", "folder", "dev_fold"]], on="rec_id", how="inner")
    missing = set(pool.rec_id) - set(out.rec_id)
    if missing:
        raise ValueError(f"{len(missing)} pool recordings have no cached windows, e.g. {sorted(missing)[:3]}")
    return out.sort_values(["rec_id", "window_idx"]).reset_index(drop=True)


def fold_fit_context(pool: pd.DataFrame, fold: int) -> tuple[list[str], NormContext]:
    """Development-mode normalisation IDs for outer dev fold `fold`: fit on the recordings of every other fold; the
    fold's own recordings are the run's evaluation set (no calibration or withheld recordings in a Q2 run)."""
    fit_ids = sorted(pool.loc[pool.dev_fold != fold, "rec_id"])
    return fit_ids, NormContext(evaluation=frozenset(pool.loc[pool.dev_fold == fold, "rec_id"]))


def fold_norm_stats(X: np.ndarray, index: pd.DataFrame, meta: pd.DataFrame, pool: pd.DataFrame, fold: int) -> dict:
    """Per-channel statistics for outer dev fold `fold`, from norm_stats in DEVELOPMENT mode (fold_fit_context)."""
    fit_ids, context = fold_fit_context(pool, fold)
    return norm_stats(X, index, meta, fit_ids=fit_ids, context=context)
