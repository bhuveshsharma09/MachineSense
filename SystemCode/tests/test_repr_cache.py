"""Representation cache: serving parity, per-track normalisation stats, stale keys fail."""
import importlib.util
import json

import numpy as np
import pandas as pd
import pytest

from ml.cache import CacheMismatchError
from ml.config import load_config
from ml.datasets import load_representation, norm_stats, track_train_mask
from ml.paths import DEFAULT_CONFIG, SYSTEM_CODE_DIR, data_dir

REC = "imbalance__20g__30.5152"


def _stacked():
    spec = importlib.util.spec_from_file_location("rc", SYSTEM_CODE_DIR / "scripts" / "10_repr_cache.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.stacked


def test_serving_path_matches_cache():
    cfg = load_config(DEFAULT_CONFIG)
    meta = pd.read_csv(SYSTEM_CODE_DIR / "metadata.csv")
    try:
        X, index = load_representation(data_dir() / "representations", cfg, meta)
        from ml.paths import data_root
        csv = data_root() / meta.set_index("rec_id").loc[REC, "path"]
    except (FileNotFoundError, RuntimeError):
        pytest.skip("representation cache or CSVs not available")
    from ml.preprocess import load_csv
    rows = np.flatnonzero(index.rec_id.to_numpy() == REC)
    served = _stacked()(load_csv(csv), float(meta.set_index("rec_id").loc[REC, "fr_measured_hz"]), cfg.preprocess,
                        cfg.representation.leading_candidate)
    assert np.array_equal(np.asarray(X[rows]), served)  # bit-identical float16


def _toy():
    index = pd.DataFrame({"rec_id": np.repeat(["a", "b", "c", "d"], 9), "window_idx": np.tile(np.arange(9), 4)})
    meta = pd.DataFrame({"rec_id": list("abcd"), "split_closed": ["train", "train", "val", "test"],
                         "loco_imbalance": ["train", "withheld", "val", "test"]})
    X = np.zeros((36, 7, 4, 3), np.float32)
    X[:9], X[9:18] = 1.0, 3.0  # a, b (train)
    X[18:] = 100.0  # c, d must never count
    return X, index, meta


def test_norm_stats_uses_training_windows_only():
    X, index, meta = _toy()
    st = norm_stats("closed", X, index, meta)
    assert st["n_windows"] == 18 and np.allclose(st["mean"], 2.0) and np.allclose(st["std"], 1.0)
    lo = norm_stats("loco_imbalance", X, index, meta)  # the withheld recording b is excluded
    assert lo["n_windows"] == 9 and np.allclose(lo["mean"], 1.0) and np.allclose(lo["std"], 0.0)
    assert track_train_mask(index, meta, "closed").sum() == 18
    with pytest.raises(ValueError):
        norm_stats("nonexistent", X, index, meta)


def test_stale_representation_cache_fails(tmp_path):
    cfg = load_config(DEFAULT_CONFIG)
    meta = pd.read_csv(SYSTEM_CODE_DIR / "metadata.csv")
    name = cfg.representation.leading_candidate
    np.save(tmp_path / f"{name}_000000000000.npy", np.zeros((9, 7, 2, 2), np.float16))
    pd.DataFrame({"rec_id": [REC] * 9, "window_idx": range(9)}).to_parquet(tmp_path / f"{name}_000000000000.index.parquet")
    (tmp_path / f"{name}_000000000000.json").write_text(json.dumps({"cache_key": "0" * 64}))
    with pytest.raises(CacheMismatchError):
        load_representation(tmp_path, cfg, meta)
