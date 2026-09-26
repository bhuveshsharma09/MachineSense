"""Representation cache: serving parity, per-track normalisation stats, stale keys fail."""
import importlib.util
import json

import numpy as np
import pandas as pd
import pytest

from ml.cache import CacheMismatchError
from ml.config import load_config
from ml.datasets import NormContext, load_representation, norm_stats, track_train_mask
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
    st = norm_stats(X, index, meta, track="closed")
    assert st["n_windows"] == 18 and np.allclose(st["mean"], 2.0) and np.allclose(st["std"], 1.0)
    lo = norm_stats(X, index, meta, track="loco_imbalance")  # the withheld recording b is excluded
    assert lo["n_windows"] == 9 and np.allclose(lo["mean"], 1.0) and np.allclose(lo["std"], 0.0)
    assert track_train_mask(index, meta, "closed").sum() == 18
    with pytest.raises(ValueError):
        norm_stats(X, index, meta, track="nonexistent")


class _RowSpy:
    """Array stand-in that records which rows are read."""

    def __init__(self, arr):
        self.arr, self.read, self.shape = arr, set(), arr.shape

    def __getitem__(self, rows):
        self.read.update(np.atleast_1d(np.arange(len(self.arr))[rows]).tolist())
        return self.arr[rows]


def _dev_toy():
    """Development-pool toy: n1-n4 normal, i1-i2 imbalance, h1 horizontal (all in the pool); outside the pool: t1
    (split_closed test), s1 (severity test, split_closed train), f1 (speed-block test, split_closed train).
    Window values = 10 × the recording's position, so every recording has a distinct mean."""
    ids = ["n1", "n2", "n3", "n4", "i1", "i2", "h1", "t1", "s1", "f1"]
    meta = pd.DataFrame({
        "rec_id": ids,
        "class10": ["normal"] * 4 + ["imbalance"] * 2 + ["horizontal_misalignment", "normal", "imbalance", "imbalance"],
        "fault_family": ["normal"] * 4 + ["imbalance"] * 2 + ["misalignment", "normal", "imbalance", "imbalance"],
        "severity": [0, 0, 0, 0, 6, 10, 0.5, 0, 35, 10],
        "split_closed": ["train", "val", "train", "train", "train", "val", "train", "test", "train", "train"],
        "split_severity": ["train", "val", "train", "train", "train", "val", "train", "test", "test", "train"],
        "split_speed_block": ["train", "val", "train", "train", "train", "val", "train", "train", "train", "test"],
    })
    index = pd.DataFrame({"rec_id": np.repeat(ids, 9), "window_idx": np.tile(np.arange(9), len(ids))})
    X = np.repeat(np.arange(len(ids), dtype=np.float32) * 10, 9)[:, None, None, None] * np.ones((1, 7, 2, 3), np.float32)
    return X, index, meta


def _rows(index, ids):
    return set(np.flatnonzero(index.rec_id.isin(ids)).tolist())


def test_norm_stats_development_mode_reads_only_given_rows():
    X, index, meta = _dev_toy()
    ctx = NormContext(calibration={"n3"}, evaluation={"n4", "i2"}, withheld={"h1"})
    # dev-fold case: fit on the fold-≠k recordings minus calibration/evaluation
    spy = _RowSpy(X)
    st = norm_stats(spy, index, meta, fit_ids=["n1", "i1"], context=ctx, chunk=4)
    assert spy.read == _rows(index, ["n1", "i1"])
    assert st["n_windows"] == 18 and st["n_recordings"] == 2 and st["track"] is None
    assert np.allclose(st["mean"], 20.0) and np.allclose(st["std"], 20.0)  # n1 = 0, i1 = 40
    # healthy-only case (Q1): normal fit recordings only
    spy = _RowSpy(X)
    st = norm_stats(spy, index, meta, fit_ids={"n1", "n2"}, context=NormContext(calibration={"n3"}, evaluation={"n4"}))
    assert spy.read == _rows(index, ["n1", "n2"]) and np.allclose(st["mean"], 5.0)
    # LOCO case: withheld class never enters, even though its recordings are in the pool
    st = norm_stats(X, index, meta, fit_ids=["n1", "n2", "i1", "i2"], context=NormContext(withheld={"h1"}))
    assert st["n_windows"] == 36 and np.allclose(st["mean"], (0 + 10 + 40 + 50) / 4)


@pytest.mark.parametrize("fit_ids, ctx", [
    (["n1", "t1"], NormContext()),  # split_closed test: outside the pool
    (["n1", "s1"], NormContext()),  # severity-test recording (split_closed train): outside the pool
    (["n1", "f1"], NormContext()),  # speed-block-test recording (split_closed train): outside the pool
    (["n1", "zz"], NormContext()),  # unknown recording
    (["n1", "n3"], NormContext(calibration={"n3"})),
    (["n1", "n4"], NormContext(evaluation={"n4"})),
    (["n1", "h1"], NormContext(withheld={"h1"})),
    ([], NormContext()),
])
def test_norm_stats_development_mode_rejects_bad_id_sets(fit_ids, ctx):
    X, index, meta = _dev_toy()
    spy = _RowSpy(X)
    with pytest.raises(ValueError):
        norm_stats(spy, index, meta, fit_ids=fit_ids, context=ctx)
    assert not spy.read  # nothing read before the check fails


def test_norm_stats_mode_arguments():
    X, index, meta = _dev_toy()
    with pytest.raises(ValueError):
        norm_stats(X, index, meta)  # neither mode
    with pytest.raises(ValueError):
        norm_stats(X, index, meta, track="closed", fit_ids=["n1"], context=NormContext())  # both
    with pytest.raises(ValueError):
        norm_stats(X, index, meta, fit_ids=["n1"])  # development mode without context
    with pytest.raises(ValueError):
        norm_stats(X, index, meta, track="closed", context=NormContext())
    # the named closed track includes recordings outside the development pool (s1, f1): closed-track runs only
    st = norm_stats(X, index, meta, track="closed")
    assert st["n_windows"] == 9 * 7 and np.allclose(st["mean"], (0 + 20 + 30 + 40 + 60 + 80 + 90) / 7)  # incl. s1, f1


def test_stale_representation_cache_fails(tmp_path):
    cfg = load_config(DEFAULT_CONFIG)
    meta = pd.read_csv(SYSTEM_CODE_DIR / "metadata.csv")
    name = cfg.representation.leading_candidate
    np.save(tmp_path / f"{name}_000000000000.npy", np.zeros((9, 7, 2, 2), np.float16))
    pd.DataFrame({"rec_id": [REC] * 9, "window_idx": range(9)}).to_parquet(tmp_path / f"{name}_000000000000.index.parquet")
    (tmp_path / f"{name}_000000000000.json").write_text(json.dumps({"cache_key": "0" * 64}))
    with pytest.raises(CacheMismatchError):
        load_representation(tmp_path, cfg, meta)
