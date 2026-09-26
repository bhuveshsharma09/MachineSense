"""Cache identity: every input that changes feature values changes the key, stale caches fail loudly, split
assignments are never cached and are joined from the current metadata."""
import pandas as pd
import pytest

from ml.cache import (CacheMismatchError, Source, SplitHashMismatchError, check_split_hash, feature_cache_key,
                      feature_cache_path, load_features, read_cache_metadata, read_feature_cache, write_feature_cache)
from ml.config import PreprocessConfig
from ml.splits import split_hash

PRE = PreprocessConfig()
SRC = {"rec_a": Source("a" * 64, 29.94), "rec_b": Source("b" * 64, 50.12)}


def _variants():
    bs, mic, env, feat = PRE.bandstop, PRE.mic, PRE.envelope, PRE.features
    return {
        "bandstop edge": PRE.model_copy(update={"bandstop": bs.model_copy(update={"stop_hi_hz": bs.stop_hi_hz + 5})}),
        "mic padding": PRE.model_copy(update={"mic": mic.model_copy(update={"burst_pad": 5})}),
        "envelope band": PRE.model_copy(update={"envelope": env.model_copy(update={"primary_band_hz": (1000.0, 4000.0)})}),
        "feature version": PRE.model_copy(update={"features": feat.model_copy(update={"version": "r1-9.9"})}),
        "feature definition": PRE.model_copy(update={"features": feat.model_copy(update={"order_tol_bins": 2.0})}),
        "window": PRE.model_copy(update={"window": PRE.window.model_copy(update={"hop_s": 0.25, "n_windows": 17})}),
    }


def test_every_input_changes_the_key():
    base = feature_cache_key(PRE, SRC)
    assert base == feature_cache_key(PRE, dict(reversed(list(SRC.items()))))  # order-independent
    for name, variant in _variants().items():
        assert feature_cache_key(variant, SRC) != base, name
    assert feature_cache_key(PRE, {**SRC, "rec_b": Source("c" * 64, 50.12)}) != base  # changed recording
    assert feature_cache_key(PRE, {**SRC, "rec_b": Source("b" * 64, 50.13)}) != base  # changed supplied speed
    assert feature_cache_key(PRE, SRC, track="loco_overhang_outer") != base


def _df():
    return pd.DataFrame({"rec_id": ["rec_a"] * 2 + ["rec_b"] * 2, "window_idx": [0, 1, 0, 1], "f_x": [1.0, 2, 3, 4]})


def _write(tmp_path, extra=None):
    key = feature_cache_key(PRE, SRC)
    path = feature_cache_path(tmp_path, "r1_features", key)
    write_feature_cache(_df(), path, key, extra or {"git_sha": "abc"})
    return path, key


def test_roundtrip_and_metadata(tmp_path):
    path, key = _write(tmp_path)
    pd.testing.assert_frame_equal(read_feature_cache(path, PRE, SRC), _df())
    meta = read_cache_metadata(path)
    assert meta["git_sha"] == "abc" and meta["cache_key"] == key


def test_stale_cache_fails_loudly(tmp_path):
    path, _ = _write(tmp_path)
    for variant in _variants().values():
        with pytest.raises(CacheMismatchError):
            read_feature_cache(path, variant, SRC)
    with pytest.raises(CacheMismatchError):
        read_feature_cache(path, PRE, {**SRC, "rec_a": Source("0" * 64, 29.94)})
    with pytest.raises(CacheMismatchError):
        read_feature_cache(path, PRE, {**SRC, "rec_a": Source("a" * 64, 29.95)})  # speed changed since
    with pytest.raises(CacheMismatchError):
        read_feature_cache(path, PRE, SRC, track="closed")
    with pytest.raises(CacheMismatchError):
        read_feature_cache(path, PRE, {"rec_a": SRC["rec_a"]})  # a recording without a source record


def test_split_columns_are_refused(tmp_path):
    key = feature_cache_key(PRE, SRC)
    with pytest.raises(ValueError):
        write_feature_cache(_df().assign(split_closed="train"), feature_cache_path(tmp_path, "x", key), key, {})


def _meta(split="train"):
    return pd.DataFrame({"rec_id": ["rec_a", "rec_b"], "npy_sha256": ["a" * 64, "b" * 64],
                         "fr_measured_hz": [29.94, 50.12], "class10": ["normal", "imbalance"],
                         "class6": ["normal", "imbalance"], "fault_family": ["normal", "imbalance"],
                         "fault_type": ["normal", "imbalance"], "severity": [0.0, 6.0],
                         "split_closed": [split, "val"], "cv_fold": [0, -1]})


def test_load_joins_current_splits(tmp_path):
    path, _ = _write(tmp_path)
    df = load_features(path, _meta("train"), PRE)
    assert df.loc[df.rec_id == "rec_a", "split_closed"].eq("train").all()
    df2 = load_features(path, _meta("test"), PRE)  # a changed split shows up immediately
    assert df2.loc[df2.rec_id == "rec_a", "split_closed"].eq("test").all()


def test_split_hash_is_checked(tmp_path):
    meta = _meta()
    check_split_hash(split_hash(meta), meta, "ok")
    check_split_hash(None, meta, "no hash recorded")
    with pytest.raises(SplitHashMismatchError):
        check_split_hash("0" * 64, meta, "stale")
    path, _ = _write(tmp_path, {"split_hash": "0" * 64})
    with pytest.raises(SplitHashMismatchError):
        load_features(path, meta, PRE)


def test_representation_settings_do_not_touch_feature_keys():
    base = feature_cache_key(PRE, SRC)
    sc = PRE.spectro
    changed = PRE.model_copy(update={"spectro": sc.model_copy(update={"n_orders": 64, "mel_n": 64})})
    assert feature_cache_key(changed, SRC) == base
