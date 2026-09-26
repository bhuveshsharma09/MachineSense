"""Cache identity: every input that changes feature values changes the key, and a stale cache fails loudly."""
import pandas as pd
import pytest

from ml.cache import (CacheMismatchError, feature_cache_key, feature_cache_path, read_cache_metadata,
                      read_feature_cache, write_feature_cache)
from ml.config import PreprocessConfig

PRE = PreprocessConfig()
SHAS = {"rec_a": "a" * 64, "rec_b": "b" * 64}


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
    base = feature_cache_key(PRE, SHAS)
    assert base == feature_cache_key(PRE, dict(reversed(list(SHAS.items()))))  # order-independent
    for name, variant in _variants().items():
        assert feature_cache_key(variant, SHAS) != base, name
    assert feature_cache_key(PRE, {**SHAS, "rec_b": "c" * 64}) != base  # a changed source recording
    assert feature_cache_key(PRE, SHAS, track="loco_overhang_outer") != base


def _df():
    return pd.DataFrame({"rec_id": ["rec_a"] * 2 + ["rec_b"] * 2, "window_idx": [0, 1, 0, 1], "f_x": [1.0, 2, 3, 4]})


def test_roundtrip_and_metadata(tmp_path):
    key = feature_cache_key(PRE, SHAS)
    path = feature_cache_path(tmp_path, "r1_features", key)
    write_feature_cache(_df(), path, key, {"split_hash": "4a93", "git_sha": "abc"})
    pd.testing.assert_frame_equal(read_feature_cache(path, PRE, SHAS), _df())
    meta = read_cache_metadata(path)
    assert meta["split_hash"] == "4a93" and meta["git_sha"] == "abc" and meta["cache_key"] == key


def test_stale_cache_fails_loudly(tmp_path):
    key = feature_cache_key(PRE, SHAS)
    path = feature_cache_path(tmp_path, "r1_features", key)
    write_feature_cache(_df(), path, key, {})
    for variant in _variants().values():
        with pytest.raises(CacheMismatchError):
            read_feature_cache(path, variant, SHAS)
    with pytest.raises(CacheMismatchError):
        read_feature_cache(path, PRE, {**SHAS, "rec_a": "0" * 64})
    with pytest.raises(CacheMismatchError):
        read_feature_cache(path, PRE, SHAS, track="closed")
    with pytest.raises(CacheMismatchError):
        read_feature_cache(path, PRE, {"rec_a": "a" * 64})  # a recording without a checksum
