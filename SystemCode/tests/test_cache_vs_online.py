"""The batch path (npy → batch_features → feature cache → read back) and the serving path (raw CSV →
recording_features) give identical features for one recording."""
import numpy as np
import pandas as pd
import pytest

from ml.cache import (feature_cache_key, feature_cache_path, read_feature_cache, sources_from_metadata,
                      write_feature_cache)
from ml.config import load_config
from ml.features import batch_features, feature_columns, recording_features
from ml.paths import DEFAULT_CONFIG, SYSTEM_CODE_DIR, data_dir
from ml.preprocess import load_csv

REC = "imbalance__20g__30.5152"


@pytest.fixture(scope="module")
def setup():
    meta_path = SYSTEM_CODE_DIR / "metadata.csv"
    if not meta_path.exists():
        pytest.skip("metadata.csv missing")
    meta = pd.read_csv(meta_path)
    row = meta[meta.rec_id == REC]
    npy = data_dir() / "raw_npy" / f"{REC}.npy"
    try:
        from ml.paths import data_root
        csv = data_root() / row.path.iloc[0]
    except FileNotFoundError:
        pytest.skip("MAFAULDA CSVs not available")
    if not npy.exists() or not csv.exists():
        pytest.skip("recording not available")
    return load_config(DEFAULT_CONFIG).preprocess, meta, row, csv


def test_batch_cache_equals_online(setup, tmp_path):
    pre, meta, row, csv = setup
    batch = batch_features(row, pre, lambda r: np.load(data_dir() / "raw_npy" / f"{r}.npy"))
    shas = sources_from_metadata(meta, [REC])
    key = feature_cache_key(pre, shas)
    path = feature_cache_path(tmp_path, "r1_test", key)
    write_feature_cache(batch, path, key, {})
    cached = read_feature_cache(path, pre, shas)

    online = recording_features(load_csv(csv), float(row.fr_measured_hz.iloc[0]), pre)  # serving: raw CSV in

    cols = feature_columns(online)
    assert cols == feature_columns(cached)
    assert np.array_equal(cached[cols].to_numpy(), online[cols].to_numpy())  # bit-identical
    assert np.array_equal(cached.speed_hz.to_numpy(), online.speed_hz.to_numpy())
    assert (cached.rec_id == REC).all() and len(cached) == 9
