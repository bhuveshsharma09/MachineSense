"""Development pool: reserved tests never enter; folders are class × severity."""
import pandas as pd
import pytest

from ml.config import DevelopmentConfig
from ml.paths import SYSTEM_CODE_DIR
from ml.splits import development_pool, folder_proxy


def test_pool_rule_on_synthetic():
    meta = pd.DataFrame({
        "rec_id": list("abcde"), "class10": ["normal", "imbalance", "imbalance", "imbalance", "horizontal_misalignment"],
        "fault_family": ["normal", "imbalance", "imbalance", "imbalance", "misalignment"],
        "severity": [0.0, 6.0, 35.0, 6.0, 0.5], "split_closed": ["train", "val", "train", "test", "train"],
        "split_severity": ["train", "val", "test", "train", "train"], "split_speed_block": ["train", "val", "train", "train", "test"]})
    pool = development_pool(meta)
    assert list(pool.rec_id) == ["a", "b"]
    assert list(pool.folder) == ["normal/-", "imbalance/6g"]
    assert folder_proxy(meta.iloc[[4]]).tolist() == ["horizontal_misalignment/0.5mm"]


def test_real_pool_excludes_every_reserved_test():
    path = SYSTEM_CODE_DIR / "metadata.csv"
    if not path.exists():
        pytest.skip("metadata.csv missing")
    cfg = DevelopmentConfig()
    pool = development_pool(pd.read_csv(path), cfg.allowed_closed, cfg.reserved_columns)
    assert set(pool.split_closed) <= {"train", "val"}
    assert (pool.split_severity != "test").all() and (pool.split_speed_block != "test").all()
    assert (pool.fr_measured_hz < 40).all()  # a consequence of the speed-block reservation
    assert pool.rec_id.is_unique
