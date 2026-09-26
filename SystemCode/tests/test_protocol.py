"""Protocol helpers: speed-stratified splits and the two threshold rules."""
import numpy as np
import pandas as pd

from ml.protocol import max_threshold, order_statistic_threshold, speed_stratified_split


def _recs(n=17):
    return pd.DataFrame({"rec_id": [f"r{i}" for i in range(n)], "fr_measured_hz": np.linspace(12, 40, n)})


def test_split_counts_and_speed_spread():
    parts = speed_stratified_split(_recs(17), {"fit": 2, "cal": 1}, seed=0)
    assert set(parts.index) == set(_recs(17).rec_id)
    assert (parts == "cal").sum() in (5, 6) and (parts == "fit").sum() in (11, 12)
    sp = _recs(17).set_index("rec_id").fr_measured_hz
    cal = sp[parts[parts == "cal"].index]
    assert cal.max() - cal.min() > 0.6 * (sp.max() - sp.min())  # calibration spans the speed range
    assert parts.equals(speed_stratified_split(_recs(17), {"fit": 2, "cal": 1}, seed=0))


def test_three_way_split():
    parts = speed_stratified_split(_recs(24), {"fit": 4, "cal": 1, "eval": 1}, seed=1)
    assert parts.value_counts().to_dict() == {"fit": 16, "cal": 4, "eval": 4}


def test_thresholds():
    assert max_threshold(np.array([0.1, 0.5, 0.3])) == 0.5
    t = order_statistic_threshold(np.arange(1, 100, dtype=float), 0.95)  # n = 99 → r = 95
    assert t["n"] == 99 and t["r"] == 95 and t["threshold"] == 95.0 and np.isclose(t["attainable_level"], 0.95)
    small = order_statistic_threshold(np.array([1.0, 2.0, 3.0]), 0.95)  # r = 4 > n → accept all
    assert small["threshold"] == float("inf") and small["attainable_level"] == 1.0
