"""Folder-grouped development folds: folders never split, every fold has every class on both sides."""
import numpy as np
import pandas as pd
import pytest

from ml.paths import SYSTEM_CODE_DIR
from ml.splits import assign_dev_folds, development_pool


def _synthetic():
    rows = []
    rng = np.random.default_rng(0)
    for cls, n_folders in [("imbalance", 6), ("underhang_ball", 3), ("vertical_misalignment", 5)]:
        for f in range(n_folders):
            for i in range(int(rng.integers(15, 30))):
                rows.append({"rec_id": f"{cls}_{f}_{i}", "class10": cls, "folder": f"{cls}/{f}",
                             "fr_measured_hz": rng.uniform(12, 40)})
    for i in range(25):
        rows.append({"rec_id": f"normal_{i}", "class10": "normal", "folder": "normal/-",
                     "fr_measured_hz": 12 + i})
    return pd.DataFrame(rows)


@pytest.fixture(params=["synthetic", "real"])
def pool(request):
    if request.param == "synthetic":
        return _synthetic()
    path = SYSTEM_CODE_DIR / "metadata.csv"
    if not path.exists():
        pytest.skip("metadata.csv missing")
    return development_pool(pd.read_csv(path))


def test_folds_are_valid(pool):
    fold, k = assign_dev_folds(pool, seed=42)
    assert k == pool[pool.class10 != "normal"].groupby("class10").folder.nunique().min()
    assert (fold >= 0).all() and set(fold) == set(range(k))  # every recording scored exactly once
    faults = pool.class10 != "normal"
    assert (pool[faults].assign(f=fold).groupby("folder").f.nunique() == 1).all()  # folders never split
    classes = set(pool.class10)
    for j in range(k):
        assert set(pool.class10[fold == j]) == classes  # held-out fold has every class
        assert set(pool.class10[fold != j]) == classes  # training side too
    # Normal is spread over the speed range in every fold.
    nsp = pool[~faults].assign(f=fold).groupby("f").fr_measured_hz
    assert (nsp.max() - nsp.min() > 0.5 * (pool[~faults].fr_measured_hz.max() - pool[~faults].fr_measured_hz.min())).all()


def test_deterministic_and_infeasible():
    p = _synthetic()
    assert assign_dev_folds(p, 1)[0].equals(assign_dev_folds(p, 1)[0])
    with pytest.raises(ValueError):
        assign_dev_folds(p[p.folder.isin(["imbalance/0", "underhang_ball/0", "vertical_misalignment/0"]) | (p.class10 == "normal")], 1)
