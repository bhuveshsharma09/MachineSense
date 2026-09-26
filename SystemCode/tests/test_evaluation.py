"""Recording-level aggregation, class6 view, bootstrap metrics and block permutation."""
import numpy as np
import pandas as pd

from ml.evaluation import (CLASS10_TO_6, aggregate_recordings, block_permutation_importance, metrics,
                           paired_bootstrap, to_class6)
from ml.labels import CLASS6, CLASS10


def test_aggregation_is_mean_log_renormalised():
    p = np.full((3, 10), 0.05)
    p[:, 1] = [0.55, 0.9, 0.0]  # a zero is clipped, not -inf
    p = p / p.sum(axis=1, keepdims=True)
    agg = aggregate_recordings(p, ["r"] * 3, floor=1e-6)
    expected = np.exp(np.log(np.clip(p, 1e-6, 1)).mean(axis=0))
    assert np.allclose(agg.loc["r"].to_numpy(), expected / expected.sum())
    assert np.isclose(agg.sum(axis=1).iloc[0], 1.0)


def test_class6_sums():
    assert set(CLASS10_TO_6.values()) == set(CLASS6)
    p10 = pd.DataFrame([np.arange(1, 11) / 55.0], columns=CLASS10)
    p6 = to_class6(p10)
    assert np.isclose(p6.sum(axis=1).iloc[0], 1.0)
    assert np.isclose(p6["underhang_bearing"].iloc[0], p10[["underhang_ball", "underhang_cage", "underhang_outer"]].sum(axis=1).iloc[0])


def test_metrics_and_paired_bootstrap():
    y = np.array(["normal", "imbalance"] * 20)
    m = metrics(y, y, CLASS10, n_boot=200, seed=0)
    assert m["macro_f1"] == 1.0 and m["macro_f1_ci"] == [1.0, 1.0]
    wrong = np.where(np.arange(40) < 10, "imbalance", y)
    d = paired_bootstrap(y, y, wrong, CLASS10, n_boot=200, seed=0)
    assert d["diff"] > 0 and d["ci"][0] > 0 and d["p_positive"] > 0.95


def test_block_permutation_moves_whole_recordings():
    rng = np.random.default_rng(0)
    n_rec, n_win = 40, 9
    labels = np.array(["normal", "imbalance"])[np.arange(n_rec) % 2]
    X = np.zeros((n_rec * n_win, 2))
    X[:, 0] = np.repeat((labels == "imbalance").astype(float), n_win)  # informative
    X[:, 1] = rng.normal(size=n_rec * n_win)  # noise
    rec_ids = np.repeat([f"r{i}" for i in range(n_rec)], n_win)
    y_rec = pd.Series(labels, index=[f"r{i}" for i in range(n_rec)])
    seen = []

    def predict(Xw):
        seen.append(Xw.copy())
        p = np.full((len(Xw), 10), 1e-3)
        p[:, CLASS10.index("imbalance")] = np.where(Xw[:, 0] > 0.5, 1, 1e-3)
        p[:, CLASS10.index("normal")] = np.where(Xw[:, 0] > 0.5, 1e-3, 1)
        return p / p.sum(axis=1, keepdims=True)

    imp = block_permutation_importance(X, rec_ids, y_rec, predict, {"informative": [0], "noise": [1]}, 3, 0)
    assert imp.set_index("group").loc["informative", "importance"] > 0.3
    assert imp.set_index("group").loc["noise", "importance"] == 0
    # Every shuffled matrix keeps each recording's 9 windows together (constant column 0 within a block).
    for Xw in seen:
        assert (Xw[:, 0].reshape(n_rec, n_win).std(axis=1) == 0).all()


def test_mixed_bootstrap_keeps_folders_whole():
    from ml.evaluation import metrics_mixed, mixed_bootstrap_draws

    classes = ["normal"] * 6 + ["imbalance"] * 9 + ["underhang_ball"] * 6
    folders = ["normal/-"] * 6 + ["imb/a"] * 3 + ["imb/b"] * 3 + ["imb/c"] * 3 + ["ub/a"] * 3 + ["ub/b"] * 3
    draws = mixed_bootstrap_draws(classes, folders, n_boot=50, seed=0)
    folders = np.asarray(folders)
    for d in draws:
        fault = [i for i in d if classes[i] != "normal"]
        # each drawn fault folder contributes all of its recordings (counts are multiples of the folder size, 3)
        counts = pd.Series(folders[fault]).value_counts()
        assert (counts % 3 == 0).all()
        assert sum(1 for i in d if classes[i] == "normal") == 6  # normal resampled individually, same count
    m = metrics_mixed(classes, classes, ["normal", "imbalance", "underhang_ball"], draws, folders)
    assert m["macro_f1"] == 1.0 and m["per_folder_accuracy"]["imb/a"]["accuracy"] == 1.0
