"""Q3 scores (ml/openset.py) on synthetic data, and the process isolation of the deep LOCO script."""
import json
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from ml.openset import knn_kth_cosine, loco_run_ids, mahalanobis_min, recording_mean, recording_msp
from ml.paths import SYSTEM_CODE_DIR


def test_mahalanobis_uses_one_within_class_covariance():
    rng = np.random.default_rng(0)
    a = rng.normal([0, 0], [1.0, 0.1], size=(400, 2))
    b = rng.normal([10, 0], [1.0, 0.1], size=(400, 2))
    fit, lab = np.vstack([a, b]), np.array(["a"] * 400 + ["b"] * 400)
    d, info = mahalanobis_min(fit, lab, np.array([[0.0, 0.5], [2.0, 0.0], [10.0, 0.0]]))
    assert info["classes"] == ["a", "b"]
    # the within-class variance is ~1 along x and ~0.01 along y: 0.5 in y is far, 2 in x is near,
    # and the between-class spread along x (0 vs 10) must NOT inflate the covariance
    # overall covariance would give x-variance ≈ 26 and a distance ≈ 0.4 for [2, 0]; within-class gives ≈ 2
    assert 1.5 < d[1] < 2.5 and d[0] > 2 * d[1] and d[2] < 0.5


def test_knn_kth_cosine():
    fit = np.array([[1.0, 0], [0, 1.0], [1.0, 1.0]])
    d = knn_kth_cosine(fit, np.array([[2.0, 0]]), k=2)
    assert np.isclose(d[0], 1 - 1 / np.sqrt(2))


def test_recording_msp_aggregates_log_probabilities():
    logits = np.array([[2.0, 0.0], [0.0, 0.0], [0.0, 5.0]])
    r = recording_msp(logits, ["r1", "r1", "r2"], ["x", "y"])
    p1 = np.exp(np.mean(np.log([[np.exp(2) / (1 + np.exp(2)), 1 / (1 + np.exp(2))], [0.5, 0.5]]), axis=0))
    p1 /= p1.sum()
    assert r.loc["r1", "pred"] == "x" and np.isclose(r.loc["r1", "msp"], p1[0]) and r.loc["r2", "pred"] == "y"
    assert recording_mean(np.array([1.0, 3.0, 5.0]), ["r1", "r1", "r2"]).to_dict() == {"r1": 2.0, "r2": 5.0}


def test_loco_run_ids_and_context():
    pool = pd.DataFrame({"rec_id": ["n1", "n2", "n3", "i1", "i2", "h1", "h2", "n4", "i3", "h3"],
                         "class10": ["normal"] * 3 + ["imbalance"] * 2 + ["horizontal_misalignment"] * 2 +
                                    ["normal", "imbalance", "horizontal_misalignment"],
                         "dev_fold": [1] * 7 + [0] * 3})
    a = pd.DataFrame({"withheld": "horizontal_misalignment", "outer_fold": 0,
                      "rec_id": ["n1", "n2", "n3", "i1", "i2"], "part": ["fit", "cal", "eval", "fit", "fit"]})
    ids = loco_run_ids(a, pool, "horizontal_misalignment", 0)
    assert ids["fit"] == ["i1", "i2", "n1"] and ids["cal"] == ["n2"] and ids["eval"] == ["n3"]
    assert ids["foldk_known"] == ["i3", "n4"] and ids["foldk_withheld"] == ["h3"]
    ctx = ids["context"]
    assert ctx.withheld == {"h1", "h2", "h3"} and ctx.calibration == {"n2"}
    assert ctx.evaluation == {"n3", "i3", "n4", "h3"} and not set(ids["fit"]) & (ctx.withheld | ctx.evaluation)
    with pytest.raises(ValueError):
        loco_run_ids(a[a.rec_id != "i2"], pool, "horizontal_misalignment", 0)


def test_loco_process_does_not_import_lightgbm():
    script = SYSTEM_CODE_DIR / "scripts" / "09_loco_cnn.py"
    code = ("import importlib.util, sys, json\n"
            "import ml.train, ml.train_ae, ml.openset\n"
            f"spec = importlib.util.spec_from_file_location('loco', {str(script)!r})\n"
            "mod = importlib.util.module_from_spec(spec)\n"
            "spec.loader.exec_module(mod)\n"
            "print(json.dumps('lightgbm' in sys.modules))")
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120, cwd=SYSTEM_CODE_DIR)
    assert r.returncode == 0, r.stderr[-2000:]
    assert json.loads(r.stdout.strip().splitlines()[-1]) is False


def test_three_way_classifier_first():
    from ml.openset import three_way
    pred = np.array(["normal", "imbalance", "normal", "imbalance", "normal", "normal"])
    msp = np.array([0.99, 0.99, 0.50, 0.50, 0.99, 0.50])
    maha = np.array([1.0, 1.0, 1.0, 1.0, 9.0, 1.0])
    ae = np.array([9.0, 9.0, 0.1, 0.1, 9.0, 9.0])
    d, ov = three_way(pred, msp, maha, ae, tau_p=0.9, tau_d=5.0, tau_e=1.0, ordering="classifier_first")
    # confident normal → healthy; confident fault → fault; unconfident normal + AE healthy → healthy (override);
    # unconfident fault → unknown even if AE healthy; far normal + AE abnormal → unknown; never a second class
    assert d.tolist() == ["healthy", "imbalance", "healthy", "unknown", "unknown", "unknown"]
    assert ov.tolist() == [False, False, True, False, False, False]
    d, _ = three_way(pred, msp, maha, ae, 0.99, 1.0, 1.0)  # ties accepted on MSP and distance
    assert d[0] == "healthy" and d[1] == "imbalance"


def test_three_way_ae_first():
    from ml.openset import three_way
    pred = np.array(["imbalance", "imbalance", "normal", "imbalance"])
    msp = np.array([0.99, 0.99, 0.99, 0.50])
    maha = np.array([1.0, 1.0, 1.0, 1.0])
    ae = np.array([0.5, 2.0, 2.0, 2.0])
    d, ov = three_way(pred, msp, maha, ae, 0.9, 5.0, 1.0, ordering="ae_first")
    # AE healthy wins even over a confident fault; top class normal with AE abnormal → unknown
    assert d.tolist() == ["healthy", "imbalance", "unknown", "unknown"] and not ov.any()


def test_outcome_categories():
    from ml.openset import outcome_category
    assert outcome_category(["healthy", "imbalance", "unknown"], ["x"] * 3, True).tolist() == \
        ["healthy", "known_fault", "unknown"]
    got = outcome_category(["imbalance", "overhang_ball", "healthy", "unknown", "imbalance", "healthy"],
                           ["imbalance", "imbalance", "imbalance", "imbalance", "normal", "normal"], False)
    assert got.tolist() == ["correct", "wrong_fault", "healthy", "unknown", "fault", "healthy"]


def test_healthy_rule_variants_b_and_c():
    from ml.openset import three_way
    pred = np.array(["normal", "normal", "normal", "imbalance", "normal"])
    msp = np.array([0.99, 0.99, 0.50, 0.99, 0.50])
    maha = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
    ae = np.array([0.5, 2.0, 0.5, 2.0, 2.0])
    a, _ = three_way(pred, msp, maha, ae, 0.9, 5.0, 1.0, "classifier_first")
    b, _ = three_way(pred, msp, maha, ae, 0.9, 5.0, 1.0, "classifier_first_no_override")
    c, _ = three_way(pred, msp, maha, ae, 0.9, 5.0, 1.0, "strict_healthy")
    assert a.tolist() == ["healthy", "healthy", "healthy", "imbalance", "unknown"]
    assert b.tolist() == ["healthy", "healthy", "unknown", "imbalance", "unknown"]  # no AE override
    assert c.tolist() == ["healthy", "unknown", "unknown", "imbalance", "unknown"]  # healthy also needs AE healthy


def test_weighted_auroc_and_oscr_match_the_unweighted_versions():
    from sklearn.metrics import roc_auc_score

    from ml.openset import oscr, weighted_auroc, weighted_oscr
    rng = np.random.default_rng(1)
    neg, pos = np.round(rng.normal(0, 1, 60), 1), np.round(rng.normal(0.8, 1, 25), 1)  # rounded: ties present
    ones_n, ones_p = np.ones(60), np.ones(25)
    assert np.isclose(weighted_auroc(neg, ones_n, pos, ones_p), roc_auc_score(np.r_[np.zeros(60), np.ones(25)], np.r_[neg, pos]))
    correct = rng.random(60) > 0.2
    assert np.isclose(weighted_oscr(neg, correct, ones_n, pos, ones_p), oscr(neg, correct, pos))
    # integer weights = repeating the rows
    wn, wp = rng.integers(0, 3, 60).astype(float), rng.integers(1, 3, 25).astype(float)
    rn, rp_ = np.repeat(neg, wn.astype(int)), np.repeat(pos, wp.astype(int))
    assert np.isclose(weighted_auroc(neg, wn, pos, wp),
                      roc_auc_score(np.r_[np.zeros(len(rn)), np.ones(len(rp_))], np.r_[rn, rp_]))
    assert np.isclose(weighted_oscr(neg, correct, wn, pos, wp), oscr(rn, np.repeat(correct, wn.astype(int)), rp_))


def test_empirical_rank_and_js():
    from ml.openset import empirical_rank, js_divergence, recording_msp, recording_probs
    cal = [1.0, 2.0, 2.0, 3.0]
    assert np.allclose(empirical_rank([0.5, 2.0, 3.0, 9.0], cal), [0 / 5, 3 / 5, 4 / 5, 4 / 5])  # ties count as ≤
    p = np.array([[1.0, 0.0], [0.5, 0.5]])
    q = np.array([[0.0, 1.0], [0.5, 0.5]])
    assert np.allclose(js_divergence(p, q), [1.0, 0.0])
    logits = np.array([[2.0, 0.0], [0.0, 0.0], [0.0, 5.0]])
    pr = recording_probs(logits, ["r1", "r1", "r2"], ["x", "y"])
    ms = recording_msp(logits, ["r1", "r1", "r2"], ["x", "y"])
    assert np.allclose(pr.sum(axis=1), 1) and np.allclose(pr.max(axis=1), ms.msp) and (pr.idxmax(axis=1) == ms.pred).all()
