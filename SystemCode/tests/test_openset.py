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
