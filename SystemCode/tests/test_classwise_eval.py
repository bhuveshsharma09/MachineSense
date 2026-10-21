"""Stage R9 helpers (ml/classwise_eval.py) on synthetic data: per-class thresholds by hand, the 50/50 shrink, the
fallback below 3 calibration recordings, the matched-q search, and that a recording uses its PREDICTED class's
threshold."""
import numpy as np
import pandas as pd
import pytest

from ml.classwise_eval import gate, known_rejections, matched_q, thresholds

# class a: 4 calibration recordings, class b: 4, class c: 2 (below the fallback minimum)
CAL = pd.DataFrame({"class10": ["a"] * 4 + ["b"] * 4 + ["c"] * 2,
                    "msp": [0.90, 0.95, 0.99, 1.00, 0.60, 0.70, 0.80, 0.90, 0.50, 0.55],
                    "maha": [1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0]})


def test_per_class_thresholds_by_hand():
    t = thresholds(CAL, 0.75, "P", ["a", "b"])["per_class"]
    # linear quantiles: maha_a at 0.75 = 3 + 0.25·(4 − 3) = 3.25; msp_a at 0.25 = 0.90 + 0.75·(0.95 − 0.90) = 0.9375
    assert t["a"][1] == pytest.approx(3.25) and t["a"][0] == pytest.approx(0.9375)
    assert t["b"][1] == pytest.approx(32.5) and t["b"][0] == pytest.approx(0.675)


def test_shrunk_is_halfway_between_per_class_and_global():
    p = thresholds(CAL, 0.9, "P", ["a", "b"])
    s = thresholds(CAL, 0.9, "S", ["a", "b"])
    g_p, g_d = p["global"]
    for c in ("a", "b"):
        assert s["per_class"][c][0] == pytest.approx(0.5 * p["per_class"][c][0] + 0.5 * g_p)
        assert s["per_class"][c][1] == pytest.approx(0.5 * p["per_class"][c][1] + 0.5 * g_d)


def test_class_with_fewer_than_3_calibration_recordings_falls_back_to_global():
    for arm in ("P", "S"):
        t = thresholds(CAL, 0.9, arm, ["a", "b", "c", "never_seen"])
        assert t["fallback"] == {"c", "never_seen"}
        assert t["per_class"]["c"] == t["global"] and t["per_class"]["never_seen"] == t["global"]


def test_matched_q_reaches_the_target_count():
    rng = np.random.default_rng(0)
    n = 200
    rec = pd.DataFrame({"role": "foldk_known", "pred": rng.choice(["a", "b"], n), "class10": "a",
                        "msp": rng.uniform(0.5, 1.0, n), "maha": rng.uniform(0.0, 50.0, n)})
    cal = pd.DataFrame({"class10": np.repeat(["a", "b"], 40), "msp": rng.uniform(0.5, 1.0, 80),
                        "maha": rng.uniform(0.0, 50.0, 80)})
    for arm in ("P", "S"):
        lo = int(known_rejections(rec, cal, 1.0, arm, ["a", "b"]))   # fewest rejections (loosest thresholds)
        hi = int(known_rejections(rec, cal, 0.5, arm, ["a", "b"]))   # most rejections
        for target in (lo, (lo + hi) // 2, hi - 1):
            m = matched_q(rec, cal, arm, ["a", "b"], target)
            assert abs(m["diff"]) <= 1 and m["count"] == int(known_rejections(rec, cal, m["q"], arm, ["a", "b"]))
            assert m["count"] == target or "jumps" in m["note"]
        out = matched_q(rec, cal, arm, ["a", "b"], lo - 3)            # unreachable: even q = 1.0 rejects more
        assert out["q"] == 1.0 and out["diff"] == 3


def test_a_recording_uses_its_predicted_class_threshold_not_its_true_class():
    table = thresholds(CAL, 0.75, "P", ["a", "b"])  # τ_d^a = 3.25, τ_d^b = 32.5; τ_p^a = 0.9375, τ_p^b = 0.675
    rec = pd.DataFrame({"pred": ["b", "a"], "class10": ["a", "b"], "msp": [0.70, 0.70], "maha": [20.0, 20.0]})
    assert gate(rec, table).tolist() == [True, False]  # same scores: passes b's thresholds, fails a's


def test_evaluate_run_end_to_end_on_a_synthetic_run():
    from ml.classwise_eval import evaluate_run
    from ml.loco_eval import decide
    rng = np.random.default_rng(3)
    roles = ["cal"] * 60 + ["foldk_known"] * 80 + ["foldk_withheld"] * 20 + ["eval"] * 20
    true = list(rng.choice(["normal", "a", "b"], 60, p=[0.1, 0.45, 0.45])) + list(rng.choice(["normal", "a", "b"], 80)) \
        + ["w"] * 20 + list(rng.choice(["a", "b"], 20))
    pred = [t if t != "w" else rng.choice(["a", "b", "normal"]) for t in true]
    rec = pd.DataFrame({"role": roles, "class10": true, "pred": pred, "msp": rng.uniform(0.6, 1.0, 180),
                        "maha": rng.uniform(0.0, 40.0, 180), "ae": rng.uniform(0, 1, 180), "withheld": "w",
                        "outer_fold": 0, "rec_id": [f"r{i}" for i in range(180)]})
    d = decide(rec)
    rows = {r["arm"]: r for r in evaluate_run(d, d.decision_B.to_numpy())}
    assert set(rows) == {"G", "P", "S"} and rows["G"]["matched_known_unknown"] == rows["G"]["target"]
    for a in ("P", "S"):
        assert abs(rows[a]["matched_known_unknown"] - rows["G"]["target"]) == abs(rows[a]["diff"])
        assert rows[a]["matched_withheld_n"] == 20 and "secondary_known_unknown" in rows[a]
