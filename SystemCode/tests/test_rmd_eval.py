"""Stage R8 evaluation helpers (ml/rmd_eval.py) on synthetic data: the gate and rule B, the matched known-side rejection
count, the outcome counts and the declared reading."""
import numpy as np
import pandas as pd

from ml.rmd_eval import gate_ok, matched_tau, outcomes, reading, rule_b


def test_gate_accepts_ties_and_rule_b_maps_decisions():
    ok = gate_ok([0.99, 0.99, 0.5, 0.99], [1.0, 2.0 + 1e-13, 1.0, 3.0], tau_p=0.99, tau_s=2.0)
    assert ok.tolist() == [True, True, False, False]
    assert rule_b(["normal", "imbalance", "imbalance"], [True, True, False]).tolist() == ["healthy", "imbalance", "unknown"]


def _rec():
    return pd.DataFrame({"role": ["foldk_known"] * 6 + ["foldk_withheld"] * 3,
                         "class10": ["normal", "imbalance", "imbalance", "imbalance", "overhang_ball", "overhang_ball",
                                     "vertical_misalignment"] + ["vertical_misalignment"] * 2,
                         "pred": ["normal", "imbalance", "imbalance", "overhang_ball", "overhang_ball", "overhang_ball",
                                  "imbalance", "normal", "imbalance"],
                         "msp": [0.99, 0.99, 0.5, 0.99, 0.99, 0.99, 0.99, 0.99, 0.99],
                         "R": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 9.0, 1.5, 2.5]})


def test_matched_tau_rejects_exactly_the_target_count():
    rec = _rec()
    for n in (1, 2, 3, 6):  # MSP alone rejects 1 known recording
        tau = matched_tau(rec, "R", 0.9, n)
        kn = rec[rec.role == "foldk_known"]
        assert int((~gate_ok(kn.msp, kn.R, 0.9, tau)).sum()) == n
    assert matched_tau(rec, "R", 0.9, 0) is None  # MSP alone already rejects more


def test_outcomes_count_withheld_and_known():
    rec = _rec()
    o = outcomes(rec, rule_b(rec.pred, gate_ok(rec.msp, rec.R, 0.9, 4.0)))
    assert o == {"withheld_n": 3, "withheld_unknown": 1, "withheld_healthy": 1, "withheld_known_fault": 1,
                 "known_n": 6, "known_unknown": 3, "known_wrong_fault": 1, "known_correct_fault": 1}


def test_reading_follows_the_declared_order():
    def tab(gains, dh=(0, 0, 0)):
        rows = []
        for s, g, h in zip((42, 43, 44), gains, dh):
            rows += [{"seed": s, "arm": "M", "unknown_recall": 0.5, "withheld_healthy": 10},
                     {"seed": s, "arm": "R", "unknown_recall": 0.5 + g, "withheld_healthy": 10 + h}]
        return pd.DataFrame(rows)
    assert reading(tab((0.06, 0.05, 0.10)))["reading"] == "helps, confirmed"
    assert reading(tab((0.06, 0.05, 0.10), (0, 1, 0)))["reading"] == "does not help"
    assert reading(tab((0.06, 0.04, 0.10)))["reading"] == "does not help"
    assert reading(tab((0.02, -0.08, 0.01)))["reading"] == "worse"
