"""Stage ML-4.2: thresholds set on the deployment scoring path. Exact comparisons, ties accepted, no tolerance (repeated
CPU runs of ml.inference.score_recording are bit-identical). Tests that need the gitignored bundle or raw recordings are
skipped without them."""
import json

import numpy as np
import pandas as pd
import pytest

from ml.arbitrator import VARIANTS, decide
from ml.calibration import thresholds_from_scores
from ml.paths import SYSTEM_CODE_DIR, artifacts_dir, data_dir

BUNDLE = artifacts_dir() / "bundle"
needs_bundle = pytest.mark.skipif(not (BUNDLE / "calibration_scores.csv").exists() or not (data_dir() / "raw_npy").exists(),
                                  reason="bundle or raw recordings not available")
BOUNDARY = {"normal__18.432": ("tau_e", "tau_b"), "overhang__outer_race__20g__18.2272": ("tau_p",),
            "overhang__cage_fault__6g__13.1072": ("tau_d",)}
DECLARED = {"normal__18.432": "healthy", "overhang__outer_race__20g__18.2272": "overhang_outer",
            "overhang__cage_fault__6g__13.1072": "overhang_cage"}


def _cal():
    return pd.read_csv(BUNDLE / "calibration_scores.csv", index_col="rec_id", float_precision="round_trip")


def _thr():
    t = json.loads((BUNDLE / "thresholds.json").read_text())
    return {k: t[k] for k in ("tau_p", "tau_d", "tau_e", "tau_b")}


def test_threshold_formulas_on_synthetic_scores():
    n = 144
    s = pd.DataFrame({"msp": np.linspace(0.5, 1.0, n), "maha": np.linspace(1, 50, n), "ae": np.linspace(0.1, 1, n),
                      "b1s": np.linspace(0.2, 2, n)}, index=[f"r{i}" for i in range(n)])
    normal = pd.Series(False, index=s.index)
    normal.iloc[[3, 7, 11]] = True
    t = thresholds_from_scores(s, normal)
    r = int(np.ceil(0.975 * (n + 1)))  # 142
    assert t["order_statistic_r"] == r
    assert t["tau_d"] == np.sort(s.maha)[r - 1] and t["tau_p"] == -np.sort(-s.msp)[r - 1]
    assert t["tau_e"] == s.ae[normal].max() and t["tau_b"] == s.b1s[normal].max()
    assert t["defining_recordings"]["tau_e"] == ["r11"]


@needs_bundle
def test_bundle_thresholds_are_the_declared_statistics_of_the_calibration_scores():
    cal = _cal()
    t = thresholds_from_scores(cal, cal.class10 == "normal")
    assert {k: t[k] for k in ("tau_p", "tau_d", "tau_e", "tau_b")} == _thr()
    assert len(cal) == 144 and int((cal.class10 == "normal").sum()) == 5
    for rid, taus in BOUNDARY.items():
        for tau in taus:
            assert t["defining_recordings"][tau] == [rid]


@needs_bundle
@pytest.mark.parametrize("rid", list(BOUNDARY))
def test_shared_function_reproduces_the_calibration_scores(rid):
    from ml.inference import get_bundle, load_recording, score_recording
    s = score_recording(*load_recording(rid), get_bundle())
    c = _cal().loc[rid]
    assert (s["msp"], s["maha"], s["ae"], s["b1s"], s["top"]) == (c.msp, c.maha, c.ae, c.b1s, c.top_class)


@needs_bundle
@pytest.mark.parametrize("rid", list(BOUNDARY))
def test_boundary_recordings_produce_the_declared_decisions(rid):
    from ml.inference import diagnose
    d = diagnose(rid, with_xai=False)["diagnosis"]
    expected = DECLARED[rid]
    assert d["rule_variant"] == "B_with_caution" and d["caution"] is None
    assert (d["fault_class"] or d["decision"]) == expected
    assert d["rule_comparison"] == {v: expected for v in ("A", "B", "C", "D")}


@needs_bundle
def test_values_on_either_side_of_each_threshold():
    t = _thr()
    lo, hi = -np.inf, np.inf
    base = dict(top_class="normal", msp=t["tau_p"], maha=t["tau_d"], ae_error=t["tau_e"], b1s=t["tau_b"])
    for v in VARIANTS:  # exactly at every threshold: accepted (ties accepted), healthy, no caution
        r = decide(v, **base, thresholds=t)
        assert r["decision"] == "healthy" and r["caution"] is None, v
    step = {"msp": np.nextafter(t["tau_p"], lo), "maha": np.nextafter(t["tau_d"], hi)}  # one float step outside
    clear = {"msp": t["tau_p"] - 1e-3, "maha": t["tau_d"] * 1.01}
    for vals in (step, clear):
        for k, x in vals.items():
            assert decide("B", **{**base, k: x}, thresholds=t)["decision"] == "unknown"
    for x in (np.nextafter(t["tau_e"], hi), t["tau_e"] * 1.01):
        assert decide("B_with_caution", **{**base, "ae_error": x}, thresholds=t)["caution"] == "ae_disagrees"
        assert decide("C", **{**base, "ae_error": x}, thresholds=t)["decision"] == "unknown"
    for x in (np.nextafter(t["tau_b"], hi), t["tau_b"] * 1.01):
        assert decide("D", **{**base, "b1s": x}, thresholds=t)["decision"] == "unknown"
        assert decide("B_with_caution", **{**base, "b1s": x}, thresholds=t)["caution"] == "b1s_disagrees"
    clearly_in = dict(top_class="normal", msp=t["tau_p"] + (1 - t["tau_p"]) / 2, maha=t["tau_d"] * 0.5,
                      ae_error=t["tau_e"] * 0.5, b1s=t["tau_b"] * 0.5)
    assert all(decide(v, **clearly_in, thresholds=t)["decision"] == "healthy" for v in VARIANTS)
