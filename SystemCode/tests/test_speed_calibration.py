"""Stage R1: the speed-conditioned transformation, its fit set and the tie rule."""
import numpy as np
import pandas as pd
import pytest

from ml.speed_calibration import (IDENTITY, SpeedModel, exceeds, fit_for_run, fit_scale, fit_speed_model,
                                  matched_threshold, standardise_run, std_threshold)


def run_frame(seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for role, n, cls in (("fit", 12, "normal"), ("cal", 3, "normal"), ("eval", 8, "normal"), ("eval", 20, "imbalance")):
        v = rng.uniform(12, 38, n)
        s = 0.2 + 0.1 * np.log(v) + rng.normal(0, 0.02 * np.log(v), n) + (1.0 if cls != "normal" else 0.0)
        rows += [{"role": role, "class10": cls, "speed_hz": a, "ae": b} for a, b in zip(v, s)]
    return pd.DataFrame(rows)


def test_identity_when_m_is_zero_and_d_is_one():
    s = np.array([0.1, 1.5, 7.0]); v = np.array([12.0, 25.0, 55.0])
    assert np.array_equal(IDENTITY.transform(s, v), s)
    m = SpeedModel(m_coef=(0.0, 0.0, 0.0), d_coef=(0.0, 0.0), x_lo=np.log(10), x_hi=np.log(40), n_fit=12)
    assert np.array_equal(m.transform(s, v), s)


def test_fit_never_touches_calibration_or_evaluation_rows():
    run = run_frame()
    base = fit_for_run(run, "ae")
    moved = run.copy()
    moved.loc[moved.role != "fit", "ae"] += 100.0                 # change every cal / eval score
    moved.loc[moved.role != "fit", "speed_hz"] *= 1.5
    assert fit_for_run(moved, "ae") == base
    fault_fit = run.copy()
    fault_fit.loc[(fault_fit.role == "eval") & (fault_fit.class10 != "normal"), "role"] = "fit"  # a fault is never a fit row
    assert fit_for_run(fault_fit, "ae") == base


def test_threshold_is_the_calibration_normal_maximum():
    run = run_frame()
    s, tau, _ = standardise_run(run, "ae")
    assert tau == s[(run.role == "cal") & (run.class10 == "normal")].max()


def test_tie_rule_is_applied():
    assert not exceeds(1.0, 1.0)
    assert not exceeds(np.nextafter(1.0, 2.0), 1.0)               # one float step above: a tie, accepted
    assert not exceeds(1.0 + 5e-13, 1.0)
    assert exceeds(1.0 + 1e-9, 1.0)
    assert not exceeds(0.5, 1.0)


def test_extrapolation_is_constant_outside_the_fitted_speeds():
    v = np.linspace(15, 30, 12)
    m = fit_speed_model(0.1 * np.log(v) ** 2 + 0.01 * np.sin(v), v, 2)
    assert m.m(5.0) == m.m(15.0) and m.m(80.0) == m.m(30.0) and m.d(80.0) == m.d(30.0)


def test_scale_recovers_a_speed_dependent_laplace_spread():
    rng = np.random.default_rng(1)
    x = rng.uniform(2.4, 3.7, 4000)
    r = rng.laplace(0, np.exp(-3 + 0.8 * x))
    a, b = fit_scale(r, x)
    assert abs(a + 3) < 0.2 and abs(b - 0.8) < 0.07


def test_degree_is_capped_and_small_fits_refused():
    with pytest.raises(ValueError):
        fit_speed_model(np.ones(12), np.linspace(12, 38, 12), 3)
    with pytest.raises(ValueError):
        fit_speed_model(np.ones(4), np.linspace(12, 38, 4), 2)


def test_matched_threshold_flags_exactly_n():
    h = np.array([0.3, 0.1, 0.9, 0.5, 0.7])
    for n in range(6):
        assert int(exceeds(h, matched_threshold(h, n)).sum()) == n
    assert std_threshold(h) == 0.9
