"""Stage R5: the healthy gates come from the arbitrator; the gated composition; the level correction and its β fit."""
import math

import numpy as np
import pandas as pd
import pytest

from ml.specialists import (accepting_sets, db_to_ln_power, gated_verdict, healthy_gates, level_corrected,
                            within_class_slope)

THR = {"tau_p": 0.9, "tau_d": 30.0, "tau_e": 0.25, "tau_b": 0.9}


def test_gates_match_the_arbitrator_inequalities_including_ties():
    for ae in (0.1, 0.25, 0.4):
        for b1s in (0.5, 0.9, 1.5):
            assert healthy_gates(ae, b1s, THR) == (ae <= 0.25 and b1s <= 0.9)
    from ml.rejection import failed_checks
    f = failed_checks(0.95, 10.0, 0.26, 0.9, **THR)
    assert healthy_gates(0.26, 0.9, THR) == (not f["ae"] and not f["b1s"])


def test_gated_composition_by_hand():
    s = pd.DataFrame({"a": [0.1, 0.1, 2.0, 0.2], "b": [2.0, 0.3, 2.0, 0.1]}, index=["p", "q", "r", "t"])
    A = accepting_sets(s, 0.35)
    assert gated_verdict(True, A["r"]) == "healthy"                    # the gates decide healthy, whatever the bank says
    assert gated_verdict(False, A["p"]) == "known_fault"
    assert gated_verdict(False, A["q"]) == "ambiguous_fault" and A["q"] == "a+b"
    assert gated_verdict(False, A["r"]) == "unfamiliar"
    with pytest.raises(ValueError):
        gated_verdict(False, "normal+a")


def test_beta_one_is_the_energy_relative_error():
    e, level_db = 0.3, -40.0
    L = db_to_ln_power(level_db)
    assert L == pytest.approx(math.log(10 ** (level_db / 10)))
    s = pd.DataFrame({"a": [math.log(e)]})
    assert level_corrected(s, [L], 1.0).iloc[0, 0] == pytest.approx(math.log(e / 10 ** (level_db / 10)))


def test_within_class_slope_ignores_between_class_level_differences():
    rng = np.random.default_rng(0)
    L = np.r_[rng.normal(-9, 0.5, 50), rng.normal(-5, 0.5, 50)]         # a quiet and a loud class
    c = np.r_[["q"] * 50, ["l"] * 50]
    s = 0.7 * L + np.where(c == "l", -10.0, 0.0) + rng.normal(0, 0.01, 100)   # between-class offset opposes the slope
    assert within_class_slope(s, L, c) == pytest.approx(0.7, abs=0.02)
    assert within_class_slope(np.ones(4), np.ones(4), ["a", "a", "b", "b"]) == 0.0
