"""Stage R0: the healthy-rejection attribution is exhaustive and agrees with the decision rule and the app's checks."""
import itertools

import numpy as np
import pandas as pd
import pytest

from ml.arbitrator import decide
from ml.rejection import CAUSES, attribute, clopper_pearson, failed_checks, joint_patterns, per_recording

THR = {"tau_p": 0.99, "tau_d": 30.0, "tau_e": 0.25, "tau_b": 0.9}


def grid() -> pd.DataFrame:
    """Every combination of pass / fail / exactly-at-the-threshold for the four checks, with a normal and a fault top class."""
    vals = {"msp": (0.999, 0.98, 0.99), "maha": (10.0, 40.0, 30.0), "ae": (0.1, 0.4, 0.25), "b1s": (0.5, 1.5, 0.9)}
    rows = [{"rec_id": f"r{i}", "top_class": top, "msp": m, "maha": d, "ae": e, "b1s": b, **THR}
            for i, (top, m, d, e, b) in enumerate(itertools.product(("normal", "imbalance"), *vals.values()))]
    return pd.DataFrame(rows)


def test_attribution_is_exhaustive():
    att = attribute(grid())
    rejected = att[att.outcome != "healthy"]
    assert len(rejected) > 0
    assert (rejected.rule_causes != "").all()                       # every non-accepted instance has a cause
    assert (rejected[[f"fail_{k}" for k in CAUSES]].any(axis=1)).all()
    assert (att[att.outcome == "healthy"].rule_causes == "").all()


def test_attribution_agrees_with_the_decision_rule():
    att = attribute(grid())
    for r in att.itertuples():
        d = decide("B_with_caution", r.top_class, r.msp, r.maha, r.ae, r.b1s, THR)
        assert (d["decision"], d["caution"]) == (r.decision, r.caution)
        confident = not (r.fail_msp or r.fail_maha)
        if not confident:
            assert r.outcome == "abstention"
        elif r.top_class != "normal":
            assert r.outcome == "known_fault" and r.rule_causes == "top_not_normal"
        else:
            assert r.outcome == ("healthy_with_caution" if (r.fail_ae or r.fail_b1s) else "healthy")


def test_ties_are_accepted():
    f = failed_checks(0.99, 30.0, 0.25, 0.9, **THR)
    assert not any(bool(v) for v in f.values())


def test_agrees_with_the_app_checks():
    """One definition of a failed check: backend.checks.checks gives the same pass / fail on the same scores."""
    from backend.checks import checks
    for r in attribute(grid()).itertuples():
        rec = {"diagnosis": {"decision": r.decision, "scores": {"msp": r.msp, "maha": r.maha, "ae_error": r.ae, "b1s": r.b1s},
                             "thresholds": THR}}
        passed = {c["key"]: c["passed"] for c in checks(rec)}
        assert {k: not passed[k] for k in ("msp", "maha", "ae", "b1s")} == {
            "msp": bool(r.fail_msp), "maha": bool(r.fail_maha), "ae": bool(r.fail_ae), "b1s": bool(r.fail_b1s)}


def test_per_recording_and_patterns_count_consistently():
    att = attribute(pd.concat([grid(), grid()]))                    # every recording scored twice
    per = per_recording(att)
    assert (per.instances == 2).all() and per[["healthy", "healthy_with_caution", "abstention", "known_fault"]].sum(axis=1).eq(2).all()
    jp = joint_patterns(att)
    assert jp.instances.sum() == int((att.outcome != "healthy").sum())


@pytest.mark.parametrize("k,n", [(0, 25), (5, 25), (25, 25)])
def test_clopper_pearson_bounds(k, n):
    lo, hi = clopper_pearson(k, n)
    assert 0 <= lo <= k / n <= hi <= 1 and (k > 0 or lo == 0) and (k < n or hi == 1)


def test_a_one_ulp_rounding_at_the_threshold_is_a_tie():
    """The order-statistic recording that defines τ_p can be stored one float step below it; that is a tie (accepted),
    as in the run that saved it, not a failed check."""
    row = {"rec_id": "t", "top_class": "normal", "msp": np.nextafter(0.99, 0.0), "maha": 10.0, "ae": 0.1, "b1s": 0.5, **THR}
    att = attribute(pd.DataFrame([row]))
    assert att.outcome.iloc[0] == "healthy" and not att.fail_msp.iloc[0] and att.tie_snapped.iloc[0] == "msp"
    far = attribute(pd.DataFrame([{**row, "msp": 0.99 - 1e-9}]))
    assert far.outcome.iloc[0] == "abstention" and far.tie_snapped.iloc[0] == ""
