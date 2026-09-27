"""Every branch of every rule variant in ml/arbitrator.py, including ties at the thresholds."""
import pytest

from ml.arbitrator import CAUTION_LABEL, UNKNOWN_LABEL, VARIANTS, decide, rule_comparison

T = {"tau_p": 0.9, "tau_d": 5.0, "tau_e": 1.0, "tau_b": 2.0}
OK = dict(msp=0.95, maha=1.0)  # both classifier checks pass
LOW_MSP = dict(msp=0.5, maha=1.0)  # MSP check fails
FAR = dict(msp=0.95, maha=9.0)  # Mahalanobis check fails


def d(variant, top, scores, ae, b1s):
    return decide(variant, top, scores["msp"], scores["maha"], ae, b1s, T)


@pytest.mark.parametrize("variant", VARIANTS)
def test_confident_fault_is_that_fault_in_every_variant(variant):
    r = d(variant, "imbalance", OK, ae=9.0, b1s=9.0)
    assert (r["decision"], r["fault_class"], r["caution"]) == ("known_fault", "imbalance", None)
    assert r["display_label"] == "known fault: imbalance"


@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize("scores", [LOW_MSP, FAR])
def test_unconfident_fault_is_unknown_in_every_variant(variant, scores):
    r = d(variant, "imbalance", scores, ae=0.1, b1s=0.1)
    assert r["decision"] == "unknown" and r["display_label"] == UNKNOWN_LABEL


@pytest.mark.parametrize("variant, expected", [("A", "healthy"), ("B", "healthy"), ("C", "healthy"), ("D", "healthy"),
                                               ("B_with_caution", "healthy")])
def test_confident_normal_with_healthy_baselines(variant, expected):
    r = d(variant, "normal", OK, ae=0.5, b1s=0.5)
    assert r["decision"] == expected and r["caution"] is None and r["display_label"] == "healthy"


@pytest.mark.parametrize("variant, expected", [("A", "healthy"), ("B", "healthy"), ("C", "unknown"), ("D", "healthy"),
                                               ("B_with_caution", "healthy")])
def test_confident_normal_with_ae_abnormal(variant, expected):
    r = d(variant, "normal", OK, ae=3.0, b1s=0.5)
    assert r["decision"] == expected
    assert r["caution"] == ("ae_disagrees" if variant == "B_with_caution" else None)


@pytest.mark.parametrize("variant, expected", [("A", "healthy"), ("B", "healthy"), ("C", "healthy"), ("D", "unknown"),
                                               ("B_with_caution", "healthy")])
def test_confident_normal_with_b1s_abnormal(variant, expected):
    r = d(variant, "normal", OK, ae=0.5, b1s=3.0)
    assert r["decision"] == expected
    assert r["caution"] == ("b1s_disagrees" if variant == "B_with_caution" else None)


def test_caution_both_and_its_label():
    r = d("B_with_caution", "normal", OK, ae=3.0, b1s=3.0)
    assert (r["decision"], r["caution"], r["display_label"]) == ("healthy", "both", CAUTION_LABEL)


@pytest.mark.parametrize("variant, expected, override", [("A", "healthy", True), ("B", "unknown", False),
                                                         ("C", "unknown", False), ("D", "unknown", False),
                                                         ("B_with_caution", "unknown", False)])
def test_unconfident_normal_with_ae_healthy_only_a_overrides(variant, expected, override):
    r = d(variant, "normal", LOW_MSP, ae=0.5, b1s=0.5)
    assert r["decision"] == expected and r["ae_override"] is override


@pytest.mark.parametrize("variant", VARIANTS)
def test_unconfident_normal_with_ae_abnormal_is_unknown(variant):
    assert d(variant, "normal", FAR, ae=3.0, b1s=0.5)["decision"] == "unknown"


def test_ties_at_every_threshold_are_accepted():
    tie = dict(msp=T["tau_p"], maha=T["tau_d"])
    for v in VARIANTS:  # MSP = τ_p and d = τ_d pass; AE = τ_e and B1s = τ_b count as healthy
        r = d(v, "normal", tie, ae=T["tau_e"], b1s=T["tau_b"])
        assert r["decision"] == "healthy" and r["caution"] is None, v
    assert d("A", "normal", dict(msp=0.5, maha=1.0), ae=T["tau_e"], b1s=0.0)["decision"] == "healthy"  # override tie
    assert d("D", "normal", tie, ae=0.0, b1s=T["tau_b"] + 1e-9)["decision"] == "unknown"


def test_rule_comparison_and_unknown_variant():
    assert rule_comparison("normal", 0.95, 1.0, 3.0, 3.0, T) == {"A": "healthy", "B": "healthy", "C": "unknown",
                                                                 "D": "unknown"}
    assert rule_comparison("imbalance", 0.95, 1.0, 3.0, 3.0, T)["A"] == "imbalance"
    with pytest.raises(ValueError):
        decide("E", "normal", 0.95, 1.0, 0.0, 0.0, T)
