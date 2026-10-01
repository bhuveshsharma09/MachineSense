"""Stage R7: aggregation, the probe's leakage guard, and the matched gate threshold."""
import numpy as np
import pandas as pd
import pytest

from ml.diagnostics import (DESCRIPTORS, fit_probe, gate_rejects, logit_scores, matched_u_threshold, probe_scores,
                            recording_embeddings)


def test_aggregations_by_hand():
    logits = np.array([[2.0, 0.0], [4.0, 0.0], [1.0, 1.0]])
    s = logit_scores(logits, ["a", "a", "b"])
    assert s.loc["a", "u_maxlogit"] == pytest.approx(-3.0) and s.loc["b", "u_maxlogit"] == pytest.approx(-1.0)
    lse = lambda v: np.log(np.exp(v).sum())
    assert s.loc["a", "u_energy"] == pytest.approx(-(lse([2, 0]) + lse([4, 0])) / 2)
    e = recording_embeddings(np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]), ["a", "a", "b"])
    assert e.loc["a"].tolist() == [2.0, 3.0] and list(e.index) == ["a", "b"]


def test_probe_recovers_a_linear_map_and_uses_fit_data_only():
    rng = np.random.default_rng(0)
    W = rng.normal(size=(8, 3))
    z_fit, z_new = rng.normal(size=(200, 8)), rng.normal(size=(60, 8))
    y_fit, y_new = z_fit @ W, z_new @ W
    p = fit_probe(z_fit, y_fit, seed=1)
    s = probe_scores(p, z_new, y_new)
    assert s["r2_mean"] > 0.99 and p["alpha"] <= 1.0
    p2 = fit_probe(z_fit, y_fit, seed=1)                       # deterministic; evaluation data never changes the fit
    probe_scores(p2, z_new + 100, y_new)
    assert np.allclose(p2["model"].coef_, p["model"].coef_)
    assert len(DESCRIPTORS) == 16 and len(set(DESCRIPTORS)) == 16


def test_gate_and_matched_threshold():
    u = np.array([0.1, 0.5, 0.9, 0.3, 0.7])
    m = np.array([1.0, 1.0, 1.0, 9.0, 1.0])                    # one recording rejected by Mahalanobis alone
    assert gate_rejects(u, 0.5 + 1e-13, m, 5.0).tolist() == [False, False, True, True, True]   # tie accepted
    for n in range(1, 5):
        t = matched_u_threshold(u, m, 5.0, n)
        assert int(gate_rejects(u, t, m, 5.0).sum()) == n
    assert matched_u_threshold(u, m, 5.0, 0) is None           # Mahalanobis alone already rejects one
