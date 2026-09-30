"""Stage R4: the withheld class never reaches a specialist's training, normalisation or calibration; the combining rule on
a hand-checked case; score calibration is per specialist."""
import json
import math

import numpy as np
import pandas as pd
import pytest

from ml.specialists import (NORMAL, accepting_sets, accepts, calibrate, empirical_rank_scores, log_ratio_scores, run_plan,
                            verdict)


@pytest.fixture(scope="module")
def data():
    from ml.config import load_config
    from ml.paths import DEFAULT_CONFIG, SYSTEM_CODE_DIR, results_dir
    from ml.splits import development_pool

    cfg = load_config(DEFAULT_CONFIG)
    meta = pd.read_csv(SYSTEM_CODE_DIR / "metadata.csv")
    pool = development_pool(meta, cfg.development.allowed_closed, cfg.development.reserved_columns).merge(
        pd.read_csv(results_dir() / cfg.development.folds_file)[["rec_id", "dev_fold"]], on="rec_id")
    return meta, pool, pd.read_csv(results_dir() / "q1q3" / "q3_fit_cal_eval_assignment.csv")


def test_withheld_class_never_reaches_a_specialist_in_any_run(data):
    from ml.splits import FAULT_CLASSES

    meta, pool, assignment = data
    cls = pool.set_index("rec_id").class10
    for w in FAULT_CLASSES:
        for k in (0, 1, 2):
            p = run_plan(assignment, pool, w, k)
            assert w not in p["specialists"] and len(p["specialists"]) == 9 and NORMAL in p["specialists"]
            for u, sp in p["specialists"].items():
                assert (cls.loc[sp["fit"]] == u).all() and (cls.loc[sp["cal"]] == u).all()   # own class only
                assert set(sp["fit"]) <= set(p["ids"]["fit"]) and set(sp["cal"]) <= set(p["ids"]["cal"])
                assert not set(sp["fit"]) & set(sp["cal"])


def test_normalisation_refuses_the_withheld_class(data):
    """The specialists' normalisation (norm_stats in development mode with the run's context) fails loudly if a withheld
    recording, or a calibration recording, is offered as a fit id."""
    from ml.datasets import norm_stats

    meta, pool, assignment = data
    p = run_plan(assignment, pool, "imbalance", 0)
    index = pd.DataFrame({"rec_id": sorted(pool.rec_id), "window_idx": 0})
    X = np.ones((len(index), 6, 2, 2), np.float32)
    fit = p["specialists"]["horizontal_misalignment"]["fit"]
    norm_stats(X, index, meta, fit_ids=fit, context=p["ids"]["context"])       # the declared use works
    w_rec = sorted(pool.loc[pool.class10 == "imbalance", "rec_id"])[0]
    with pytest.raises(ValueError):
        norm_stats(X, index, meta, fit_ids=fit + [w_rec], context=p["ids"]["context"])
    with pytest.raises(ValueError):
        norm_stats(X, index, meta, fit_ids=fit + p["ids"]["cal"][:1], context=p["ids"]["context"])


def test_saved_specialist_normalisation_excludes_the_withheld_class(data):
    """Proof from the stage's own outputs (when present): every saved specialist normalisation file lists exactly the
    plan's fit recordings of its class, and none of the withheld class."""
    from ml.paths import results_dir

    meta, pool, assignment = data
    files = sorted((results_dir() / "specialists").glob("w-*/fold*_*_norm.json"))
    if not files:
        pytest.skip("Stage R4 outputs not present")
    cls = pool.set_index("rec_id").class10
    for f in files:
        d = json.loads(f.read_text())
        assert d["specialist"] != d["withheld"] and not (cls.loc[d["fit_ids"]] == d["withheld"]).any()
        assert d["fit_ids"] == run_plan(assignment, pool, d["withheld"], d["fold"])["specialists"][d["specialist"]]["fit"]


def hand_case():
    rows = {  # rec: (role, class, e_normal, e_a)
        "n1": ("cal", "normal", 1.0, 10.0), "n2": ("cal", "normal", 3.0, 10.0),
        "a1": ("cal", "a", 10.0, 2.0), "a2": ("cal", "a", 10.0, 4.0), "a3": ("cal", "a", 10.0, 8.0),
        "x": ("eval", "normal", 2.0, 100.0), "y": ("eval", "a", 2.0, 4.8), "z": ("foldk_withheld", "w", 20.0, 40.0)}
    df = pd.DataFrame(rows, index=["role", "class10", "normal", "a"]).T
    return df[["normal", "a"]].astype(float), df.role, df.class10


def test_log_ratio_scores_by_hand():
    E, roles, classes = hand_case()
    s = log_ratio_scores(E, roles, classes)
    # normal: median of its own calibration errors (1, 3) = 2; leave-one-out: n1 → 3, n2 → 1
    assert s.loc["n1", "normal"] == pytest.approx(math.log(1 / 3)) and s.loc["n2", "normal"] == pytest.approx(math.log(3))
    assert s.loc["x", "normal"] == pytest.approx(0.0) and s.loc["a1", "normal"] == pytest.approx(math.log(5))
    # a: median (2, 4, 8) = 4; leave-one-out a1 → 6, a2 → 5, a3 → 3
    assert s.loc["a1", "a"] == pytest.approx(math.log(2 / 6)) and s.loc["a3", "a"] == pytest.approx(math.log(8 / 3))
    assert s.loc["y", "a"] == pytest.approx(math.log(4.8 / 4)) and s.loc["z", "a"] == pytest.approx(math.log(10))


def test_combining_rule_by_hand():
    E, roles, classes = hand_case()
    s = log_ratio_scores(E, roles, classes)
    A = accepting_sets(s, 0.5)
    assert A["n1"] == "normal" and A["x"] == "normal" and verdict(A["x"]) == "healthy"
    assert A["y"] == "a+normal" and verdict(A["y"]) == "ambiguous_healthy_fault"        # ambiguity kept, not collapsed
    assert A["a1"] == "a" and verdict(A["a1"]) == "known_fault"
    assert A["z"] == "" and verdict(A["z"]) == "unfamiliar"
    assert verdict("a+b") == "ambiguous_fault"
    assert accepts(0.5 + 1e-13, 0.5) and not accepts(0.5 + 1e-9, 0.5)                   # R0 tie rule


def test_calibration_uses_only_calibration_recordings():
    rng = np.random.default_rng(0)
    n = 60
    roles = pd.Series(["cal"] * n + ["foldk_withheld"] * 20)
    classes = pd.Series(["normal", "a"] * (n // 2) + ["w"] * 20)
    E = pd.DataFrame({"normal": rng.lognormal(0, 0.3, n + 20), "a": rng.lognormal(0, 0.3, n + 20)})
    s = log_ratio_scores(E, roles, classes)
    t = calibrate(s, roles)
    E2 = E.copy(); E2.loc[roles == "foldk_withheld"] *= 1000.0
    s2 = log_ratio_scores(E2, roles, classes)
    assert calibrate(s2, roles)["threshold"] == t["threshold"]                  # the withheld rows move nothing
    assert np.allclose(s2[roles == "cal"], s[roles == "cal"])
    assert t["threshold"] == np.sort(s.min(axis=1)[roles == "cal"])[math.ceil(0.975 * (n + 1)) - 1]
    with pytest.raises(ValueError):                                             # a withheld calibration row is refused
        log_ratio_scores(E, pd.Series(["cal"] * (n + 20)), classes)


def test_score_calibration_is_per_specialist():
    E, roles, classes = hand_case()
    s = log_ratio_scores(E, roles, classes)
    E2 = E.copy(); E2.loc[["a1", "a2", "a3"], "normal"] = [50.0, 60.0, 70.0]   # specialist normal on class-a recordings
    s2 = log_ratio_scores(E2, roles, classes)
    assert np.allclose(s2["a"], s["a"])                                         # specialist a is untouched
    assert np.allclose(s2.loc[["n1", "n2", "x", "y", "z"], "normal"], s.loc[["n1", "n2", "x", "y", "z"], "normal"])
    r = empirical_rank_scores(E, roles, classes)
    assert r["normal"].max() <= 2 / 3 + 1e-12                                   # the declared cap with 2 calibration normals
