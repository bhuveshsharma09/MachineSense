"""Stage R5 addendum: the --seed plumbing leaves seed 42 exactly as it was — same paths, same run plan, same thresholds and
the same saved R5 numbers — and seeds 43 / 44 never write a seed-42 file."""
import importlib.util
import json

import numpy as np
import pandas as pd
import pytest

from ml.paths import SYSTEM_CODE_DIR, results_dir
from tests.conftest import need_full_data

SAVED = (results_dir() / "scores" / "cnn6ae6_s42_loco_decisions.parquet",
         results_dir() / "scores" / "specialists_w-imbalance_s42_loco_fold0_errors.parquet")  # gitignored saved scores


def script(name):
    spec = importlib.util.spec_from_file_location(name.replace(".py", ""), SYSTEM_CODE_DIR / "scripts" / name)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


@pytest.fixture(scope="module")
def s34():
    return script("34_train_specialists.py")


def test_default_seed_is_42_and_keeps_every_seed42_path(s34):
    assert s34.SEED == 42
    import argparse, re
    assert re.search(r'add_argument\("--seed", type=int, default=42', (SYSTEM_CODE_DIR / "scripts" / "34_train_specialists.py").read_text())
    assert s34.errors_path("imbalance", 0).name == "specialists_w-imbalance_s42_loco_fold0_errors.parquet"
    assert s34.weights_path("imbalance", 0, "overhang_ball").parts[-3:] == ("w-imbalance_s42", "fold0", "overhang_ball.pt")
    assert s34.spec_res() == results_dir() / "specialists"
    assert s34.out_file("r4_specialists", "json") == results_dir() / "research" / "r4_specialists.json"
    s34.set_seed(43)
    try:
        assert s34.spec_res() == results_dir() / "specialists" / "s43"
        assert s34.out_file("r4_ambiguity", "csv").name == "r4_ambiguity_s43.csv"
        assert "_s43_" in s34.errors_path("imbalance", 0).name and "w-imbalance_s43" in str(s34.weights_path("imbalance", 0, "x"))
    finally:
        s34.set_seed(42)


def test_default_reproduces_the_seed42_run_plan(s34):
    """Every saved seed-42 run used the plan run_plan gives now: the same specialists and the same fit recordings."""
    need_full_data(*SAVED)
    from ml.specialists import run_plan
    from ml.splits import FAULT_CLASSES
    cfg, meta, pool, assignment = s34.setup()
    for c in FAULT_CLASSES:
        for k in (0, 1, 2):
            p = run_plan(assignment, pool, c, k)
            tr = json.loads((s34.spec_res() / f"w-{c}" / f"fold{k}_train.json").read_text())
            assert set(tr["specialists"]) == set(p["specialists"])
            assert all(tr["specialists"][u]["n_fit_recordings"] == len(p["specialists"][u]["fit"]) for u in p["specialists"])
            e = pd.read_parquet(s34.errors_path(c, k))
            assert {col[2:] for col in e.columns if col.startswith("e_")} == set(p["specialists"])


def test_calibration_reproduces_the_seed42_thresholds(s34):
    need_full_data(*SAVED)
    t = s34.calibrated_thresholds(42).set_index(["withheld", "outer_fold"])
    saved = pd.read_csv(results_dir() / "loco" / "decision_thresholds.csv").set_index(["withheld", "outer_fold"]).loc[t.index]
    for q in ("tau_p", "tau_d", "tau_e"):
        assert np.abs(t[q] - saved[q]).max() <= 1e-12


def test_seed42_r5_numbers_reproduce_after_the_change():
    need_full_data(*SAVED)
    saved = json.loads((results_dir() / "research" / "r5_decision_design.json").read_text())
    res = script("35_decision_design.py").compute(42, ("R4", "E1"))
    for name in ("R4", "E1"):
        a, b = res["summary"][name], saved["variants"][name]
        assert a["matched"]["variant_recall"] == b["matched"]["variant_recall"]
        assert a["matched"]["baseline_recall"] == b["matched"]["baseline_recall"]
        assert a["matched"]["paired_bootstrap"] == b["matched"]["paired_bootstrap"]
        assert {str(k): v for k, v in a["matched"]["per_fold"].items()} == b["matched"]["per_fold"]
        assert a["withheld_healthy"] == b["withheld_healthy"] and a["calibrated"] == b["calibrated"]
        assert a["reading"] == b["reading"]
    assert res["s0"]["floor_gates_pass"] == saved["step0_floor"]["floor_gates_pass"]
