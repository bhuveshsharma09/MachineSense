"""Split integrity: on synthetic metadata, and on the committed metadata.csv once it has split columns."""
import numpy as np
import pandas as pd
import pytest

from ml.labels import CLASS10
from ml.paths import SYSTEM_CODE_DIR
from ml.splits import (CLOSED_FRACTIONS, FAULT_CLASSES, SPEED_BLOCK_HZ, _quota, assign_splits, is_severity_test,
                       split_columns, split_hash)

SIZES = {"normal": 49, "imbalance": 333, "horizontal_misalignment": 197, "vertical_misalignment": 301,
         "underhang_ball": 186, "underhang_cage": 188, "underhang_outer": 184,
         "overhang_ball": 137, "overhang_cage": 188, "overhang_outer": 188}
SEVERITIES = {"normal": [0.0], "imbalance": [6, 10, 15, 20, 25, 30, 35],
              "horizontal_misalignment": [0.5, 1.0, 1.5, 2.0],
              "vertical_misalignment": [0.51, 0.63, 1.27, 1.40, 1.78, 1.90]}


def _synthetic() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    rows = []
    for cls, n in SIZES.items():
        family = ("bearing" if cls.startswith(("underhang", "overhang")) else
                  "misalignment" if "misalignment" in cls else cls)
        sev = SEVERITIES.get(cls, [0, 6, 20, 35])
        for i in range(n):
            rows.append({"rec_id": f"{cls}__{i:03d}", "class10": cls, "fault_family": family,
                         "severity": float(sev[i % len(sev)]), "fr_measured_hz": rng.uniform(11.8, 60.7)})
    return pd.DataFrame(rows)


def _real() -> pd.DataFrame | None:
    path = SYSTEM_CODE_DIR / "metadata.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    return df if "split_closed" in df.columns else None


FRAMES = {"synthetic": lambda: assign_splits(_synthetic(), seed=42), "real": _real}


@pytest.fixture(params=list(FRAMES))
def df(request):
    frame = FRAMES[request.param]()
    if frame is None:
        pytest.skip("metadata.csv has no split columns yet")
    return frame


def test_one_row_per_recording_and_no_missing_split(df):
    assert df["rec_id"].is_unique
    cols = split_columns(df)
    assert len(cols) == 4 + len(FAULT_CLASSES)  # closed, speed block, severity, cv_fold + LOCO
    assert df[cols].notna().all().all()


def test_counts_add_up(df):
    for col in split_columns(df):
        assert df[col].value_counts().sum() == len(df)


def test_closed_values_and_stratification(df):
    assert set(df["split_closed"]) == set(CLOSED_FRACTIONS)
    frac = pd.crosstab(df["class10"], df["split_closed"], normalize="index")
    for split, target in CLOSED_FRACTIONS.items():
        assert (frac[split] - target).abs().max() <= 0.02, frac


def test_loco_withheld_class_absent_from_train_val(df):
    assert "loco_normal" not in df.columns
    for cls in FAULT_CLASSES:
        col = df[f"loco_{cls}"]
        assert set(col[df["class10"] == cls]) == {"withheld"}
        assert set(col[df["class10"] != cls]) <= {"train", "val", "test"}
        # Known classes keep the closed-set assignment, so LOCO folds are comparable.
        assert (col[df["class10"] != cls] == df.loc[df["class10"] != cls, "split_closed"]).all()


def test_speed_block(df):
    fast = df["fr_measured_hz"] >= SPEED_BLOCK_HZ
    assert (df.loc[fast, "split_speed_block"] == "test").all()
    assert set(df.loc[~fast, "split_speed_block"]) <= {"train", "val"}


def test_severity_block(df):
    held = is_severity_test(df)
    faulty = df["class10"] != "normal"
    assert (df.loc[held & faulty, "split_severity"] == "test").all()
    assert set(df.loc[~held & faulty, "split_severity"]) <= {"train", "val"}
    # Every fault class keeps some training data and has a held-out test severity.
    for cls in FAULT_CLASSES:
        s = df.loc[df["class10"] == cls, "split_severity"]
        assert (s == "train").any() and (s == "test").any(), cls


def test_cv_folds_cover_train_val_only(df):
    trval = df["split_closed"] != "test"
    assert (df.loc[~trval, "cv_fold"] == -1).all()
    assert set(df.loc[trval, "cv_fold"]) == set(range(5))
    sizes = pd.crosstab(df.loc[trval, "class10"], df.loc[trval, "cv_fold"])
    assert (sizes.max(axis=1) - sizes.min(axis=1)).max() <= 1


def test_deterministic_for_seed():
    a, b = assign_splits(_synthetic(), seed=42), assign_splits(_synthetic(), seed=42)
    assert split_hash(a) == split_hash(b)
    assert split_hash(a) != split_hash(assign_splits(_synthetic(), seed=43))


def test_quota_exact():
    for n in range(1, 400):
        counts = _quota(n, list(CLOSED_FRACTIONS.values()))
        assert counts.sum() == n and (np.abs(counts - n * np.array([0.7, 0.15, 0.15])) < 1).all()


def test_real_metadata_has_all_classes():
    real = _real()
    if real is None:
        pytest.skip("metadata.csv has no split columns yet")
    assert len(real) == 1951 and set(real["class10"]) == set(CLASS10)
