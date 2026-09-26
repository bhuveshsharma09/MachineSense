"""Folder path → labels, for every folder pattern in CLAUDE.md §3."""
from collections import Counter
from pathlib import Path

import pytest

from ml.config import EXPECTED_CLASS_COUNTS, EXPECTED_TOTAL_FILES
from ml.labels import CLASS6, CLASS10, parse_labels


def test_normal():
    lab = parse_labels("normal/12.288.csv")
    assert (lab.fault_family, lab.fault_type, lab.class10, lab.class6) == ("normal",) * 2 + ("normal",) * 2
    assert lab.severity == 0.0 and lab.severity_unit == ""
    assert lab.fr_file_hz == pytest.approx(12.288)


@pytest.mark.parametrize("mass", ["6g", "10g", "15g", "20g", "25g", "30g", "35g"])
def test_imbalance(mass):
    lab = parse_labels(f"imbalance/{mass}/61.8496.csv")
    assert lab.class10 == lab.class6 == "imbalance"
    assert lab.fault_family == "imbalance"
    assert lab.severity == float(mass[:-1]) and lab.severity_unit == "g"
    assert lab.fr_file_hz == pytest.approx(61.8496)


@pytest.mark.parametrize("mm", ["0.5mm", "1.0mm", "1.5mm", "2.0mm"])
def test_horizontal_misalignment(mm):
    lab = parse_labels(f"horizontal-misalignment/{mm}/20.2752.csv")
    assert lab.class10 == lab.class6 == "horizontal_misalignment"
    assert (lab.fault_family, lab.fault_type) == ("misalignment", "horizontal")
    assert lab.severity == float(mm[:-2]) and lab.severity_unit == "mm"


@pytest.mark.parametrize("mm", ["0.51mm", "0.63mm", "1.27mm", "1.40mm", "1.78mm", "1.90mm"])
def test_vertical_misalignment(mm):
    lab = parse_labels(f"vertical-misalignment/{mm}/20.2752.csv")
    assert lab.class10 == lab.class6 == "vertical_misalignment"
    assert (lab.fault_family, lab.fault_type) == ("misalignment", "vertical")
    assert lab.severity == float(mm[:-2]) and lab.severity_unit == "mm"


@pytest.mark.parametrize("position", ["underhang", "overhang"])
@pytest.mark.parametrize("folder, fault", [("ball_fault", "ball"), ("cage_fault", "cage"), ("outer_race", "outer")])
@pytest.mark.parametrize("mass", ["0g", "6g", "20g", "35g"])
def test_bearing(position, folder, fault, mass):
    lab = parse_labels(f"{position}/{folder}/{mass}/30.1056.csv")
    assert lab.class10 == f"{position}_{fault}"
    assert lab.class6 == f"{position}_bearing"
    assert (lab.fault_family, lab.fault_type) == ("bearing", fault)
    # Bearing sub-folders are added imbalance mass, not damage size.
    assert lab.severity == float(mass[:-1]) and lab.severity_unit == "g"


def test_absolute_path_with_parent_dirs():
    lab = parse_labels(Path("/some/where/mafaulda/overhang/cage_fault/6g/40.96.csv"))
    assert lab.class10 == "overhang_cage" and lab.severity == 6.0 and lab.fr_file_hz == pytest.approx(40.96)


@pytest.mark.parametrize("bad", [
    "normal/extra/12.288.csv",
    "imbalance/12.288.csv",
    "imbalance/6mm/12.288.csv",
    "horizontal-misalignment/0.5g/12.288.csv",
    "underhang/inner_race/6g/12.288.csv",
    "underhang/ball_fault/12.288.csv",
    "something_else/12.288.csv",
    "normal/12.288.txt",
])
def test_rejects_unknown_layouts(bad):
    with pytest.raises(ValueError):
        parse_labels(bad)


def test_class_lists_match_expected_counts():
    assert set(EXPECTED_CLASS_COUNTS) == set(CLASS10)
    assert sum(EXPECTED_CLASS_COUNTS.values()) == EXPECTED_TOTAL_FILES
    assert len(CLASS10) == 10 and len(CLASS6) == 6


def test_real_tree_counts():
    """If the dataset is present, every file parses and class counts match CLAUDE.md §3."""
    from ml.paths import data_root

    try:
        root = data_root()
    except FileNotFoundError:
        pytest.skip("MAFAULDA data root not available")
    counts = Counter(parse_labels(p.relative_to(root)).class10 for p in root.rglob("*.csv"))
    assert dict(counts) == EXPECTED_CLASS_COUNTS
