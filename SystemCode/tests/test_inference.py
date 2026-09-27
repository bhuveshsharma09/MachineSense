"""Deployable inference (Stage ML-4): golden outputs, reserved-recording refusal, input checks, the speed warning, and
bundle refusal on a hash mismatch. Tests that need the (gitignored) bundle or the raw recordings skip without them."""
import json
import shutil

import numpy as np
import pandas as pd
import pytest

from ml.inference import (CALIBRATION_SPEED_WARNING, GOLDEN_RECORDINGS, SPEED_WARNING, InputError, ReservedRecordingError, check_rec_id, diagnose, speed_warnings,
                          stable_output as stable)
from ml.paths import SYSTEM_CODE_DIR, artifacts_dir, data_dir

BUNDLE = artifacts_dir() / "bundle"
GOLDEN = SYSTEM_CODE_DIR / "tests" / "golden" / "diagnose_golden.json"
GOLDEN_RECS = list(GOLDEN_RECORDINGS)
needs_bundle = pytest.mark.skipif(not (BUNDLE / "manifest.json").exists() or not (data_dir() / "raw_npy").exists(),
                                  reason="bundle or raw recordings not available")


def test_reserved_recordings_are_refused():
    meta = pd.read_csv(SYSTEM_CODE_DIR / "metadata.csv")
    for col in ("split_closed", "split_severity", "split_speed_block"):
        rid = meta.loc[meta[col] == "test", "rec_id"].iloc[0]
        with pytest.raises(ReservedRecordingError, match="reserved"):
            diagnose(rid)  # refused before the bundle is loaded
    with pytest.raises(InputError):
        check_rec_id("no_such_recording")


def test_speed_warning_rule():
    assert speed_warnings(20.0, (11.8, 39.75)) == []
    assert speed_warnings(11.8, (11.8, 39.75)) == [] and speed_warnings(39.75, (11.8, 39.75)) == []
    assert speed_warnings(45.0, (11.8, 39.75)) == [SPEED_WARNING]
    assert speed_warnings(10.0, (11.8, 39.75)) == [SPEED_WARNING]
    assert speed_warnings(float("nan"), (11.8, 39.75)) == [SPEED_WARNING]


def test_calibration_normal_speed_warning_rule():
    train, cal = (11.8, 39.75), (18.0, 38.4)
    assert speed_warnings(25.0, train, cal) == []
    assert speed_warnings(15.0, train, cal) == [CALIBRATION_SPEED_WARNING]  # trained, but no calibration normal this slow
    assert speed_warnings(39.0, train, cal) == [CALIBRATION_SPEED_WARNING]
    assert speed_warnings(45.0, train, cal) == [SPEED_WARNING, CALIBRATION_SPEED_WARNING]
    assert speed_warnings(18.0, train, cal) == [] and speed_warnings(38.4, train, cal) == []  # inclusive bounds
    assert speed_warnings(15.0, train) == []  # the calibration check is optional


@needs_bundle
def test_calibration_normal_speed_warning_in_diagnose():
    slow = diagnose("underhang__outer_race__0g__14.7456", with_xai=False)  # 14.48 Hz: trained, not calibrated for healthy
    w = slow["signal_quality"]["warnings"]
    assert CALIBRATION_SPEED_WARNING in w and SPEED_WARNING not in w
    lo, hi = slow["signal_quality"]["calibration_normal_speed_range_hz"]
    assert 18.0 < lo < 18.1 and 38.3 < hi < 38.4  # measured speeds of the 5 calibration normals (nominal 18.4–39.3 Hz)


@needs_bundle
def test_speed_warning_in_diagnose():
    fast = diagnose("underhang__outer_race__6g__40.96", with_xai=False)  # 39.995 Hz, above the training maximum
    assert SPEED_WARNING in fast["signal_quality"]["warnings"]
    slow = diagnose("normal__18.432", with_xai=False)
    assert SPEED_WARNING not in slow["signal_quality"]["warnings"]


@needs_bundle
def test_wrong_shape_is_refused():
    with pytest.raises(InputError):
        diagnose(np.zeros((1000, 8)))


@needs_bundle
@pytest.mark.parametrize("rec_id", GOLDEN_RECS)
def test_golden_outputs_are_identical_across_runs(rec_id):
    a, b = stable(diagnose(rec_id)), stable(diagnose(rec_id))
    assert a == b
    assert set(a) == {"input", "signal_quality", "diagnosis", "physics", "xai", "limitations", "provenance"}
    assert a["diagnosis"]["decision"] in ("healthy", "known_fault", "unknown")
    assert set(a["diagnosis"]["rule_comparison"]) == {"A", "B", "C", "D"}
    if not GOLDEN.exists():
        pytest.fail("no golden reference; run scripts/16_update_golden.py after checking the change")
    g = json.loads(GOLDEN.read_text())
    if g["manifest_sha256"] != a["provenance"]["manifest_sha256"]:
        pytest.fail(f"the golden reference belongs to manifest {g['manifest_sha256'][:12]}, the bundle is "
                    f"{a['provenance']['manifest_sha256'][:12]}; run scripts/16_update_golden.py after checking the change")
    assert a == g["outputs"][rec_id]


@needs_bundle
def test_repo_config_changes_do_not_change_diagnose(monkeypatch):
    import ml.config
    import ml.inference as inf
    base = stable(diagnose(GOLDEN_RECS[1], with_xai=False))
    d = ml.config.Config().model_dump(mode="json")
    d["preprocess"]["mic"]["burst_z"] = 5.0
    d["preprocess"]["bandstop"]["enabled"] = False
    d["cnn"]["dropout"] = 0.5
    changed = ml.config.Config.model_validate(d)
    monkeypatch.setattr(ml.config, "load_config", lambda *a, **k: changed)
    monkeypatch.setattr(inf, "load_config", lambda *a, **k: changed)
    inf._refusal_config.cache_clear()
    try:
        assert stable(diagnose(GOLDEN_RECS[1], with_xai=False)) == base
    finally:
        inf._refusal_config.cache_clear()


@needs_bundle
def test_bundle_with_a_tampered_file_is_refused(tmp_path):
    from ml.bundle import BundleIntegrityError, load_bundle
    d = tmp_path / "bundle"
    shutil.copytree(BUNDLE, d)
    t = json.loads((d / "thresholds.json").read_text())
    t["tau_p"] = 0.5
    (d / "thresholds.json").write_text(json.dumps(t))
    with pytest.raises(BundleIntegrityError, match="sha256 mismatch"):
        load_bundle(d)
