"""Bundle v1 integrity: every file is hash-checked; a mismatch, a missing file or an extra file is refused."""
import json

import pytest

from ml.bundle import BundleIntegrityError, Manifest, file_entries, now_iso, verify_bundle, write_manifest


def _fake(tmp_path):
    (tmp_path / "a.safetensors").write_bytes(b"weights")
    (tmp_path / "thresholds.json").write_text(json.dumps({"tau_p": 0.9}))
    m = Manifest(created_at=now_iso(), git_sha="x", preprocess_config_sha256="p", split_hash="s", fs_hz=50000,
                 channels=["tacho"], class10=["normal"], class6=["normal"], representation={}, thresholds={}, rule={},
                 data={}, limitations=[],
                 files=file_entries(tmp_path, {"a": ("a.safetensors", "classifier", "w"),
                                               "thresholds": ("thresholds.json", "thresholds", "t")}))
    write_manifest(tmp_path, m)
    return tmp_path


def test_intact_bundle_verifies(tmp_path):
    assert len(verify_bundle(_fake(tmp_path)).files) == 2


def test_hash_mismatch_is_refused(tmp_path):
    d = _fake(tmp_path)
    (d / "a.safetensors").write_bytes(b"weightz")
    with pytest.raises(BundleIntegrityError, match="sha256 mismatch"):
        verify_bundle(d)


def test_missing_and_extra_files_are_refused(tmp_path):
    d = _fake(tmp_path)
    (d / "extra.joblib").write_bytes(b"?")
    with pytest.raises(BundleIntegrityError, match="not in the manifest"):
        verify_bundle(d)
    (d / "extra.joblib").unlink()
    (d / "thresholds.json").unlink()
    with pytest.raises(BundleIntegrityError, match="missing"):
        verify_bundle(d)


def test_wrong_schema_is_refused(tmp_path):
    d = _fake(tmp_path)
    m = json.loads((d / "manifest.json").read_text())
    m["schema_version"] = "0.1"
    (d / "manifest.json").write_text(json.dumps(m))
    with pytest.raises(BundleIntegrityError, match="schema"):
        verify_bundle(d)
