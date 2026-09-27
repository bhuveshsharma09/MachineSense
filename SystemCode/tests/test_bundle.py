"""Bundle v1 integrity: every file is hash-checked; a mismatch, a missing file or an extra file is refused."""
import json

import pytest

from ml.bundle import (BundleIntegrityError, Manifest, config_hash, file_entries, now_iso, sha256_file, verify_bundle,
                       write_manifest)
from ml.config import Config

SPEC = {"a": ("a.safetensors", "classifier", "w"), "thresholds": ("thresholds.json", "thresholds", "t"),
        "inference_config": ("inference_config.json", "config", "c")}


def _fake(tmp_path):
    (tmp_path / "a.safetensors").write_bytes(b"weights")
    (tmp_path / "thresholds.json").write_text(json.dumps({"tau_p": 0.9}))
    (tmp_path / "inference_config.json").write_text(json.dumps(Config().model_dump(mode="json")))
    m = Manifest(created_at=now_iso(), git_sha="x", unrelated_worktree_changes=[], inference_config_sha256=config_hash(Config()),
                 preprocess_config_sha256="p", split_hash="s", fs_hz=50000, channels=["tacho"], class10=["normal"],
                 class6=["normal"], representation={}, thresholds={}, rule={}, data={}, test_eligibility={},
                 limitations=[], files=file_entries(tmp_path, SPEC))
    write_manifest(tmp_path, m)
    return tmp_path


def test_tampered_packaged_config_is_refused_even_with_a_matching_file_hash(tmp_path):
    d = _fake(tmp_path)
    c = json.loads((d / "inference_config.json").read_text())
    c["preprocess"]["mic"]["burst_z"] = 5.0  # a changed setting, with the manifest's file hash updated to match
    (d / "inference_config.json").write_text(json.dumps(c))
    m = json.loads((d / "manifest.json").read_text())
    for e in m["files"]:
        if e["file"] == "inference_config.json":
            e["sha256"] = sha256_file(d / "inference_config.json")
    (d / "manifest.json").write_text(json.dumps(m))
    with pytest.raises(BundleIntegrityError, match="config hash"):
        verify_bundle(d)


def test_intact_bundle_verifies(tmp_path):
    assert len(verify_bundle(_fake(tmp_path)).files) == 3


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
