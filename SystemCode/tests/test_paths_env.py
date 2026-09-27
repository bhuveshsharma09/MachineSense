"""Item 0 (Stage ML-5): ml resolves its data and bundle directories from MACHINESENSE_DATA_DIR / MACHINESENSE_BUNDLE_DIR
with the previous paths as defaults, and bundle tests fail (not skip) under MACHINESENSE_REQUIRE_BUNDLE=1."""
import os
import subprocess
import sys

from ml.paths import SYSTEM_CODE_DIR, artifacts_dir, bundle_dir, data_dir


def test_defaults_are_unchanged(monkeypatch):
    monkeypatch.delenv("MACHINESENSE_DATA_DIR", raising=False)
    monkeypatch.delenv("MACHINESENSE_BUNDLE_DIR", raising=False)
    assert data_dir() == SYSTEM_CODE_DIR / "data"
    assert bundle_dir() == artifacts_dir() / "bundle"


def test_env_overrides(monkeypatch, tmp_path):
    monkeypatch.setenv("MACHINESENSE_DATA_DIR", str(tmp_path / "d"))
    monkeypatch.setenv("MACHINESENSE_BUNDLE_DIR", "some/relative/bundle")
    assert data_dir() == (tmp_path / "d").resolve()
    assert bundle_dir() == (SYSTEM_CODE_DIR / "some/relative/bundle").resolve()  # relative to SystemCode/, not the cwd


def test_required_mode_fails_and_plain_mode_skips(tmp_path):
    cmd = [sys.executable, "-m", "pytest", "-q", "-rsE", "-p", "no:cacheprovider",
           "tests/test_inference.py::test_manifest_records_per_task_eligibility"]
    env = {**os.environ, "MACHINESENSE_BUNDLE_DIR": str(tmp_path / "missing"), "MACHINESENSE_REQUIRE_BUNDLE": "1"}
    p = subprocess.run(cmd, cwd=SYSTEM_CODE_DIR, env=env, capture_output=True, text=True, timeout=300)
    assert p.returncode != 0 and "MACHINESENSE_REQUIRE_BUNDLE=1" in p.stdout and "SKIPPED" not in p.stdout
    env.pop("MACHINESENSE_REQUIRE_BUNDLE")
    p = subprocess.run(cmd, cwd=SYSTEM_CODE_DIR, env=env, capture_output=True, text=True, timeout=300)
    assert p.returncode == 0 and "SKIPPED" in p.stdout
