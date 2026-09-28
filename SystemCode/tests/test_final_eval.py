"""Stage F0: the F1 script refuses without a committed §12, the dry run never touches a reserved rec_id, the final models
verify on load, and ml.final scores equal ml.inference.score_recording on the bundle."""
import importlib.util
import json
import shutil

import pandas as pd
import pytest

from ml.config import load_config
from ml.final import final_dir, verify_final
from ml.paths import DEFAULT_CONFIG, SYSTEM_CODE_DIR, require_bundle
from ml.splits import FAULT_CLASSES, development_pool, folder_proxy

spec = importlib.util.spec_from_file_location("final_eval", SYSTEM_CODE_DIR / "scripts" / "30_final_eval.py")
F1 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(F1)


def _need_final():
    if not (final_dir() / "manifest.json").exists():
        if require_bundle():
            pytest.fail("MACHINESENSE_REQUIRE_BUNDLE=1: artifacts/final/ is missing", pytrace=False)
        pytest.skip("artifacts/final/ not built")


def _meta_pool():
    meta = pd.read_csv(SYSTEM_CODE_DIR / "metadata.csv")
    meta["folder"] = folder_proxy(meta)
    dc = load_config(DEFAULT_CONFIG).development
    return meta, development_pool(meta, dc.allowed_closed, dc.reserved_columns)


def _reserved(meta):
    return set(meta.rec_id[(meta.split_closed == "test") | (meta.split_severity == "test") | (meta.split_speed_block == "test")])


def test_refuses_without_committed_section_12(monkeypatch, tmp_path):
    monkeypatch.setattr(F1, "git", lambda *a: "# CLAUDE.md without the protocol\n" if a[:1] == ("show",) else "")
    with pytest.raises(F1.Refused, match="§12"):
        F1.run(True, 1, ["imbalance"], 5, out=tmp_path / "x")
    assert not (tmp_path / "x").exists()


def test_refuses_when_section_12_is_modified(monkeypatch, tmp_path):
    fake = {"show": F1.PROTOCOL_HEADING, "diff": "CLAUDE.md\n"}
    monkeypatch.setattr(F1, "git", lambda *a: fake.get(a[0], ""))
    with pytest.raises(F1.Refused, match="uncommitted"):
        F1.run(True, 1, ["imbalance"], 5, out=tmp_path / "x")


def test_real_run_refuses_uncommitted_code_and_an_existing_output(monkeypatch, tmp_path):
    monkeypatch.setattr(F1, "committed_protocol", lambda: "abc")
    monkeypatch.setattr(F1, "git", lambda *a: "" if a[0] == "ls-files" else "")
    with pytest.raises(F1.Refused, match="not committed"):
        F1.preconditions(False, tmp_path / "final")
    monkeypatch.setattr(F1, "committed_code", lambda: None)
    (tmp_path / "final").mkdir()
    with pytest.raises(F1.Refused, match="runs once"):
        F1.preconditions(False, tmp_path / "final")


def test_limit_is_refused_outside_the_dry_run():
    meta, pool = _meta_pool()
    with pytest.raises(F1.Refused):
        F1.build_sets(meta, pool, False, ["imbalance"], limit=3)


def test_dry_run_sets_are_development_recordings_only():
    meta, pool = _meta_pool()
    sets = F1.build_sets(meta, pool, True, list(FAULT_CLASSES))
    ids = set().union(*sets["bundle"].values(), sets["r18"], *[s for d in sets["loco"].values() for s in d.values()])
    assert ids and ids <= set(pool.rec_id) and not ids & _reserved(meta)
    for c, d in sets["loco"].items():
        assert all(meta.set_index("rec_id").class10[r] == c for r in d["unknown_primary"])
        assert not any(meta.set_index("rec_id").class10[r] == c for r in d["known"])


def test_dry_run_loader_refuses_a_reserved_recording():
    meta, pool = _meta_pool()
    loader = F1.Loader(meta, set(pool.rec_id), dry_run=True)
    reserved_id = min(_reserved(meta))  # never read: the loader must refuse before opening the file
    with pytest.raises(F1.ReservedAccess):
        loader.load(reserved_id)
    assert loader.touched == []


@pytest.mark.needs_bundle("manifest.json")
def test_dry_run_end_to_end_touches_development_recordings_only(tmp_path):
    _need_final()
    meta, pool = _meta_pool()
    out = F1.run(True, 2, ["imbalance"], 20, out=tmp_path / "dry")
    assert out["loader_touched"] and set(out["loader_touched"]) <= set(pool.rec_id)
    assert not set(out["loader_touched"]) & _reserved(meta)
    for name in ("provenance", "q1", "q2", "q3", "headline"):
        d = json.loads((tmp_path / "dry" / f"{name}.json").read_text())
        assert d["label"] == F1.DRY_LABEL
    prov = json.loads((tmp_path / "dry" / "provenance.json").read_text())
    assert prov["dry_run"] and len(prov["script_sha256"]) == 64 and prov["split_hash"]
    assert "DRY RUN" in (tmp_path / "dry" / "README.md").read_text()


@pytest.mark.needs_bundle("manifest.json")
def test_final_scoring_equals_score_recording_on_the_bundle():
    from ml.final import recording_inputs, score_cnn
    from ml.inference import get_bundle, load_recording, score_recording
    from ml.labels import CLASS10
    b = get_bundle()
    x, fr = load_recording("normal__18.432")
    s = score_recording(x, fr, b)
    f = score_cnn(recording_inputs(x, fr, b.config), b.cnn, b.normalisation["cnn6"], CLASS10, b.mahalanobis, b.ae,
                  b.normalisation["ae6"], b.model_config["ae"], b.b1s, b.model_config["prob_floor"])
    assert all(s[k] == f[k] for k in ("msp", "maha", "ae", "b1s", "top"))


def test_final_models_verify_and_refuse_a_tampered_file(tmp_path):
    _need_final()
    m = verify_final()
    assert m["split_hash"] and m["git_sha"] and len(m["loco_classes"]) == 9
    d = tmp_path / "final"
    shutil.copytree(final_dir(), d)
    t = d / "loco" / "imbalance" / "thresholds.json"
    t.write_text(t.read_text().replace('"tau_p"', '"tau_p" ', 1))
    from ml.bundle import BundleIntegrityError
    with pytest.raises(BundleIntegrityError):
        verify_final(d)
