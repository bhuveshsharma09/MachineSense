"""Stage App-2a backend: the evidence endpoints return only ml-derived values; the healthy reference is always a
training normal of the development pool; the showcase uses development recordings only; the upload guard refuses a
reserved recording; cases persist across a restart; required-test mode fails without the bundle."""
import io
import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from ml.paths import SYSTEM_CODE_DIR, data_dir

REC = "underhang__outer_race__0g__14.7456"


def _pool(client):
    return pd.DataFrame(client.app.state.engine.recordings())


@pytest.fixture(scope="module")
def case(client):
    return client.post("/api/diagnose", json={"rec_id": REC}).json()


def test_channels_are_ml_feature_values_against_a_training_normal(client, case):
    from backend.evidence import channel_values
    from ml.features import recording_features
    from ml.inference import load_recording
    eng = client.app.state.engine
    ch = client.get(f"/api/cases/{case['case_id']}/channels").json()
    x, fr = load_recording(REC)
    pre = eng.bundle.config.preprocess
    mine = channel_values(recording_features(x, fr, pre))
    ref_id = ch["reference"]["rec_id"]
    xr, frr = load_recording(ref_id)
    ref = channel_values(recording_features(xr, frr, pre))
    for row in ch["channels"]:
        assert row["values"] == pytest.approx(mine[row["channel"]], rel=1e-12)
        assert row["reference_values"] == pytest.approx(ref[row["channel"]], rel=1e-12)
    pool = _pool(client).set_index("rec_id")
    assert ref_id != REC and pool.loc[ref_id, "class10"] == "normal"
    assert pool.loc[ref_id, "used_for"] == "model training"  # never a calibration (val) or reserved recording
    assert len(ch["marked"]) <= 2 and set(ch["marked"]) <= set(ch["exceeding"]) and "not a calibrated anomaly test" in ch["label"]
    metrics = {k for row in ch["channels"] for k in row["values"]}
    assert not any("ae" in k.split("_") or "error" in k for k in metrics)  # no autoencoder error as channel evidence


def test_reference_is_always_a_training_normal(client):
    from backend.evidence import healthy_reference
    from ml.inference import _metadata
    eng = client.app.state.engine
    meta = _metadata().reset_index()
    pool = _pool(client).set_index("rec_id")
    for speed in np.linspace(5, 70, 27):
        r = healthy_reference(meta, eng.bundle.config, float(speed))
        assert r.split_closed == "train" and r.class10 == "normal" and pool.loc[r.rec_id, "used_for"] == "model training"
    val_normals = pool[(pool.class10 == "normal") & (pool.used_for == "threshold calibration")]
    for rid, row in val_normals.iterrows():  # a calibration normal's own speed still picks a training normal
        assert healthy_reference(meta, eng.bundle.config, row.speed_hz, exclude=rid).split_closed == "train"


def test_signals_are_ml_order_spectra_and_cleaned_waveforms(client, case):
    from ml.features import comb_lines, order_spectrum
    from ml.inference import load_recording
    from ml.preprocess import clean_recording
    eng = client.app.state.engine
    pre = eng.bundle.config.preprocess
    s = client.get(f"/api/cases/{case['case_id']}/signals").json()
    x, _ = load_recording(REC)
    fr = case["signal_quality"]["shaft_speed_hz"]
    xc, _ = clean_recording(x, pre)
    orders = np.asarray(s["orders"])
    i = pre.channels.index("under_radial")
    ref = order_spectrum(np.asarray(xc[:, i], dtype=np.float64), pre.fs_hz, fr, orders)
    assert np.allclose(s["channels"]["under_radial"]["spectrum"], ref, rtol=0, atol=1e-8)
    seg = np.asarray(xc[pre.fs_hz:pre.fs_hz + pre.fs_hz // 2, i], dtype=np.float64).reshape(1000, -1)
    assert np.allclose(s["channels"]["under_radial"]["waveform"]["max"], seg.max(axis=1), atol=1e-6)
    for k, v in comb_lines(pre).items():
        assert s["cursors_orders"][k] == pytest.approx(list(v))
    assert s["shaft_speed_hz"] == fr


def test_evidence_views_for_an_upload_use_its_stored_array(client):
    x = np.load(data_dir() / "raw_npy" / "normal__18.432.npy")
    buf = io.StringIO()
    np.savetxt(buf, x, fmt="%.9g", delimiter=",")
    up = client.post("/api/diagnose", files={"file": ("u.csv", buf.getvalue().encode(), "text/csv")}).json()
    ch = client.get(f"/api/cases/{up['case_id']}/channels").json()
    assert ch["reference"]["used_for"] == "model training (healthy)"
    assert client.get(f"/api/cases/{up['case_id']}/signals").status_code == 200


def test_showcase_uses_development_recordings_only(client):
    from backend.showcase import select
    from backend.settings import showcase_file
    pool = set(_pool(client).rec_id)
    rows = [{"rec_id": r, "class10": "normal", "used_for": "model training", "speed_hz": 20.0, "decision": "healthy",
             "display_label": "healthy", "fault_class": None, "caution": None, "top_class": "normal",
             "lr_top": "normal", "msp": 1.0, "rule_comparison": {"A": "healthy", "B": "healthy"}, "warnings": [],
             "quality_warnings": []} for r in sorted(pool)[:3]]
    sel = select(rows)
    assert {s["rec_id"] for s in sel["showcase"]} <= pool
    assert "further inspection needed" in sel["states_without_a_development_recording"]  # missing, never invented
    if showcase_file().exists():
        committed = json.loads(showcase_file().read_text())
        assert committed["showcase"] and {s["rec_id"] for s in committed["showcase"]} <= pool
        assert client.get("/api/showcase").json() == committed


def test_upload_guard_refuses_a_reserved_recording_written_to_csv(client, reserved_ids):
    rid = reserved_ids[0]
    x = np.load(data_dir() / "raw_npy" / f"{rid}.npy")
    buf = io.StringIO()
    np.savetxt(buf, x, fmt="%.5g", delimiter=",")  # the dataset's own CSV precision
    r = client.post("/api/diagnose", files={"file": ("r.csv", buf.getvalue().encode(), "text/csv")})
    assert r.status_code == 403 and r.json()["detail"] == "this appears to be a reserved test recording"
    from backend.guard import ReservedGuard
    g = client.app.state.engine.guard
    assert g.match(x) == rid  # an exact copy of the array
    assert g.match(np.load(data_dir() / "raw_npy" / f"{REC}.npy")) is None  # development recordings pass


def test_cases_persist_across_an_app_restart(bundle_path, guard_path, tmp_path):
    from fastapi.testclient import TestClient

    from backend.app import create_app
    db = tmp_path / "cases.db"
    with TestClient(create_app(bundle_path, cases_path=db, guard_path=guard_path)) as c:
        rec = c.post("/api/diagnose", json={"rec_id": "normal__18.432"}).json()
        act = c.get(f"/api/cases/{rec['case_id']}/activity").json()
    with TestClient(create_app(bundle_path, cases_path=db, guard_path=guard_path)) as c:
        assert c.get(f"/api/cases/{rec['case_id']}").json() == rec
        assert c.get(f"/api/cases/{rec['case_id']}/activity").json() == act
        assert rec["case_id"] in {x["case_id"] for x in c.get("/api/cases").json()["cases"]}
    steps = [s["step"] for s in act["steps"]]
    assert steps == ["bundle_verified", "signal_quality", "models", "decision_rule", "physics", "explanations", "report"]
    assert all(s["kind"] == "deterministic step" and s["ms"] is not None for s in act["steps"])


def test_report_sections_are_tagged(client, case):
    rep = client.get(f"/api/cases/{case['case_id']}/report").json()
    assert [s["n"] for s in rep["sections"]] == ["01", "02", "03", "04"]
    items = [i for s in rep["sections"] for i in s["items"]]
    assert {i["tag"] for i in items} <= {"Measured", "Model", "Guidance", "Missing"}
    assert all("guidance library not yet connected" in i["text"] for i in items if i["tag"] == "Guidance")
    assert any("not evidence of accuracy" in i["text"] for i in items)


def test_required_mode_fails_when_the_bundle_is_missing(tmp_path):
    env = {**os.environ, "MACHINESENSE_REQUIRE_BUNDLE": "1", "MACHINESENSE_BUNDLE_DIR": str(tmp_path / "no_bundle")}
    cmd = [sys.executable, "-m", "pytest", "-q", "-rsE", "-p", "no:cacheprovider",
           "tests/backend/test_api.py::test_health_reports_the_verified_bundle"]
    p = subprocess.run(cmd, cwd=SYSTEM_CODE_DIR, env=env, capture_output=True, text=True, timeout=300)
    assert p.returncode != 0 and "ERROR" in p.stdout and "MACHINESENSE_REQUIRE_BUNDLE=1" in p.stdout
    assert "SKIPPED" not in p.stdout
    env.pop("MACHINESENSE_REQUIRE_BUNDLE")
    p = subprocess.run(cmd, cwd=SYSTEM_CODE_DIR, env=env, capture_output=True, text=True, timeout=300)
    assert p.returncode == 0 and "SKIPPED" in p.stdout
