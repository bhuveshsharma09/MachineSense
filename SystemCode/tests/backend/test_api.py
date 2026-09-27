"""Stage App-1 backend: the API serves exactly ml.inference.diagnose(), verifies the bundle at startup, never lists or
accepts reserved recordings, rejects bad uploads clearly, and returns records that satisfy the schema."""
import io
import json
import shutil

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from backend.schemas import RUNTIME_ONLY_FIELDS, EvidenceRecord
from ml.paths import SYSTEM_CODE_DIR, data_dir

REC = "underhang__outer_race__0g__14.7456"


def _without_runtime(d: dict) -> dict:
    d = json.loads(json.dumps(d))
    for k in RUNTIME_ONLY_FIELDS:
        d.pop(k, None)
    return d


def test_api_record_equals_diagnose_output(client):
    """Runtime-only fields excluded: case_id, created_at, source (added by the API) and timing_ms (measured per call)."""
    from ml.inference import diagnose
    api = client.post("/api/diagnose", json={"rec_id": REC})
    assert api.status_code == 200
    engine = client.app.state.engine
    assert _without_runtime(api.json()) == _without_runtime(diagnose(REC, bundle=engine.bundle))
    assert api.json()["source"] == "dataset"


def test_upload_of_a_development_recording_matches_the_dataset_path(client):
    x = np.load(data_dir() / "raw_npy" / f"{REC}.npy")
    buf = io.StringIO()
    np.savetxt(buf, x, fmt="%.9g", delimiter=",")
    up = client.post("/api/diagnose", files={"file": ("rec.csv", buf.getvalue().encode(), "text/csv")})
    assert up.status_code == 200, up.text
    ds = client.post("/api/diagnose", json={"rec_id": REC}).json()
    u = up.json()
    assert u["source"] == "upload" and u["input"]["rec_id"] is None
    assert u["diagnosis"]["scores"] == ds["diagnosis"]["scores"]
    assert u["diagnosis"]["decision"] == ds["diagnosis"]["decision"]


def test_backend_refuses_to_start_with_a_hash_mismatched_bundle(bundle_path, tmp_path):
    from fastapi.testclient import TestClient

    from backend.app import create_app
    from ml.bundle import BundleIntegrityError
    d = tmp_path / "bundle"
    shutil.copytree(bundle_path, d)
    t = json.loads((d / "thresholds.json").read_text())
    t["tau_p"] = 0.5
    (d / "thresholds.json").write_text(json.dumps(t))
    with pytest.raises(BundleIntegrityError):
        with TestClient(create_app(d)):
            pass


def test_reserved_recordings_are_neither_listed_nor_accepted(client):
    meta = pd.read_csv(SYSTEM_CODE_DIR / "metadata.csv")
    reserved = meta[(meta.split_closed == "test") | (meta.split_severity == "test") | (meta.split_speed_block == "test")]
    listed = {r["rec_id"] for r in client.get("/api/recordings").json()["recordings"]}
    assert len(listed) == 804 and not listed & set(reserved.rec_id)
    for col in ("split_closed", "split_severity", "split_speed_block"):
        rid = meta.loc[meta[col] == "test", "rec_id"].iloc[0]
        r = client.post("/api/diagnose", json={"rec_id": rid})
        assert r.status_code == 403 and "reserved" in r.json()["detail"]


@pytest.mark.parametrize("payload, needle", [
    (b"", "empty"),
    (b"1,2,3\n4,5,6\n", "expected 8 columns"),
    (b"1,2,3,4,5,6,7,8\n" * 10, "expected 250000 rows"),
    (b"a,b,c,d,e,f,g,h\nx,y,z,w,v,u,t,s\n", "could not be read as a numeric CSV"),
])
def test_bad_uploads_are_rejected_with_a_clear_error(client, payload, needle):
    r = client.post("/api/diagnose", files={"file": ("bad.csv", payload, "text/csv")})
    assert r.status_code == 422 and needle in r.json()["detail"]


def test_bad_requests(client):
    assert client.post("/api/diagnose", json={}).status_code == 422
    assert client.post("/api/diagnose", json={"rec_id": "no_such_recording"}).status_code == 404
    assert client.get("/api/cases/nope").status_code == 404


def test_full_record_validates_against_the_schema(client):
    rec = client.post("/api/diagnose", json={"rec_id": "normal__18.432"}).json()
    m = EvidenceRecord.model_validate(rec)
    assert m.to_json() == rec
    for bad in ({**rec, "extra": 1}, {**rec, "diagnosis": {**rec["diagnosis"], "unexpected": 1}},
                {k: v for k, v in rec.items() if k != "limitations"}):
        with pytest.raises(ValidationError):
            EvidenceRecord.model_validate(bad)


def test_cases_and_the_fixed_report(client):
    rec = client.post("/api/diagnose", json={"rec_id": "normal__18.432"}).json()
    assert client.get(f"/api/cases/{rec['case_id']}").json() == rec
    a = client.get(f"/api/cases/{rec['case_id']}/report").json()
    b = client.get(f"/api/cases/{rec['case_id']}/report").json()
    assert a == b and "no language model" in a["generated_by"]
    assert rec["diagnosis"]["display_label"] in a["report"] and "Secondary logistic-regression evidence" in a["report"]


def test_health_reports_the_verified_bundle(client, bundle_path):
    h = client.get("/api/health").json()
    assert h["bundle_verified"] is True and h["status"] == "ok"
    assert h["manifest_sha256"] == client.app.state.engine.bundle.manifest_sha256
    assert h["test_eligibility"]["q3_leave_one_class_out_unfamiliar_fault"]["eligible"] is False
