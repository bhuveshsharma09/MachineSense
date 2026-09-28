"""Stage App-2a.3: re-running a stored case (uploads on their stored array), the request / input ids echoed with every
diagnosis so the UI can drop stale responses, and the case-store migration."""
import io
import os
from pathlib import Path

import numpy as np

from ml.paths import data_dir

REC = "normal__18.432"


def _upload(client, rec=REC, request_id="req-1", upload_id="up-1"):
    x = np.load(data_dir() / "raw_npy" / f"{rec}.npy")
    buf = io.StringIO()
    np.savetxt(buf, x, fmt="%.9g", delimiter=",")
    return client.post("/api/diagnose", files={"file": ("u.csv", buf.getvalue().encode(), "text/csv")},
                       data={"upload_id": upload_id}, headers={"X-Request-Id": request_id})


def test_diagnose_echoes_request_and_input_ids(client):
    r = client.post("/api/diagnose", json={"rec_id": REC}, headers={"X-Request-Id": "abc"})
    assert r.headers["x-request-id"] == "abc" and r.headers["x-input-id"] == REC
    u = _upload(client, request_id="def", upload_id="up-9")
    assert u.headers["x-request-id"] == "def" and u.headers["x-input-id"] == "up-9"


def test_rerun_of_an_upload_gives_identical_scores_and_decision(client):
    orig = _upload(client).json()
    meta = client.get(f"/api/cases/{orig['case_id']}/meta").json()
    assert meta["can_rerun"] and meta["source"] == "upload" and meta["rerun_note"] is None
    r = client.post(f"/api/cases/{orig['case_id']}/rerun", headers={"X-Request-Id": "rr"})
    assert r.status_code == 200 and r.headers["x-request-id"] == "rr"
    new = r.json()
    assert new["case_id"] != orig["case_id"] and new["source"] == "upload"
    assert new["diagnosis"]["scores"] == orig["diagnosis"]["scores"]
    assert new["diagnosis"]["decision"] == orig["diagnosis"]["decision"]
    assert new["diagnosis"]["display_label"] == orig["diagnosis"]["display_label"]
    assert client.get(f"/api/cases/{new['case_id']}/meta").json()["rerun_of"] == orig["case_id"]
    assert client.get(f"/api/cases/{new['case_id']}/activity").json()["rerun_of"] == orig["case_id"]


def test_rerun_of_a_dataset_case(client):
    orig = client.post("/api/diagnose", json={"rec_id": REC}).json()
    new = client.post(f"/api/cases/{orig['case_id']}/rerun").json()
    assert new["input"]["rec_id"] == REC and new["diagnosis"]["scores"] == orig["diagnosis"]["scores"]


def test_rerun_with_a_missing_stored_array_is_refused_clearly(client):
    orig = _upload(client).json()
    path = client.app.state.engine.store.get(orig["case_id"])["array_path"]
    os.remove(path)
    meta = client.get(f"/api/cases/{orig['case_id']}/meta").json()
    assert not meta["can_rerun"] and "re-upload the file to run again" in meta["rerun_note"]
    r = client.post(f"/api/cases/{orig['case_id']}/rerun")
    assert r.status_code == 409 and "re-upload the file to run again" in r.json()["detail"]
    assert client.post("/api/cases/no-such-case/rerun").status_code == 404
    assert client.get("/api/cases/no-such-case/meta").status_code == 404


def test_rerun_still_applies_the_upload_guard(reserved_raw_allowed, client, reserved_ids):
    """A stored array that matches a reserved recording (planted directly; never uploaded) is refused on rerun."""
    orig = _upload(client).json()
    path = Path(client.app.state.engine.store.get(orig["case_id"])["array_path"])
    np.save(path, np.load(data_dir() / "raw_npy" / f"{reserved_ids[0]}.npy").astype(np.float32))
    r = client.post(f"/api/cases/{orig['case_id']}/rerun")
    assert r.status_code == 403 and "reserved" in r.json()["detail"]


def test_store_migrates_an_old_database(tmp_path):
    import json
    import sqlite3

    from backend.store import CaseStore
    db = tmp_path / "old.db"
    with sqlite3.connect(db) as c:
        c.execute("CREATE TABLE cases (case_id TEXT PRIMARY KEY, created_at TEXT NOT NULL, source TEXT NOT NULL, "
                  "rec_id TEXT, record TEXT NOT NULL, activity TEXT NOT NULL, array_path TEXT)")
        c.execute("INSERT INTO cases VALUES ('a', 't', 'dataset', 'r', ?, '[]', NULL)", (json.dumps({"x": 1}),))
    s = CaseStore(db)
    assert s.get("a")["rerun_of"] is None and s.recent()[0]["case_id"] == "a"
