"""Record real API responses as fixtures for the frontend smoke test (run from SystemCode/ with the venv and
MACHINESENSE_BUNDLE_DIR set):  PYTHONPATH=. python frontend/tests/make_fixtures.py [trace]
Signals are downsampled (every 8th order bin, 100 waveform buckets) to keep the fixtures small. With the argument
"trace", only the pipeline-trace fixture (App-4.2) is recorded and the other fixtures are left as they are."""
import json
import sys
from pathlib import Path

import httpx
from fastapi.testclient import TestClient

from backend.app import create_app

OUT = Path(__file__).resolve().parent / "fixtures"
CASES = {"known_fault": "overhang__cage_fault__0g__34.2016", "unknown": "imbalance__10g__13.9264",
         "caution": "normal__20.2752", "poor_quality": "imbalance__25g__15.1552", "healthy": "normal__13.1072",
         "disagree": "normal__27.4432", "narrow": "overhang__cage_fault__6g__13.1072", "plain_healthy": "normal__18.432"}


def trim(sig: dict) -> dict:
    sig["orders"] = sig["orders"][::8]
    for c in sig["channels"].values():
        c["spectrum"] = c["spectrum"][::8]
        w = c["waveform"]
        k = len(w["min"]) // 100
        w["min"], w["max"], w["dt_s"] = w["min"][::k], w["max"][::k], w["dt_s"] * k
    return sig


def _refuse(request):  # no language model while recording fixtures: explanations are the template
    raise httpx.ConnectError("no model while recording fixtures", request=request)


def record_trace(c) -> None:
    """GET /trace of a real case (the known-fault showcase recording, after its template explanation is stored)."""
    cid = c.post("/api/diagnose", json={"rec_id": CASES["known_fault"]}).json()["case_id"]
    c.post(f"/api/cases/{cid}/explanation")
    json.dump(c.get(f"/api/cases/{cid}/trace").json(), open(OUT / "trace.json", "w"), ensure_ascii=False)


ONLY_TRACE = sys.argv[1:] == ["trace"]
with TestClient(create_app(llm_transport=httpx.MockTransport(_refuse))) as c:
    OUT.mkdir(exist_ok=True)
    record_trace(c)
    if ONLY_TRACE:
        print("trace fixture written to", OUT / "trace.json")
        sys.exit(0)
    recs = c.get("/api/recordings").json()["recordings"]
    json.dump({"recordings": [r for r in recs if r["rec_id"] in CASES.values()], "health": c.get("/api/health").json(),
               "showcase": c.get("/api/showcase").json()}, open(OUT / "common.json", "w"))
    for name, rid in CASES.items():
        r = c.post("/api/diagnose", json={"rec_id": rid}).json()
        cid = r["case_id"]
        json.dump({"record": r, "activity": c.get(f"/api/cases/{cid}/activity").json(),
                   "report": c.get(f"/api/cases/{cid}/report").json(), "channels": c.get(f"/api/cases/{cid}/channels").json(),
                   "signals": trim(c.get(f"/api/cases/{cid}/signals").json()),
                   "explanation": c.post(f"/api/cases/{cid}/explanation").json()}, open(OUT / f"{name}.json", "w"))
    bad = c.post("/api/diagnose", files={"file": ("bad.csv", b"1,2,3\n4,5,6\n", "text/csv")})
    json.dump({"status": bad.status_code, "detail": bad.json()["detail"]}, open(OUT / "error_422.json", "w"))
print("fixtures written to", OUT)
