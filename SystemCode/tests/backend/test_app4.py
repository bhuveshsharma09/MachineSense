"""Stage App-4.1: the read-only pipeline trace. Every number in it comes from the saved record, the stored activity log,
the packaged config or the bundle manifest (or is a count recomputed from the record); timings are copies of stored
activity steps; statuses match the case; the edges are the real dependencies of ml.inference.diagnose; reading the trace
never changes the saved record."""
import io
import numbers

import numpy as np
import pytest

from ml.paths import data_dir

REC = "underhang__outer_race__0g__14.7456"      # dataset case, no signal-quality warnings expected to matter
WARN = "imbalance__25g__15.1552"                # signal-quality warnings (showcase signal-quality case)


@pytest.fixture(scope="module")
def dataset_case(client):
    return client.post("/api/diagnose", json={"rec_id": REC}).json()


@pytest.fixture(scope="module")
def warn_case(client):
    return client.post("/api/diagnose", json={"rec_id": WARN}).json()


@pytest.fixture(scope="module")
def upload_case(client):
    x = np.load(data_dir() / "raw_npy" / f"{REC}.npy")
    buf = io.StringIO()
    np.savetxt(buf, x, fmt="%.9g", delimiter=",")
    r = client.post("/api/diagnose", files={"file": ("rec.csv", buf.getvalue().encode(), "text/csv")})
    assert r.status_code == 200, r.text
    return r.json()


def trace(client, case):
    r = client.get(f"/api/cases/{case['case_id']}/trace")
    assert r.status_code == 200, r.text
    return r.json()


def blocks(t):
    def walk(bs):
        for b in bs:
            yield b
            yield from walk(b.get("children", []))
    return {b["id"]: b for b in walk(t["blocks"])}


def independent_sources(client, case_id):
    """Every source fetched through its own endpoint or object, not through the trace code."""
    eng = client.app.state.engine
    e = client.get(f"/api/cases/{case_id}/explanation")
    return {"record": client.get(f"/api/cases/{case_id}").json(),
            "activity": client.get(f"/api/cases/{case_id}/activity").json()["steps"],
            "config": eng.bundle.config.model_dump(mode="json"), "manifest": eng.bundle.manifest.model_dump(mode="json"),
            "explanation": e.json() if e.status_code == 200 else {},
            "store": {"rerun_of": client.get(f"/api/cases/{case_id}/meta").json()["rerun_of"]}}


def check_every_value(t, src):
    """Walk the trace: every sourced value equals its source; every count is recomputed; no bare number exists except
    timing_ms. Returns (number of sourced values checked, timing values)."""
    from backend.checks import checks
    from backend.trace import resolve

    n, timings = 0, []

    def walk(x, key=None):
        nonlocal n
        if isinstance(x, dict):
            if "from" in x and "value" in x:
                src, path = x["from"].split(":", 1)
                if src == "derived":
                    if path.startswith("len(record:"):
                        assert x["value"] == len(resolve(path[4:-1], src_all)), x
                    elif path.startswith("backend.checks.checks(record)."):
                        k, field = path.split(".")[-2:]
                        assert x["value"] == next(c for c in checks(src_all["record"]) if c["key"] == k)[field], x
                    else:
                        assert not isinstance(x["value"], numbers.Number) or isinstance(x["value"], bool), x
                else:
                    assert x["value"] == resolve(x["from"], src_all), x
                n += 1
                return
            if set(x) == {"value", "reason"}:
                assert x["value"] is None and x["reason"]
                return
            for k, v in x.items():
                walk(v, k)
        elif isinstance(x, list):
            for v in x:
                walk(v, key)
        elif isinstance(x, numbers.Number) and not isinstance(x, bool):
            assert key == "timing_ms", f"bare number {x!r} under {key!r}"
            timings.append(x)

    src_all = src
    walk(t)
    return n, timings


def test_every_number_comes_from_the_saved_case_or_the_bundle(client, dataset_case):
    t = trace(client, dataset_case)
    src = independent_sources(client, dataset_case["case_id"])
    n, _ = check_every_value(t, src)
    assert n > 80
    # the record's own numbers appear unchanged (spot checks against GET /api/cases/{id})
    rec = src["record"]
    b = blocks(t)
    assert b["cnn6"]["output"]["msp"]["value"] == rec["diagnosis"]["scores"]["msp"]
    assert b["ae6"]["technical"]["ae_channel_errors"]["value"] == rec["diagnosis"]["scores"]["ae_channel_errors"]
    assert "ae_channel_errors" not in b["ae6"]["output"] and "not usable as fault location" in b["ae6"]["technical"]["ae_channel_errors_caveat"]


def test_timings_are_stored_activity_steps_never_invented(client, dataset_case, upload_case):
    for case in (dataset_case, upload_case):
        t = trace(client, case)
        steps = {s["step"]: s["ms"] for s in client.get(f"/api/cases/{case['case_id']}/activity").json()["steps"]}
        _, timings = check_every_value(t, independent_sources(client, case["case_id"]))
        assert set(timings) <= set(steps.values())
        for b in blocks(t).values():
            if b["timing_ms"] is not None:
                named = [p.split(":", 1)[1] for p in b["provenance"] if p.startswith("activity:")]
                assert any(steps[s] == b["timing_ms"] for s in named), b["id"]
        bs = blocks(t)
        assert bs["models"]["timing_ms"] == steps["models"] and bs["decision"]["timing_ms"] == steps["decision_rule"]
        for bid in ("input", "validation", "preparation", "representations", "cnn6", "evidence_record", "explanation"):
            assert bs[bid]["timing_ms"] is None


def test_dataset_case_statuses_and_validation(client, dataset_case):
    t = trace(client, dataset_case)
    b = blocks(t)
    assert t["trace_version"] == "1.0" and t["case_id"] == dataset_case["case_id"] and len(t["blocks"]) == 10
    assert [x["id"] for x in t["blocks"]] == ["input", "validation", "signal_quality", "preparation", "representations",
                                              "models", "decision", "evidence_record", "explanation", "presentation"]
    applied = {c["function"] + "|" + c["check"]: c["applied"]["value"] for c in b["validation"]["output"]["checks"]}
    assert applied["ml.inference.check_rec_id|rec_id is known and not in any reserved test"] is True
    assert not any(v for k, v in applied.items() if k.startswith("backend.engine.parse_upload"))
    assert b["signal_quality"]["status"] == ("warning" if dataset_case["signal_quality"]["warnings"] else "completed")
    assert b["class6"]["technical"]["separate_classifier"] is False and "DERIVED" in b["class6"]["operation"]
    assert b["lr"]["output"]["feeds_decision"] is False and "does not receive it" in b["lr"]["operation"]
    assert "does not mean a new fault was found" in b["decision"]["output"]["unknown_meaning"]
    assert [c["check"] for c in b["decision"]["output"]["checks"]] == ["msp >= tau_p", "maha <= tau_d", "ae <= tau_e",
                                                                       "b1s <= tau_b"]
    assert "applied to the whole recording before windowing" in b["preparation"]["operation"]
    assert t["bundle_matches_record"]["value"] is True


def test_case_without_explanation_then_with_the_template(client, dataset_case):
    b = blocks(trace(client, dataset_case))
    assert b["explanation"]["status"] == "not_run" and b["explanation"]["tag"] == "Missing"
    assert b["explanation"]["output"]["generator"] == {"value": None, "reason": "no explanation stored for this case"}
    views = {v["view"]: v for v in b["presentation"]["output"]["views"]}
    assert views["explanation"]["available"]["value"] is False
    client.post(f"/api/cases/{dataset_case['case_id']}/explanation", json={})  # no model reachable in tests → template
    t = trace(client, dataset_case)
    e = blocks(t)["explanation"]
    assert e["status"] == "warning" and e["output"]["template_fallback_used"]["value"] is True
    assert e["output"]["source"]["value"] == "template" and "unreachable" in e["output"]["fallback_reason"]["value"]
    check_every_value(t, independent_sources(client, dataset_case["case_id"]))


def test_signal_quality_warning_case(client, warn_case):
    assert warn_case["signal_quality"]["warnings"]
    t = trace(client, warn_case)
    b = blocks(t)
    assert b["signal_quality"]["status"] == "warning"
    assert b["signal_quality"]["output"]["warnings"]["value"] == warn_case["signal_quality"]["warnings"]
    check_every_value(t, independent_sources(client, warn_case["case_id"]))


def test_upload_case_and_its_rerun(client, upload_case):
    t = trace(client, upload_case)
    b = blocks(t)
    checks = {c["check"]: c for c in b["validation"]["output"]["checks"]}
    for name in ("column count", "row count (5 s at 50 kHz)", "all values finite",
                 "reserved-recording fingerprint guard (best effort)", "array shape", "shaft speed measurable"):
        assert checks[name]["applied"]["value"] is True and checks[name]["outcome"]["value"] == "passed"
    assert checks["rec_id is known and not in any reserved test"]["applied"]["value"] is False
    assert b["input"]["output"]["rec_id"]["value"] is None and b["input"]["output"]["source"]["value"] == "upload"
    check_every_value(t, independent_sources(client, upload_case["case_id"]))
    rr = client.post(f"/api/cases/{upload_case['case_id']}/rerun").json()
    t2 = trace(client, rr)
    c2 = {c["check"]: c for c in blocks(t2)["validation"]["output"]["checks"]}
    assert c2["column count"]["applied"]["value"] is False and upload_case["case_id"] in c2["column count"]["not_applied_reason"]
    assert c2["reserved-recording fingerprint guard (best effort)"]["applied"]["value"] is True
    assert blocks(t2)["presentation"]["output"]["rerun_of"]["value"] == upload_case["case_id"]


def test_ports_and_edges_are_the_real_dependencies(client, dataset_case):
    t = trace(client, dataset_case)
    b = blocks(t)
    ports = {f"{x['id']}.{p['id']}" for x in b.values() for p in x["ports_in"] + x["ports_out"]}
    for x in b.values():
        for p in x["ports_in"] + x["ports_out"]:
            assert 2 <= len(p["short"]) <= 4 and p["short"].isupper(), p  # e.g. "RAW", "R1", "P10"
    edges = {(e["from"], e["to"]) for e in t["edges"]}
    for e in t["edges"]:
        assert e["from"] in ports and e["to"] in ports and e["kind"] in ("signal", "model", "decision", "explanation")
        assert e["tag"] in ("Measured", "Model", "Guidance")
    required = [("physics_features.features", "lr.features"), ("physics_features.features", "b1s.features"),
                ("order_spectrogram.spectrogram", "cnn6.spectrogram"), ("order_spectrogram.spectrogram", "ae6.spectrogram"),
                ("cnn6.embedding", "mahalanobis.embedding"), ("cnn6.probs", "decision.top_msp"),
                ("mahalanobis.distance", "decision.maha"), ("ae6.error", "decision.ae"), ("b1s.score", "decision.b1s")]
    assert set(required) <= edges
    from_lr = [e for e in t["edges"] if e["from"].startswith("lr.")]
    assert [e["to"] for e in from_lr] == ["evidence_record.secondary"] and from_lr[0]["kind"] == "model"
    from_sq = [e["to"] for e in t["edges"] if e["from"].startswith("signal_quality.")]
    assert from_sq == ["evidence_record.quality"]
    assert not any(e["to"].startswith("decision.") and not e["from"].split(".")[0] in ("cnn6", "mahalanobis", "ae6", "b1s")
                   for e in t["edges"])
    # the decision function's actual inputs are exactly these four models (ml.arbitrator.decide signature)
    import inspect

    from ml.arbitrator import decide
    assert list(inspect.signature(decide).parameters) == ["variant", "top_class", "msp", "maha", "ae_error", "b1s",
                                                          "thresholds"]


def test_unknown_case_is_404(client):
    assert client.get("/api/cases/does-not-exist/trace").status_code == 404


def test_reading_the_trace_never_changes_the_saved_record(client, warn_case):
    cid = warn_case["case_id"]
    before = client.get(f"/api/cases/{cid}").json()
    acts = client.get(f"/api/cases/{cid}/activity").json()
    for _ in range(2):
        trace(client, warn_case)
    assert client.get(f"/api/cases/{cid}").json() == before == warn_case
    assert client.get(f"/api/cases/{cid}/activity").json() == acts
