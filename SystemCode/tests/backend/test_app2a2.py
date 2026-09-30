"""Stage App-2a.2 presentation fixes: check relations and fault wording (1), margins and the narrow-margin / disagreement
rule (2), score formatting (3), the typical-healthy-variation band and the change marker (4), and /api/health paths (11)."""
import copy
import json
import os
from pathlib import Path

import pandas as pd
import pytest

from backend.checks import NARROW_REL, assessment, checks, fmt_score
from ml.paths import SYSTEM_CODE_DIR

FAULT = "underhang__outer_race__0g__14.7456"   # known fault; AE and B1s above their healthy limits
CAUTION = "normal__20.2752"                      # healthy with caution: B1s +0.063 % above τ_b
TIE = "overhang__cage_fault__6g__13.1072"        # Mahalanobis distance exactly at τ_d (a threshold-defining recording)
VARIANTS = "normal__27.4432"                     # rule D disagrees because B1s is clearly above τ_b


@pytest.fixture(scope="module")
def rec(client):
    cache = {}

    def get(rid):
        if rid not in cache:
            r = client.post("/api/diagnose", json={"rec_id": rid}).json()
            cache[rid] = {"record": r, "report": client.get(f"/api/cases/{r['case_id']}/report").json()}
        return cache[rid]
    return get


def _by(cs):
    return {c["key"]: c for c in cs}


# 1 ---------------------------------------------------------------------------------------------------------------
def test_relations_and_fault_wording(rec):
    c = _by(checks(rec(FAULT)["record"]))
    assert rec(FAULT)["record"]["diagnosis"]["decision"] == "known_fault"
    for k in ("ae", "b1s"):
        assert c[k]["relation"] == ">" and c[k]["status"] == "above healthy limit — consistent with a fault"
        assert c[k]["tone"] == "neutral"
    assert c["msp"]["relation"] == "≥" and c["maha"]["relation"] == "≤" and c["msp"]["status"] == "pass"
    h = _by(checks(rec(CAUTION)["record"]))
    assert h["b1s"]["relation"] == ">" and h["b1s"]["status"] == "above healthy limit — caution" and h["b1s"]["tone"] == "warn"
    assert h["ae"]["relation"] == "≤" and h["ae"]["status"] == "within healthy limit"


def test_abstention_wording_does_not_assert_a_fault(client):
    """App-4.4: on "further inspection needed" the healthy-baseline checks above their limit read "not consistent with
    healthy", never "consistent with a fault"; the report, sections and assessment all carry it."""
    rec = client.post("/api/diagnose", json={"rec_id": "imbalance__10g__13.9264"}).json()
    assert rec["diagnosis"]["decision"] == "unknown"
    c = _by(checks(rec))
    for k in ("ae", "b1s"):
        assert c[k]["relation"] == ">" and c[k]["status"] == "above healthy limit — not consistent with healthy"
        assert c[k]["tone"] == "neutral"
    rep = client.get(f"/api/cases/{rec['case_id']}/report").json()
    blob = rep["report"] + json.dumps(rep["sections"]) + json.dumps(rep["assessment"])
    assert "consistent with a fault" not in blob and "not consistent with healthy" in rep["report"]


# 2 ---------------------------------------------------------------------------------------------------------------
def test_margins_are_value_minus_threshold(rec):
    r = rec(CAUTION)["record"]
    s, t = r["diagnosis"]["scores"], r["diagnosis"]["thresholds"]
    c = _by(checks(r))
    assert c["b1s"]["margin_abs"] == s["b1s"] - t["tau_b"]
    assert c["b1s"]["margin_rel"] == pytest.approx((s["b1s"] - t["tau_b"]) / t["tau_b"])
    assert c["msp"]["margin_rel"] == pytest.approx((s["msp"] - t["tau_p"]) / (1 - t["tau_p"]))  # the 1 − MSP scale
    assert c["b1s"]["narrow"] and abs(c["b1s"]["margin_rel"]) <= NARROW_REL


def test_narrow_margin_replaces_checks_disagree(rec):
    a = rec(CAUTION)["report"]["assessment"]
    assert a["variants_disagree"] and a["variants_disagree_narrow"] and not a["disagree"]
    assert a["narrow_notes"] == [n for n in a["narrow_notes"] if n.startswith("Narrowly above the speed-aware baseline (B1s)")]
    t = rec(TIE)["report"]["assessment"]
    assert t["narrow_notes"] == ["Exactly at the familiarity (Mahalanobis distance) threshold (ties are accepted)."]
    assert not t["disagree"]


def test_checks_disagree_only_for_real_disagreement(rec):
    v = rec(VARIANTS)["report"]["assessment"]
    assert v["disagree"] and v["disagree_reasons"] == ["the alternative decision rules reach different verdicts"]
    r = copy.deepcopy(rec(FAULT)["record"])
    assert not assessment(r)["disagree"]
    r["diagnosis"]["lr_secondary"]["top_class"] = "imbalance"  # synthetic: CNN-6 ≠ LR
    assert assessment(r)["disagree_reasons"] == ["CNN-6 and the secondary logistic regression rank different classes first"]


def test_no_rule_codes_or_stages_in_user_text(rec):
    for rid in (FAULT, CAUTION, VARIANTS):
        rep = rec(rid)["report"]
        text = json.dumps(rep["sections"], ensure_ascii=False) + rep["report"] + json.dumps(rep["assessment"], ensure_ascii=False)
        for bad in ("Stage 3D", "App-2b", "B_with_caution", "rule A", "rule B", "rule C", "rule D"):
            assert bad not in text, (rid, bad)
        assert text.count("Decision rule: provisional research setting") >= 1
        s1 = json.dumps(rep["sections"][0], ensure_ascii=False)
        assert s1.count("provisional research setting") == 1  # stated once


# 3 ---------------------------------------------------------------------------------------------------------------
def test_score_format():
    assert fmt_score(0.999978) == "> 0.9999" and fmt_score(1.0) == "> 0.9999"
    assert fmt_score(0.00001) == "< 0.0001" and fmt_score(0.5) == "0.5000"
    assert fmt_score(0.99999992, 6) == "> 0.999999"


def test_report_has_no_certain_looking_scores(rec):
    rep = rec(FAULT)["report"]
    text = json.dumps(rep["sections"], ensure_ascii=False) + rep["report"]
    assert "1.0000" not in text and "100.0 %" not in text
    d = rec(FAULT)["record"]["diagnosis"]
    top = max(d["class10_probabilities"].values())
    assert f"{d['top_class'].replace('_', ' ')} {fmt_score(top)}" in text
    classes = [i["text"] for s in rep["sections"] for i in s["items"] if i["text"].startswith("Most probable classes")]
    assert classes and "%" not in classes[0]


# 4 ---------------------------------------------------------------------------------------------------------------
def test_variation_band_uses_training_normals_only(client):
    var = client.app.state.engine.variation
    meta = pd.read_csv(SYSTEM_CODE_DIR / "metadata.csv").set_index("rec_id")
    pool = {r["rec_id"]: r for r in client.app.state.engine.recordings()}
    assert var["recordings"] and all(pool[r]["used_for"] == "model training" and pool[r]["class10"] == "normal"
                                     for r in var["recordings"])
    assert all(meta.loc[r, "split_closed"] == "train" for r in var["recordings"])
    assert var["method"]["min_amplitude_v"] == 0.005 and "95th percentile" in var["method"]["statistic"]
    assert all(v["kurtosis"] is None for v in var["bands"].values() if "kurtosis" in v)


def test_min_amplitude_and_direction():
    from backend.evidence import compare
    vals = {"over_radial": {"rms": 0.03, "kurtosis": 0.1, "order_1x": 0.004, "order_2x": 0.02}}
    ref = {"over_radial": {"rms": 0.01, "kurtosis": -0.2, "order_1x": 0.0002, "order_2x": 0.04}}
    bands = {"over_radial": {"rms": 2.0, "order_1x": 3.0, "order_2x": 1.5}}
    m = compare(vals, ref, bands)[0]["metrics"]
    assert not m["order_1x"]["comparable"] and m["order_1x"]["factor"] is None   # both below 0.005 V
    assert not m["kurtosis"]["comparable"]                                       # excess kurtosis never ratioed
    assert m["rms"]["direction"] == "up" and m["rms"]["factor"] == pytest.approx(3.0) and m["rms"]["exceeds"]
    assert m["order_2x"]["direction"] == "down" and m["order_2x"]["factor"] == pytest.approx(2.0) and m["order_2x"]["exceeds"]


def test_no_change_marker_on_plain_healthy_verdicts(client, rec):
    """(a) No lime marker on any plain healthy verdict; (b) a DESCRIPTIVE count of the calibration normals that exceed
    at least one band (reported, not asserted to a value; the band is not tuned on them)."""
    pool = [r for r in client.app.state.engine.recordings() if r["class10"] == "normal"]
    cal = [r["rec_id"] for r in pool if r["used_for"] == "threshold calibration"]
    assert len(cal) == 5
    exceeding, plain = 0, 0
    for rid in cal + ["normal__13.1072", CAUTION]:
        r = rec(rid)["record"]
        ch = client.get(f"/api/cases/{r['case_id']}/channels").json()
        d = r["diagnosis"]
        if d["decision"] == "healthy" and not d["caution"]:
            plain += 1
            assert ch["marked"] == [] and ch["healthy_note"] == "within what the healthy checks accept"
        else:
            assert len(ch["marked"]) <= 2 and set(ch["marked"]) <= set(ch["exceeding"])
        if rid in cal:
            exceeding += bool(ch["exceeding"])
    assert plain >= 5
    print(f"\n[descriptive] {exceeding} of {len(cal)} calibration normals exceed at least one typical-healthy-variation band")


def test_caution_and_faults_keep_the_marker(client, rec):
    for rid in (CAUTION, FAULT):
        r = rec(rid)["record"]
        ch = client.get(f"/api/cases/{r['case_id']}/channels").json()
        assert ch["marked"] == ch["exceeding"][:2]


# 11 --------------------------------------------------------------------------------------------------------------
def test_health_reports_bundle_path_relative_to_systemcode(client, bundle_path):
    rel = client.get("/api/health").json()["bundle_dir"]
    assert not os.path.isabs(rel) and not rel.startswith("/") and "Users" not in rel
    assert (SYSTEM_CODE_DIR / rel).resolve() == Path(bundle_path).resolve()
