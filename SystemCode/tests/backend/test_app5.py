"""Stage App-5: section 04 of the report shows cited steps from the curated guidance library, selected by
ml.guidance.select; nothing is offered without a covering entry, and verification status stays visible."""
import copy

import pytest

from backend.report import build_sections

BEARING = "underhang__outer_race__0g__14.7456"
IMBALANCE = "imbalance__10g__19.2512"
UNKNOWN = "imbalance__10g__13.9264"  # a calibration recording the bundle rejects (MSP below τ_p)


@pytest.fixture(scope="module")
def reports(client):
    out = {}
    for rec_id in (BEARING, IMBALANCE, UNKNOWN):
        case = client.post("/api/diagnose", json={"rec_id": rec_id}).json()
        out[rec_id] = (case, client.get(f"/api/cases/{case['case_id']}/report").json())
    return out


def s4(rep):
    return next(s for s in rep["sections"] if s["n"] == "04")["items"]


def test_bearing_verdict_cites_the_defect_frequency_entry(reports):
    case, rep = reports[BEARING]
    assert case["diagnosis"]["decision"] == "known_fault" and case["diagnosis"]["fault_class"] == "underhang_outer"
    items = s4(rep)
    assert "bearing-defect-frequencies" in {i["cite"] for i in items}
    for i in items:
        assert i["tag"] == "Guidance" and f"source: {i['cite']}" in i["text"] and i["hover"]
    bdf = [i for i in items if i["cite"] == "bearing-defect-frequencies"]
    assert all("Citation not yet checked against the full text" in i["text"] for i in bdf)
    assert "Randall" in bdf[0]["hover"] and "https://doi.org/10.1016/j.ymssp.2010.07.017" in bdf[0]["hover"]


def test_unknown_verdict_puts_further_inspection_first(reports):
    case, rep = reports[UNKNOWN]
    assert case["diagnosis"]["decision"] == "unknown"
    cites = [i["cite"] for i in s4(rep)]
    assert cites[0] == "further-inspection-needed"
    assert not {"imbalance-1x", "bearing-defect-frequencies", "misalignment-2x-axial"} & set(cites)  # no class entry


def test_imbalance_steps_carry_the_ai_drafted_note(reports):
    case, rep = reports[IMBALANCE]
    assert case["diagnosis"]["fault_class"] == "imbalance"
    imb = [i for i in s4(rep) if i["cite"] == "imbalance-1x"]
    assert len(imb) == 5 and all("AI-drafted, not yet checked against a published source" in i["text"] for i in imb)
    assert "balance-quality-grades" not in {i["cite"] for i in s4(rep)}  # paywalled stub never offered


def test_missing_envelope_evidence_gives_a_missing_item_not_advice(reports, client):
    case, _ = reports[BEARING]
    names = [n for n in client.app.state.engine.bundle.feature_dictionary.index if not n.startswith("f_env_")]
    rec = copy.deepcopy(case)
    rec["signal_quality"]["warnings"] = []
    items = s4({"sections": build_sections(rec, None, names)})
    assert items == [{"tag": "Missing", "text": "No curated guidance covers this finding (underhang outer); "
                                                "no inspection steps are offered."}]


def healthy_copy(case, warnings):
    rec = copy.deepcopy(case)
    rec["diagnosis"].update(decision="healthy", fault_class=None)
    rec["signal_quality"]["warnings"] = warnings
    return rec


def test_healthy_verdict_gets_one_model_line_not_a_gap(reports):
    case, _ = reports[BEARING]
    for warnings in ([], ["healthy thresholds were not calibrated at this speed; a healthy verdict is less reliable"]):
        assert s4({"sections": build_sections(healthy_copy(case, warnings))}) == [
            {"tag": "Model", "text": "No inspection steps are offered for a healthy verdict."}]


def test_healthy_verdict_with_a_matching_warning_shows_its_acquisition_step(reports):
    case, _ = reports[BEARING]
    items = s4({"sections": build_sections(healthy_copy(case, ["5 glitched tacho intervals"]))})
    assert [i["cite"] for i in items] == ["acquisition-and-sensor-checks"] and "glitched tacho" in items[0]["text"]


def test_other_condition_with_nothing_selected_keeps_the_missing_item(reports):
    case, _ = reports[BEARING]
    rec = copy.deepcopy(case)
    rec["signal_quality"]["warnings"] = []
    assert [i["tag"] for i in s4({"sections": build_sections(rec)})] == ["Missing"]  # no feature names → no bearing entry
