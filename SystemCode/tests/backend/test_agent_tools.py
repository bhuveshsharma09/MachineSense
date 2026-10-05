"""Stage App-7: the assistant's four read-only tools on a stored case: values come from the stored record (and the report's
own builders), every fact carries a ref, lookup_guidance is section 04 exactly, and bad input gives an error dict."""
import json
import math

import pytest

from backend.agent.tools import CHANNELS, NAMES, SCHEMAS, Tools, leaves, summary
from backend.report import guidance_items

BEARING = "underhang__outer_race__0g__14.7456"
GRADIENT = ("grad", "cam", "saliency", "attribution", "occlusion", "xai")


@pytest.fixture(scope="module")
def setup(client):
    case = client.post("/api/diagnose", json={"rec_id": BEARING}).json()
    return Tools(client.app.state.engine), case, client


def numbers(result):
    return [v for _, v in leaves(result) if isinstance(v, (int, float)) and not isinstance(v, bool)]


def test_every_tool_is_json_serialisable_and_numbers_carry_refs(setup):
    tools, case, _ = setup
    for name in NAMES:
        out = tools.call(name, case["case_id"], {})
        json.dumps(out)
        assert "error" not in out and numbers(out) or name == "lookup_guidance", name

        def bare(o, path=""):  # every number sits in a leaf {"value", "ref"}, never loose
            if isinstance(o, dict):
                return [] if "ref" in o else [p for k, v in o.items() for p in bare(v, f"{path}.{k}")]
            if isinstance(o, list):
                return [p for i, v in enumerate(o) for p in bare(v, f"{path}[{i}]")]
            return [path] if isinstance(o, (int, float)) and not isinstance(o, bool) else []
        assert bare(out) == [], (name, bare(out))
        assert not any(g in json.dumps(out).lower() for g in GRADIENT), name  # §11 rule 11


def test_diagnose_reads_the_stored_record(setup):
    tools, case, _ = setup
    out, d = tools.diagnose(case["case_id"]), case["diagnosis"]
    assert out["verdict"]["value"] == d["decision"] and out["fault_class"]["value"] == d["fault_class"]
    assert "according to the dataset label" in out["location"]["value"]
    by = {c["score"]["ref"]: c for c in out["checks"]}
    assert by["diagnosis.scores.msp"]["score"]["value"] == d["scores"]["msp"]
    assert by["diagnosis.scores.msp"]["threshold"]["value"] == d["thresholds"]["tau_p"]
    assert by["diagnosis.scores.maha"]["threshold"]["ref"] == "diagnosis.thresholds.tau_d"
    top = sorted(d["class10_probabilities"].items(), key=lambda kv: -kv[1])[:2]
    assert [(t["class"], t["probability"]["value"]) for t in out["top_classes"]] == top


def test_check_signal_reads_the_signal_quality_block(setup):
    tools, case, _ = setup
    out, q = tools.check_signal(case["case_id"]), case["signal_quality"]
    assert out["shaft_speed_hz"]["value"] == q["shaft_speed_hz"] and out["speed_stable"]["value"] == q["speed_stable"]
    assert [w["value"] for w in out["warnings"]] == q["warnings"]
    lo, hi = q["calibration_normal_speed_range_hz"]
    assert out["inside_calibration_speed_range"]["value"] == (lo <= q["shaft_speed_hz"] <= hi)


def test_get_evidence_is_the_reports_channel_evidence(setup):
    tools, case, client = setup
    out = tools.get_evidence(case["case_id"])
    ch = client.get(f"/api/cases/{case['case_id']}/channels").json()
    assert [c["channel"] for c in out["channels"]] == [r["channel"] for r in ch["channels"]]
    assert [m["value"] for m in out["marked_channels"]] == ch["marked"]
    row = {r["channel"]: r for r in ch["channels"]}["over_radial"]
    got = next(c for c in out["channels"] if c["channel"] == "over_radial")["metrics"]["rms"]
    assert got["value"]["value"] == pytest.approx(row["metrics"]["rms"]["value"], rel=1e-12)
    assert math.isclose(got["change_factor"]["value"], row["metrics"]["rms"]["factor"], rel_tol=1e-12)
    one = tools.get_evidence(case["case_id"], channel="under_axial")
    assert [c["channel"] for c in one["channels"]] == ["under_axial"]


def test_lookup_guidance_is_section_04_exactly(setup):
    tools, case, client = setup
    out = tools.lookup_guidance(case["case_id"])
    expected = guidance_items(case, client.app.state.engine.bundle.feature_dictionary.index)
    assert [{k: v for k, v in i.items() if k != "ref"} for i in out["items"]] == expected
    s4 = client.get(f"/api/cases/{case['case_id']}/report").json()["sections"][3]["items"]
    assert expected == s4
    assert len({i["ref"] for i in out["items"]}) == len(out["items"])


def test_unknown_case_channel_tool_or_argument_gives_an_error_dict(setup):
    tools, case, _ = setup
    for name in NAMES:
        assert "error" in tools.call(name, "no-such-case", {})
    assert "error" in tools.get_evidence(case["case_id"], channel="tacho_x")
    assert "error" in tools.call("read_file", case["case_id"], {})
    assert "error" in tools.call("diagnose", case["case_id"], {"case_id": "other"})  # the model cannot pick a case


def test_schemas_and_summaries(setup):
    tools, case, _ = setup
    assert all("case_id" not in s["function"]["parameters"]["properties"] for s in SCHEMAS)
    assert set(SCHEMAS[2]["function"]["parameters"]["properties"]["channel"]["enum"]) == set(CHANNELS)
    for name in NAMES:
        line = summary(name, tools.call(name, case["case_id"], {}))
        assert line and "\n" not in line
