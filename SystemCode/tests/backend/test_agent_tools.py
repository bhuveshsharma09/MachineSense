"""Stage App-7: the assistant's four read-only tools on a stored case: values come from the stored record (and the report's
own builders), every fact carries a ref, lookup_guidance is section 04 exactly, and bad input gives an error dict."""
import json

import pytest

from backend.agent.tools import (CHANNELS, EVIDENCE_RANK_BASIS, NAMES, SCHEMAS, Tools, fixed, leaves, measured,
                                 probability, score, summary)
from backend.checks import fmt_score
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
    assert by["diagnosis.scores.msp"]["score"]["value"] == score(d["scores"]["msp"])
    assert by["diagnosis.scores.msp"]["threshold"]["value"] == score(d["thresholds"]["tau_p"])
    assert by["diagnosis.scores.maha"]["threshold"] == {"value": round(d["thresholds"]["tau_d"], 2),
                                                        "ref": "diagnosis.thresholds.tau_d"}  # the report shows 2 decimals
    top = sorted(d["class10_probabilities"].items(), key=lambda kv: -kv[1])[:2]
    assert [(t["class"], t["rank"]["value"], t["probability"]["value"]) for t in out["top_classes"]] == \
        [(c, k + 1, probability(p)) for k, (c, p) in enumerate(top)]
    assert out["top_classes"][0]["rank"]["ref"] == "diagnosis.top_classes.0.rank" and out["top_classes_rank_basis"]


def test_check_signal_reads_the_signal_quality_block(setup):
    tools, case, _ = setup
    out, q = tools.check_signal(case["case_id"]), case["signal_quality"]
    assert out["shaft_speed_hz"]["value"] == round(q["shaft_speed_hz"], 2) and out["speed_stable"]["value"] == q["speed_stable"]
    assert [w["value"] for w in out["warnings"]] == q["warnings"]
    lo, hi = q["calibration_normal_speed_range_hz"]
    assert out["inside_calibration_speed_range"]["value"] == (lo <= q["shaft_speed_hz"] <= hi)


def test_get_evidence_is_the_reports_channel_evidence(setup):
    tools, case, client = setup
    out = tools.get_evidence(case["case_id"])
    ch = client.get(f"/api/cases/{case['case_id']}/channels").json()
    assert sorted(c["channel"] for c in out["channels"]) == sorted(r["channel"] for r in ch["channels"])
    assert [m["value"] for m in out["marked_channels"]] == ch["marked"]
    row = {r["channel"]: r for r in ch["channels"]}["over_radial"]
    got = next(c for c in out["channels"] if c["channel"] == "over_radial")["metrics"]["rms"]
    assert got["value"]["value"] == measured(row["metrics"]["rms"]["value"])
    assert got["change_factor"]["value"] == round(row["metrics"]["rms"]["factor"], 1)
    one = tools.get_evidence(case["case_id"], channel="under_axial")
    assert [c["channel"] for c in one["channels"]] == ["under_axial"]
    rank = next(c for c in out["channels"] if c["channel"] == "under_axial")["rank"]
    assert one["channels"][0]["rank"] == rank and one["largest_change"] == out["largest_change"]  # global rank


def test_get_evidence_ranks_by_change_beyond_the_healthy_band(setup):
    """Rank 1 is the channel furthest beyond its band (the report's first marked or exceeding channel), named in plain
    words; largest_change is rank 1."""
    tools, case, client = setup
    out = tools.get_evidence(case["case_id"])
    ch = client.get(f"/api/cases/{case['case_id']}/channels").json()
    ranked = [c for c in out["channels"] if isinstance(c["rank"], dict)]
    assert [c["rank"]["value"] for c in ranked] == list(range(1, len(ranked) + 1))
    assert [c["channel"] for c in ranked][:len(ch["exceeding"])] == ch["exceeding"]
    beyond = {r["channel"]: r["beyond_band"] for r in ch["channels"] if r["beyond_band"] is not None}
    assert [c["channel"] for c in ranked] == sorted(beyond, key=lambda k: -beyond[k])
    top = out["largest_change"]
    assert top["channel"]["value"] == ranked[0]["channel"] and top["rank"] == {"value": 1, "ref": "evidence.largest_change.rank"}
    assert out["rank_basis"] == top["rank_basis"] == EVIDENCE_RANK_BASIS == "change beyond the healthy band"
    assert ranked[0]["rank"]["ref"] == f"evidence.{ranked[0]['channel']}.rank"


def test_values_are_rounded_to_the_reports_display_precision():
    assert score(0.99997759) == 0.999978 and score(0.9999999) == "> 0.999999" and score(0.4115553) == 0.411555
    assert score(1.661334) == 1.6613 and score(10.436428) == 10.44 and probability(0.99998773) == "> 0.9999"
    assert probability(0.4) == 0.4 and fixed(1.8222750078995185, 1) == 1.8 and fixed(33.41008, 2) == 33.41
    assert measured(0.012922000778) == 0.0129 and measured(1.66133) == 1.66 and measured(123.456) == 123.0
    assert fmt_score(0.99998773) == "> 0.9999"


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
