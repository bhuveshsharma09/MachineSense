"""App-9 addendum C: the post-confirmation procedure schema (procedure-1) and the gate. The entries here are SYNTHETIC test
fixtures written to a temporary folder; the real library (knowledge/procedures/) has no entries."""
import pytest
import yaml

from ml.procedures import NONE_AVAILABLE, PROCEDURES_DIR, Procedure, gate, load_procedures


def entry(**kw):
    base = {"schema_version": "procedure-1", "id": "synthetic-overhang-cage", "version": "1.0", "title": "SYNTHETIC test entry",
            "machine": "test rig", "parts": ["out_brg"], "fault_classes": ["overhang_cage"], "prerequisites": ["SYNTHETIC prerequisite"],
            "safety_stop": ["SYNTHETIC stop condition"], "steps": ["SYNTHETIC step"], "source": "SYNTHETIC source, p. 1",
            "source_url": None, "licence": "test only", "verification": "reviewed", "reviewer": "Test Reviewer", "reviewed_on": "2026-10-26"}
    return Procedure(**(base | kw))


EV = lambda status, fc=None, part=None: {"status": status, "fault_class": fc, "part": part}


def test_the_real_library_has_no_entries_and_says_none_is_available():
    assert load_procedures(PROCEDURES_DIR) == []
    g = gate([EV("confirmed", "overhang_cage", "out_brg")], load_procedures(PROCEDURES_DIR))
    assert g["open"] is False and g["message"] == NONE_AVAILABLE and g["procedures"] == []


def test_gate_opens_only_on_a_latest_confirmation_with_a_reviewed_matching_entry():
    e = [entry()]
    assert gate([EV("confirmed", "overhang_cage", "out_brg")], e)["open"] is True
    assert gate([EV("confirmed", "overhang_cage", "out_brg")], e)["procedures"][0]["id"] == "synthetic-overhang-cage"
    for events, why in (([], "no technician confirmation"),
                        ([EV("confirmed", "overhang_cage", "out_brg"), EV("rejected")], "rejected"),
                        ([EV("confirmed", "overhang_cage", "out_brg"), EV("withdrawn")], "withdrawn"),
                        ([EV("confirmed", "overhang_ball", "out_brg")], "no procedure entry covers"),
                        ([EV("confirmed", "overhang_cage", "in_brg")], "no procedure entry covers")):
        g = gate(events, e)
        assert g["open"] is False and g["procedures"] == [] and g["message"] == NONE_AVAILABLE and why in g["reason"]


@pytest.mark.parametrize("status", ["awaiting_source", "unavailable", "draft"])
def test_an_unreviewed_entry_is_never_offered(status):
    g = gate([EV("confirmed", "overhang_cage", "out_brg")], [entry(verification=status, reviewer=None, reviewed_on=None)])
    assert g["open"] is False and status in g["reason"]


def test_schema_rejects_a_reviewed_entry_without_reviewer_and_unknown_labels():
    with pytest.raises(ValueError):
        entry(reviewer=None)
    with pytest.raises(ValueError):
        entry(parts=["s_in"])
    with pytest.raises(ValueError):
        entry(fault_classes=["normal"])
    with pytest.raises(ValueError):
        entry(safety_stop=[])
    with pytest.raises(ValueError):
        Procedure(**(entry().model_dump(mode="json") | {"torque_nm": 40}))  # no field the schema does not define


def test_loader_checks_ids(tmp_path):
    (tmp_path / "synthetic-overhang-cage.yaml").write_text(yaml.safe_dump(entry().model_dump(mode="json")))
    assert [p.id for p in load_procedures(tmp_path)] == ["synthetic-overhang-cage"]
    (tmp_path / "other-name.yaml").write_text(yaml.safe_dump(entry().model_dump(mode="json")))
    with pytest.raises(ValueError):
        load_procedures(tmp_path)
