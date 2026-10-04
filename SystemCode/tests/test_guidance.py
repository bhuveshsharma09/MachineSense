"""Stage App-5: the guidance library loads and validates, and the selection rule is exactly the declared one."""
import random
import re

import pytest
import yaml
from pydantic import ValidationError

from ml.guidance import ENTRIES_DIR, Entry, load_entries, select

GRADIENT = re.compile(r"grad|cam|saliency|attribution|occlusion", re.I)


def base(**kw):
    d = {"id": "x", "title": "t", "source": "internal", "source_url": None, "licence": "l", "verification": "internal",
         "drafted_with_ai": True, "applies_to": ["imbalance"], "evidence_required": ["speed_hz"], "excludes": [],
         "steps": ["inspect something"], "notes": "n"}
    d.update(kw)
    return Entry.model_validate(d)


def test_every_entry_loads_and_its_id_is_its_stem():
    entries = load_entries()
    stems = sorted(p.stem for p in ENTRIES_DIR.glob("*.yaml"))
    assert sorted(e.id for e in entries) == stems and len(entries) == 8


def test_id_mismatch_and_duplicates_are_errors(tmp_path):
    good = yaml.safe_load((ENTRIES_DIR / "imbalance-1x.yaml").read_text())
    (tmp_path / "other-name.yaml").write_text(yaml.safe_dump(good))
    with pytest.raises(ValueError, match="file stem"):
        load_entries(tmp_path)


def test_no_entry_relies_on_gradient_attribution():
    for e in load_entries():
        assert not [n for n in e.evidence_required if GRADIENT.search(n)], e.id      # CLAUDE.md §11 rule 11


def test_paywalled_stubs_have_no_steps():
    stubs = [e for e in load_entries() if e.verification == "needs_paywalled_source"]
    assert stubs and all(not e.steps for e in stubs)


def test_missing_evidence_drops_the_entry():
    e = base(evidence_required=["speed_hz", "f_under_radial_o1"])
    assert select([e], condition="imbalance", available_evidence={"speed_hz", "f_under_radial_o1"}, warnings=[]) == [e]
    assert select([e], condition="imbalance", available_evidence={"speed_hz"}, warnings=[]) == []


def test_paywalled_entry_is_never_offered_even_when_everything_matches():
    e = base(verification="needs_paywalled_source", steps=[])
    assert select([e], condition="imbalance", available_evidence={"speed_hz"}, warnings=[]) == []


def test_excludes_match_a_warning_case_insensitively():
    e = base(excludes=["Microphone bursts"])
    assert select([e], condition="imbalance", available_evidence={"speed_hz"}, warnings=["12 microphone bursts removed"]) == []
    assert select([e], condition="imbalance", available_evidence={"speed_hz"}, warnings=["possible clipping on over_radial"]) == [e]


def test_applies_to_any_and_condition():
    a, b = base(id="a", applies_to="any"), base(id="b", applies_to=["vertical_misalignment"])
    assert [x.id for x in select([a, b], condition="imbalance", available_evidence={"speed_hz"}, warnings=[])] == ["a"]
    assert [x.id for x in select([a, b], condition="unknown", available_evidence={"speed_hz"}, warnings=[])] == ["a"]


def test_unknown_is_a_verdict_label():
    e = base(applies_to=["unknown"])
    assert select([e], condition="unknown", available_evidence={"speed_hz"}, warnings=[]) == [e]
    assert select([e], condition="imbalance", available_evidence={"speed_hz"}, warnings=[]) == []


def test_order_is_deterministic():
    entries = [base(id=f"{v}-{i}", verification=v) for v in ("ai_drafted", "internal", "citation_unverified", "open_verified")
               for i in (2, 1)]
    want = ["internal-1", "internal-2", "open_verified-1", "open_verified-2", "citation_unverified-1",
            "citation_unverified-2", "ai_drafted-1", "ai_drafted-2"]
    for seed in (0, 1):
        shuffled = entries[:]
        random.Random(seed).shuffle(shuffled)
        assert [e.id for e in select(shuffled, condition="imbalance", available_evidence={"speed_hz"}, warnings=[])] == want


def test_schema_failures():
    d = yaml.safe_load((ENTRIES_DIR / "imbalance-1x.yaml").read_text())
    with pytest.raises(ValidationError):
        Entry.model_validate({k: v for k, v in d.items() if k != "drafted_with_ai"})
    with pytest.raises(ValidationError):
        Entry.model_validate({**d, "verification": "peer_reviewed"})
    with pytest.raises(ValidationError):
        Entry.model_validate({**d, "unexpected": 1})
    with pytest.raises(ValidationError):
        Entry.model_validate({**d, "applies_to": ["not_a_class"]})
