"""Stage App-6: "Explore an unseen fault". The committed selection (backend/unseen.json) contains no reserved recording and
the required cases; every shown verdict equals ml.arbitrator.decide on the shown scores; each hidden class is absent from
its model's classes and calibration; the API serves the saved values without the weights, refuses reserved ids, and — when
the weights and raw recordings are present — recomputes values that agree with the saved ones."""
import json
import math

import pandas as pd
import pytest

from backend.unseen import CATEGORIES, TOLERANCE_REL, split_eligible
from ml.arbitrator import DEFAULT_VARIANT, decide, rule_comparison
from ml.paths import SYSTEM_CODE_DIR
from ml.rejection import failed_checks

UNSEEN = SYSTEM_CODE_DIR / "backend" / "unseen.json"


@pytest.fixture(scope="module")
def curated():
    if not UNSEEN.exists():
        pytest.fail("backend/unseen.json is missing (python -m backend.unseen build)")
    return json.loads(UNSEEN.read_text())


@pytest.fixture(scope="module")
def meta():
    return pd.read_csv(SYSTEM_CODE_DIR / "metadata.csv").set_index("rec_id")


def all_rec_ids(curated):
    ids = set()
    for r in curated["runs"].values():
        ids |= {e["rec_id"] for e in r["examples"]} | set(r["familiar_excluded"]) | set(r.get("calibration_ids", []))
        ids |= set(r.get("defining_recordings", []))
    return ids


def test_no_reserved_recording_in_the_selection(curated, meta):
    ids = all_rec_ids(curated)
    assert ids and ids <= set(meta.index)
    assert all(split_eligible(meta.loc[r]) for r in ids)
    text = UNSEEN.read_text()
    reserved = meta.index[~meta.apply(split_eligible, axis=1)]
    assert not [r for r in reserved if f'"{r}"' in text]                 # not even as a string anywhere in the file


def test_withheld_class_absent_from_training_and_calibration(curated):
    for c, r in curated["runs"].items():
        assert c not in r["classes_learned"] and len(r["classes_learned"]) == 9
        assert c not in r["calibration_classes"]
        for e in r["examples"]:
            if e["role"] == "withheld":
                assert e["true_class"] == c and e["result"]["true_class"] == c
            else:
                assert e["true_class"] in r["classes_learned"] and e["rec_id"] not in r["familiar_excluded"]


def test_shown_decision_is_the_arbitrators(curated):
    for c, r in curated["runs"].items():
        for e in r["examples"]:
            res, s, t = e["result"], e["result"]["scores"], e["result"]["thresholds"]
            assert t == r["thresholds"]
            assert res["decision"] == decide(DEFAULT_VARIANT, res["proposed_label"], s["msp"], s["maha"], s["ae"], s["b1s"], t)
            assert res["rule_comparison"] == rule_comparison(res["proposed_label"], s["msp"], s["maha"], s["ae"], s["b1s"], t)
            f = failed_checks(s["msp"], s["maha"], s["ae"], s["b1s"], **t)
            assert res["checks"]["confidence"]["passed"] == (not f["msp"])
            assert res["checks"]["familiarity"]["passed"] == (not f["maha"])
            both = res["checks"]["confidence"]["passed"] and res["checks"]["familiarity"]["passed"]
            assert (res["decision"]["decision"] == "unknown") == (not both)  # the default rule: the two checks decide


def test_curated_categories(curated):
    cats = {e["category"] for r in curated["runs"].values() for e in r["examples"]}
    assert {"successful rejection", "honest failure"} <= cats
    for c, r in curated["runs"].items():
        got = {e["category"] for e in r["examples"]}
        assert got | set(r["categories_without_an_eligible_recording"]) == set(CATEGORIES)   # reported, never substituted
        for e in r["examples"]:
            d = e["result"]["decision"]["decision"]
            if e["category"] == "successful rejection":
                assert e["role"] == "withheld" and d == "unknown"
            elif e["category"] == "honest failure":
                assert e["role"] == "withheld" and d != "unknown"
            else:
                assert e["role"] == "familiar" and d != "unknown" and not e["result"]["decision"]["caution"]


# ----------------------------------------------------------------------------------------------------------- API
@pytest.fixture(scope="module")
def static_client(bundle_path, guard_path, cases_path, tmp_path_factory):
    """The app as on a clean clone: no leave-one-class-out weights (final_directory points at an empty folder)."""
    from fastapi.testclient import TestClient

    from backend.app import create_app
    from tests.backend.conftest import no_llm_transport
    empty = tmp_path_factory.mktemp("no_final")
    with TestClient(create_app(bundle_path, cases_path=cases_path, guard_path=guard_path, llm_transport=no_llm_transport(),
                               final_directory=empty)) as c:
        yield c


def test_api_serves_saved_values_without_the_weights(static_client, curated):
    s = static_client.get("/api/unseen").json()
    assert s["built"] and not s["live"]["available"] and "not present" in s["live"]["reason"]
    assert "result" not in json.dumps(s["runs"])                      # results only when an example is run
    e = curated["runs"]["imbalance"]["examples"][0]
    got = static_client.get("/api/unseen/run", params={"withheld": "imbalance", "rec_id": e["rec_id"], "role": e["role"]})
    assert got.status_code == 200 and got.json() == e["result"] and got.json()["source"] == "saved"


def test_api_refuses_reserved_and_ineligible(static_client, meta):
    t = meta[["split_closed", "split_speed_block", "split_severity"]] == "test"
    for col in t.columns:
        rid = meta.index[t[col] & (t.sum(axis=1) == 1)][0]
        cls = meta.at[rid, "class10"]
        for role in ("withheld", "familiar"):
            r = static_client.get("/api/unseen/run", params={"withheld": cls if cls != "normal" else "imbalance",
                                                             "rec_id": rid, "role": role})
            assert r.status_code == 403 and "reserved" in r.json()["detail"]
    r = static_client.get("/api/unseen/run", params={"withheld": "imbalance", "rec_id": "normal__18.432", "role": "familiar"})
    assert r.status_code == 403                                        # a threshold-defining recording is never offered
    eligible = meta[meta.apply(split_eligible, axis=1) & (meta.class10 == "imbalance")].index[-1]
    r = static_client.get("/api/unseen/run", params={"withheld": "imbalance", "rec_id": eligible, "role": "withheld"})
    assert r.status_code in (200, 404)                                 # 404 unless it happens to be a curated example


def test_live_values_agree_with_the_saved_selection(bundle_path, curated):
    from backend.unseen import UnseenService
    from ml.bundle import load_bundle
    from ml.final import final_dir
    from ml.paths import data_dir
    if not (final_dir() / "manifest.json").exists():
        pytest.skip("artifacts/final/ (the leave-one-class-out weights) is not present: live agreement cannot be checked")
    if not (data_dir() / "raw_npy").exists():
        pytest.skip("the raw recordings are not present: live agreement cannot be checked")
    svc = UnseenService(load_bundle(bundle_path, device="cpu"))
    assert svc.live() is not None, svc._live_reason
    assert svc.live().manifest_sha256 == curated["final_manifest_sha256"]
    for c, r in curated["runs"].items():
        for e in r["examples"]:
            live, saved = svc.run(c, e["rec_id"], e["role"]), e["result"]
            assert live["source"] == "live" and live["decision"] == saved["decision"]
            assert live["proposed_label"] == saved["proposed_label"] and live["rule_comparison"] == saved["rule_comparison"]
            for k, v in saved["scores"].items():
                assert math.isclose(live["scores"][k], v, rel_tol=TOLERANCE_REL, abs_tol=TOLERANCE_REL), (c, e["rec_id"], k)
