"""Demo-pack mode: a fresh clone without MAFAULDA runs the app on the committed demo pack (SystemCode/demo/, built by
scripts/40_demo_pack.py). These tests need the bundle only; they point the data directory at the demo pack, so they run
the same way in a clone and in the full setup. Every shipped recording must be a development-pool recording that the
reserved-fingerprint guard does not match; each demo recording must give the verdict the full setup gave."""
import hashlib
import io
import json
import os

import numpy as np
import pandas as pd
import pytest

from backend.settings import demo_dir, guard_index
from ml.paths import DATA_DIR_ENV, SYSTEM_CODE_DIR

MANIFEST = json.loads((demo_dir() / "manifest.json").read_text())
DEMO_IDS = [r["rec_id"] for r in MANIFEST["recordings"]]
REF_IDS = [r["rec_id"] for r in MANIFEST["healthy_references"]]


@pytest.fixture(scope="module")
def demo_env():
    """The data directory points at the demo pack for this module only."""
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv(DATA_DIR_ENV, str(demo_dir()))
        yield


@pytest.fixture(scope="module")
def demo_client(bundle_only, demo_env, tmp_path_factory):
    from fastapi.testclient import TestClient

    from backend.app import create_app
    from tests.backend.conftest import no_llm_transport
    with TestClient(create_app(bundle_only, cases_path=tmp_path_factory.mktemp("demo") / "cases.db",
                               guard_path=guard_index(), llm_transport=no_llm_transport())) as c:
        yield c


def test_pack_files_match_the_manifest_and_stay_small():
    files = sorted((demo_dir() / "raw_npy").glob("*.npy"))
    assert sorted(p.stem for p in files) == sorted(DEMO_IDS + REF_IDS)
    want = {r["rec_id"]: r["sha256"] for r in MANIFEST["recordings"] + MANIFEST["healthy_references"]}
    assert all(hashlib.sha256(p.read_bytes()).hexdigest() == want[p.stem] for p in files)
    assert max(p.stat().st_size for p in files) < 50e6 and sum(p.stat().st_size for p in demo_dir().rglob("*") if p.is_file()) < 100e6
    v = MANIFEST["healthy_reference_values"]
    assert hashlib.sha256((demo_dir() / v["file"]).read_bytes()).hexdigest() == v["sha256"]


def test_every_shipped_recording_is_in_the_development_pool_and_not_reserved(bundle_only):
    from backend.guard import ReservedGuard, reserved_rec_ids
    from ml.bundle import load_bundle
    from ml.splits import development_pool
    cfg = load_bundle(bundle_only, device="cpu").config
    meta = pd.read_csv(SYSTEM_CODE_DIR / "metadata.csv")
    pool = set(development_pool(meta, cfg.development.allowed_closed, cfg.development.reserved_columns).rec_id)
    reserved = set(reserved_rec_ids(meta, cfg))
    guard = ReservedGuard.load(guard_index())
    m = meta.set_index("rec_id")
    for r in DEMO_IDS + REF_IDS:
        assert r in pool and r not in reserved, r
        assert all(m.loc[r, c] != "test" for c in ("split_closed", "split_severity", "split_speed_block")), r
        assert guard.match(np.load(demo_dir() / "raw_npy" / f"{r}.npy")) is None, r
    assert set(MANIFEST["healthy_reference_values"]["recordings"]) <= pool - reserved


def test_demo_pack_is_used_only_when_the_full_data_are_missing(tmp_path, monkeypatch):
    import ml.paths
    from backend.settings import use_demo_pack
    monkeypatch.setenv(DATA_DIR_ENV, "unset")  # recorded, so use_demo_pack's own setting is undone after the test
    monkeypatch.delenv(DATA_DIR_ENV)
    monkeypatch.setattr(ml.paths, "SYSTEM_CODE_DIR", tmp_path)  # a checkout whose data/raw_npy is missing
    assert use_demo_pack() is True and os.environ[DATA_DIR_ENV] == str(demo_dir())
    monkeypatch.delenv(DATA_DIR_ENV)
    (tmp_path / "data" / "raw_npy").mkdir(parents=True)  # the full data present: they stay the default
    assert use_demo_pack() is False and DATA_DIR_ENV not in os.environ


def test_only_demo_recordings_are_listed_and_offered(demo_client):
    r = demo_client.get("/api/recordings").json()
    assert sorted(x["rec_id"] for x in r["recordings"]) == sorted(DEMO_IDS) and r["data_mode"] == "demo-pack"
    assert {s["rec_id"] for s in demo_client.get("/api/showcase").json()["showcase"]} <= set(DEMO_IDS)
    h = demo_client.get("/api/health").json()
    assert h["bundle_verified"] and h["data"] == {"mode": "demo-pack", "recordings_available": len(DEMO_IDS)}


@pytest.mark.parametrize("rec", MANIFEST["recordings"], ids=lambda r: r["rec_id"])
def test_each_demo_recording_gives_the_full_setup_verdict_with_its_evidence(demo_client, rec):
    r = demo_client.post("/api/diagnose", json={"rec_id": rec["rec_id"]})
    assert r.status_code == 200, r.text
    d = r.json()["diagnosis"]
    assert {k: d[k] for k in rec["expected"]} == rec["expected"]
    cid = r.json()["case_id"]
    ch = demo_client.get(f"/api/cases/{cid}/channels")
    assert ch.status_code == 200 and ch.json()["reference"]["rec_id"] == rec["healthy_reference"]
    assert demo_client.get(f"/api/cases/{cid}/signals").json()["reference"]["rec_id"] == rec["healthy_reference"]
    assert demo_client.get(f"/api/cases/{cid}/report").status_code == 200


def test_a_recording_outside_the_pack_gets_a_clear_message(demo_client):
    meta = pd.read_csv(SYSTEM_CODE_DIR / "metadata.csv")
    other = next(r for r in demo_client.app.state.engine.pool().rec_id if r not in DEMO_IDS + REF_IDS)
    r = demo_client.post("/api/diagnose", json={"rec_id": other})
    assert r.status_code == 404 and "not in the demo pack" in r.json()["detail"]
    reserved = meta.loc[meta.split_closed == "test", "rec_id"].iloc[0]  # refused as reserved, as before (never read)
    assert demo_client.post("/api/diagnose", json={"rec_id": reserved}).status_code == 403


def test_precomputed_reference_values_equal_the_values_from_the_raw_recordings(bundle_only, demo_env):
    from backend.evidence import _features_for_rec, _precomputed, channel_values, training_values
    from ml.bundle import load_bundle
    from ml.config import config_sha256
    cfg = load_bundle(bundle_only, device="cpu").config
    pre = _precomputed(config_sha256(cfg))
    assert sorted(pre) == sorted(MANIFEST["healthy_reference_values"]["recordings"])
    for r in REF_IDS:  # shipped references: recomputed from their raw file
        assert channel_values(_features_for_rec(r, cfg.model_dump_json())) == pre[r]
    missing = next(r for r in pre if r not in REF_IDS)  # a training normal not shipped: served from the precomputed values
    assert training_values(missing, cfg) == pre[missing]


def test_an_uploaded_csv_is_diagnosed_with_channel_evidence(demo_client):
    x = np.load(demo_dir() / "raw_npy" / f"{DEMO_IDS[0]}.npy")
    buf = io.StringIO()
    np.savetxt(buf, x, fmt="%.9g", delimiter=",")
    up = demo_client.post("/api/diagnose", files={"file": ("demo.csv", buf.getvalue().encode(), "text/csv")})
    assert up.status_code == 200, up.text
    assert up.json()["diagnosis"]["decision"] == MANIFEST["recordings"][0]["expected"]["decision"]
    assert demo_client.get(f"/api/cases/{up.json()['case_id']}/channels").status_code == 200
