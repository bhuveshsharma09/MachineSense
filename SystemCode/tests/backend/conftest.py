"""Backend tests need the approved bundle (MACHINESENSE_BUNDLE_DIR, see backend/README.md) and the raw development
recordings. Without the bundle the bundle-dependent tests are skipped, unless MACHINESENSE_REQUIRE_BUNDLE=1: then they FAIL
when the bundle is missing or does not verify (use this mode before every commit). Without the full raw recordings
(data/raw_npy; e.g. a fresh clone with only the demo pack) the tests that need them are skipped, unless
MACHINESENSE_REQUIRE_FULL_DATA=1; the demo-pack tests (test_demo_pack.py) need the bundle only."""
import os

import pandas as pd
import pytest

from backend.settings import bundle_dir, require_bundle
from ml.paths import SYSTEM_CODE_DIR, data_dir

GUARD_TEST_RESERVED = 3  # the test app's guard index covers a few reserved recordings (the full index takes ~15 s)


def _unavailable(reason: str):
    if require_bundle():
        pytest.fail(f"MACHINESENSE_REQUIRE_BUNDLE=1: {reason}", pytrace=False)
    pytest.skip(reason)


def raw_unavailable(reason: str):
    """The full raw recordings are missing: skip (a demo-pack clone), or FAIL under MACHINESENSE_REQUIRE_FULL_DATA=1."""
    if os.environ.get("MACHINESENSE_REQUIRE_FULL_DATA", "") == "1":
        pytest.fail(f"MACHINESENSE_REQUIRE_FULL_DATA=1: {reason}", pytrace=False)
    pytest.skip(reason)


@pytest.fixture(scope="session")
def bundle_path(bundle_only):
    if not (data_dir() / "raw_npy").exists():
        raw_unavailable(f"the full development recordings are not at {data_dir() / 'raw_npy'} (demo-pack clone)")
    return bundle_only


@pytest.fixture(scope="session")
def bundle_only():
    """The approved bundle, verified; no raw recordings needed."""
    d = bundle_dir()
    if not (d / "manifest.json").exists():
        _unavailable(f"approved bundle not found at {d} (set MACHINESENSE_BUNDLE_DIR)")
    if require_bundle():
        from ml.bundle import load_bundle
        try:
            load_bundle(d, device="cpu")
        except Exception as e:  # noqa: BLE001
            pytest.fail(f"MACHINESENSE_REQUIRE_BUNDLE=1: the bundle at {d} does not verify ({e})", pytrace=False)
    return d


@pytest.fixture(scope="session")
def reserved_ids(bundle_path):
    from backend.guard import reserved_rec_ids
    from ml.bundle import load_bundle
    meta = pd.read_csv(SYSTEM_CODE_DIR / "metadata.csv")
    return reserved_rec_ids(meta, load_bundle(bundle_path, device="cpu").config)


def no_reserved_read() -> bool:
    """MACHINESENSE_NO_RESERVED_READ=1 (Stage D1): no test may open a reserved raw array. The guard fixture then uses the
    existing index file as is (never rebuilt), and tests that would open a reserved signal skip."""
    import os
    return os.environ.get("MACHINESENSE_NO_RESERVED_READ", "") == "1"


@pytest.fixture
def reserved_raw_allowed():
    """Request this in a test that opens a reserved raw array: it skips under MACHINESENSE_NO_RESERVED_READ=1."""
    if no_reserved_read():
        pytest.skip("MACHINESENSE_NO_RESERVED_READ=1: this test opens a reserved raw array")


@pytest.fixture(scope="session")
def guard_path(tmp_path_factory, reserved_ids):
    if no_reserved_read():
        from backend.settings import guard_index
        if not guard_index().exists():
            _unavailable(f"MACHINESENSE_NO_RESERVED_READ=1 needs the existing index {guard_index()} (never rebuilt here)")
        return guard_index()
    from backend.guard import build_index
    p = tmp_path_factory.mktemp("guard") / "reserved_fingerprints.json"
    build_index(reserved_ids[:GUARD_TEST_RESERVED], data_dir() / "raw_npy", p)
    return p


@pytest.fixture(scope="session")
def cases_path(tmp_path_factory):
    return tmp_path_factory.mktemp("cases") / "cases.db"


@pytest.fixture(scope="session", autouse=True)
def uploads_path(tmp_path_factory):
    """Uploaded arrays go to a temporary folder in tests, never to backend/data."""
    import os
    p = tmp_path_factory.mktemp("uploads")
    old = os.environ.get("MACHINESENSE_UPLOADS_DIR")
    os.environ["MACHINESENSE_UPLOADS_DIR"] = str(p)
    yield p
    if old is None:
        os.environ.pop("MACHINESENSE_UPLOADS_DIR", None)
    else:
        os.environ["MACHINESENSE_UPLOADS_DIR"] = old


def no_llm_transport():
    """No test ever reaches a live language model: every explanation request fails as 'connection refused'."""
    import httpx

    def refuse(request):
        raise httpx.ConnectError("no language model in tests", request=request)
    return httpx.MockTransport(refuse)


@pytest.fixture(scope="session", autouse=True)
def explanations_log_path(tmp_path_factory):
    import os
    p = tmp_path_factory.mktemp("explain") / "explanations.jsonl"
    old = os.environ.get("MACHINESENSE_EXPLANATIONS_LOG")
    os.environ["MACHINESENSE_EXPLANATIONS_LOG"] = str(p)
    yield p
    if old is None:
        os.environ.pop("MACHINESENSE_EXPLANATIONS_LOG", None)
    else:
        os.environ["MACHINESENSE_EXPLANATIONS_LOG"] = old


@pytest.fixture(scope="session")
def client(bundle_path, guard_path, cases_path):
    from fastapi.testclient import TestClient

    from backend.app import create_app
    with TestClient(create_app(bundle_path, cases_path=cases_path, guard_path=guard_path,
                               llm_transport=no_llm_transport())) as c:
        yield c
