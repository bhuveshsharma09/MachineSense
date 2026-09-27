"""Backend tests need the approved bundle (MACHINESENSE_BUNDLE_DIR, see backend/README.md) and the raw development
recordings. Without them the bundle-dependent tests are skipped, unless MACHINESENSE_REQUIRE_BUNDLE=1: then they FAIL
when the bundle is missing or does not verify (use this mode before every commit)."""
import pandas as pd
import pytest

from backend.settings import bundle_dir, require_bundle
from ml.paths import SYSTEM_CODE_DIR, data_dir

GUARD_TEST_RESERVED = 3  # the test app's guard index covers a few reserved recordings (the full index takes ~15 s)


def _unavailable(reason: str):
    if require_bundle():
        pytest.fail(f"MACHINESENSE_REQUIRE_BUNDLE=1: {reason}", pytrace=False)
    pytest.skip(reason)


@pytest.fixture(scope="session")
def bundle_path():
    d = bundle_dir()
    if not (d / "manifest.json").exists():
        _unavailable(f"approved bundle not found at {d} (set MACHINESENSE_BUNDLE_DIR)")
    if not (data_dir() / "raw_npy").exists():
        _unavailable(f"raw development recordings not found at {data_dir() / 'raw_npy'}")
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


@pytest.fixture(scope="session")
def guard_path(tmp_path_factory, reserved_ids):
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


@pytest.fixture(scope="session")
def client(bundle_path, guard_path, cases_path):
    from fastapi.testclient import TestClient

    from backend.app import create_app
    with TestClient(create_app(bundle_path, cases_path=cases_path, guard_path=guard_path)) as c:
        yield c
