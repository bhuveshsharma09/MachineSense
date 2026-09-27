"""Backend tests need the approved bundle (set MACHINESENSE_BUNDLE_DIR, see backend/README.md) and the raw development
recordings; without them they are skipped."""
import pytest

from backend.engine import bundle_dir
from ml.paths import data_dir


@pytest.fixture(scope="session")
def bundle_path():
    d = bundle_dir()
    if not (d / "manifest.json").exists() or not (data_dir() / "raw_npy").exists():
        pytest.skip("approved bundle (MACHINESENSE_BUNDLE_DIR) or raw recordings not available")
    return d


@pytest.fixture(scope="session")
def client(bundle_path):
    from fastapi.testclient import TestClient

    from backend.app import create_app
    with TestClient(create_app(bundle_path)) as c:
        yield c
