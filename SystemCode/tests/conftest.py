"""Bundle-dependent tests are marked @pytest.mark.needs_bundle(<file in the bundle>, …). They need the configured bundle
(ml.paths.bundle_dir(): $MACHINESENSE_BUNDLE_DIR, else artifacts/bundle) and the raw recordings (data_dir()/raw_npy).
Without the bundle they skip, unless MACHINESENSE_REQUIRE_BUNDLE=1: then they FAIL. Without the full raw recordings (a
fresh clone with only the demo pack) they skip, unless MACHINESENSE_REQUIRE_FULL_DATA=1. The same applies to tests that
read gitignored research outputs (results/scores/, artifacts/final/): see need_full_data."""
import os

import pytest


def need_full_data(*paths):
    """Skip a test whose local-only inputs (raw recordings, gitignored research outputs) are missing, as in a fresh
    clone; FAIL instead under MACHINESENSE_REQUIRE_FULL_DATA=1 (the full local setup)."""
    from pathlib import Path
    missing = [str(p) for p in paths if not Path(p).exists()]
    if not missing:
        return
    msg = "not in this checkout (full local data only): " + ", ".join(missing)
    if os.environ.get("MACHINESENSE_REQUIRE_FULL_DATA", "") == "1":
        pytest.fail("MACHINESENSE_REQUIRE_FULL_DATA=1: " + msg, pytrace=False)
    pytest.skip(msg)

from ml.paths import bundle_dir, data_dir, require_bundle


def pytest_configure(config):
    config.addinivalue_line("markers", "needs_bundle(*files): needs the configured bundle and the raw recordings")


def pytest_runtest_setup(item):
    m = item.get_closest_marker("needs_bundle")
    if m is None:
        return
    files = m.args or ("manifest.json",)
    missing = [str(bundle_dir() / f) for f in files if not (bundle_dir() / f).exists()]
    if missing:
        msg = "bundle not available: " + ", ".join(missing)
        if require_bundle():
            pytest.fail("MACHINESENSE_REQUIRE_BUNDLE=1: " + msg, pytrace=False)
        pytest.skip(msg)
    if not (data_dir() / "raw_npy").exists():
        msg = f"the full development recordings are not at {data_dir() / 'raw_npy'} (demo-pack clone)"
        if os.environ.get("MACHINESENSE_REQUIRE_FULL_DATA", "") == "1":
            pytest.fail("MACHINESENSE_REQUIRE_FULL_DATA=1: " + msg, pytrace=False)
        pytest.skip(msg)
