"""Bundle-dependent tests are marked @pytest.mark.needs_bundle(<file in the bundle>, …). They need the configured bundle
(ml.paths.bundle_dir(): $MACHINESENSE_BUNDLE_DIR, else artifacts/bundle) and the raw recordings (data_dir()/raw_npy).
Without them they skip, unless MACHINESENSE_REQUIRE_BUNDLE=1: then they FAIL."""
import pytest

from ml.paths import bundle_dir, data_dir, require_bundle


def pytest_configure(config):
    config.addinivalue_line("markers", "needs_bundle(*files): needs the configured bundle and the raw recordings")


def pytest_runtest_setup(item):
    m = item.get_closest_marker("needs_bundle")
    if m is None:
        return
    files = m.args or ("manifest.json",)
    missing = [str(bundle_dir() / f) for f in files if not (bundle_dir() / f).exists()]
    if not (data_dir() / "raw_npy").exists():
        missing.append(str(data_dir() / "raw_npy"))
    if missing:
        msg = "bundle or raw recordings not available: " + ", ".join(missing)
        if require_bundle():
            pytest.fail("MACHINESENSE_REQUIRE_BUNDLE=1: " + msg, pytrace=False)
        pytest.skip(msg)
