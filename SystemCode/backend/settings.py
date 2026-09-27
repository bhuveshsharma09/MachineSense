"""Every path the app needs, as environment variables with defaults relative to SystemCode/ (no personal folders).

| Setting | Env var | Default |
| --- | --- | --- |
| approved model bundle | MACHINESENSE_BUNDLE_DIR | SystemCode/artifacts/bundle |
| case history (SQLite) | MACHINESENSE_CASES_DB | SystemCode/backend/data/cases.db |
| uploaded arrays (for evidence views) | MACHINESENSE_UPLOADS_DIR | SystemCode/backend/data/uploads |
| reserved-recording fingerprint index | MACHINESENSE_GUARD_INDEX | SystemCode/backend/data/reserved_fingerprints.json |
| showcase list | MACHINESENSE_SHOWCASE | SystemCode/backend/showcase.json |
| raw development recordings | (fixed by ml.paths.data_dir) | SystemCode/data/raw_npy |
| require the bundle in tests | MACHINESENSE_REQUIRE_BUNDLE=1 | off |
"""
from __future__ import annotations

import os
from pathlib import Path

from ml.paths import SYSTEM_CODE_DIR, data_dir

BACKEND_DIR = Path(__file__).resolve().parent


def _p(env: str, default: Path) -> Path:
    return Path(os.environ.get(env) or default).expanduser().resolve()


def bundle_dir(explicit=None) -> Path:
    return Path(explicit).resolve() if explicit else _p("MACHINESENSE_BUNDLE_DIR", SYSTEM_CODE_DIR / "artifacts" / "bundle")


def cases_db() -> Path:
    return _p("MACHINESENSE_CASES_DB", BACKEND_DIR / "data" / "cases.db")


def uploads_dir() -> Path:
    return _p("MACHINESENSE_UPLOADS_DIR", BACKEND_DIR / "data" / "uploads")


def guard_index() -> Path:
    return _p("MACHINESENSE_GUARD_INDEX", BACKEND_DIR / "data" / "reserved_fingerprints.json")


def showcase_file() -> Path:
    return _p("MACHINESENSE_SHOWCASE", BACKEND_DIR / "showcase.json")


def raw_dir() -> Path:
    """Raw development recordings (read by ml.inference.load_recording; not configurable without changing ml/)."""
    return data_dir() / "raw_npy"


def require_bundle() -> bool:
    return os.environ.get("MACHINESENSE_REQUIRE_BUNDLE", "") == "1"
