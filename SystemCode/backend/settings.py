"""Every path the app needs, as environment variables with defaults relative to SystemCode/ (no personal folders).

| Setting | Env var | Default |
| --- | --- | --- |
| approved model bundle | MACHINESENSE_BUNDLE_DIR | SystemCode/artifacts/bundle |
| case history (SQLite) | MACHINESENSE_CASES_DB | SystemCode/backend/data/cases.db |
| uploaded arrays (for evidence views) | MACHINESENSE_UPLOADS_DIR | SystemCode/backend/data/uploads |
| reserved-recording fingerprint index | MACHINESENSE_GUARD_INDEX | SystemCode/backend/data/reserved_fingerprints.json |
| showcase list | MACHINESENSE_SHOWCASE | SystemCode/backend/showcase.json |
| unseen-fault selection (App-6) | MACHINESENSE_UNSEEN | SystemCode/backend/unseen.json |
| unseen-fault sweep cache (gitignored) | MACHINESENSE_UNSEEN_SWEEP | SystemCode/backend/data/unseen_sweep.json |
| typical healthy variation (built from training normals) | MACHINESENSE_VARIATION | SystemCode/backend/data/healthy_variation.json |
| raw development recordings | (fixed by ml.paths.data_dir) | SystemCode/data/raw_npy |
| require the bundle in tests | MACHINESENSE_REQUIRE_BUNDLE=1 | off |
| LLM endpoint / model / timeout / on-off (App-3) | MACHINESENSE_LLM_BASE_URL / _MODEL / _TIMEOUT_S / _ENABLED | http://localhost:1234/v1 / google/gemma-4-e4b / 20 / true |
| LLM API key (optional; read from SystemCode/.env only) | MACHINESENSE_LLM_API_KEY in .env | none |
| explanation call log | MACHINESENSE_EXPLANATIONS_LOG | SystemCode/backend/data/explanations.jsonl |
| assistant: tool calls per question (App-7) | MACHINESENSE_AGENT_MAX_TOOL_CALLS | 6 |
| assistant: revisions after a failed evidence review | MACHINESENSE_AGENT_MAX_REVISIONS | 1 |
| assistant: wall-clock limit per question, seconds (also each model request's timeout) | MACHINESENSE_AGENT_TIMEOUT_S | 60 |
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


def unseen_file() -> Path:
    return _p("MACHINESENSE_UNSEEN", BACKEND_DIR / "unseen.json")


def unseen_sweep_file() -> Path:
    return _p("MACHINESENSE_UNSEEN_SWEEP", BACKEND_DIR / "data" / "unseen_sweep.json")


def variation_file() -> Path:
    return _p("MACHINESENSE_VARIATION", BACKEND_DIR / "data" / "healthy_variation.json")


def raw_dir() -> Path:
    """Raw development recordings (read by ml.inference.load_recording; not configurable without changing ml/)."""
    return data_dir() / "raw_npy"


def require_bundle() -> bool:
    return os.environ.get("MACHINESENSE_REQUIRE_BUNDLE", "") == "1"


# ----------------------------------------------------------------------------------------------------------- LLM (App-3)
# The explanation layer (backend/explain/) is optional: with no model reachable the app uses the fixed template.
LLM_KEY_NAME = "MACHINESENSE_LLM_API_KEY"


def llm_base_url() -> str:
    return os.environ.get("MACHINESENSE_LLM_BASE_URL") or "http://localhost:1234/v1"


def llm_model() -> str:
    return os.environ.get("MACHINESENSE_LLM_MODEL") or "google/gemma-4-e4b"


def llm_timeout_s() -> float:
    return float(os.environ.get("MACHINESENSE_LLM_TIMEOUT_S") or 20)


def llm_enabled() -> bool:
    return (os.environ.get("MACHINESENSE_LLM_ENABLED") or "true").strip().lower() not in ("0", "false", "no", "off")


def llm_api_key(env_file: Path | None = None) -> str | None:
    """An API key, if any, is read from SystemCode/.env ONLY (never from the process environment); LM Studio needs none."""
    p = Path(env_file or SYSTEM_CODE_DIR / ".env")
    if not p.exists():
        return None
    for line in p.read_text().splitlines():
        k, _, v = line.partition("=")
        if k.strip() == LLM_KEY_NAME and v.strip():
            return v.strip().strip('"').strip("'")
    return None


def explanations_log() -> Path:
    return _p("MACHINESENSE_EXPLANATIONS_LOG", BACKEND_DIR / "data" / "explanations.jsonl")


# ----------------------------------------------------------------------------------------------- assistant (App-7)
# The fixed budget of every investigation (CLAUDE.md §11 rule 3); when it runs out the answer falls back to the report.
def agent_max_tool_calls() -> int:
    return int(os.environ.get("MACHINESENSE_AGENT_MAX_TOOL_CALLS") or 6)


def agent_max_revisions() -> int:
    return int(os.environ.get("MACHINESENSE_AGENT_MAX_REVISIONS") or 1)


def agent_timeout_s() -> float:
    return float(os.environ.get("MACHINESENSE_AGENT_TIMEOUT_S") or 60)
