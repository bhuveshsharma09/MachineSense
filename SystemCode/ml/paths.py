"""Resolve the data root and project directories from config and env vars; nothing is hard-coded."""
from __future__ import annotations

import hashlib
import os
from pathlib import Path

from pydantic import BaseModel

from ml.config import Config, config_sha256, load_config

SYSTEM_CODE_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = SYSTEM_CODE_DIR.parent
CONFIGS_DIR = SYSTEM_CODE_DIR / "configs"
DEFAULT_CONFIG = CONFIGS_DIR / "base.yaml"

DATA_ROOT_ENV = "MACHINESENSE_DATA_ROOT"


def _resolve(p: str | Path) -> Path:
    """Expand ~ and make relative paths relative to SystemCode/, not the current working directory."""
    p = Path(p).expanduser()
    return (p if p.is_absolute() else SYSTEM_CODE_DIR / p).resolve()


def data_root(cfg: Config | None = None) -> Path:
    """Folder that directly contains normal/, imbalance/, ...: $MACHINESENSE_DATA_ROOT, else the config."""
    env = os.environ.get(DATA_ROOT_ENV)
    if env:
        root = _resolve(env)
    else:
        cfg = cfg or load_config(DEFAULT_CONFIG)
        root = _resolve(cfg.paths.data_root)
    if not root.is_dir():
        raise FileNotFoundError(f"data root {root} does not exist; set {DATA_ROOT_ENV} or paths.data_root")
    return root


def mlflow_uri(cfg: Config | None = None) -> str:
    """Tracking URI from the config; a relative sqlite path is anchored in SystemCode/, not the cwd."""
    cfg = cfg or load_config(DEFAULT_CONFIG)
    uri = cfg.paths.mlflow_uri
    prefix = "sqlite:///"
    if uri.startswith(prefix) and not Path(uri[len(prefix):]).is_absolute():
        return prefix + str(SYSTEM_CODE_DIR / uri[len(prefix):])
    return uri


def mlruns_dir() -> Path:
    """MLflow artifact store (gitignored)."""
    return SYSTEM_CODE_DIR / "mlruns"


def data_dir() -> Path:
    """Gitignored derived data: raw_npy/, windows.parquet, cache/."""
    return SYSTEM_CODE_DIR / "data"


def artifacts_dir() -> Path:
    return SYSTEM_CODE_DIR / "artifacts"


def results_dir() -> Path:
    return SYSTEM_CODE_DIR / "results"


def cache_file(name: str, *sections: BaseModel, ext: str) -> Path:
    """data/cache/<name>_<sha12>.<ext>, named by the sha256 of the config sections that produced it,
    so a changed setting can never silently reuse a stale cache."""
    sha = hashlib.sha256("".join(config_sha256(s) for s in sections).encode()).hexdigest()[:12]
    return data_dir() / "cache" / f"{name}_{sha}.{ext}"
