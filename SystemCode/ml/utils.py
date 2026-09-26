"""Seeding, device selection and MLflow run setup."""
from __future__ import annotations

import os
import random
import subprocess
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np

from ml import paths
from ml.config import Config, config_sha256

DEVICE_ENV = "MACHINESENSE_DEVICE"


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy and torch (CPU, CUDA, MPS). MPS is still not bit-deterministic."""
    import torch

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def get_device():
    """torch.device from $MACHINESENSE_DEVICE (mps | cuda | cpu); defaults to cpu for inference, tests and the app."""
    import torch

    name = os.environ.get(DEVICE_ENV, "cpu").strip().lower()
    if name not in ("cpu", "mps", "cuda"):
        raise ValueError(f"{DEVICE_ENV}={name!r}; expected mps, cuda or cpu")
    if name == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError(f"{DEVICE_ENV}=mps but MPS is not available on this machine")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"{DEVICE_ENV}=cuda but CUDA is not available on this machine")
    return torch.device(name)


def git_sha() -> str:
    """Current commit of the repo, suffixed '-dirty' if tracked files have uncommitted changes."""
    try:
        sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=paths.REPO_ROOT, capture_output=True,
                             text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=paths.REPO_ROOT,
                               capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return f"{sha}-dirty" if dirty else sha


def _flatten(d: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(_flatten(v, f"{key}."))
        else:
            out[key] = v
    return out


@contextmanager
def mlflow_run(cfg: Config, run_name: str | None = None, experiment: str = "machinesense",
               tracking_uri: str | None = None, artifact_root: Path | None = None) -> Iterator[Any]:
    """Start an MLflow run that logs the resolved config, its sha256s, the seed and the git sha.

    Called from scripts and notebooks only (it writes to the tracking store). Defaults to
    sqlite:///mlruns.db and mlruns/ in SystemCode/; tests pass a temp tracking_uri and artifact_root.
    """
    import mlflow

    mlflow.set_tracking_uri(tracking_uri or paths.mlflow_uri(cfg))
    exp = mlflow.get_experiment_by_name(experiment)
    if exp is None:
        root = Path(artifact_root or paths.mlruns_dir()).resolve()
        exp_id = mlflow.create_experiment(experiment, artifact_location=root.as_uri())
    else:
        exp_id = exp.experiment_id

    resolved = cfg.model_dump(mode="json")
    with mlflow.start_run(experiment_id=exp_id, run_name=run_name) as run:
        mlflow.log_params({k: str(v) for k, v in _flatten(resolved).items()})
        mlflow.set_tags({
            "git_sha": git_sha(),
            "config_sha256": config_sha256(cfg),
            "preprocess_sha256": config_sha256(cfg.preprocess),
            "seed": str(cfg.seed),
        })
        mlflow.log_dict(resolved, "config.json")
        yield run
