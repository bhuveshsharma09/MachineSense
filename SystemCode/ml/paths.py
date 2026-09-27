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
DATA_DIR_ENV = "MACHINESENSE_DATA_DIR"        # derived data (raw_npy/, cache/, …); default SystemCode/data
BUNDLE_DIR_ENV = "MACHINESENSE_BUNDLE_DIR"    # the approved model bundle; default SystemCode/artifacts/bundle
REQUIRE_BUNDLE_ENV = "MACHINESENSE_REQUIRE_BUNDLE"


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
    """Gitignored derived data: raw_npy/, windows.parquet, cache/. $MACHINESENSE_DATA_DIR, else SystemCode/data."""
    env = os.environ.get(DATA_DIR_ENV)
    return _resolve(env) if env else SYSTEM_CODE_DIR / "data"


def bundle_dir() -> Path:
    """The approved model bundle: $MACHINESENSE_BUNDLE_DIR, else SystemCode/artifacts/bundle."""
    env = os.environ.get(BUNDLE_DIR_ENV)
    return _resolve(env) if env else artifacts_dir() / "bundle"


def require_bundle() -> bool:
    """MACHINESENSE_REQUIRE_BUNDLE=1: bundle-dependent tests fail instead of skipping."""
    return os.environ.get(REQUIRE_BUNDLE_ENV, "") == "1"


def artifacts_dir() -> Path:
    return SYSTEM_CODE_DIR / "artifacts"


def results_dir() -> Path:
    return SYSTEM_CODE_DIR / "results"


def cache_file(name: str, *sections: BaseModel, ext: str) -> Path:
    """data/cache/<name>_<sha12>.<ext>, named by the sha256 of the config sections that produced it,
    so a changed setting can never silently reuse a stale cache."""
    sha = hashlib.sha256("".join(config_sha256(s) for s in sections).encode()).hexdigest()[:12]
    return data_dir() / "cache" / f"{name}_{sha}.{ext}"


class RunPaths:
    """Every output path of one trained variant × seed (× fold), Stage 3B on. The name `<variant>_s<seed>_<protocol>`
    appears in every score, embedding, weight, normalisation-stat and result path and in the MLflow run names, so two
    runs that differ only in seed (or variant) never write to the same file. Stage 3A's seed-42 CNN outputs keep their
    original paths (they predate this scheme) and are never overwritten by it."""

    def __init__(self, family: str, variant: str, seed: int, protocol: str = "dev", withheld: str | None = None):
        self.family, self.variant, self.seed, self.protocol, self.withheld = family, variant, int(seed), protocol, withheld
        self.name = f"{variant}_s{int(seed)}_{protocol}" if withheld is None else \
            f"{variant}_w-{withheld}_s{int(seed)}_{protocol}"

    def run_name(self, fold: int | None = None) -> str:
        return self.name if fold is None else f"{self.name}_fold{int(fold)}"

    def tags(self, fold: int | None = None) -> dict[str, str]:
        t = {"family": self.family, "variant": self.variant, "seed": str(self.seed), "protocol": self.protocol}
        if self.withheld is not None:
            t["withheld"] = self.withheld
        return t if fold is None else {**t, "fold": str(int(fold))}

    def scores(self, kind: str = "scores") -> Path:
        """results/scores/<name>_<kind>.parquet (kind: scores, logits, channel_errors, …)."""
        return results_dir() / "scores" / f"{self.name}_{kind}.parquet"

    def embeddings(self, fold: int | None = None) -> Path:
        tail = "" if fold is None else f"_fold{int(fold)}"
        return results_dir() / "scores" / f"{self.name}{tail}_embeddings.npz"

    def weights(self, fold: int) -> Path:
        return artifacts_dir() / self.family / self.name / f"fold{int(fold)}.pt"

    def norm_stats(self, fold: int) -> Path:
        return results_dir() / self.family / self.name / f"fold{int(fold)}_norm.json"

    def fold_result(self, fold: int, kind: str) -> Path:
        """results/<family>/<name>/fold<k>_<kind>.json."""
        return results_dir() / self.family / self.name / f"fold{int(fold)}_{kind}.json"

    def result(self, kind: str) -> Path:
        """results/<family>/<name>_<kind>.json (kind: train, eval, …)."""
        return results_dir() / self.family / f"{self.name}_{kind}.json"

    def all_paths(self, folds=(0, 1, 2)) -> list[Path]:
        out = [self.scores(), self.scores("logits"), self.scores("channel_errors"), self.embeddings(),
               self.result("train"), self.result("eval")]
        return out + [p for k in folds for p in (self.weights(k), self.norm_stats(k), self.embeddings(k),
                                                 self.scores(f"fold{k}_logits"), self.fold_result(k, "train"))]
