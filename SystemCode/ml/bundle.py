"""Model bundle v1 (schema "1.0", Stage ML-4): what the app loads, pinned by hashes. Torch is imported lazily (only by
load_bundle), so this module stays importable in LightGBM processes; LightGBM is never part of the bundle.

artifacts/bundle/ (gitignored): cnn6.safetensors, ae6.safetensors, mahalanobis.safetensors (state_dict / arrays only),
lr_r1.joblib and b1s_vib.joblib (scikit-learn / numpy objects, documented inside), thresholds.json, normalisation.json,
rule.json, model_config.json, inference_config.json (the complete Config used at inference, Stage ML-4.1),
feature_dictionary.csv and manifest.json with a sha256 for every file. verify_bundle() refuses a missing, extra or
altered file and a packaged configuration whose config hash differs from the manifest's; load_bundle() verifies before
loading anything. Inference uses the packaged configuration, never the repo's current configs/."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict

SCHEMA_VERSION = "1.0"
MANIFEST = "manifest.json"


class BundleIntegrityError(RuntimeError):
    """The bundle on disk does not match its manifest (missing, extra or altered file, or wrong schema)."""


class FileEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str
    file: str  # relative to the bundle directory
    role: str
    sha256: str
    description: str


class Manifest(BaseModel):
    """Everything inference needs to reproduce a verdict, pinned by hashes."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = SCHEMA_VERSION
    created_at: str
    git_sha: str  # the committed implementation revision (HEAD)
    unrelated_worktree_changes: list[str]  # working-tree changes present when the manifest was written (listed, never hidden)
    inference_config_sha256: str  # config_sha256 of the packaged Config (inference_config.json)
    preprocess_config_sha256: str
    split_hash: str
    fs_hz: int
    channels: list[str]
    class10: list[str]
    class6: list[str]
    representation: dict[str, Any]
    files: list[FileEntry]
    thresholds: dict[str, Any]
    rule: dict[str, Any]
    data: dict[str, Any]
    test_eligibility: dict[str, Any]
    limitations: list[str]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def file_entries(directory: Path, spec: dict[str, tuple[str, str, str]]) -> list[FileEntry]:
    """spec: name → (file, role, description); hashes are computed from disk."""
    return [FileEntry(name=n, file=f, role=r, sha256=sha256_file(Path(directory) / f), description=d)
            for n, (f, r, d) in spec.items()]


def write_manifest(directory: Path, manifest: Manifest) -> Path:
    path = Path(directory) / MANIFEST
    path.write_text(json.dumps(manifest.model_dump(mode="json"), indent=1, ensure_ascii=False))
    return path


def verify_bundle(directory: Path) -> Manifest:
    """Check the schema version and every file hash; refuse missing, extra or altered files."""
    directory = Path(directory)
    path = directory / MANIFEST
    if not path.exists():
        raise BundleIntegrityError(f"no {MANIFEST} in {directory}")
    manifest = Manifest.model_validate(json.loads(path.read_text()))
    if manifest.schema_version != SCHEMA_VERSION:
        raise BundleIntegrityError(f"schema {manifest.schema_version!r}, expected {SCHEMA_VERSION!r}")
    listed = {e.file for e in manifest.files}
    present = {p.name for p in directory.iterdir() if p.is_file() and p.name != MANIFEST and not p.name.startswith(".")}
    if present - listed:
        raise BundleIntegrityError(f"files not in the manifest: {sorted(present - listed)}")
    for e in manifest.files:
        f = directory / e.file
        if not f.exists():
            raise BundleIntegrityError(f"missing bundle file {e.file}")
        if sha256_file(f) != e.sha256:
            raise BundleIntegrityError(f"sha256 mismatch for {e.file}; refusing to load")
    if config_hash(packaged_config(directory, manifest)) != manifest.inference_config_sha256:
        raise BundleIntegrityError("the packaged inference configuration does not match the manifest's config hash")
    return manifest


def packaged_config(directory: Path, manifest: Manifest | None = None):
    """The Config packaged in the bundle (inference_config.json), validated."""
    from ml.config import Config

    files = {e.name: e.file for e in (manifest.files if manifest else [])}
    return Config.model_validate(json.loads((Path(directory) / files.get("inference_config", "inference_config.json")).read_text()))


def config_hash(cfg) -> str:
    from ml.config import config_sha256
    return config_sha256(cfg)


def bundle_config(directory: Path):
    """Verify the bundle (hashes and config hash) and return its packaged Config, without loading any model."""
    return packaged_config(directory, verify_bundle(directory))


@dataclass
class Bundle:
    """A verified, loaded bundle (CPU by default)."""
    directory: Path
    manifest: Manifest
    manifest_sha256: str
    cnn: Any
    ae: Any
    mahalanobis: dict
    lr: dict
    b1s: dict
    thresholds: dict
    normalisation: dict
    rule: dict
    model_config: dict
    config: Any  # the packaged ml.config.Config
    feature_dictionary: Any  # pandas DataFrame indexed by feature name


def load_bundle(directory: Path, device: str = "cpu") -> Bundle:
    """Verify every hash, then load the models (weights via safetensors, sklearn objects via joblib)."""
    import joblib
    from safetensors.numpy import load_file as load_np
    from safetensors.torch import load_file

    from ml.config import AeConfig, CnnConfig
    from ml.models.ae import build_ae
    from ml.models.cnn import ScratchCNN

    directory = Path(directory)
    manifest = verify_bundle(directory)
    files = {e.name: directory / e.file for e in manifest.files}
    cfg = json.loads(files["model_config"].read_text())
    norm = json.loads(files["normalisation"].read_text())
    cnn = ScratchCNN(len(norm["cnn6"]["channels"]), len(manifest.class10), CnnConfig(**cfg["cnn"]))
    cnn.load_state_dict(load_file(str(files["cnn6"]), device=device))
    ae = build_ae("frame", len(norm["ae6"]["channels"]), AeConfig(**cfg["ae"]))
    ae.load_state_dict(load_file(str(files["ae6"]), device=device))
    maha = load_np(str(files["mahalanobis"]))
    import pandas as pd
    return Bundle(directory=directory, manifest=manifest, manifest_sha256=sha256_file(directory / MANIFEST),
                  cnn=cnn.to(device).eval(), ae=ae.to(device).eval(),
                  mahalanobis={"classes": norm["mahalanobis"]["classes"], "centroids": maha["centroids"],
                               "precision": maha["precision"]},
                  lr=joblib.load(files["lr_r1"]), b1s=joblib.load(files["b1s_vib"]),
                  thresholds=json.loads(files["thresholds"].read_text()), normalisation=norm,
                  rule=json.loads(files["rule"].read_text()), model_config=cfg,
                  config=packaged_config(directory, manifest),
                  feature_dictionary=pd.read_csv(files["feature_dictionary"]).set_index("name"))


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")
