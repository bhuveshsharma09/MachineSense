"""Identity of derived artifacts: every cache is named by the sha256 of everything that changes its values, and a
cache whose identity does not match the current code, config and source data fails loudly on load.

Normalisation statistics are never cached here; they are fitted per track at training time.
"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path

import pandas as pd

from ml.config import PreprocessConfig


class CacheMismatchError(RuntimeError):
    """A cache was built from a different config, feature definition, source recording or track."""


def feature_cache_key(pre: PreprocessConfig, npy_sha256: Mapping[str, str], track: str | None = None) -> str:
    """sha256 over: the full preprocess config (cleaning, band-stop, envelope band, windows, feature settings and
    feature version), the feature definitions (feature_dictionary), the source recordings' npy checksums, and the
    track ("shared" for track-independent features)."""
    from ml.features import feature_dictionary

    payload = {
        "preprocess": pre.model_dump(mode="json"),
        "feature_version": pre.features.version,
        "feature_definitions": hashlib.sha256(feature_dictionary(pre).to_csv(index=False).encode()).hexdigest(),
        "sources": sorted([rid, sha] for rid, sha in npy_sha256.items()),
        "track": track or "shared",
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def feature_cache_path(directory: Path, name: str, key: str, track: str | None = None) -> Path:
    return Path(directory) / f"{name}_{track or 'shared'}_{key[:12]}.parquet"


def write_feature_cache(df: pd.DataFrame, path: Path, key: str, extra: Mapping[str, str]) -> None:
    """Parquet with the cache key and provenance (split hash, git sha, …) in the file's schema metadata."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    table = pa.Table.from_pandas(df, preserve_index=False)
    meta = {**(table.schema.metadata or {}), b"machinesense.cache_key": key.encode(),
            **{f"machinesense.{k}".encode(): str(v).encode() for k, v in extra.items()}}
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table.replace_schema_metadata(meta), path)


def read_cache_metadata(path: Path) -> dict[str, str]:
    import pyarrow.parquet as pq

    raw = pq.read_schema(path).metadata or {}
    return {k.decode()[len("machinesense."):]: v.decode() for k, v in raw.items() if k.startswith(b"machinesense.")}


def read_feature_cache(path: Path, pre: PreprocessConfig, npy_sha256: Mapping[str, str],
                       track: str | None = None) -> pd.DataFrame:
    """Load a feature cache, recomputing its key from the current config and the checksums of the recordings it
    contains. Raises CacheMismatchError if anything differs."""
    meta = read_cache_metadata(path)
    df = pd.read_parquet(path)
    ids = sorted(df.rec_id.unique())
    missing = [r for r in ids if r not in npy_sha256]
    if missing:
        raise CacheMismatchError(f"{path.name}: no checksum for {len(missing)} recordings, e.g. {missing[:3]}")
    expected = feature_cache_key(pre, {r: npy_sha256[r] for r in ids}, track)
    stored = meta.get("cache_key")
    if stored != expected:
        raise CacheMismatchError(f"{path.name}: cache key {str(stored)[:12]}… does not match the current config, "
                                 f"feature definitions, source checksums or track ({expected[:12]}…); rebuild it")
    if not Path(path).name.endswith(f"_{expected[:12]}.parquet"):
        raise CacheMismatchError(f"{path.name}: file name does not carry the key {expected[:12]}")
    return df
