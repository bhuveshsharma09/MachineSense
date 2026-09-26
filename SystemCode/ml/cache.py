"""Identity of derived artifacts: every cache is named by the sha256 of everything that changes its values, and a
cache whose identity does not match the current code, config and source data fails loudly on load.

Feature caches never contain split assignments: those are joined from the current metadata.csv at load time, so a
split change can never be hidden inside a cache. Any artifact that does record a split hash (for example a score
file) is checked against the current split hash on load. Normalisation statistics are never cached here; they are
fitted per track at training time.
"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import NamedTuple

import pandas as pd

from ml.config import PreprocessConfig

SPLIT_PREFIXES = ("split_", "loco_", "cv_fold")
JOIN_COLUMNS = ("class10", "class6", "fault_family", "fault_type", "severity", "fr_measured_hz")


class CacheMismatchError(RuntimeError):
    """A cache was built from a different config, feature definition, source recording, speed or track."""


class SplitHashMismatchError(RuntimeError):
    """An artifact was produced under different split assignments than the current metadata.csv."""


class Source(NamedTuple):
    """What a recording contributes to a feature cache's identity."""

    npy_sha256: str
    fr_hz: float  # the supplied recording speed (fr_measured_hz), used as the per-window fallback speed


def sources_from_metadata(meta: pd.DataFrame, rec_ids=None) -> dict[str, Source]:
    m = meta if rec_ids is None else meta[meta.rec_id.isin(set(rec_ids))]
    return {r: Source(s, float(f)) for r, s, f in zip(m.rec_id, m.npy_sha256, m.fr_measured_hz)}


def feature_cache_key(pre: PreprocessConfig, sources: Mapping[str, Source], track: str | None = None) -> str:
    """sha256 over: the preprocess config except the representation-only `spectro` section (cleaning, band-stop,
    low-pass, envelope band, windows, feature settings and version), the feature definitions (feature_dictionary), each source recording's npy checksum AND supplied
    speed, and the track ("shared" for track-independent features)."""
    from ml.features import feature_dictionary

    payload = {
        # R1 features do not depend on the representation-only `spectro` section, so it is left out of their key.
        "preprocess": pre.model_dump(mode="json", exclude={"spectro"}),
        "feature_version": pre.features.version,
        "feature_definitions": hashlib.sha256(feature_dictionary(pre).to_csv(index=False).encode()).hexdigest(),
        "sources": sorted([rid, s.npy_sha256, repr(float(s.fr_hz))] for rid, s in sources.items()),
        "track": track or "shared",
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def feature_cache_path(directory: Path, name: str, key: str, track: str | None = None) -> Path:
    """<name>_<key12>.parquet for track-independent features; <name>_<track>_<key12>.parquet per track."""
    return Path(directory) / (f"{name}_{track}_{key[:12]}.parquet" if track else f"{name}_{key[:12]}.parquet")


def write_feature_cache(df: pd.DataFrame, path: Path, key: str, extra: Mapping[str, str]) -> None:
    """Parquet with the cache key and provenance (git sha, …) in the file's schema metadata. Refuses split columns."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    bad = [c for c in df.columns if c.startswith(SPLIT_PREFIXES)]
    if bad:
        raise ValueError(f"feature caches must not contain split columns: {bad[:5]}")
    table = pa.Table.from_pandas(df, preserve_index=False)
    meta = {**(table.schema.metadata or {}), b"machinesense.cache_key": key.encode(),
            **{f"machinesense.{k}".encode(): str(v).encode() for k, v in extra.items()}}
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table.replace_schema_metadata(meta), path)


def read_cache_metadata(path: Path) -> dict[str, str]:
    import pyarrow.parquet as pq

    raw = pq.read_schema(path).metadata or {}
    return {k.decode()[len("machinesense."):]: v.decode() for k, v in raw.items() if k.startswith(b"machinesense.")}


def check_split_hash(stored: str | None, meta: pd.DataFrame, what: str) -> None:
    """Fail if an artifact recorded a split hash that differs from the current metadata.csv."""
    from ml.splits import split_hash

    if stored is not None and stored != split_hash(meta):
        raise SplitHashMismatchError(f"{what} was made with split hash {stored[:12]}…, current is "
                                     f"{split_hash(meta)[:12]}…; regenerate it")


def read_feature_cache(path: Path, pre: PreprocessConfig, sources: Mapping[str, Source],
                       track: str | None = None) -> pd.DataFrame:
    """Load a feature cache, recomputing its key from the current config and the checksums and speeds of the
    recordings it contains. Raises CacheMismatchError if anything differs."""
    meta = read_cache_metadata(path)
    df = pd.read_parquet(path)
    ids = sorted(df.rec_id.unique())
    missing = [r for r in ids if r not in sources]
    if missing:
        raise CacheMismatchError(f"{path.name}: no source record for {len(missing)} recordings, e.g. {missing[:3]}")
    expected = feature_cache_key(pre, {r: sources[r] for r in ids}, track)
    stored = meta.get("cache_key")
    if stored != expected:
        raise CacheMismatchError(f"{path.name}: cache key {str(stored)[:12]}… does not match the current config, "
                                 f"feature definitions, source checksums, speeds or track ({expected[:12]}…); rebuild it")
    if not Path(path).name.endswith(f"_{expected[:12]}.parquet"):
        raise CacheMismatchError(f"{path.name}: file name does not carry the key {expected[:12]}")
    return df


def load_features(path: Path, meta: pd.DataFrame, pre: PreprocessConfig, track: str | None = None) -> pd.DataFrame:
    """Validated feature cache with the CURRENT labels and split assignments joined from metadata.csv."""
    check_split_hash(read_cache_metadata(path).get("split_hash"), meta, path.name)
    df = read_feature_cache(path, pre, sources_from_metadata(meta), track)
    cols = ["rec_id", *JOIN_COLUMNS, *[c for c in meta.columns if c.startswith(SPLIT_PREFIXES)]]
    return df.merge(meta[cols], on="rec_id", how="left", validate="many_to_one")


# ----------------------------------------------------------------------------- representation cache
def representation_cache_key(pre: PreprocessConfig, sources: Mapping[str, Source], candidate: str, version: str,
                             track: str | None = None) -> str:
    """sha256 over: the FULL preprocess config (cleaning, band-stop, low-pass, windows, speed and every spectro/Mel
    setting), the candidate name, the representation version, each source recording's npy checksum and supplied speed,
    and the track ("shared")."""
    payload = {"preprocess": pre.model_dump(mode="json"), "candidate": candidate, "version": version,
               "sources": sorted([rid, s.npy_sha256, repr(float(s.fr_hz))] for rid, s in sources.items()),
               "track": track or "shared"}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
