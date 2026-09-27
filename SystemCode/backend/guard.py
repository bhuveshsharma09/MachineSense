"""Upload guard (best effort): refuse uploads that appear to be reserved test recordings.

A GUARD, NOT A GUARANTEE: it recognises reserved recordings uploaded as-is or re-exported with the dataset's CSV
precision (5 significant figures); a modified copy (resampled, cropped, rescaled, noise added) is not recognised.

Fingerprint of a (250000, 8) float32 array:
- sha256 of the float32 bytes (exact copies);
- a stats signature: sha256 of the first 1,000 samples of every channel rounded to CSV precision (5 significant
  figures), confirmed by the per-channel mean and standard deviation (relative tolerance 1e-3).
The index of reserved recordings is built read-only from SystemCode/data/raw_npy (`python -m backend.guard build`).
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

HEAD = 1000
SIG_FIGS = 5
STATS_RTOL = 1e-3


def round_sig(x: np.ndarray, sig: int = SIG_FIGS) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    mag = np.floor(np.log10(np.where(x == 0, 1.0, np.abs(x))))
    scale = 10.0 ** (sig - 1 - mag)
    return np.round(x * scale) / scale


def fingerprint(x: np.ndarray) -> dict:
    a = np.ascontiguousarray(x, dtype=np.float32)
    head = round_sig(a[:HEAD]).astype(np.float64)
    return {"sha256": hashlib.sha256(a.tobytes()).hexdigest(),
            "head_sha256": hashlib.sha256(np.ascontiguousarray(head).tobytes()).hexdigest(),
            "mean": a.astype(np.float64).mean(axis=0).tolist(), "std": a.astype(np.float64).std(axis=0).tolist()}


def reserved_rec_ids(meta, cfg) -> list[str]:
    from ml.splits import development_pool
    pool = set(development_pool(meta, cfg.development.allowed_closed, cfg.development.reserved_columns).rec_id)
    return sorted(set(meta.rec_id) - pool)


def build_index(rec_ids, raw_dir: Path, out: Path) -> dict:
    """Read-only over the raw npy files; writes {rec_id: fingerprint}."""
    idx = {}
    for r in rec_ids:
        p = Path(raw_dir) / f"{r}.npy"
        if p.exists():
            idx[r] = fingerprint(np.load(p))
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"n": len(idx), "fingerprints": idx}))
    return idx


class ReservedGuard:
    def __init__(self, index: dict):
        self.index = index
        self.by_sha = {v["sha256"]: k for k, v in index.items()}
        self.by_head = {}
        for k, v in index.items():
            self.by_head.setdefault(v["head_sha256"], []).append(k)

    @classmethod
    def load(cls, path: Path) -> "ReservedGuard":
        return cls(json.loads(Path(path).read_text())["fingerprints"])

    def match(self, x: np.ndarray) -> str | None:
        """The reserved rec_id the array appears to be, or None."""
        fp = fingerprint(x)
        if fp["sha256"] in self.by_sha:
            return self.by_sha[fp["sha256"]]
        for r in self.by_head.get(fp["head_sha256"], []):
            v = self.index[r]
            if np.allclose(fp["mean"], v["mean"], rtol=STATS_RTOL, atol=1e-6) and \
               np.allclose(fp["std"], v["std"], rtol=STATS_RTOL, atol=1e-6):
                return r
        return None


def main() -> int:
    import pandas as pd

    from backend.settings import guard_index, raw_dir
    from ml.config import load_config
    from ml.paths import DEFAULT_CONFIG, SYSTEM_CODE_DIR

    if len(sys.argv) < 2 or sys.argv[1] != "build":
        print("usage: python -m backend.guard build")
        return 2
    meta = pd.read_csv(SYSTEM_CODE_DIR / "metadata.csv")
    ids = reserved_rec_ids(meta, load_config(DEFAULT_CONFIG))
    idx = build_index(ids, raw_dir(), guard_index())
    print(f"{len(idx)} reserved recordings fingerprinted → {guard_index()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
