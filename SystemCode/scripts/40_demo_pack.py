"""Build the committed demo pack (SystemCode/demo/) so a fresh clone can run the app without MAFAULDA: 5 development
recordings, the healthy training recordings their channel evidence is compared with, the precomputed healthy reference
values of every healthy training recording (so an uploaded CSV at any speed gets channel evidence), and manifest.json
with each file's sha256 and the verdict the full setup gives. Run from SystemCode/ on the full data:
    PYTHONPATH=. python scripts/40_demo_pack.py

Every shipped recording is checked with the existing split code and the reserved-fingerprint guard: it must be in the
development pool, no reserved-test column may be "test", and the guard must not match its array. Nothing reserved is
read: the guard compares fingerprints with the existing index file.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

DEMO = [  # (rec_id, why it is in the pack); the report walkthrough's cases (Section 7.5)
    ("normal__24.576", "healthy"),
    ("imbalance__30g__19.456", "known fault: imbalance"),
    ("underhang__cage_fault__6g__35.0208", "known fault: bearing (envelope view)"),
    ("imbalance__10g__13.9264", "further inspection needed"),
    ("horizontal-misalignment__1.0mm__24.9856", "known fault: horizontal misalignment"),
]
RESERVED_COLUMNS = ("split_closed", "split_severity", "split_speed_block")
NOTE = ("Demo pack: 5 development recordings (closed validation split, used for threshold calibration, so their verdicts "
        "are in-sample) and the healthy training recordings their channel evidence is compared with. They are not test "
        "data; reserved test recordings are never shipped.")


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main() -> int:
    from backend.engine import Engine
    from backend.evidence import REFERENCE_VALUES, training_values
    from backend.guard import reserved_rec_ids
    from backend.settings import demo_dir, demo_mode, raw_dir
    from ml.config import config_sha256
    from ml.inference import _metadata

    if demo_mode() or not raw_dir().exists():
        print("the full development recordings (data/raw_npy) are needed to build the demo pack", file=sys.stderr)
        return 1
    eng = Engine(cases_path=Path(tempfile.mkdtemp()) / "demo_pack.db")
    cfg, meta = eng.bundle.config, _metadata().reset_index()
    pool, reserved = set(eng.pool().rec_id), set(reserved_rec_ids(meta, cfg))
    m = meta.set_index("rec_id")
    healthy_training = sorted(m[(m.index.isin(pool)) & (m.split_closed == "train") & (m.class10 == "normal")].index)

    def check(rec_id: str) -> None:
        row = m.loc[rec_id]
        problems = [p for p, bad in [("not in the development pool", rec_id not in pool), ("reserved", rec_id in reserved),
                                     *[(f"{c} = test", row[c] == "test") for c in RESERVED_COLUMNS],
                                     ("matches a reserved fingerprint", eng.guard.match(np.load(raw_dir() / f"{rec_id}.npy")) is not None)]
                    if bad]
        if problems:
            raise SystemExit(f"{rec_id}: {', '.join(problems)}; not shipped")

    out = demo_dir() / "raw_npy"
    out.mkdir(parents=True, exist_ok=True)
    recordings, refs = [], {}
    for rec_id, why in DEMO:
        check(rec_id)
        r = eng.diagnose(rec_id=rec_id)
        ref = eng.channels(r["case_id"])["reference"]["rec_id"]
        if ref not in healthy_training:
            raise SystemExit(f"{rec_id}: reference {ref} is not a healthy training recording")
        refs.setdefault(ref, []).append(rec_id)
        d, row = r["diagnosis"], m.loc[rec_id]
        recordings.append({"rec_id": rec_id, "why": why, "class10": row.class10, "speed_hz": float(row.fr_measured_hz),
                           **{c: row[c] for c in RESERVED_COLUMNS}, "used_for": "threshold calibration",
                           "expected": {"decision": d["decision"], "display_label": d["display_label"],
                                        "top_class": d["top_class"], "caution": d["caution"]},
                           "healthy_reference": ref})
    for ref in refs:
        check(ref)
    keep = {f"{x['rec_id']}.npy" for x in recordings} | {f"{r}.npy" for r in refs}
    for p in out.glob("*.npy"):
        if p.name not in keep:
            p.unlink()
    for name in sorted(keep):
        shutil.copyfile(raw_dir() / name, out / name)
    files = {name: sha256(out / name) for name in sorted(keep)}
    for x in recordings:
        x["sha256"] = files[f"{x['rec_id']}.npy"]

    values = {"note": "channel_values (backend.evidence) of every healthy training recording, computed from data/raw_npy "
                      "with the bundle configuration; used only when that recording's raw file is not on disk",
              "config_sha256": config_sha256(cfg), "values": {r: training_values(r, cfg) for r in healthy_training}}
    (demo_dir() / REFERENCE_VALUES).write_text(json.dumps(values, indent=1))
    manifest = {"note": NOTE, "bundle_manifest_sha256": eng.bundle.manifest_sha256, "config_sha256": config_sha256(cfg),
                "recordings": recordings,
                "healthy_references": [{"rec_id": r, "class10": "normal", "used_for": "model training (healthy)",
                                        "split_closed": m.loc[r].split_closed, "reference_for": refs[r],
                                        "sha256": files[f"{r}.npy"]} for r in sorted(refs)],
                "healthy_reference_values": {"file": REFERENCE_VALUES, "recordings": healthy_training,
                                             "sha256": sha256(demo_dir() / REFERENCE_VALUES)}}
    (demo_dir() / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    size = sum((out / n).stat().st_size for n in keep) / 1e6
    print(f"{len(recordings)} demo recordings + {len(refs)} healthy references ({size:.1f} MB) → {out}")
    for x in recordings:
        print(f"  {x['rec_id']}: {x['expected']['display_label']} (reference {x['healthy_reference']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
