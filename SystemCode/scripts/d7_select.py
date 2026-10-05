"""Stage D7: sweep every eligible development recording through the app's engine (the verdict the app would show), save
results/d7/pool_verdicts.csv, then select the 15 pilot cases (ml.pilot.select_cases) into results/d7/cases.csv. Stops and
reports the counts if a stratum is short (CLAUDE.md §7 D7 item 1). Run from SystemCode/:
    PYTHONPATH=. python scripts/d7_select.py            # sweep (~15 min) + select
    PYTHONPATH=. python scripts/d7_select.py --reuse    # select from the saved sweep
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

import pandas as pd

from ml.paths import SYSTEM_CODE_DIR
from ml.pilot import SEED, InsufficientStratum, select_cases, stratum_counts

OUT = SYSTEM_CODE_DIR / "results" / "d7"


def acquisition_substrings() -> list[str]:
    from ml.guidance import load_entries
    e = next(e for e in load_entries() if e.id == "acquisition-and-sensor-checks")
    return [w.lower() for s in e.steps if not isinstance(s, str) for w in s.when_warning]


def sweep() -> tuple[pd.DataFrame, dict]:
    from backend.checks import assessment
    from backend.engine import Engine
    from backend.unseen import split_eligible
    from ml.inference import _metadata
    meta = _metadata().reset_index()
    pool = sorted(meta.loc[meta.apply(split_eligible, axis=1), "rec_id"])
    eng = Engine(cases_path=Path(tempfile.mkdtemp()) / "d7_sweep.db")
    subs = acquisition_substrings()
    cls = meta.set_index("rec_id").class10
    rows, t0 = [], time.perf_counter()
    for i, rec in enumerate(pool):
        r = eng.diagnose(rec_id=rec)
        d, w = r["diagnosis"], r["signal_quality"]["warnings"]
        a = assessment(r)
        rows.append({"rec_id": rec, "class10": cls[rec], "decision": d["decision"], "fault_class": d["fault_class"],
                     "caution": d["caution"] or "", "variants_disagree": a["variants_disagree"],
                     "cnn_lr_agree": d["top_class"] == d["lr_secondary"]["top_class"], "narrow": bool(a["narrow_notes"]),
                     "acquisition_warning": any(s in x.lower() for x in w for s in subs), "warnings": " | ".join(w)})
        if i % 100 == 0:
            print(f"{i}/{len(pool)} {time.perf_counter() - t0:.0f} s", flush=True)
    return pd.DataFrame(rows), {"n_eligible": len(pool), "bundle_manifest_sha256": eng.bundle.manifest_sha256,
                                "sweep_seconds": round(time.perf_counter() - t0, 1)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reuse", action="store_true", help="select from the saved results/d7/pool_verdicts.csv")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.reuse:
        verdicts = pd.read_csv(OUT / "pool_verdicts.csv", keep_default_na=False)
    else:
        verdicts, info = sweep()
        verdicts.to_csv(OUT / "pool_verdicts.csv", index=False)
        (OUT / "pool_verdicts_info.json").write_text(json.dumps(info, indent=1))
    print(json.dumps(stratum_counts(verdicts), indent=1))
    try:
        cases = select_cases(verdicts, SEED)
    except InsufficientStratum as e:
        print("STOP: a stratum has too few eligible cases; nothing is filled another way (CLAUDE.md §7 D7 item 1)")
        print(json.dumps(e.counts, indent=1))
        (OUT / "selection_stopped.json").write_text(json.dumps(e.counts, indent=1))
        return 2
    cases.to_csv(OUT / "cases.csv", index=False)
    print(cases.to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
