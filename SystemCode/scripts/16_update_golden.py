"""Update the golden reference of tests/test_inference.py — an explicit step, run only after a deliberate bundle or
inference change (CLAUDE.md §7): diagnose() on the fixed golden recordings, a diff summary against the stored reference,
then the new reference is written for the current bundle manifest.

    python scripts/16_update_golden.py
"""
from __future__ import annotations

import json
import sys

from ml.inference import GOLDEN_RECORDINGS, diagnose, get_bundle, stable_output
from ml.paths import SYSTEM_CODE_DIR

GOLDEN = SYSTEM_CODE_DIR / "tests" / "golden" / "diagnose_golden.json"


def leaves(o, path=""):
    if isinstance(o, dict):
        for k, v in o.items():
            yield from leaves(v, f"{path}.{k}" if path else k)
    elif isinstance(o, list):
        for i, v in enumerate(o):
            yield from leaves(v, f"{path}[{i}]")
    else:
        yield path, o


def main() -> int:
    b = get_bundle()
    new = {"manifest_sha256": b.manifest_sha256, "outputs": {r: stable_output(diagnose(r)) for r in GOLDEN_RECORDINGS}}
    old = json.loads(GOLDEN.read_text()) if GOLDEN.exists() else None
    print(f"manifest: {old['manifest_sha256'][:12] if old else '(no reference)'} → {new['manifest_sha256'][:12]}")
    for r in GOLDEN_RECORDINGS:
        if not old or r not in old["outputs"]:
            print(f"  {r}: new")
            continue
        a, c = dict(leaves(old["outputs"][r])), dict(leaves(new["outputs"][r]))
        changed = sorted(k for k in a.keys() | c.keys() if a.get(k, "<absent>") != c.get(k, "<absent>"))
        top = sorted({k.split(".")[0] + "." + k.split(".")[1].split("[")[0] if "." in k else k for k in changed})
        print(f"  {r}: {len(changed)} of {len(c)} values changed" + (f" (in {', '.join(top[:8])})" if changed else ""))
        for k in changed[:5]:
            print(f"      {k}: {str(a.get(k, '<absent>'))[:60]} → {str(c.get(k, '<absent>'))[:60]}")
    GOLDEN.parent.mkdir(parents=True, exist_ok=True)
    GOLDEN.write_text(json.dumps(new, indent=1, sort_keys=True))
    print(f"→ {GOLDEN.relative_to(SYSTEM_CODE_DIR)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
