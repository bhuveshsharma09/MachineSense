"""Showcase recordings: real development recordings chosen by querying diagnose() outputs for each app state.

    python -m backend.showcase build     # sweeps every development-pool recording (~15 min), then selects

The sweep is cached in backend/data/showcase_sweep.json (gitignored); the selection with reasons is written to
backend/showcase.json. A state with no development recording is reported as missing, never fabricated."""
from __future__ import annotations

import json
import sys
import time

from backend.settings import BACKEND_DIR, showcase_file

RANGE_WARNINGS = ()  # filled on import of ml.inference in _range_warnings()
SWEEP = BACKEND_DIR / "data" / "showcase_sweep.json"
FAMILY = {"imbalance": "imbalance", "horizontal_misalignment": "horizontal misalignment",
          "vertical_misalignment": "vertical misalignment", "underhang_ball": "underhang bearing",
          "underhang_cage": "underhang bearing", "underhang_outer": "underhang bearing",
          "overhang_ball": "overhang bearing", "overhang_cage": "overhang bearing", "overhang_outer": "overhang bearing"}


def _range_warnings() -> tuple[str, str]:
    from ml.inference import CALIBRATION_SPEED_WARNING, SPEED_WARNING
    return SPEED_WARNING, CALIBRATION_SPEED_WARNING


def sweep(engine) -> list[dict]:
    from ml.inference import diagnose
    global RANGE_WARNINGS
    RANGE_WARNINGS = _range_warnings()
    rows = []
    for r in engine.recordings():
        o = diagnose(r["rec_id"], bundle=engine.bundle, with_xai=False)
        d, q = o["diagnosis"], o["signal_quality"]
        rows.append({"rec_id": r["rec_id"], "class10": r["class10"], "used_for": r["used_for"], "speed_hz": r["speed_hz"],
                     "decision": d["decision"], "display_label": d["display_label"], "fault_class": d["fault_class"],
                     "caution": d["caution"], "top_class": d["top_class"], "lr_top": d["lr_secondary"]["top_class"],
                     "msp": d["scores"]["msp"], "rule_comparison": d["rule_comparison"], "warnings": q["warnings"],
                     "quality_warnings": [w for w in q["warnings"] if w not in RANGE_WARNINGS],
                     "scores": {k: v for k, v in d["scores"].items() if k != "ae_channel_errors"},
                     "thresholds": d["thresholds"]})
    return rows


def _assess(r: dict) -> dict:
    """backend.checks.assessment on the swept scores (the same function the report and the UI use)."""
    from backend.checks import assessment
    return assessment({"diagnosis": {"decision": r["decision"], "scores": r["scores"], "thresholds": r["thresholds"],
                                     "rule_comparison": r["rule_comparison"], "top_class": r["top_class"],
                                     "lr_secondary": {"top_class": r["lr_top"]}}})


CLEAR_MARGIN = 0.05  # App-3: the healthy example sits clearly inside every threshold


def threshold_defining() -> set:
    """Recordings that define a bundle threshold (they sit exactly on τ_p, τ_d, τ_e or τ_b)."""
    from backend.settings import bundle_dir
    t = json.loads((bundle_dir() / "thresholds.json").read_text())
    return {r for v in t.get("defining_recordings", {}).values() for r in v}


def select(rows: list[dict]) -> dict:
    """First match by name within each state (deterministic); calibration recordings preferred over training ones. One
    recording per state: a recording already shown is skipped; if a state has no other recording, the existing entry is
    labelled with both states."""
    order = sorted(rows, key=lambda r: (r["used_for"] != "threshold calibration", r["rec_id"]))
    for r in rows:
        r["_a"] = _assess(r) if "scores" in r else {"narrow_notes": [], "variants_disagree_narrow": False, "disagree": False}
    pick, missing = [], []

    def add(state, pred, reason, key=None):
        cands = [r for r in order if pred(r)]
        if key:
            cands = sorted(cands, key=key)
        if not cands:
            missing.append(state)
            return
        used = {p["rec_id"]: p for p in pick}
        fresh = [r for r in cands if r["rec_id"] not in used]
        if not fresh:
            p = used[cands[0]["rec_id"]]
            p["state"] += " + " + state
            p["states"].append(state)
            p["reason"] += "; " + reason(cands[0])
            return
        r = fresh[0]
        pick.append({"state": state, "states": [state], "rec_id": r["rec_id"], "class10": r["class10"],
                     "used_for": r["used_for"], "display_label": r["display_label"], "reason": reason(r)})

    agree = lambda r: len(set(r["rule_comparison"].values())) == 1
    clean = lambda r: not r["warnings"]
    defining = threshold_defining()
    clear = lambda r: "_a" in r and "checks" in r["_a"] and all(abs(c["margin_rel"]) >= CLEAR_MARGIN for c in r["_a"]["checks"])
    add("healthy", lambda r: r["class10"] == "normal" and r["decision"] == "healthy" and not r["caution"] and agree(r)
        and clean(r) and r["rec_id"] not in defining and clear(r),
        lambda r: "normal recording; healthy, no caution, all rule variants agree, no warnings; not a threshold-defining "
                  f"recording, every check at least {CLEAR_MARGIN:.0%} from its threshold (MSP on the 1 − MSP scale)")
    for fam in dict.fromkeys(FAMILY.values()):
        add(f"known fault · {fam}", lambda r, fam=fam: r["decision"] == "known_fault" and r["fault_class"] == r["class10"]
            and FAMILY.get(r["class10"]) == fam and not r["caution"] and agree(r) and r["top_class"] == r["lr_top"] and clean(r),
            lambda r: f"{r['class10']}: known fault matching its label; rule variants, CNN-6 and LR agree; no warnings",
            key=lambda r: (r["used_for"] != "threshold calibration", -r["msp"], r["rec_id"]))
    add("healthy with caution", lambda r: r["decision"] == "healthy" and r["caution"],
        lambda r: f"{r['class10']} verdict healthy with caution '{r['caution']}'")
    add("further inspection needed", lambda r: r["decision"] == "unknown",
        lambda r: f"{r['class10']} recording; the classifier checks did not pass, so the verdict is unknown")
    add("rule variants disagree", lambda r: not agree(r),
        lambda r: "decision-rule variants disagree: " + ", ".join(f"{k} {v}" for k, v in r["rule_comparison"].items())
        + ("; explained by a narrow margin" if r["_a"]["variants_disagree_narrow"] else ""))
    add("narrow margin", lambda r: bool(r["_a"]["narrow_notes"]), lambda r: " ".join(r["_a"]["narrow_notes"]))
    add("CNN-6 and LR disagree", lambda r: r["top_class"] != r["lr_top"],
        lambda r: f"CNN-6 top class {r['top_class']}, secondary logistic regression {r['lr_top']}")
    add("signal-quality warning", lambda r: bool(r["quality_warnings"]),
        lambda r: "signal-quality warning: " + "; ".join(r["quality_warnings"]))
    rw = _range_warnings()
    add("speed-range warning", lambda r: any(w in rw for w in r["warnings"]),
        lambda r: f"verdict {r['display_label']}; speed-range warning: " + "; ".join(w for w in r["warnings"] if w in rw),
        key=lambda r: (r["decision"] != "healthy", r["used_for"] != "threshold calibration", r["rec_id"]))  # shown on healthy
    counts = {s: sum(1 for r in rows if p(r)) for s, p in {
        "healthy with caution": lambda r: r["decision"] == "healthy" and r["caution"],
        "further inspection needed": lambda r: r["decision"] == "unknown",
        "rule variants disagree": lambda r: not agree(r),
        "narrow margin": lambda r: bool(r["_a"]["narrow_notes"]),
        "checks disagree (banner)": lambda r: r["_a"]["disagree"],
        "CNN-6 and LR disagree": lambda r: r["top_class"] != r["lr_top"],
        "signal-quality warning": lambda r: bool(r["quality_warnings"])}.items()}
    return {"note": "real development-pool recordings chosen from diagnose() outputs; every one was used to build the "
                    "model (training or threshold calibration), so results on them are not evidence of accuracy",
            "showcase": pick, "states_without_a_development_recording": missing, "counts_in_sweep": counts,
            "n_swept": len(rows)}


def main() -> int:
    from backend.engine import Engine
    if sys.argv[1:] != ["build"]:
        print("usage: python -m backend.showcase build")
        return 2
    eng = Engine()
    if SWEEP.exists():
        rows = json.loads(SWEEP.read_text())
    else:
        t = time.perf_counter()
        rows = sweep(eng)
        SWEEP.parent.mkdir(parents=True, exist_ok=True)
        SWEEP.write_text(json.dumps(rows))
        print(f"swept {len(rows)} recordings in {time.perf_counter() - t:.0f} s")
    sel = select(rows)
    showcase_file().write_text(json.dumps(sel, indent=1, ensure_ascii=False))
    print(json.dumps(sel, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
