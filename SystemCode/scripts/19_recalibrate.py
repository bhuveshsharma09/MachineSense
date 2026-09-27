"""Recalibrate the bundle thresholds on the deployment scoring path (Stage ML-4.2): the 144 calibration recordings
(closed val ∩ development pool, 5 normals) scored by ml.inference.score_recording on the CPU — twice, to check that
repeated runs are bit-identical — then τ_p, τ_d, τ_e, τ_b with the unchanged formulas (ml.calibration). Writes
thresholds.json (with the previous values kept as `previous`) and calibration_scores.csv into the bundle, rebuilds the
manifest (scripts/15_bundle_manifest.py) and writes results/bundle/recalibration.json. No model weight changes.

    MACHINESENSE_DEVICE=cpu python scripts/19_recalibrate.py
"""
from __future__ import annotations

import importlib.util
import json
import sys

import pandas as pd

from ml.arbitrator import VARIANTS, decide
from ml.calibration import acceptance, score_calibration_set, thresholds_from_scores
from ml.inference import get_bundle
from ml.paths import SYSTEM_CODE_DIR, artifacts_dir, results_dir
from ml.splits import development_pool

B = artifacts_dir() / "bundle"


def main() -> int:
    b = get_bundle()
    cfg = b.config
    meta = pd.read_csv(SYSTEM_CODE_DIR / "metadata.csv")
    pool = development_pool(meta, cfg.development.allowed_closed, cfg.development.reserved_columns)
    va = pool[pool.split_closed == "val"].set_index("rec_id").sort_index()
    ids = list(va.index)
    s1 = score_calibration_set(b, ids)
    s2 = score_calibration_set(b, ids)
    cols = ["msp", "maha", "ae", "b1s"]
    identical = bool((s1[cols].to_numpy() == s2[cols].to_numpy()).all() and (s1.top_class == s2.top_class).all())
    diffs = {c: float((s1[c] - s2[c]).abs().max()) for c in cols}
    is_normal = va.class10 == "normal"
    old = json.loads((B / "thresholds.json").read_text())
    old_t = {k: old[k] for k in ("tau_p", "tau_d", "tau_e", "tau_b")}
    new = thresholds_from_scores(s1, is_normal)
    new["scoring_path"] = "ml.inference.score_recording on the CPU (the application's path); repeated runs bit-identical: " + str(identical)
    new["tolerance"] = None if identical else "REQUIRED — see recalibration.json"
    new["rule"] = old["rule"]
    new["previous"] = {**old_t, "scoring_path": "Stage ML-4 threshold setting: representation cache and feature cache, "
                                                "training device (MPS)", "superseded_by": "Stage ML-4.2 recalibration"}
    s1.assign(class10=va.class10).to_csv(B / "calibration_scores.csv", float_format="%.17g")
    (B / "thresholds.json").write_text(json.dumps(new, indent=1, ensure_ascii=False))
    t = {k: new[k] for k in ("tau_p", "tau_d", "tau_e", "tau_b")}
    # decisions before and after, for every calibration recording and every variant
    changes = []
    for r in ids:
        x = s1.loc[r]
        for v in VARIANTS:
            a = decide(v, x.top_class, x.msp, x.maha, x.ae, x.b1s, old_t)
            n = decide(v, x.top_class, x.msp, x.maha, x.ae, x.b1s, t)
            fa = (a["fault_class"] or a["decision"]) + (f" ({a['caution']})" if a["caution"] else "")
            fn = (n["fault_class"] or n["decision"]) + (f" ({n['caution']})" if n["caution"] else "")
            if fa != fn:
                changes.append({"rec_id": r, "class10": va.loc[r, "class10"], "variant": v, "old": fa, "new": fn})
    normals = []
    for r in va.index[is_normal]:
        x = s1.loc[r]
        normals.append({"rec_id": r, "measured_speed_hz": float(va.loc[r, "fr_measured_hz"]), "pred10": x.top_class,
                        "msp": x.msp, "msp_ge_tau_p": bool(x.msp >= t["tau_p"]), "maha": x.maha,
                        "maha_le_tau_d": bool(x.maha <= t["tau_d"]), "ae": x.ae, "ae_le_tau_e": bool(x.ae <= t["tau_e"]),
                        "b1s": x.b1s, "b1s_le_tau_b": bool(x.b1s <= t["tau_b"]),
                        "verdicts": {v: (lambda o: (o["fault_class"] or o["decision"]) + (f" (caution: {o['caution']})" if o["caution"] else ""))(
                            decide(v, x.top_class, x.msp, x.maha, x.ae, x.b1s, t)) for v in VARIANTS}})
    report = {"old_thresholds": old_t, "new_thresholds": t, "defining_recordings": new["defining_recordings"],
              "order_statistic": {"n": new["n_calibration"], "r": new["order_statistic_r"]},
              "repeated_cpu_runs_identical": identical, "repeated_run_max_abs_difference": diffs,
              "acceptance_new": acceptance(s1, t), "acceptance_old_thresholds_on_new_scores": acceptance(s1, old_t),
              "normals": normals, "changed_decisions_vs_old_thresholds": changes}
    (results_dir() / "bundle" / "recalibration.json").write_text(json.dumps(report, indent=1, default=float))
    spec = importlib.util.spec_from_file_location("manifest", SYSTEM_CODE_DIR / "scripts" / "15_bundle_manifest.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.main()
    print(json.dumps({k: report[k] for k in ("old_thresholds", "new_thresholds", "defining_recordings",
                                             "repeated_cpu_runs_identical", "repeated_run_max_abs_difference",
                                             "acceptance_new")}, indent=1, default=float))
    print(pd.DataFrame(normals).drop(columns="verdicts").to_string())
    for n in normals:
        print(n["rec_id"], n["verdicts"])
    print(pd.DataFrame(changes).to_string() if changes else "no decision changes")
    return 0


if __name__ == "__main__":
    sys.exit(main())
