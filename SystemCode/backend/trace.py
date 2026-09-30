"""Stage App-4.1: the read-only pipeline trace of a saved case ("How this diagnosis was made"). A separate, versioned,
derived view: it never re-runs inference, never changes the evidence record and adds no decision logic.

Every value carries its source: {"value": v, "from": "<source>:<path>"} with source one of
  record       the stored evidence record (GET /api/cases/{id})
  activity     the stored activity log (per-step wall-clock ms measured at diagnosis time)
  config       the bundle's packaged configuration (inference_config.json)
  manifest     the bundle manifest
  explanation  the stored explanation of the case (App-3), if any
  store        the case row (rerun_of)
  derived      computed for this view from the above (a count, a comparison done by backend.checks, a code-path fact)
or {"value": None, "reason": "..."} when the value was never recorded. timing_ms is a bare number copied from one
activity step, or None ("not recorded"). Blocks follow the code path of ml.inference.diagnose; ports and edges are the
real data dependencies of that code (no edge is drawn that the code does not have)."""
from __future__ import annotations

from typing import Any

from backend.checks import checks as record_checks

TRACE_VERSION = "1.0"
# edge kind → the UI's evidence TAG (frontend/src/lib/theme.js); "Missing" is for not_run / unavailable blocks
KIND_TAG = {"signal": "Measured", "model": "Model", "decision": "Model", "explanation": "Guidance"}
STATUS_TAG = {"not_run": "Missing", "unavailable": "Missing"}
NOT_TIMED = "not recorded: this step is not wrapped by the activity timer"
IN_MODELS_STEP = ("not recorded separately: cleaning, windowing, features and the representation run inside "
                  "ml.inference.score_recording, whose time is the models step")
UNKNOWN_MEANING = ("\"Further inspection needed\" means the checks did not accept any known class for this recording; it "
                   "does not mean a new fault was found.")
AE_CHANNEL_CAVEAT = ("Per-channel autoencoder error is not usable as fault location or anomaly evidence (backend/evidence.py; "
                     "CLAUDE.md, Stage 3B sensor-evidence finding).")


# ---------------------------------------------------------------------------------------------------------- values
def _get(obj: Any, path: str) -> Any:
    """Dotted path into dicts; on a list of dicts, a segment selects the item whose "name" equals it."""
    for seg in path.split("."):
        if isinstance(obj, list):
            obj = next(i for i in obj if isinstance(i, dict) and i.get("name") == seg)
        else:
            obj = obj[seg]
    return obj


def resolve(ref: str, sources: dict) -> Any:
    """The value a "<source>:<path>" reference points to (sources: record, activity, config, manifest, explanation,
    store). activity:<step> is that step's ms. Used by the builder and by the tests."""
    src, path = ref.split(":", 1)
    if src == "activity":
        return next(s["ms"] for s in sources["activity"] if s["step"] == path)
    return _get(sources[src], path)


class _Refs:
    def __init__(self, sources: dict):
        self.s = sources

    def __call__(self, ref: str) -> dict:
        return {"value": resolve(ref, self.s), "from": ref}


def _derived(value: Any, how: str) -> dict:
    return {"value": value, "from": f"derived:{how}"}


def _missing(reason: str) -> dict:
    return {"value": None, "reason": reason}


def _port(pid: str, label: str, short: str) -> dict:
    return {"id": pid, "label": label, "short": short}


def _uses_derived(x: Any) -> bool:
    if isinstance(x, dict):
        if str(x.get("from", "")).startswith("derived:"):
            return True
        return any(_uses_derived(v) for k, v in x.items() if k != "children")
    if isinstance(x, list):
        return any(_uses_derived(v) for v in x)
    return False


def _block(bid: str, title_mechanic: str, title_developer: str, purpose: str, input: Any, operation: str, output: Any,
           status: str, provenance: list[str], technical: Any, timing_ms: float | None, ports_in: list, ports_out: list,
           children: list | None = None, parent: str | None = None) -> dict:
    b = {"id": bid, "title_mechanic": title_mechanic, "title_developer": title_developer, "purpose": purpose,
         "input": input, "operation": operation, "output": output, "status": status, "provenance": provenance,
         "technical": technical, "timing_ms": timing_ms, "ports_in": ports_in, "ports_out": ports_out}
    b["derived"] = _uses_derived(b)
    if status in STATUS_TAG:
        b["tag"] = STATUS_TAG[status]
    if children is not None:
        b["children"] = children
    if parent is not None:
        b["parent"] = parent
    return b


def _ms(activity: list[dict], step: str) -> float | None:
    """timing_ms of a block: the stored activity step's ms, never estimated."""
    s = next((a for a in activity if a["step"] == step), None)
    return None if s is None else s["ms"]


# ---------------------------------------------------------------------------------------------------------- blocks
def _input(R, rec: dict, stored_signal: bool) -> dict:
    upload = rec["source"] == "upload"
    return _block(
        "input", "Recording", "Input", "Which recording was diagnosed and where it came from.",
        None,
        "An uploaded CSV is parsed to a (250000, 8) array; a development-pool recording is read from data/raw_npy by "
        "rec_id (ml.inference.load_recording).",
        {"source": R("record:source"), "rec_id": R("record:input.rec_id"), "origin": R("record:input.origin"),
         "created_at": R("record:created_at"), "case_id": R("record:case_id"),
         "stored_signal_available": _derived(stored_signal, "uploads: stored array present; dataset: raw_npy by rec_id")},
        "completed", ["record:source", "record:input", "record:created_at"],
        {"channels": R("config:preprocess.channels"), "fs_hz": R("config:preprocess.fs_hz"),
         "n_rows": R("config:preprocess.n_rows"),
         "note": "an uploaded array is stored so the case can be re-run" if upload else
                 "reserved-test recordings are never read by this path"},
        None, [], [_port("raw", "raw 8-channel recording", "RAW")])


def _validation(R, rec: dict, rerun_of: str | None) -> dict:
    upload = rec["source"] == "upload"
    csv_applied = upload and rerun_of is None
    csv_reason = (None if csv_applied else
                  "dataset recording: no file was uploaded" if not upload else
                  f"re-run of case {rerun_of}: the stored array is reused; the CSV checks were applied when it was uploaded")

    def item(check, function, applied, reason=None, expected=None):
        d = {"check": check, "function": function, "applied": _derived(applied, "source and rerun_of of the case")}
        if expected is not None:
            d["expected"] = expected
        if applied:
            d["outcome"] = _derived("passed", "the case was saved; a failed check stores no case")
        else:
            d["not_applied_reason"] = reason or "not on this input path"
        return d
    items = [
        item("rec_id is known and not in any reserved test", "ml.inference.check_rec_id", not upload,
             "upload: there is no rec_id to check"),
        item("column count", "backend.engine.parse_upload", csv_applied, csv_reason, R("config:preprocess.n_channels")),
        item("row count (5 s at 50 kHz)", "backend.engine.parse_upload", csv_applied, csv_reason, R("config:preprocess.n_rows")),
        item("all values finite", "backend.engine.parse_upload", csv_applied, csv_reason),
        item("reserved-recording fingerprint guard (best effort)", "backend.guard.ReservedGuard.match", upload,
             "dataset recording: refused by rec_id instead"),
        item("array shape", "ml.inference.diagnose", upload, "dataset recording: read from raw_npy by rec_id"),
        item("shaft speed measurable", "ml.inference.diagnose (estimate_speed)", upload,
             "dataset recording: the metadata speed is used"),
    ]
    return _block(
        "validation", "Checks on the file", "Validation and reserved-recording guard",
        "Refuse inputs that cannot be diagnosed and every reserved-test recording.",
        {"source": R("record:source")},
        "Uploads: parse_upload (columns, rows, finite values), then the reserved-recording fingerprint guard, then the "
        "shape and speed checks in diagnose(). Dataset recordings: check_rec_id refuses unknown and reserved rec_ids "
        "before anything is loaded. Which checks applied is reconstructed from the code path for this case.",
        {"checks": items}, "completed", ["record:source", "store:rerun_of"], None, None,
        [_port("raw", "raw 8-channel recording", "RAW")], [_port("accepted", "accepted recording", "OK")])


def _signal_quality(R, rec: dict, activity: list) -> dict:
    q = rec["signal_quality"]
    return _block(
        "signal_quality", "Signal quality", "Signal quality (ml.inference._signal_quality)",
        "Report speed, tacho, microphone, clipping and spike statistics and their warnings.",
        {"recording": "accepted recording", "shaft_speed_hz": R("record:signal_quality.shaft_speed_hz")},
        "Per-window speeds from the untouched tachometer, tacho glitch count, microphone burst count, quality "
        "statistics; warnings are compared against the training and calibration-normal speed ranges. The warnings go "
        "into the evidence record only; no model receives them.",
        {k: R(f"record:signal_quality.{k}") for k in q},
        "warning" if q["warnings"] else "completed", ["record:signal_quality", "activity:signal_quality"],
        {"unstable_speed_threshold_pct": _missing("a code constant (ml.inference.UNSTABLE_SPEED_PCT), not saved in "
                                                  "the record, config or manifest"),
         "quality_settings": R("config:preprocess.quality")},
        _ms(activity, "signal_quality"),
        [_port("raw", "accepted recording", "RAW"), _port("speed", "measured shaft speed", "SPD")],
        [_port("quality", "signal-quality statistics and warnings", "SQ")])


def _preparation(R, rec: dict) -> dict:
    q = rec["signal_quality"]
    upload = rec["source"] == "upload"
    return _block(
        "preparation", "Preparing the signal", "Speed, cleaning and windowing",
        "Measure the shaft speed, clean the whole recording and cut it into windows.",
        {"recording": "accepted recording"},
        "Shaft speed: " + ("measured from the tachometer and vibration (ml.preprocess.estimate_speed)." if upload else
                           "the metadata measured speed of the recording.") +
        " clean_recording (microphone burst removal, the fixed ~22 kHz band-stop, the 10 kHz low-pass; tachometer "
        "untouched) is applied to the whole recording before windowing. It is invoked inside feature building "
        "(ml.features.recording_features), representation building (ml.spectro.recording_representation) and the "
        "physics envelope spectrum (ml.inference._physics), not as a separate step. Per-window speeds come from "
        "window_speeds_detail on the untouched tachometer with the same arguments in the signal-quality check, feature "
        "building and representation building.",
        {"shaft_speed_hz": R("record:signal_quality.shaft_speed_hz"),
         "window_speeds_hz": R("record:signal_quality.window_speeds_hz"),
         "window_speed_fallback": R("record:signal_quality.window_speed_fallback"),
         "speed_range_pct": R("record:signal_quality.speed_range_pct"),
         "speed_stable": R("record:signal_quality.speed_stable"),
         "mic_bursts_removed": R("record:signal_quality.mic_bursts_removed"),
         "window_count": _derived(len(q["window_speeds_hz"]), "len(record:signal_quality.window_speeds_hz)"),
         "win_s": R("config:preprocess.window.win_s"), "hop_s": R("config:preprocess.window.hop_s"),
         "n_windows_configured": R("config:preprocess.window.n_windows"),
         "lowpass": R("config:preprocess.lowpass"), "bandstop": R("config:preprocess.bandstop"),
         "mic_burst_settings": R("config:preprocess.mic")},
        "warning" if any(q["window_speed_fallback"]) else "completed",
        ["record:signal_quality", "config:preprocess"],
        {"speed_source": _derived("tachometer and vibration" if upload else "metadata fr_measured_hz", "source of the case"),
         "measured_speed_ratio": R("config:preprocess.measured_speed_ratio")},
        None, [_port("raw", "accepted recording", "RAW")],
        [_port("speed", "measured shaft speed", "SPD"), _port("prepared", "cleaned recording, windows, per-window speeds", "PREP")])


def _representations(R) -> dict:
    cand = "config:preprocess.spectro.candidates.C_order8rev"
    r1 = _block(
        "physics_features", "Vibration measurements", "Physics features (R1)",
        "Per-window physics features: order amplitudes, envelope comb, microphone bands, speed.",
        {"prepared": "cleaned windows and per-window speeds"},
        "ml.features.recording_features on the cleaned recording, one row per window.",
        {"version": R("config:preprocess.features.version"),
         "rows": R("config:preprocess.window.n_windows"),
         "n_features": _missing("the number of R1 feature columns is not declared in the packaged config; the column "
                                "lists are stored in the LR and B1s model files and feature_dictionary.csv"),
         "accel_orders": R("config:preprocess.features.accel_orders"),
         "envelope_channels": R("config:preprocess.features.envelope_channels"),
         "mic_bands_hz": R("config:preprocess.features.mic_bands_hz")},
        "completed", ["config:preprocess.features"], None, None,
        [_port("prepared", "cleaned windows and per-window speeds", "PREP")], [_port("features", "R1 feature table", "R1")],
        parent="representations")
    spec = _block(
        "order_spectrogram", "Vibration pictures", "Order-domain spectrogram (representation C)",
        "Per-window order spectrograms of the six accelerometer axes plus the aligned microphone log-Mel.",
        {"prepared": "cleaned windows and per-window speeds"},
        "ml.spectro.stacked_representation: order-domain resampling at the per-window speed; stored as float16, cast to "
        "float32 on the CPU before the models. Not recomputed for this view.",
        {"candidate": R("config:representation.leading_candidate"), "version": R("config:representation.version"),
         "windows": R("config:preprocess.window.n_windows"),
         "order_bins": R("config:preprocess.spectro.n_orders"), "order_min": R("config:preprocess.spectro.order_min"),
         "order_max": R("config:preprocess.spectro.order_max"), "frames": R(f"{cand}.frames_per_window"),
         "vibration_channels": R("config:preprocess.spectro.channels"),
         "microphone_mel_bins": R("config:preprocess.spectro.mel_n"),
         "n_revs": R(f"{cand}.n_revs"), "samples_per_rev": R(f"{cand}.samples_per_rev"),
         "anti_alias_orders": R(f"{cand}.aa_orders"),
         "cnn_input_channels": R("config:cnn.vibration_channels"),
         "cnn_channels_note": R("manifest:representation.cnn_channels"),
         "ae_input_channels": R("manifest:files.ae6.description")},
        "completed", ["config:representation", "config:preprocess.spectro", "manifest:representation"], None, None,
        [_port("prepared", "cleaned windows and per-window speeds", "PREP")],
        [_port("spectrogram", "order spectrograms", "SPEC")], parent="representations")
    return _block(
        "representations", "Turning the signal into features", "Representations",
        "Two representations of the same cleaned windows: physics features and order-domain spectrograms.",
        {"prepared": "cleaned windows and per-window speeds"},
        "Built inside ml.inference.score_recording; the dimensions below are the packaged configuration's.",
        None, "completed", ["config:preprocess", "config:representation"], None, None, [], [],
        children=[r1, spec])


def _models(R, rec: dict, activity: list) -> dict:
    d = rec["diagnosis"]
    t = "record:diagnosis.thresholds"
    cnn = _block(
        "cnn6", "Pattern classifier", "CNN-6 (main classifier)",
        "Rank the ten known conditions from the vibration spectrograms.",
        {"spectrogram": "6 vibration order spectrograms per window"},
        "Scratch 2D CNN on the normalised spectrograms of each window; recording probabilities from the window "
        "log-probabilities.",
        {"class10_probabilities": R("record:diagnosis.class10_probabilities"), "top_class": R("record:diagnosis.top_class"),
         "msp": R("record:diagnosis.scores.msp"), "model_name": R("record:diagnosis.model_name")},
        "completed", ["record:diagnosis", "manifest:files.cnn6"],
        {"aggregation": "mean log-probability over the 9 windows, renormalised",
         "weights_sha256": R("record:provenance.model_sha256.cnn6"), "file": R("manifest:files.cnn6.description")},
        None, [_port("spectrogram", "order spectrograms", "SPEC")],
        [_port("probs", "ten-class probabilities, top class, MSP", "P10"), _port("embedding", "penultimate embedding", "EMB")],
        parent="models")
    c6 = _block(
        "class6", "Six-group view", "Class6 view (derived)",
        "The same prediction grouped into six conditions (bearing faults per position).",
        {"probs": "CNN-6 ten-class probabilities"},
        "DERIVED from the ten-class output by summing probabilities with the class10 → class6 mapping; not a separate "
        "classifier.",
        {"class6_probabilities": R("record:diagnosis.class6_probabilities"),
         "class6_mapping": R("record:diagnosis.class6_mapping")},
        "completed", ["record:diagnosis.class6_probabilities", "record:diagnosis.class6_mapping"],
        {"separate_classifier": False}, None,
        [_port("probs", "ten-class probabilities", "P10")], [_port("probs6", "six-class probabilities", "P6")], parent="models")
    maha = _block(
        "mahalanobis", "Familiarity check", "Mahalanobis distance on the CNN-6 embedding",
        "How close the recording's embedding lies to the known classes seen in training.",
        {"embedding": "CNN-6 penultimate-layer embedding (same network, not a separate model)"},
        "Distance to the class centroids with one shared Ledoit–Wolf covariance (bundle mahalanobis file).",
        {"maha": R("record:diagnosis.scores.maha"), "tau_d": R(f"{t}.tau_d"), "direction": "lower = more familiar"},
        "completed", ["record:diagnosis.scores.maha", "manifest:files.mahalanobis"],
        {"aggregation": "per-window minimum over classes, then mean", "file": R("manifest:files.mahalanobis.description")},
        None, [_port("embedding", "penultimate embedding", "EMB")], [_port("distance", "Mahalanobis distance", "MAHA")],
        parent="models")
    ae = _block(
        "ae6", "Healthy-pattern check (autoencoder)", "AE-6 reconstruction error",
        "How well an autoencoder trained on healthy recordings only reconstructs this recording.",
        {"spectrogram": "6 vibration order spectrograms per frame"},
        "Frame-level autoencoder on the normalised spectrograms; reconstruction error per channel.",
        {"ae_error": R("record:diagnosis.scores.ae_error"), "tau_e": R(f"{t}.tau_e"), "direction": "lower = more healthy-like"},
        "completed", ["record:diagnosis.scores.ae_error", "manifest:files.ae6"],
        {"aggregation": "per-channel error, equal-weight mean, then mean over windows",
         "ae_channel_errors": R("record:diagnosis.scores.ae_channel_errors"), "ae_channel_errors_caveat": AE_CHANNEL_CAVEAT,
         "file": R("manifest:files.ae6.description")},
        None, [_port("spectrogram", "order spectrograms", "SPEC")], [_port("error", "AE error", "AE")], parent="models")
    b1s = _block(
        "b1s", "Healthy-pattern check (speed-aware)", "B1s speed-aware healthy baseline",
        "How far the vibration features lie from healthy recordings at the same speed.",
        {"features": "R1 vibration features and speed"},
        "Speed-aware polynomial baseline (ml.anomaly.SpeedPoly) fitted on healthy training recordings.",
        {"b1s": R("record:diagnosis.scores.b1s"), "tau_b": R(f"{t}.tau_b"), "direction": "lower = more healthy-like"},
        "completed", ["record:diagnosis.scores.b1s", "manifest:files.b1s_vib"],
        {"aggregation": "mean over windows", "file": R("manifest:files.b1s_vib.description")},
        None, [_port("features", "R1 feature table", "R1")], [_port("score", "B1s score", "B1S")], parent="models")
    lr = _block(
        "lr", "Second opinion", "Logistic regression (secondary)",
        "A simple classifier on the physics features; its feature contributions explain the result.",
        {"features": "R1 feature table"},
        "Quantile transform + logistic regression on the capped R1 features; window probabilities aggregated per "
        "recording. ml.arbitrator.decide() does not receive it.",
        {"top_class": R("record:diagnosis.lr_secondary.top_class"),
         "probabilities": R("record:diagnosis.lr_secondary.probabilities"),
         "feeds_decision": False, "role": "secondary classifier and explanation source"},
        "completed", ["record:diagnosis.lr_secondary", "manifest:files.lr_r1"],
        {"file": R("manifest:files.lr_r1.description")},
        None, [_port("features", "R1 feature table", "R1")], [_port("probs", "secondary probabilities", "LR")],
        parent="models")
    return _block(
        "models", "Model checks", "Models (ml.inference.score_recording)",
        "Score the recording with every model of the verified bundle.",
        {"representations": "physics features and order spectrograms"},
        "One shared scoring function (Stage ML-4.2) used by the application and by threshold calibration; float32 on "
        "the CPU. Each child states its own window → recording aggregation.",
        None, "completed", ["record:diagnosis", "activity:models"],
        {"timing_note": "the models step also contains cleaning, windowing, features and the representation"},
        _ms(activity, "models"), [], [], children=[cnn, c6, maha, ae, b1s, lr])


RULE_USE = {
    "A": "classifier first (MSP and Mahalanobis); a normal top class with AE error ≤ τ_e is healthy even when the "
         "classifier checks fail (AE override)",
    "B": "classifier first, no AE override",
    "C": "healthy needs a confident normal top class AND AE error ≤ τ_e (exploratory)",
    "D": "B, and healthy also needs B1s ≤ τ_b (otherwise unknown)",
    "B_with_caution": "B's decisions; a healthy verdict with AE > τ_e and/or B1s > τ_b carries a caution",
}


def _decision(R, rec: dict, activity: list) -> dict:
    t = "record:diagnosis.thresholds"
    names = {"msp": ("msp >= tau_p", "scores.msp", "tau_p"), "maha": ("maha <= tau_d", "scores.maha", "tau_d"),
             "ae": ("ae <= tau_e", "scores.ae_error", "tau_e"), "b1s": ("b1s <= tau_b", "scores.b1s", "tau_b")}
    per_check = []
    for c in record_checks(rec):
        label, sk, tk = names[c["key"]]
        per_check.append({"check": label, "name": c["name"], "role": c["role"],
                          "value": R(f"record:diagnosis.{sk}"), "threshold": R(f"{t}.{tk}"),
                          "passed": _derived(c["passed"], f"backend.checks.checks(record).{c['key']}.passed"),
                          "status": _derived(c["status"], f"backend.checks.checks(record).{c['key']}.status")})
    return _block(
        "decision", "Decision", "Decision rule (ml.arbitrator.decide)",
        "Turn the four scores into healthy / known fault / further inspection needed.",
        {"top_class": R("record:diagnosis.top_class"), "msp": R("record:diagnosis.scores.msp"),
         "maha": R("record:diagnosis.scores.maha"), "ae_error": R("record:diagnosis.scores.ae_error"),
         "b1s": R("record:diagnosis.scores.b1s"), "thresholds": R(t)},
        "decide(rule, top_class, msp, maha, ae_error, b1s, thresholds): the classifier checks (MSP ≥ τ_p and "
        "Mahalanobis ≤ τ_d, ties accepted) accept a known class; the healthy-baseline checks (AE, B1s) are used as the "
        "rule variant states. The logistic regression is not an input. " + UNKNOWN_MEANING,
        {"decision": R("record:diagnosis.decision"), "fault_class": R("record:diagnosis.fault_class"),
         "display_label": R("record:diagnosis.display_label"), "caution": R("record:diagnosis.caution"),
         "rule_variant": R("record:diagnosis.rule_variant"), "ae_override": R("record:diagnosis.ae_override"),
         "rule_status": R("record:diagnosis.rule_status"), "thresholds": R(t), "checks": per_check,
         "rule_comparison": R("record:diagnosis.rule_comparison"),
         "rule_comparison_note": "read-only evidence: what rules A–D decide for the same scores; not used for this verdict",
         "unknown_meaning": UNKNOWN_MEANING},
        "completed", ["record:diagnosis", "activity:decision_rule", "manifest:thresholds"],
        {"how_each_rule_uses_the_checks": RULE_USE,
         "threshold_calibration": {k: R(f"manifest:thresholds.{k}") for k in
                                   ("acceptance_target", "n_calibration", "order_statistic_r", "defining_recordings",
                                    "scoring_path", "tolerance")}},
        _ms(activity, "decision_rule"),
        [_port("top_msp", "top class and MSP", "MSP"), _port("maha", "Mahalanobis distance", "MAHA"),
         _port("ae", "AE error", "AE"), _port("b1s", "B1s score", "B1S")],
        [_port("verdict", "verdict, caution, rule comparison", "VRD")])


def _evidence_record(R, rec: dict, activity: list) -> dict:
    p = "record:provenance"
    tech = {k: R(f"activity:{k}") for k in ("physics", "explanations", "bundle_verified")
            if any(a["step"] == k for a in activity)}
    tech_note = ("activity steps shown here for reference: physics (envelope spectrum and cursors), explanations (LR "
                 "contributions and the CNN map), bundle_verified (server start, not per case)")
    return _block(
        "evidence_record", "Saved result", "Evidence record (schema-validated)",
        "The saved, schema-validated record every view reads; its provenance pins the code, bundle and data.",
        {"quality": "signal quality", "scores": "model scores", "verdict": "decision", "secondary": "LR output",
         "physics": "envelope spectrum and cursors"},
        "backend.schemas.EvidenceRecord (extra = forbid) validates diagnose()'s output with case_id, created_at and "
        "source; the record is stored unchanged in the case database.",
        {"provenance": {k: R(f"{p}.{k}") for k in ("git_sha", "bundle_git_sha", "manifest_sha256", "model_sha256",
                                                    "preprocess_config_sha256", "split_hash", "schema_version")},
         "limitations": R("record:limitations"),
         "has_explanations_xai": _derived(rec.get("xai") is not None, "record:xai is present")},
        "completed", ["record:provenance", "record:limitations"],
        {"timings_ms": tech, "timings_note": tech_note, "total_ms": R("record:timing_ms.total")},
        None,
        [_port("quality", "signal-quality statistics and warnings", "SQ"), _port("scores", "model scores", "SCR"),
         _port("class6", "six-class probabilities", "P6"), _port("secondary", "secondary classifier output", "LR"),
         _port("verdict", "verdict", "VRD"), _port("physics", "envelope spectrum and cursors", "PHY")],
        [_port("record", "evidence record", "REC")])


def _explanation(R, explanation: dict | None) -> dict:
    ports = ([_port("record", "evidence record (compact view, no signal)", "REC")],
             [_port("text", "plain-language explanation", "EXP")])
    if explanation is None:
        return _block(
            "explanation", "Plain-language explanation", "Explanation (App-3)",
            "A plain-language explanation of the saved evidence (language model with guards, or the fixed template).",
            None, "Not generated for this case yet; the trace never generates one.",
            {"generator": _missing("no explanation stored for this case")}, "not_run", ["explanation:none"], None, None,
            *ports)
    E = "explanation"
    reason = explanation.get("fallback_reason")
    outcome = ("generated text passed every check" if explanation["source"] == "llm" else
               f"fixed template used: {reason}" if reason else "fixed template used")
    return _block(
        "explanation", "Plain-language explanation", "Explanation (App-3)",
        "A plain-language explanation of the saved evidence (language model with guards, or the fixed template).",
        {"record": "compact view of the evidence record; never the raw signal"},
        "backend.explain.explain_case: the language model's JSON is checked by application code (schema, numbers only "
        "from the record, no raw scores, forbidden phrases, directions); any failure falls back to the fixed template.",
        {"source": R(f"{E}:source"), "badge": R(f"{E}:badge"),
         "template_fallback_used": _derived(explanation["source"] == "template", "explanation:source == template"),
         "fallback_reason": R(f"{E}:fallback_reason"),
         "validation_outcome": _derived(outcome, "explanation:source and fallback_reason"),
         "attempts": R(f"{E}:attempts")},
        "completed" if reason is None else "warning", ["explanation:source", "explanation:fallback_reason"],
        {k: R(f"{E}:{k}") for k in ("provider", "model", "prompt_version", "created_at", "latency_ms") if k in explanation},
        None, *ports)


def _presentation(R, rec: dict, case_id: str, rerun_of: str | None, stored_signal: bool, has_explanation: bool,
                  activity: list) -> dict:
    base = f"/api/cases/{case_id}"

    def view(name, path, available, reason=None):
        d = {"view": name, "endpoint": base + path, "available": _derived(available, "stored case, signal and explanation")}
        if not available:
            d["reason"] = reason
        return d
    views = [view("evidence record", "", True), view("activity", "/activity", True),
             view("fixed report (no language model)", "/report", True),
             view("measured channel evidence", "/channels", stored_signal, "the stored upload array is missing"),
             view("signals and spectra", "/signals", stored_signal, "the stored upload array is missing"),
             view("explanation", "/explanation", has_explanation, "not generated yet (POST to generate)"),
             view("pipeline trace", "/trace", True)]
    return _block(
        "presentation", "What you see", "Case history and views",
        "The saved case and the views built from it.",
        {"record": "evidence record", "explanation": "explanation, if generated"},
        "The case is saved in SQLite with its activity log; every view reads the saved record (the report and channel "
        "views are rebuilt on request from it and the stored signal).",
        {"case_id": R("record:case_id"), "saved_at": R("record:created_at"), "rerun_of": R("store:rerun_of"),
         "views": views},
        "completed", ["record:case_id", "store:rerun_of", "activity:report"],
        {"report_note": "the report step was timed once at diagnosis time; /report rebuilds it on request"},
        _ms(activity, "report"),
        [_port("record", "evidence record", "REC"), _port("explanation", "plain-language explanation", "EXP")], [])


# ----------------------------------------------------------------------------------------------------------- edges
EDGES = [  # (from, to, kind, label) — the real dependencies in ml.inference.diagnose
    ("input.raw", "validation.raw", "signal", "raw recording"),
    ("validation.accepted", "preparation.raw", "signal", "accepted recording"),
    ("validation.accepted", "signal_quality.raw", "signal", "accepted recording"),
    ("preparation.speed", "signal_quality.speed", "signal", "measured shaft speed (speed warnings, window fallback)"),
    ("signal_quality.quality", "evidence_record.quality", "signal", "statistics and warnings (record only; no model)"),
    ("preparation.prepared", "physics_features.prepared", "signal", "cleaned windows, per-window speeds"),
    ("preparation.prepared", "order_spectrogram.prepared", "signal", "cleaned windows, per-window speeds"),
    ("preparation.prepared", "evidence_record.physics", "signal", "envelope spectrum and cursors (ml.inference._physics)"),
    ("physics_features.features", "lr.features", "signal", "R1 features"),
    ("physics_features.features", "b1s.features", "signal", "R1 vibration features and speed"),
    ("order_spectrogram.spectrogram", "cnn6.spectrogram", "signal", "6 vibration order spectrograms"),
    ("order_spectrogram.spectrogram", "ae6.spectrogram", "signal", "6 vibration order spectrograms"),
    ("cnn6.embedding", "mahalanobis.embedding", "model", "penultimate-layer embedding (same network)"),
    ("cnn6.probs", "class6.probs", "model", "ten-class probabilities (summed by the mapping)"),
    ("cnn6.probs", "decision.top_msp", "model", "top class and MSP"),
    ("mahalanobis.distance", "decision.maha", "model", "Mahalanobis distance"),
    ("ae6.error", "decision.ae", "model", "AE error (override in A, healthy check in C, caution in B_with_caution)"),
    ("b1s.score", "decision.b1s", "model", "B1s (healthy check in D, caution in B_with_caution)"),
    ("cnn6.probs", "evidence_record.scores", "model", "probabilities, MSP; CNN attribution map (exploratory)"),
    ("mahalanobis.distance", "evidence_record.scores", "model", "Mahalanobis distance"),
    ("ae6.error", "evidence_record.scores", "model", "AE error and per-channel errors"),
    ("b1s.score", "evidence_record.scores", "model", "B1s score"),
    ("class6.probs6", "evidence_record.class6", "model", "six-class view"),
    ("lr.probs", "evidence_record.secondary", "model", "secondary probabilities and feature contributions"),
    ("decision.verdict", "evidence_record.verdict", "decision", "verdict, caution, rule comparison"),
    ("evidence_record.record", "explanation.record", "explanation", "compact view of the record"),
    ("evidence_record.record", "presentation.record", "decision", "evidence record"),
    ("explanation.text", "presentation.explanation", "explanation", "plain-language explanation"),
]


def _all_blocks(blocks: list[dict]):
    for b in blocks:
        yield b
        yield from _all_blocks(b.get("children", []))


def build_trace(case_id: str, entry: dict, explanation: dict | None, config: dict, manifest: dict,
                manifest_sha256: str, stored_signal: bool) -> dict:
    """The trace of a saved case. entry: CaseStore.get(case_id); config / manifest: the loaded bundle's packaged Config
    and manifest as JSON; manifest_sha256: the loaded bundle's; stored_signal: the case's signal can be read."""
    rec, activity = entry["record"], entry["activity"]
    sources = {"record": rec, "activity": activity, "config": config, "manifest": manifest,
               "explanation": explanation or {}, "store": {"rerun_of": entry.get("rerun_of")}}
    R = _Refs(sources)
    blocks = [_input(R, rec, stored_signal), _validation(R, rec, entry.get("rerun_of")),
              _signal_quality(R, rec, activity), _preparation(R, rec), _representations(R), _models(R, rec, activity),
              _decision(R, rec, activity), _evidence_record(R, rec, activity), _explanation(R, explanation),
              _presentation(R, rec, case_id, entry.get("rerun_of"), stored_signal, explanation is not None, activity)]
    ports = {f"{b['id']}.{p['id']}" for b in _all_blocks(blocks) for p in b["ports_in"] + b["ports_out"]}
    edges = []
    for a, z, kind, label in EDGES:
        assert a in ports and z in ports, (a, z)  # an edge always connects declared ports
        edges.append({"from": a, "to": z, "kind": kind, "label": label, "tag": KIND_TAG[kind]})
    same = rec["provenance"]["manifest_sha256"] == manifest_sha256
    return {"case_id": case_id, "trace_version": TRACE_VERSION,
            "bundle_matches_record": _derived(same, "record:provenance.manifest_sha256 == loaded bundle manifest sha256"),
            "bundle_note": None if same else ("this case was produced by a different bundle; config and manifest values "
                                              "describe the loaded bundle, not necessarily the one that ran"),
            "value_sources": ["record", "activity", "config", "manifest", "explanation", "store", "derived"],
            "edge_kinds": KIND_TAG, "blocks": blocks, "edges": edges}
