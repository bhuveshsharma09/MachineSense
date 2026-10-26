// Case mode of the pipeline page (App-4.3): what happened to ONE saved recording, read from its trace. Nothing is
// computed here: every value is a reference into the trace ({p: path, f: formatter}), rendered by the page with the
// existing formatting helpers so a number reads as it does in the main panel. The tests resolve the same paths in the
// fixture and compare. Mechanic and Developer show the same verdict (display_label) and the same evidence; only depth
// differs.
import { fmtCheck, fmtScore, human } from "./model.js";
import { finding, plainWarning, speedText } from "./mechanic.js";

// ---------------------------------------------------------------------------------------------------------- access
export function at(trace, path) {
  let n = trace;
  for (const k of path.split(".")) {
    if (n === null || n === undefined) return undefined;
    n = n[/^\d+$/.test(k) && Array.isArray(n) ? Number(k) : k];
  }
  return n;
}
const isSourced = (n) => n !== null && typeof n === "object" && !Array.isArray(n) && "value" in n && ("from" in n || "reason" in n);
export const valueOf = (n) => (isSourced(n) ? n.value : n);
export const missing = (n) => isSourced(n) && n.value === null && typeof n.reason === "string";
export const isDerived = (n) => isSourced(n) && String(n.from || "").startsWith("derived:");

export function blockPaths(trace) {
  const out = {};
  const walk = (bs, prefix) => (bs || []).forEach((b, i) => {
    out[b.id] = `${prefix}.${i}`;
    if (b.children) walk(b.children, `${prefix}.${i}.children`);
  });
  walk(trace.blocks, "blocks");
  return out;
}

// ---------------------------------------------------------------------------------------------------------- format
// Formatters named by key in the rendered page (data-fmt) and used again by the tests. They are the existing helpers.
const decisionWords = { healthy: "healthy", known_fault: "known fault", unknown: 'unknown (shown as "further inspection needed")' };
export const FMT = {
  score: (v) => fmtScore(v),                         // model scores in [0, 1], as the main panel
  check: (v) => fmtCheck(v),                         // check values and thresholds, as the report (value_text)
  hz: (v) => `${v.toFixed(2)} Hz`,                   // Developer panel: measured shaft speed
  pct: (v) => `${v.toFixed(2)} %`,                   // Developer panel: speed range within the recording
  ms: (v) => `${v.toFixed(1)} ms`,                   // Developer panel: activity step times
  speed: (v) => speedText(v),                        // Mechanic panel: rotation speed
  int: (v) => String(v),
  full: (v) => (v === null || v === undefined ? "not recorded" : typeof v === "boolean" ? (v ? "yes" : "no") : String(v)),
  text: (v) => String(v),
  cls: (v) => human(v),
  hash: (v) => `${String(v).slice(0, 12)}…`,
  yesno: (v) => (v ? "yes" : "no"),
  passfail: (v) => (v ? "passed" : "not passed"),
  stable: (v) => (v ? "speed steady" : "speed not steady"),
  none: (v) => (v === null || v === undefined ? "none" : human(v)),
  noneText: (v) => (v === null || v === undefined ? "none" : String(v)),
  rerun: (v) => (v === null || v === undefined ? "not a re-run" : `${String(v).slice(0, 12)}…`),
  recid: (v) => (v === null || v === undefined ? "none (uploaded file)" : String(v)),
  rule: (v) => (v === "unknown" ? "further inspection needed" : v === "healthy" ? "healthy" : `known fault: ${human(v)}`),
  decision: (v) => decisionWords[v] || String(v),
  warnPlain: (v) => plainWarning(v),
  source: (v) => (v === "llm" ? "AI-written (checked)" : v === "template" ? "fixed template" : String(v)),
  // the saved records' rule status (written before F1) as the rule in use (App-8.6 item 1, Bhuvesh)
  ruleStatus: () => "B_with_caution, frozen 28 Sep 2026 before the final evaluation",
};

export const STATUS_WORD = { completed: "Completed", warning: "Warning", failed: "Failed", not_run: "Not run", unavailable: "Unavailable" };

// ---------------------------------------------------------------------------------------------------------- lines
const R = (p, f = "full") => ({ p, f });
const checkIndex = (trace, key) => {
  const P = blockPaths(trace).decision;
  const list = at(trace, `${P}.output.checks`) || [];
  const i = list.findIndex((c) => c.check.startsWith(key));
  return i < 0 ? null : `${P}.output.checks.${i}`;
};
const passed = (trace, key) => { const c = checkIndex(trace, key); return c ? valueOf(at(trace, `${c}.passed`)) : null; };

// Why a block has its status (warning, failed, not run), in the level's words; null when completed.
export function statusReason(trace, id, level) {
  const P = blockPaths(trace)[id];
  const b = at(trace, P);
  if (!b || b.status === "completed") return null;
  if (id === "signal_quality") {
    const w = valueOf(at(trace, `${P}.output.warnings`)) || [];
    return w.map((_, i) => [R(`${P}.output.warnings.value.${i}`, level === "developer" ? "text" : "warnPlain")]);
  }
  if (id === "preparation") return [["Some windows used the recording speed instead of their own; see the per-window fallback flags."]];
  if (id === "explanation") {
    if (b.status === "not_run") return [[b.operation_case || "Not generated yet."]];
    return [[level === "developer" ? "Fallback: " : "The fixed template was used because: ", R(`${P}.output.fallback_reason`, "text")]];
  }
  return [["The trace marks this step ", STATUS_WORD[b.status] || b.status, "."]];
}

// One short line per block (the block's second line in case mode), after the status word.
export function caseLine(trace, id, level) {
  const P = blockPaths(trace)[id];
  if (!P) return [];
  const b = at(trace, P);
  const o = `${P}.output`;
  const dev = level === "developer";
  switch (id) {
    case "input": {
      const upload = valueOf(at(trace, `${o}.source`)) === "upload";
      return upload ? ["Uploaded file · ", R(`${o}.origin`, "text")] : ["Dataset recording · ", R(`${o}.rec_id`, "recid")];
    }
    case "validation":
      return b.status === "completed" ? ["The checks that apply to this input passed"] : [];
    case "signal_quality": {
      const w = valueOf(at(trace, `${o}.warnings`)) || [];
      const first = w.length ? [" · ", R(`${o}.warnings.value.0`, dev ? "text" : "warnPlain")] : [" · no quality warnings"];
      return dev ? [R(`${o}.shaft_speed_hz`, "hz"), " · ", R(`${o}.speed_stable`, "stable"), ...first]
        : [R(`${o}.speed_stable`, "stable"), ...first];
    }
    case "preparation":
      return dev ? [R(`${o}.shaft_speed_hz`, "hz"), " · windows: ", R(`${o}.window_count`, "int"), " · microphone bursts removed: ", R(`${o}.mic_bursts_removed`, "int")]
        : [R(`${o}.shaft_speed_hz`, "speed")];
    case "physics_features":
      if (!dev) return ["Measurements for each window"];
      return missing(at(trace, `${o}.n_features`)) ? ["rows: ", R(`${o}.rows`, "int"), " · features: see feature dictionary"]
        : [R(`${o}.n_features`, "int"), " physics features per window · ", R(`${o}.rows`, "int"), " windows"];
    case "order_spectrogram":
      return dev ? ["windows: ", R(`${o}.windows`, "int"), " · order bins: ", R(`${o}.order_bins`, "int"), " · frames: ", R(`${o}.frames`, "int")]
        : ["A vibration picture for each window"];
    case "cnn6":
      return dev ? [R(`${o}.top_class`, "cls"), " · MSP ", R(`${o}.msp`, "score"), " · ", R(`${checkIndex(trace, "msp")}.passed`, "passfail")]
        : ["Best match: ", R(`${o}.top_class`, "cls"), " (model prediction)"];
    case "class6":
      return ["DERIVED from the ten-class output; not a separate classifier"];
    case "mahalanobis":
      return dev ? [R(`${o}.maha`, "check"), " vs limit ", R(`${o}.tau_d`, "check"), " · ", R(`${checkIndex(trace, "maha")}.passed`, "passfail"), " · lower = more familiar"]
        : [passed(trace, "maha") ? "The pattern looks familiar to the model" : "The pattern is unfamiliar to the model"];
    case "ae6":
      return dev ? [R(`${o}.ae_error`, "check"), " vs limit ", R(`${o}.tau_e`, "check"), " · ", R(`${checkIndex(trace, "ae")}.status`, "text")]
        : [passed(trace, "ae") ? "Looks like the healthy recordings" : "Differs from the healthy recordings"];
    case "b1s":
      return dev ? [R(`${o}.b1s`, "check"), " vs limit ", R(`${o}.tau_b`, "check"), " · ", R(`${checkIndex(trace, "b1s")}.status`, "text")]
        : [passed(trace, "b1s") ? "Looks like the healthy recordings at this speed" : "Differs from the healthy recordings at this speed"];
    case "lr":
      return [dev ? "" : "Second opinion: ", R(`${o}.top_class`, "cls"), " · not an input to the decision"].filter((x) => x !== "");
    case "decision":
      return [{ verdict: `${o}.display_label` }, ...(valueOf(at(trace, `${o}.decision`)) === "known_fault" ? [{ tag: "model prediction" }] : [])];
    case "evidence_record":
      return dev ? ["schema ", R(`${o}.provenance.schema_version`, "text"), " · manifest ", R(`${o}.provenance.manifest_sha256`, "hash")]
        : ["Saved with its versions and limitations"];
    case "explanation":
      if (b.status === "not_run") return ["not generated yet"];
      if (!valueOf(at(trace, `${o}.template_fallback_used`))) return [R(`${o}.source`, "source")];
      return dev ? ["template fallback: ", R(`${o}.fallback_reason`, "text")] : ["Written by the fixed template; the AI-written explanation could not be used"];
    case "presentation":
      return dev ? ["saved · ", R(`${o}.rerun_of`, "rerun")] : [valueOf(at(trace, `${o}.rerun_of`)) ? "Saved; a re-run of an earlier case" : "Saved"];
    default:
      return [];
  }
}

// Mechanic drawer: plain sentences, the same evidence as the Developer drawer without numbers, thresholds or hashes.
export function mechanicDetail(trace, id) {
  const paths = blockPaths(trace);
  const P = paths[id];
  const o = `${P}.output`;
  const dec = paths.decision;
  const d = (k) => valueOf(at(trace, `${dec}.output.${k}`));
  switch (id) {
    case "validation": {
      const cs = at(trace, `${o}.checks`) || [];
      return cs.map((c, i) => (valueOf(c.applied) ? [c.check, ": ", R(`${o}.checks.${i}.outcome`, "text")] : null)).filter(Boolean);
    }
    case "signal_quality": {
      const w = valueOf(at(trace, `${o}.warnings`)) || [];
      return [[R(`${o}.shaft_speed_hz`, "speed")], [R(`${o}.speed_stable`, "stable")],
        ...(w.length ? w.map((_, i) => [R(`${o}.warnings.value.${i}`, "warnPlain")]) : [["No recording-quality warnings."]])];
    }
    case "preparation":
      return [[R(`${o}.shaft_speed_hz`, "speed")], ["Cut into ", R(`${o}.window_count`, "int"), " overlapping windows."],
        ["Microphone bursts removed: ", R(`${o}.mic_bursts_removed`, "int")]];
    case "cnn6":
      return [["Best match: ", R(`${o}.top_class`, "cls"), " (model prediction, not a confirmed physical diagnosis)."],
        [passed(trace, "msp") ? "The classifier's confidence check passed." : "The classifier's confidence check did not pass."]];
    case "class6":
      return [["The same prediction grouped into six conditions (the bearing faults at each position together). It is worked out from the ten-condition result; it is not a separate model."]];
    case "mahalanobis":
      return [[passed(trace, "maha") ? "The pattern looks familiar: it is close to the examples the model learned from." : "The pattern is unfamiliar: it is far from the examples the model learned from."]];
    case "ae6":
      return [[passed(trace, "ae") ? "The healthy-pattern check agreed: the recording looks like the healthy recordings." : "The healthy-pattern check disagreed: the recording differs from the healthy recordings."]];
    case "b1s":
      return [[passed(trace, "b1s") ? "The speed-aware healthy check agreed." : "The speed-aware healthy check disagreed: the recording differs from healthy recordings at this speed."]];
    case "lr":
      return [["Second opinion: ", R(`${o}.top_class`, "cls"), "."], ["It helps explain the result; it is not an input to the decision."]];
    case "decision": {
      const cs = at(trace, `${o}.checks`) || [];
      return [
        [{ verdict: `${o}.display_label` }, ...(d("decision") === "known_fault" ? [{ tag: "model prediction" }] : [])],
        [finding({ diagnosis: { decision: d("decision"), fault_class: d("fault_class"), caution: d("caution") } })],
        ...(d("decision") === "unknown" ? [[R(`${o}.unknown_meaning`, "text")]] : []),
        ...cs.map((c, i) => [c.name, ": ", R(`${o}.checks.${i}.passed`, "passfail")]),
        [{ comparison: `${o}.rule_comparison` }],
      ];
    }
    case "evidence_record": {
      const lim = valueOf(at(trace, `${o}.limitations`)) || [];
      return lim.map((_, i) => [R(`${o}.limitations.value.${i}`, "text")]);
    }
    case "explanation": {
      const b = at(trace, P);
      if (b.status === "not_run") return [["Not generated yet."]];
      return [["Written by: ", R(`${o}.source`, "source")], ["Checks: ", R(`${o}.validation_outcome`, "text")]];
    }
    case "presentation": {
      const views = at(trace, `${o}.views`) || [];
      return views.map((v, i) => (valueOf(v.available) ? null : [v.view, ": not available (", R(`${o}.views.${i}.reason`, "text"), ")"])).filter(Boolean);
    }
    case "input": {
      const upload = valueOf(at(trace, `${o}.source`)) === "upload";
      return [[upload ? "An uploaded file." : "A recording from the development dataset: ", ...(upload ? [] : [R(`${o}.rec_id`, "recid")])],
        ["Recorded ", R(`${o}.created_at`, "text")]];
    }
    default:
      return [];
  }
}
