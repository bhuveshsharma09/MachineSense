// Choose the dashboard card from structured tool refs and citations, never from AI-written answer prose.
const TARGETS = {
  finding: { view: "mechanic", card: "finding", label: "See the finding on the dashboard" },
  checks: { view: "mechanic", card: "suggested-checks", label: "See suggested checks on the dashboard" },
  sensors: { view: "developer", card: "sensor-channels", label: "See the sensor comparison on the dashboard" },
  quality: { view: "developer", card: "signal-quality", label: "See signal quality on the dashboard" },
  decision: { view: "developer", card: "decision-checks", label: "See decision checks on the dashboard" },
  spectrum: { view: "mechanic", card: "measurements", label: "See the measurements on the dashboard" },
  report: { view: "developer", card: "fixed-report", label: "See the report on the dashboard" },
};

export function dashboardTarget(question, answer) {
  if (!answer || answer.source === "social" || ["status", "procedure"].includes(answer.fallback_kind)) return null;
  if (answer.source === "fallback") return TARGETS.report;
  const refs = (answer.claims || []).flatMap((claim) => claim.refs || []);
  const tools = (answer.actions || []).map((action) => action.tool);
  const citations = answer.citations || [];
  const has = (prefix) => refs.some((ref) => ref.startsWith(prefix));
  const questionText = String(question || "").toLowerCase();
  const guidance = citations.some((c) => c.kind !== "procedure") || has("guidance.") || tools.includes("lookup_guidance");
  if (guidance && /\b(inspect|next|guidance|steps?)\b/.test(questionText)) return TARGETS.checks;
  if (/\b(sensor|channel|accelerometer|microphone)\b/.test(questionText) && (has("evidence.") || tools.includes("get_evidence"))) return TARGETS.sensors;
  if (/\bwhy\b.*\b(flag|diagnos|result|say)\b/.test(questionText)) return TARGETS.finding;
  if (has("assessment.") || has("diagnosis.checks.")) return TARGETS.decision;
  if (has("signal_quality.") || tools.includes("check_signal")) return TARGETS.quality;
  if (has("evidence.") || tools.includes("get_evidence")) return TARGETS.sensors;
  if (has("physics.")) return TARGETS.spectrum;
  if (guidance) return TARGETS.checks;
  if (has("diagnosis.") || tools.includes("diagnose")) return TARGETS.finding;
  return null;
}
