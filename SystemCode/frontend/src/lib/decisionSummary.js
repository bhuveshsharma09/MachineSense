// Shared plain-language decision summary for the pipeline and dashboard. These checks are read from the saved case;
// the healthy checks describe evidence and can only add a caution to an otherwise healthy verdict.
import { human } from "./model.js";

export const checkResults = (checks) => Object.fromEntries((checks || []).map((c) => [c.key || c.check?.split(" ")[0], c.passed?.value ?? c.passed]));

export function healthyCheckPhrase(checks, decision) {
  const by = checkResults(checks);
  if (by.ae == null || by.b1s == null) return "Healthy checks not available";
  if (by.ae && by.b1s) return "Like the healthy recordings";
  if (!by.ae && !by.b1s) return "Unlike the healthy recordings";
  return decision === "healthy" ? "The two healthy checks disagree; this adds a caution"
    : "The two healthy checks disagree";
}

export function decisionSummary({ topClass, decision, caution, checks }) {
  const by = checkResults(checks);
  const match = topClass ? human(topClass) : "not available";
  const confidence = by.msp == null ? "Confidence check not available" : by.msp ? `Confident match: ${match}` : `Best match not confident enough: ${match}`;
  const familiarity = by.maha == null ? "Familiarity check not available" : by.maha ? "Familiar pattern" : "Unfamiliar pattern";
  const healthy = healthyCheckPhrase(checks, decision);
  const outcome = decision === "known_fault" ? "Known fault · model prediction" : decision === "unknown" ? "Further inspection needed"
    : caution ? "Healthy · caution" : "Healthy";
  return {
    match, confidence, familiarity, healthy, outcome,
    checks: by,
    lines: [`Best match: ${match}.`, `${confidence}. ${familiarity}.`, `${healthy}. Verdict: ${outcome}.`],
  };
}
