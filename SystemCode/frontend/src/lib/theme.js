// Design tokens from design/prototype_v2 (MachineSense v2).
export const K = {
  ink: "oklch(0.23 0.006 260)", ink2: "oklch(0.40 0.008 260)", ink3: "oklch(0.50 0.008 260)",
  white: "oklch(0.995 0.002 85)", violet: "oklch(0.54 0.21 277)", orange: "oklch(0.66 0.19 42)",
  orangeT: "oklch(0.53 0.17 40)", amber: "oklch(0.80 0.14 75)", amberT: "oklch(0.50 0.11 65)",
  lime: "oklch(0.89 0.17 118)", limeT: "oklch(0.45 0.11 125)", green: "oklch(0.64 0.11 160)",
  greenT: "oklch(0.47 0.09 160)",
};
export const al = (c, x) => c.slice(0, -1) + " / " + x + ")";
// Pipeline graph palette (App-4.3 restyle, from the reference "Diagnostics Flow" UI): neutral grey for measured signals,
// the electric blue for model outputs and decisions, lime for explanations; keyed by the evidence tag the trace names.
export const PIPE = {
  Measured: { c: "oklch(0.52 0.008 260)", b: "oklch(0.52 0.008 260 / 0.12)" },
  Model: { c: K.violet, b: "oklch(0.54 0.21 277 / 0.12)" },
  Guidance: { c: "oklch(0.86 0.19 118)", b: "oklch(0.89 0.17 118 / 0.35)" },
  Missing: { c: K.amberT, b: "oklch(0.80 0.14 75 / 0.2)" },
};
// per-kind accent where one tag covers two kinds: decisions in orange, apart from model outputs (App-4.4)
export const PIPE_KIND = { decision: { c: K.orange, b: "oklch(0.66 0.19 42 / 0.14)" } };
export const TAG = {
  Measured: { c: K.ink, b: "oklch(0.23 0.006 260 / 0.07)" },
  Model: { c: "oklch(0.44 0.09 250)", b: "oklch(0.55 0.1 250 / 0.12)" },
  Guidance: { c: "oklch(0.42 0.07 190)", b: "oklch(0.55 0.08 190 / 0.14)" },
  Missing: { c: K.amberT, b: "oklch(0.80 0.14 75 / 0.2)" },
};
