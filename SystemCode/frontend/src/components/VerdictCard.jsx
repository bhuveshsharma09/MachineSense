const TONE = { healthy: "border-emerald-400 bg-emerald-50", known_fault: "border-rose-400 bg-rose-50",
               unknown: "border-amber-400 bg-amber-50" };

export default function VerdictCard({ d }) {
  const label = d.decision === "unknown" ? "further inspection needed" : d.display_label;
  return (
    <section className={`card border-2 ${TONE[d.decision]}`}>
      <h2 className="card-title">Verdict</h2>
      <div className="flex flex-wrap items-center gap-3">
        <span className="text-2xl font-semibold">{label}</span>
        {d.caution && (
          <span className="rounded-full bg-amber-200 px-3 py-1 text-xs font-semibold text-amber-900"
                title="healthy — caution: a healthy-baseline check disagrees; consider a recheck">
            caution: {d.caution.replaceAll("_", " ")}
          </span>
        )}
      </div>
      {d.decision === "known_fault" && (
        <p className="mt-2 text-sm text-slate-700">Predicted fault location according to the dataset labels.</p>
      )}
      <dl className="mt-3 grid grid-cols-2 gap-x-6 gap-y-1 text-sm">
        <dt className="text-slate-500">Rule</dt><dd>{d.rule_variant} <span className="text-slate-500">(provisional demo default)</span></dd>
        <dt className="text-slate-500">Classifier confidence (MSP)</dt>
        <dd>{d.scores.msp.toFixed(6)} {d.scores.msp >= d.thresholds.tau_p ? "≥" : "<"} {d.thresholds.tau_p.toFixed(6)}</dd>
        <dt className="text-slate-500">Familiarity (Mahalanobis)</dt>
        <dd>{d.scores.maha.toFixed(2)} {d.scores.maha <= d.thresholds.tau_d ? "≤" : ">"} {d.thresholds.tau_d.toFixed(2)}</dd>
        <dt className="text-slate-500">Autoencoder error</dt>
        <dd>{d.scores.ae_error.toFixed(4)} {d.scores.ae_error <= d.thresholds.tau_e ? "≤" : ">"} {d.thresholds.tau_e.toFixed(4)}</dd>
        <dt className="text-slate-500">Speed-aware baseline (B1s)</dt>
        <dd>{d.scores.b1s.toFixed(4)} {d.scores.b1s <= d.thresholds.tau_b ? "≤" : ">"} {d.thresholds.tau_b.toFixed(4)}</dd>
      </dl>
      <div className="mt-3 text-sm">
        <span className="text-slate-500">Other rule variants: </span>
        {Object.entries(d.rule_comparison).map(([k, v]) => (
          <span key={k} className="mr-3"><strong>{k}</strong> {v === "unknown" ? "further inspection needed" : v.replaceAll("_", " ")}</span>
        ))}
      </div>
      <p className="mt-2 text-sm text-slate-600">Secondary logistic regression: {d.lr_secondary.top_class.replaceAll("_", " ")}</p>
    </section>
  );
}
