export default function SignalQualityPanel({ q }) {
  return (
    <section className="card">
      <h2 className="card-title">Signal quality</h2>
      {q.warnings.length > 0 ? (
        <ul className="mb-3 space-y-1" aria-label="warnings">
          {q.warnings.map((w, i) => (
            <li key={i} className="rounded-md border border-amber-400 bg-amber-100 px-2 py-1 text-sm font-medium text-amber-900">⚠ {w}</li>
          ))}
        </ul>
      ) : (
        <p className="mb-3 text-sm text-emerald-700">No signal-quality warnings.</p>
      )}
      <dl className="grid grid-cols-2 gap-y-1 text-sm">
        <dt className="text-slate-500">Shaft speed</dt><dd>{q.shaft_speed_hz.toFixed(2)} Hz</dd>
        <dt className="text-slate-500">Speed variation</dt><dd>{q.speed_range_pct.toFixed(2)} % {q.speed_stable ? "(stable)" : "(not stable)"}</dd>
        <dt className="text-slate-500">Trained speeds</dt><dd>{q.train_speed_range_hz.map((v) => v.toFixed(1)).join("–")} Hz</dd>
        <dt className="text-slate-500">Healthy calibration</dt><dd>{q.calibration_normal_speed_range_hz.map((v) => v.toFixed(1)).join("–")} Hz</dd>
        <dt className="text-slate-500">Tacho glitches</dt><dd>{q.tacho_glitches}</dd>
        <dt className="text-slate-500">Mic bursts removed</dt><dd>{q.mic_bursts_removed}</dd>
        <dt className="text-slate-500">Missing samples</dt><dd>{q.nan_count}</dd>
        <dt className="text-slate-500">Clipping</dt><dd>{q.clipping_channels.join(", ") || "none"}</dd>
      </dl>
    </section>
  );
}
