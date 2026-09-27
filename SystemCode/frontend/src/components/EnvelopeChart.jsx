import { CartesianGrid, Legend, Line, LineChart, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";

const CURSOR_COLOURS = { shaft: "#0f172a", ftf: "#7c3aed", bsf: "#0891b2", bpfo: "#dc2626", bpfi: "#ea580c" };

export default function EnvelopeChart({ physics }) {
  const e = physics.envelope_spectrum;
  const chans = Object.keys(e.channels);
  const rows = e.orders.map((o, i) => Object.fromEntries([["order", o], ...chans.map((c) => [c, e.channels[c][i]])]));
  const bf = physics.bearing_frequencies_hz;
  return (
    <section className="card">
      <h2 className="card-title">Envelope spectrum ({e.band_hz[0] / 1000}–{e.band_hz[1] / 1000} kHz band), in orders of the measured speed</h2>
      <p className="text-sm text-slate-600 mb-2">
        Shaft {physics.shaft_speed_hz.toFixed(2)} Hz ({physics.shaft_speed_rpm.toFixed(0)} rpm) · BPFO {bf.bpfo.toFixed(1)} Hz ·
        BPFI {bf.bpfi.toFixed(1)} Hz · 2×BSF {bf.bsf_defect_line_2x.toFixed(1)} Hz · FTF {bf.ftf.toFixed(1)} Hz
      </p>
      <div style={{ width: "100%", height: 300 }}>
        <ResponsiveContainer>
          <LineChart data={rows} margin={{ right: 20 }}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="order" type="number" domain={[0, 12]} ticks={[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]}
                   label={{ value: "order (× shaft speed)", position: "insideBottom", offset: -2 }} />
            <YAxis tickFormatter={(v) => v.toExponential(0)} width={60} />
            <Tooltip labelFormatter={(o) => `order ${Number(o).toFixed(2)} = ${(o * physics.shaft_speed_hz).toFixed(1)} Hz`}
                     formatter={(v) => Number(v).toExponential(3)} />
            <Legend verticalAlign="top" />
            {chans.map((c, i) => <Line key={c} dataKey={c} dot={false} stroke={i ? "#2563eb" : "#16a34a"} strokeWidth={1} isAnimationActive={false} />)}
            {Object.entries(physics.cursors_orders).flatMap(([k, os]) => os.map((o, j) => (
              <ReferenceLine key={`${k}${j}`} x={o} stroke={CURSOR_COLOURS[k] || "#64748b"} strokeDasharray="4 3"
                             label={j === 0 ? { value: k.toUpperCase(), position: "top", fontSize: 10, fill: CURSOR_COLOURS[k] } : undefined} />
            )))}
          </LineChart>
        </ResponsiveContainer>
      </div>
    </section>
  );
}
