import { Bar, BarChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";

export default function ProbabilityChart({ d }) {
  const rows = Object.entries(d.class10_probabilities).map(([k, v]) => ({ name: k.replaceAll("_", " "), p: 100 * v }));
  const six = Object.entries(d.class6_probabilities).map(([k, v]) => `${k.replaceAll("_", " ")} ${(100 * v).toFixed(1)} %`);
  return (
    <section className="card">
      <h2 className="card-title">Class probabilities (CNN-6)</h2>
      <div style={{ width: "100%", height: 280 }}>
        <ResponsiveContainer>
          <BarChart data={rows} layout="vertical" margin={{ left: 40, right: 20 }}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis type="number" domain={[0, 100]} unit=" %" />
            <YAxis type="category" dataKey="name" width={170} tick={{ fontSize: 12 }} />
            <Tooltip formatter={(v) => `${Number(v).toFixed(2)} %`} />
            <Bar dataKey="p" fill="#334155" isAnimationActive={false} />
          </BarChart>
        </ResponsiveContainer>
      </div>
      <p className="text-xs text-slate-600">Six-class view: {six.join(" · ")}</p>
    </section>
  );
}
