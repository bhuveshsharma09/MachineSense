import { useState } from "react";
import * as Switch from "@radix-ui/react-switch";

function heatColour(v) {
  const t = Math.max(0, Math.min(1, v));
  return `rgb(${Math.round(255 - 200 * t)}, ${Math.round(255 - 170 * t)}, ${Math.round(255 - 60 * t)})`;
}

export default function ExplanationPanel({ xai }) {
  const [advanced, setAdvanced] = useState(false);
  if (!xai) return null;
  const lr = xai.lr, cnn = xai.cnn;
  const maxC = Math.max(...lr.top_features.map((f) => Math.abs(f.contribution)), 1e-9);
  return (
    <section className="card">
      <h2 className="card-title">Explanations</h2>
      <h3 className="text-lg font-semibold">Secondary logistic-regression evidence</h3>
      <p className="text-sm text-slate-600 mb-2">
        {lr.label[0].toUpperCase() + lr.label.slice(1)} (for its class: {lr.class.replaceAll("_", " ")}). These contributions
        explain the secondary model, not the main neural network's decision.
      </p>
      <table className="w-full text-sm">
        <thead><tr className="text-left text-slate-500"><th>Feature</th><th>Channel</th><th>Definition</th><th className="w-48">Contribution</th></tr></thead>
        <tbody>
          {lr.top_features.map((f) => (
            <tr key={f.feature} className="border-t border-slate-100 align-top">
              <td className="py-1 font-mono text-xs">{f.feature}</td>
              <td className="py-1">{f.channel}</td>
              <td className="py-1 text-slate-600">{f.definition}</td>
              <td className="py-1">
                <div className="flex items-center gap-2">
                  <div className="h-2 bg-slate-700" style={{ width: `${(100 * Math.abs(f.contribution)) / maxC}%` }} />
                  <span className="text-xs">{f.contribution.toFixed(3)}</span>
                </div>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
      <div className="mt-4 flex items-center gap-3">
        <Switch.Root id="adv" checked={advanced} onCheckedChange={setAdvanced}
                     className="relative h-6 w-11 rounded-full bg-slate-300 data-[state=checked]:bg-slate-900">
          <Switch.Thumb className="block h-5 w-5 translate-x-0.5 rounded-full bg-white transition-transform data-[state=checked]:translate-x-[22px]" />
        </Switch.Root>
        <label htmlFor="adv" className="text-sm font-medium">Advanced: CNN gradient attribution</label>
      </div>
      {advanced && (
        <div className="mt-3">
          <p className="text-sm text-amber-900 bg-amber-50 border border-amber-200 rounded p-2 mb-2">
            {cnn.label[0].toUpperCase() + cnn.label.slice(1)}. What the model attended to for {cnn.class.replaceAll("_", " ")};
            not proof of the fault's location. {xai.note}
          </p>
          <div className="overflow-x-auto">
            <table className="text-xs">
              <tbody>
                {cnn.heat_map.channels.map((ch, i) => (
                  <tr key={ch}>
                    <td className="pr-2 text-right text-slate-600">{ch}</td>
                    {cnn.heat_map.values[i].map((v, j) => (
                      <td key={j} title={`${ch}, orders ${cnn.heat_map.order_groups[j][0].toFixed(2)}–${cnn.heat_map.order_groups[j][1].toFixed(2)}: ${v.toFixed(2)}`}
                          style={{ background: heatColour(v), width: 16, height: 16 }} />
                    ))}
                  </tr>
                ))}
                <tr>
                  <td />
                  {cnn.heat_map.order_groups.map((g, j) => (
                    <td key={j} className="text-[9px] text-slate-500">{j % 8 === 0 ? g[0].toFixed(1) : ""}</td>
                  ))}
                </tr>
              </tbody>
            </table>
          </div>
          <p className="mt-1 text-xs text-slate-500">Order groups on a log axis (0.5–150 orders). Top channels:{" "}
            {cnn.top_channels.map((c) => `${c.channel} ${(100 * c.share).toFixed(0)} %`).join(", ")}.</p>
        </div>
      )}
    </section>
  );
}
