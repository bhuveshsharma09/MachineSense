// Small SVG charts drawn from real arrays (order spectra, envelope spectra, min/max waveforms).
export const W = 640, H = 200;

export function linePath(xs, ys, x0, x1, ymax) {
  let d = "", first = true;
  for (let i = 0; i < xs.length; i++) {
    if (xs[i] < x0 || xs[i] > x1) continue;
    const x = ((xs[i] - x0) / (x1 - x0)) * W, y = H - Math.min(ys[i] / ymax, 1) * H;
    d += (first ? "M" : "L") + x.toFixed(1) + "," + y.toFixed(1);
    first = false;
  }
  return { line: d, area: d ? d + `L${W},${H}L0,${H}Z` : "" };
}

export function rangeMax(xs, ys, x0, x1) {
  let m = 0;
  for (let i = 0; i < xs.length; i++) if (xs[i] >= x0 && xs[i] <= x1 && ys[i] > m) m = ys[i];
  return m || 1;
}

export function bandPath(mins, maxs, ymax) {
  const n = mins.length, x = (i) => ((i / (n - 1)) * W).toFixed(1), y = (v) => (H / 2 - (v / ymax) * (H / 2)).toFixed(1);
  let d = "";
  for (let i = 0; i < n; i++) d += (i ? "L" : "M") + x(i) + "," + y(maxs[i]);
  for (let i = n - 1; i >= 0; i--) d += "L" + x(i) + "," + y(mins[i]);
  return d + "Z";
}

export function Svg({ paths }) {
  return (
    <svg viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="none" aria-hidden="true"
      style={{ position: "absolute", inset: 0, width: "100%", height: "100%", overflow: "visible" }}>
      {paths.map((p, i) => (
        <path key={i} d={p.d} fill={p.fill || "none"} stroke={p.stroke || "none"} strokeWidth={p.w || 0}
          strokeDasharray={p.dash} vectorEffect="non-scaling-stroke" strokeLinejoin="round" />
      ))}
    </svg>
  );
}
