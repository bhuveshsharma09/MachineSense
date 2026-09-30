import { forwardRef, useCallback, useEffect, useImperativeHandle, useMemo, useRef, useState } from "react";
import { K, TAG } from "../lib/theme.js";


// App-4.2: the SVG canvas of the pipeline architecture. Structure only: titles, one subtitle line (the block's purpose),
// ports and edges from the trace; no case value is rendered. Pan and zoom move the SVG viewBox (pointer drag, wheel,
// arrow keys, + / -); blocks are focusable in trace order.
const ZOOM = { min: 0.2, max: 2.5, step: 1.25 };
const PAN_STEP = 80;  // screen px per arrow key
const LABEL_MIN_SCALE = 0.6;  // edge labels are hidden when zoomed out below readable size (they stay in the drawer)
export const LEGIBLE = 0.8;  // App-4.2.1: below this scale block titles are no longer comfortably readable
const INSET = { top: 96, side: 16, bottom: 16 };  // the header floats over the canvas; fit frames the area below it

export const edgeColour = (kind, edgeKinds) => TAG[edgeKinds?.[kind]]?.c || K.ink3;
const clamp = (v, lo, hi) => Math.min(hi, Math.max(lo, v));
const usable = (size) => ({ w: Math.max(1, size.w - 2 * INSET.side), h: Math.max(1, size.h - INSET.top - INSET.bottom) });
const topAligned = (bounds, size, scale) => ({ cx: bounds.x + bounds.w / 2, cy: bounds.y + (size.h / 2 - INSET.top) / scale, scale });

// Fit to screen with a legibility floor: the whole graph when that is legible (scale ≥ LEGIBLE); otherwise the width at
// a legible scale, top-aligned and pannable vertically; on a screen too narrow even for that, LEGIBLE and pan both ways.
export function fitView(bounds, size) {
  const u = usable(size);
  const whole = Math.min(u.w / bounds.w, u.h / bounds.h);
  if (whole >= LEGIBLE) {
    const scale = clamp(whole, ZOOM.min, 1.2);
    return { cx: bounds.x + bounds.w / 2, cy: bounds.y + bounds.h / 2 - (INSET.top - INSET.bottom) / 2 / scale, scale };
  }
  return topAligned(bounds, size, clamp(Math.max(u.w / bounds.w, LEGIBLE), LEGIBLE, 1.2));
}

// the default view: scale 1 (no text below its nominal size), top-aligned, centred on the graph
export const homeView = (bounds, size) => topAligned(bounds, size, 1);

const PipelineCanvas = forwardRef(function PipelineCanvas({ layout, edgeKinds, level, selected, onSelect, initialSize, onViewChange,
  showAllLabels = false }, ref) {
  const wrap = useRef(null);
  const [size, setSize] = useState(initialSize || { w: 1440, h: 800 });
  const [view, setView] = useState(() => homeView(layout.bounds, initialSize || { w: 1440, h: 800 }));
  const [hover, setHover] = useState(null);
  const drag = useRef(null);

  useEffect(() => {
    const el = wrap.current;
    if (!el || typeof ResizeObserver === "undefined") return undefined;
    const ro = new ResizeObserver(([e]) => setSize({ w: e.contentRect.width, h: e.contentRect.height }));
    ro.observe(el);
    return () => ro.disconnect();
  }, []);
  useEffect(() => { onViewChange?.(view, size); }, [view, size]);

  const zoomBy = useCallback((f, at) => setView((v) => {
    const s = clamp(v.scale * f, ZOOM.min, ZOOM.max);
    if (!at) return { ...v, scale: s };
    // keep the point under the cursor fixed
    const wx = v.cx + (at.x - size.w / 2) / v.scale, wy = v.cy + (at.y - size.h / 2) / v.scale;
    return { cx: wx - (at.x - size.w / 2) / s, cy: wy - (at.y - size.h / 2) / s, scale: s };
  }), [size]);
  const panBy = (dx, dy) => setView((v) => ({ ...v, cx: v.cx + dx / v.scale, cy: v.cy + dy / v.scale }));
  const centreOn = (x, y) => setView((v) => ({ ...v, cx: x, cy: y }));

  useImperativeHandle(ref, () => ({
    zoomIn: () => zoomBy(ZOOM.step), zoomOut: () => zoomBy(1 / ZOOM.step),
    fit: () => setView(fitView(layout.bounds, size)), reset: () => setView(homeView(layout.bounds, size)), centreOn,
  }), [layout, size, zoomBy]);

  useEffect(() => {  // wheel zoom needs a non-passive listener to stop the page from scrolling
    const el = wrap.current;
    if (!el) return undefined;
    const onWheel = (e) => {
      e.preventDefault();
      const r = el.getBoundingClientRect();
      zoomBy(Math.exp(-e.deltaY * 0.0015), { x: e.clientX - r.left, y: e.clientY - r.top });
    };
    el.addEventListener("wheel", onWheel, { passive: false });
    return () => el.removeEventListener("wheel", onWheel);
  }, [zoomBy]);

  const onKeyDown = (e) => {
    const k = e.key;
    if (k === "ArrowLeft") panBy(-PAN_STEP, 0);
    else if (k === "ArrowRight") panBy(PAN_STEP, 0);
    else if (k === "ArrowUp") panBy(0, -PAN_STEP);
    else if (k === "ArrowDown") panBy(0, PAN_STEP);
    else if (k === "+" || k === "=") zoomBy(ZOOM.step);
    else if (k === "-" || k === "_") zoomBy(1 / ZOOM.step);
    else return;
    e.preventDefault();
  };
  const onPointerDown = (e) => {
    if (e.button !== 0 || e.target.closest?.(".pl-block, .pl-frame-cap")) return;
    drag.current = { x: e.clientX, y: e.clientY };
    e.currentTarget.setPointerCapture?.(e.pointerId);
  };
  const onPointerMove = (e) => {
    if (!drag.current) return;
    panBy(drag.current.x - e.clientX, drag.current.y - e.clientY);
    drag.current = { x: e.clientX, y: e.clientY };
  };
  const onPointerUp = () => { drag.current = null; };

  // keep a block brought into focus by Tab inside the viewport
  const reveal = (n) => {
    const vw = size.w / view.scale, vh = size.h / view.scale;
    const x0 = view.cx - vw / 2, y0 = view.cy - vh / 2;
    if (n.x < x0 || n.x + n.w > x0 + vw || n.y < y0 || n.y + n.h > y0 + vh) centreOn(n.x + n.w / 2, n.y + n.h / 2);
  };

  const vb = [view.cx - size.w / view.scale / 2, view.cy - size.h / view.scale / 2, size.w / view.scale, size.h / view.scale];
  // edge labels: the hovered or selected block's edges by default, every edge with "show all labels" (App-4.2.1)
  const focus = [hover, selected].filter(Boolean);
  const visibleKey = showAllLabels ? "*" : focus.join("|");
  const labelAt = useMemo(() => {
    const touches = (e, id) => e.from.split(".")[0] === id || e.to.split(".")[0] === id;
    const vis = new Set(layout.edges.filter((e) => showAllLabels || focus.some((id) => touches(e, id))).map((e) => e.id));
    return vis.size ? layout.labels(vis) : {};
  }, [layout, visibleKey]);
  const titleOf = (b) => layout.title(b, level);
  const connected = new Set(hover ? layout.edges.filter((e) => e.from.startsWith(hover + ".") || e.to.startsWith(hover + ".")).map((e) => e.id) : []);
  const blockEl = (n) => {
    const b = n.block, sel = selected === b.id;
    const act = () => onSelect(b.id);
    return (
      <foreignObject key={b.id} x={n.x - 4} y={n.y - 4} width={n.w + 8} height={n.h + 8} style={{ overflow: "visible" }}>
        <div className={"pl-block" + (sel ? " glass-soft is-sel" : "")} role="button" tabIndex={0} aria-label={titleOf(b)} aria-pressed={sel}
          title={titleOf(b)} data-block={b.id} style={{ width: n.w, height: n.h }}
          onClick={act} onKeyDown={(e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); act(); } }}
          onFocus={() => reveal(n)} onMouseEnter={() => setHover(b.id)} onMouseLeave={() => setHover(null)}>
          <div className="pl-title">{titleOf(b)}</div>
          {b.purpose && <div className="pl-sub">{b.purpose}</div>}
        </div>
      </foreignObject>
    );
  };
  const portEl = (key, p) => {
    const c = p.kind ? edgeColour(p.kind, edgeKinds) : null;
    const below = p.side === "out";
    return (
      <g key={key} data-port={key}>
        <circle cx={p.x} cy={p.y} r={2.5} fill={c || "oklch(0.995 0.002 85)"} stroke={c || K.ink3} strokeWidth={1}>
          <title>{p.label}</title>
        </circle>
        <text x={p.x + 5} y={below ? p.y + 13 : p.y - 5} className="pl-port-label">{p.short}</text>
      </g>
    );
  };

  return (
    <div ref={wrap} className="pl-canvas-wrap" tabIndex={-1} onKeyDown={onKeyDown}>
      <svg className="pl-canvas" viewBox={vb.join(" ")} width="100%" height="100%" preserveAspectRatio="xMidYMid meet"
        role="group" aria-label="Pipeline architecture diagram"
        onPointerDown={onPointerDown} onPointerMove={onPointerMove} onPointerUp={onPointerUp} onPointerCancel={onPointerUp}>
        {layout.frames.map((f) => (
          <g key={f.id} data-frame={f.id}>
            <rect x={f.x} y={f.y} width={f.w} height={f.h} rx={18} fill="none" stroke="oklch(0.23 0.006 260 / 0.12)" strokeDasharray="5 5" />
          </g>
        ))}
        {layout.unplacedY !== null && (
          <text x={layout.bounds.x + 20} y={layout.unplacedY - 22} className="pl-frame-label">UNPLACED · NO LAYOUT ENTRY</text>
        )}
        <g className="pl-edges">
          {layout.edges.map((e) => {
            const c = edgeColour(e.kind, edgeKinds);
            const [s, z] = [e.points[0], e.points[e.points.length - 1]];
            const dim = hover && !connected.has(e.id);
            return (
              <g key={e.id} data-edge={e.id} data-kind={e.kind} opacity={dim ? 0.28 : 1}>
                <path d={e.d} fill="none" stroke={c} strokeWidth={1.25} strokeLinejoin="round"><title>{e.label || e.kind}</title></path>
                <circle cx={s[0]} cy={s[1]} r={2.25} fill={c} />
                <circle cx={z[0]} cy={z[1]} r={2.25} fill={c} />
                {e.label && labelAt[e.id] && view.scale >= LABEL_MIN_SCALE && (
                  <text x={labelAt[e.id].x} y={labelAt[e.id].y} textAnchor={labelAt[e.id].anchor} className="pl-edge-label" fill={c}>{e.label}</text>
                )}
              </g>
            );
          })}
        </g>
        {layout.order.map((id) => {  // trace order = tab order
          const f = layout.frames.find((x) => x.id === id);
          if (f) {
            const sel = selected === id;
            return (
              <foreignObject key={id} x={f.captionX} y={f.y - 11} width={f.captionW} height={22} style={{ overflow: "visible" }}>
                <div className={"pl-frame-cap mono" + (sel ? " is-sel" : "")} role="button" tabIndex={0} aria-label={titleOf(f.block)}
                  aria-pressed={sel} data-block={id} onClick={() => onSelect(id)}
                  onKeyDown={(e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); onSelect(id); } }}>
                  {titleOf(f.block)}
                </div>
              </foreignObject>
            );
          }
          const n = layout.nodes[id];
          return n ? blockEl(n) : null;
        })}
        <g className="pl-ports">{Object.entries(layout.ports).map(([k, p]) => portEl(k, p))}</g>
      </svg>
    </div>
  );
});

export default PipelineCanvas;

// The overview (App-4.2.1): every group frame and block as an outline, the edges faintly, and the visible area; a click
// centres the canvas there.
export function MiniMap({ layout, view, size, onCentre, width = 240 }) {
  const b = layout.bounds;
  const h = Math.round((width * b.h) / b.w);
  const k = width / b.w;
  const vw = view ? size.w / view.scale : 0, vh = view ? size.h / view.scale : 0;
  const click = (e) => {
    const r = e.currentTarget.getBoundingClientRect();
    onCentre(b.x + (e.clientX - r.left) / k, b.y + (e.clientY - r.top) / k);
  };
  const px = 1 / k;  // one screen pixel in graph units
  return (
    <svg className="pl-minimap" width={width} height={h} viewBox={`${b.x} ${b.y} ${b.w} ${b.h}`} onClick={click}
      role="img" aria-label="Overview of the whole pipeline with the visible area">
      {layout.edges.map((e) => <path key={e.id} d={e.d} fill="none" stroke={K.ink3} strokeOpacity={0.35} strokeWidth={px} />)}
      {layout.frames.map((f) => <rect key={f.id} x={f.x} y={f.y} width={f.w} height={f.h} rx={18} fill="none" stroke={K.ink3}
        strokeOpacity={0.5} strokeWidth={px} strokeDasharray={`${3 * px} ${3 * px}`} />)}
      {Object.values(layout.nodes).map((n) => <rect key={n.id} x={n.x} y={n.y} width={n.w} height={n.h} rx={14}
        fill="oklch(0.99 0.002 85 / 0.72)" stroke={K.ink} strokeOpacity={0.55} strokeWidth={px} />)}
      {view && <rect x={view.cx - vw / 2} y={view.cy - vh / 2} width={vw} height={vh} fill="none" stroke={K.violet} strokeWidth={1.5 * px} />}
    </svg>
  );
}
