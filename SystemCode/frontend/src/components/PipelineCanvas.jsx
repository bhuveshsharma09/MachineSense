import { forwardRef, useCallback, useEffect, useImperativeHandle, useMemo, useRef, useState } from "react";
import { K, PIPE, PIPE_CASING, PIPE_KIND, PIPE_TEXT } from "../lib/theme.js";
import { CaseLine, DecisionPanel } from "./PipelineCase.jsx";
import { DEC } from "../lib/pipelineLayout.js";
import { STATUS_WORD } from "../lib/pipelineCase.js";
import { PREVIEW_BLOCKS, Thumb } from "./PipelinePreview.jsx";


// App-4.2: the SVG canvas of the pipeline architecture. Structure only: titles, one subtitle line (the block's purpose),
// ports and edges from the trace; no case value is rendered. Pan and zoom move the SVG viewBox (pointer drag, wheel,
// arrow keys, + / -); blocks are focusable in trace order.
const ZOOM = { min: 0.2, max: 2.5, step: 1.25 };
const PAN_STEP = 80;  // screen px per arrow key
const LABEL_MIN_SCALE = 0.6;  // edge labels are hidden when zoomed out below readable size (they stay in the drawer)
const PORT_LABEL_MIN_SCALE = 0.9;  // App-4.3: the 9.5px port micro-labels only where they stay about 10px or larger
export const LEGIBLE = 0.8;  // App-4.2.1: below this scale block titles are no longer comfortably readable
export const INSET = { top: 100, side: 40, bottom: 36 };  // screen margins of fit to screen; the header floats above ``top``

export const edgeColour = (kind, edgeKinds) => PIPE_KIND[kind]?.c || PIPE[edgeKinds?.[kind]]?.c || K.ink3;
// App-4.5 contrast: label text and a casing under lines whose colour is too light to read on the canvas
export const edgeTextColour = (kind, edgeKinds) => PIPE_KIND[kind]?.text || PIPE_TEXT[edgeKinds?.[kind]] || edgeColour(kind, edgeKinds);
export const edgeCasing = (kind, edgeKinds) => PIPE_CASING[edgeKinds?.[kind]] || null;
const clamp = (v, lo, hi) => Math.min(hi, Math.max(lo, v));
// top: the header's height when it grows (App-4.5: the different-bundle notice adds a line), so fit never hides a block
const usable = (size, top = INSET.top) => ({ w: Math.max(1, size.w - 2 * INSET.side), h: Math.max(1, size.h - top - INSET.bottom) });

// Fit to screen with a legibility floor: the whole graph when that is legible (scale ≥ LEGIBLE); otherwise (App-8, the
// graph reads left to right) the height at a legible scale, left-aligned and pannable horizontally; on a screen too short
// even for that, LEGIBLE and pan both ways.
export function fitView(bounds, size, top = INSET.top) {
  const u = usable(size, top);
  const whole = Math.min(u.w / bounds.w, u.h / bounds.h);
  if (whole >= LEGIBLE) {
    const scale = clamp(whole, ZOOM.min, 1.2);
    return { cx: bounds.x + bounds.w / 2, cy: bounds.y + bounds.h / 2 - (top - INSET.bottom) / 2 / scale, scale };
  }
  const scale = clamp(Math.max(u.h / bounds.h, LEGIBLE), LEGIBLE, 1.2);
  return { cx: bounds.x + (size.w / 2 - INSET.side) / scale, cy: bounds.y + bounds.h / 2 - (top - INSET.bottom) / 2 / scale, scale };
}


const PipelineCanvas = forwardRef(function PipelineCanvas({ layout, edgeKinds, level, selected, onSelect, initialSize, onViewChange,
  showAllLabels = false, mode = "architecture", trace = null, preview = null, topInset = INSET.top }, ref) {
  const wrap = useRef(null);
  const [size, setSize] = useState(initialSize || { w: 1440, h: 800 });
  // App-4.3: the page opens at fit to screen (port micro-labels are hidden at that scale, so no text is below ~10px)
  const [view, setView] = useState(() => fitView(layout.bounds, initialSize || { w: 1440, h: 800 }, topInset));
  const fitted = useRef(false);
  const [hover, setHover] = useState(null);
  const drag = useRef(null);

  useEffect(() => {
    const el = wrap.current;
    if (!el || typeof ResizeObserver === "undefined") return undefined;
    const ro = new ResizeObserver(([e]) => {
      const s = { w: e.contentRect.width, h: e.contentRect.height };
      setSize(s);
      if (!fitted.current && s.w > 0 && s.h > 0) { fitted.current = true; setView(fitView(layout.bounds, s, topInset)); }  // fit the real size once
    });
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
    fit: () => setView(fitView(layout.bounds, size, topInset)), reset: () => setView(fitView(layout.bounds, size, topInset)), centreOn,
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
  // App-4.5: the accessible name carries the status (not only the dot) and, on the decision, the verdict
  const nameOf = (b) => {
    if (mode !== "case" || !trace) return titleOf(b);
    const verdict = b.id === "decision" ? b.output?.display_label?.value : null;
    return `${titleOf(b)}, ${STATUS_WORD[b.status] || b.status}${verdict ? `, ${verdict}` : ""}`;
  };
  const connected = new Set(hover ? layout.edges.filter((e) => e.from.startsWith(hover + ".") || e.to.startsWith(hover + ".")).map((e) => e.id) : []);
  const blockEl = (n) => {
    const b = n.block, sel = selected === b.id;
    const act = () => onSelect(b.id);
    const dec = b.id === "decision" && layout.direction === "lr";
    return (
      <foreignObject key={b.id} x={n.x - 4} y={n.y - 4} width={n.w + 8} height={n.h + 8} style={{ overflow: "visible" }}>
        <div className={"pl-block" + (sel ? " glass-soft is-sel" : "")} role="button" tabIndex={0} aria-label={nameOf(b)} aria-pressed={sel}
          title={titleOf(b)} data-block={b.id} style={{ width: n.w, height: n.h }} data-kind={dec ? "decision" : undefined}
          onClick={act} onKeyDown={(e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); act(); } }}
          onFocus={() => reveal(n)} onMouseEnter={() => setHover(b.id)} onMouseLeave={() => setHover(null)}>
          {dec ? <DecisionPanel trace={mode === "case" ? trace : null} block={b} level={level} geom={DEC} />
          : mode === "case" && trace && preview && PREVIEW_BLOCKS[b.id] ? (
            <div className="pl-bl-col">
              <div className="pl-bl-text"><div className="pl-title">{titleOf(b)}</div><CaseLine trace={trace} id={b.id} level={level} /></div>
              <Thumb kind={PREVIEW_BLOCKS[b.id]} st={preview} />
            </div>
          ) : (<>
            <div className="pl-title">{titleOf(b)}</div>
            {mode === "case" && trace ? <CaseLine trace={trace} id={b.id} level={level} /> : b.purpose && <div className="pl-sub">{b.purpose}</div>}
          </>)}
        </div>
      </foreignObject>
    );
  };
  const portEl = (key, p) => {
    const c = p.kind ? edgeColour(p.kind, edgeKinds) : null;
    const below = p.side === "out";
    return (
      <g key={key} data-port={key}>
        <circle cx={p.x} cy={p.y} r={4.5} fill={c || "oklch(0.975 0.002 260)"} stroke={c ? "oklch(0.99 0.002 260 / 0.9)" : K.ink3}
          strokeWidth={c ? 1.5 : 1.25}>
          <title>{p.label}</title>
        </circle>
        {view.scale >= PORT_LABEL_MIN_SCALE && (p.lx !== undefined
          ? <text x={p.lx} y={p.ly} textAnchor={p.anchor} className="pl-port-label">{p.short}</text>
          : <text x={p.x + 5} y={below ? p.y + 13 : p.y - 5} className="pl-port-label">{p.short}</text>)}
      </g>
    );
  };

  return (
    <div ref={wrap} className="pl-canvas-wrap" tabIndex={-1} onKeyDown={onKeyDown}>
      <svg className="pl-canvas" viewBox={vb.join(" ")} width="100%" height="100%" preserveAspectRatio="xMidYMid meet"
        role="group" aria-label="Pipeline architecture diagram"
        onPointerDown={onPointerDown} onPointerMove={onPointerMove} onPointerUp={onPointerUp} onPointerCancel={onPointerUp}>
        {(layout.stages || []).map((s) => (
          <foreignObject key={s.n} x={s.x} y={0} width={s.w} height={60} style={{ overflow: "visible" }}>
            <div className="pl-stage" data-stage={s.n}><span className="pl-stage-n">{s.n}</span>
              <div><div className="pl-stage-t">{s.title}</div><div className="pl-stage-d">{mode === "case" ? s.desc : s.arch}</div></div></div>
          </foreignObject>
        ))}
        {layout.frames.map((f) => (
          <g key={f.id} data-frame={f.id}>
            <rect x={f.x} y={f.y} width={f.w} height={f.h} rx={22} fill="oklch(0.985 0.002 260 / 0.45)" stroke="oklch(0.23 0.006 260 / 0.09)" />
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
                {edgeCasing(e.kind, edgeKinds) && <path d={e.d} fill="none" stroke={edgeCasing(e.kind, edgeKinds)} strokeWidth={3.75} strokeLinejoin="round" strokeLinecap="round" />}
                <path d={e.d} fill="none" stroke={c} strokeWidth={1.75} strokeLinejoin="round" strokeLinecap="round"><title>{e.label || e.kind}</title></path>
                <circle cx={s[0]} cy={s[1]} r={4.5} fill={c} />
                <circle cx={z[0]} cy={z[1]} r={4.5} fill={c} />
                {e.label && labelAt[e.id] && view.scale >= LABEL_MIN_SCALE && (
                  <EdgeLabel at={labelAt[e.id]} id={e.id} text={e.label} fill={edgeTextColour(e.kind, edgeKinds)} />
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
                <div className={"pl-frame-cap mono" + (sel ? " is-sel" : "")} role="button" tabIndex={0} aria-label={nameOf(f.block)}
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

// an edge label; one too long for its space is clipped (the full text stays in the DOM, the tooltip and the drawer) and
// followed by an ellipsis
function EdgeLabel({ at, id, text, fill }) {
  if (!at.clipW) return <text x={at.x} y={at.y} textAnchor={at.anchor} className="pl-edge-label" fill={fill}>{text}</text>;
  const cid = "lc-" + id.replace(/[^a-z0-9]/gi, "-");
  return <>
    <clipPath id={cid}><rect x={at.x - 3} y={at.y - 12} width={at.clipW + 3} height={16} /></clipPath>
    <text x={at.x} y={at.y} className="pl-edge-label" fill={fill} clipPath={`url(#${cid})`}>{text}</text>
    <text x={at.x + at.clipW} y={at.y} className="pl-edge-label" fill={fill}>…</text>
  </>;
}

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
