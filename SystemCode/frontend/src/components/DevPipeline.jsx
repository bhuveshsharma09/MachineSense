import { useEffect, useLayoutEffect, useRef, useState } from "react";
import { getTrace } from "../api.js";
import { PIPE, PIPE_KIND } from "../lib/theme.js";
import { STAGES, devCode, devTitle } from "../lib/pipelineLayout.js";
import { at, blockPaths, caseLine, valueOf } from "../lib/pipelineCase.js";
import { edgeColour } from "./PipelineCanvas.jsx";
import { Parts, StatusWord, V, Verdict } from "./PipelineCase.jsx";

// Stage App-8.3: section 06 of the Developer view, the designer's embedded pipeline diagram. One row per pipeline stage
// (lib/pipelineLayout STAGES), one card per step, wires from the trace's own edges. Every value is read from this case's
// trace and rendered by the pipeline page's helpers (caseLine, Parts, V), so it reads as on the full pipeline page.
// Hover a step to trace its wires; click to open it. Nothing is computed or re-run here.
const KIND_WORDS = { signal: "measured signal", model: "model output", decision: "decision", explanation: "explanation" };
// columns inside a stage row, ordered so most wires run straight down
const ORDER = { Features: ["physics_features", "order_spectrogram"], Models: ["lr", "b1s", "cnn6", "mahalanobis", "class6", "ae6"] };
const HIDDEN_KEYS = /^(rule_variant|rule_status|rule_comparison|checks|thresholds|limitations|views|class6_mapping)$/;
const DEC_KEYS = [["msp", "Confidence", "≥", "<"], ["maha", "Familiarity", "≤", ">"], ["ae", "Healthy pattern", "≤", ">"], ["b1s", "Healthy at this speed", "≤", ">"]];
const msText = (v) => (v == null ? null : v < 1 ? "<1 ms" : `${Math.round(v).toLocaleString("en-US")} ms`);

const blockOf = (trace, id) => at(trace, blockPaths(trace)[id]);
const portShort = (b, port, dir) => (b?.[dir === "in" ? "ports_in" : "ports_out"] || []).find((p) => p.id === port)?.short || port;

// scalar output values of a step, with their trace paths, for the open card (never the rule's internal name)
function leaves(trace, id) {
  const P = blockPaths(trace)[id], out = blockOf(trace, id)?.output || {};
  return Object.entries(out).filter(([k, n]) => !HIDDEN_KEYS.test(k) && n && typeof n === "object" && "value" in n
    && (n.value === null || typeof n.value !== "object")).slice(0, 8).map(([k]) => [k, `${P}.output.${k}`]);
}

export default function DevPipeline({ caseId, initialTrace = null, fetchTrace = getTrace, initialOpen = [], refreshKey = "" }) {
  const [st, setSt] = useState(initialTrace ? { trace: initialTrace } : { loading: true });
  // App-8.8 item 2: refetched when the case's explanation appears or changes (the previous trace stays shown meanwhile)
  useEffect(() => {
    if (initialTrace) return undefined;
    const ac = new AbortController();
    fetchTrace(caseId, { signal: ac.signal }).then((trace) => setSt({ trace }))
      .catch(() => { if (!ac.signal.aborted) setSt((s) => (s.trace ? s : { error: true })); });
    return () => ac.abort();
  }, [caseId, refreshKey]);
  const href = `#/case/${encodeURIComponent(caseId)}/pipeline`;
  if (!st.trace) return <div className="dvw-pipe"><p className="dvw-muted">{st.error ? "The pipeline trace could not be loaded; the full diagram may still open." : "Loading the pipeline…"}</p>
    <a href={href}>Full-screen diagram ›</a></div>;
  return <Grid trace={st.trace} href={href} initialOpen={initialOpen} />;
}

function Grid({ trace, href, initialOpen }) {
  const paths = blockPaths(trace);
  const rows = STAGES.map((s) => ({ ...s, ids: (ORDER[s.title] || s.ids).filter((id) => paths[id]) }));
  const ids = rows.flatMap((r) => r.ids);
  const openable = ids.filter((id) => id !== "decision");
  const [open, setOpen] = useState(() => Object.fromEntries(initialOpen.map((id) => [id, true])));
  const [hot, setHot] = useState(null);
  const allOpen = openable.every((id) => open[id]);
  const edges = (trace.edges || []).map((e) => ({ ...e, a: e.from.split(".")[0], b: e.to.split(".")[0], pa: e.from.split(".")[1], pb: e.to.split(".")[1] }))
    .filter((e) => ids.includes(e.a) && ids.includes(e.b));
  const linked = hot ? new Set([hot, ...edges.filter((e) => e.a === hot || e.b === hot).flatMap((e) => [e.a, e.b])]) : null;
  const kinds = Object.keys(trace.edge_kinds || {});
  const stageMs = (s) => {
    const parent = s.ids.includes("cnn6") ? "models" : s.ids.includes("order_spectrogram") ? "representations" : null;
    const own = parent ? blockOf(trace, parent)?.timing_ms : null;
    const sum = s.ids.map((id) => blockOf(trace, id)?.timing_ms).filter((v) => v != null);
    return msText(own ?? (sum.length ? sum.reduce((a, b) => a + b, 0) : null));
  };
  const total = ["input", "validation", "signal_quality", "preparation", "representations", "models", "decision", "evidence_record", "explanation", "presentation"]
    .map((id) => paths[id] ? blockOf(trace, id)?.timing_ms : null).filter((v) => v != null).reduce((a, b) => a + b, 0);

  // wires: measured from the rendered cards, redrawn when a card opens or the page resizes
  const box = useRef(null);
  const [wires, setWires] = useState([]);
  useLayoutEffect(() => {
    const el = box.current;
    if (!el) return undefined;
    const draw = () => {
      const o = el.getBoundingClientRect();
      const r = (id) => { const n = el.querySelector(`[data-pn="${id}"]`); if (!n) return null; const q = n.getBoundingClientRect();
        return { l: q.left - o.left, t: q.top - o.top, r: q.right - o.left, b: q.bottom - o.top, cx: (q.left + q.right) / 2 - o.left }; };
      setWires(edges.map((e, i) => {
        const A = r(e.a), B = r(e.b);
        if (!A || !B) return null;
        let d;
        if (Math.abs(A.t - B.t) < 4) {                                  // same row: side to side
          const y = A.t + 26, x0 = A.r < B.l ? A.r : A.l, x1 = A.r < B.l ? B.l : B.r, m = (x0 + x1) / 2;
          d = `M${x0},${y} C${m},${y} ${m},${y} ${x1},${y}`;
        } else {                                                         // down a stage (or more)
          const x0 = A.cx, y0 = A.b, x1 = B.cx, y1 = B.t, k = Math.max(30, (y1 - y0) * 0.45);
          d = `M${x0},${y0} C${x0},${y0 + k} ${x1},${y1 - k} ${x1},${y1}`;
        }
        return { key: i, d, c: edgeColour(e.kind, trace.edge_kinds), a: e.a, b: e.b, label: e.label };
      }).filter(Boolean));
    };
    draw();
    const ro = typeof ResizeObserver !== "undefined" ? new ResizeObserver(draw) : null;
    ro?.observe(el);
    return () => ro?.disconnect();
  }, [trace, open]);

  const dim = (id) => linked && !linked.has(id);
  const P = paths.decision, checks = at(trace, `${P}.output.checks`) || [];
  return (
    <div className="dvw-pipe">
      <div className="dvw-pipe-bar">
        <span className="dvw-muted">Every step that ran for this case, top to bottom. Hover a step to trace its wires; click to open it.</span>
        <button type="button" className="dvw-ghost" onClick={() => setOpen(allOpen ? {} : Object.fromEntries(openable.map((id) => [id, true])))}>
          {allOpen ? "Collapse all" : "Expand all"}</button>
        <a href={href}>Full-screen diagram ›</a>
      </div>
      <div className="dvw-pg" ref={box} onMouseLeave={() => setHot(null)}>
        <svg className="dvw-wires" aria-hidden="true">
          {wires.map((w) => { const on = linked && (w.a === hot || w.b === hot);
            return <path key={w.key} d={w.d} fill="none" stroke={w.c} strokeWidth={on ? 2.2 : 1.4} opacity={linked ? (on ? 1 : 0.12) : 0.55}><title>{w.label}</title></path>; })}
        </svg>
        {rows.map((s, i) => (
          <div key={s.title} className="dvw-pg-stage" style={{ gridRow: i + 1, order: i * 100 }}>
            <span className="dvw-pg-n mono">{String(i + 1).padStart(2, "0")}</span>
            <b>{s.title}</b><span className="dvw-muted">{s.desc}</span>
            {stageMs(s) && <span className="mono dvw-muted">{stageMs(s)}</span>}
          </div>))}
        <div className="dvw-pg-legend" style={{ gridRow: 1, order: 10000 }}>
          <span className="dvw-kick-s">Connections carry</span>
          <div>{kinds.map((k) => { const c = edgeColour(k, trace.edge_kinds);
            return <span key={k}><i style={{ background: c }} />{KIND_WORDS[k] || k}<em className="tag" style={{ background: (PIPE_KIND[k] || PIPE[trace.edge_kinds[k]])?.b }}>{trace.edge_kinds[k]}</em></span>; })}</div>
          <span className="dvw-muted">Codes at each port name what the wire carries. Some results skip ahead to the saved record.</span>
          {total > 0 && <span className="mono dvw-muted">{`Recorded step times ${msText(total)} in total`}</span>}
        </div>
        {rows.flatMap((s, r) => s.ids.map((id, c) => {
          const b = blockOf(trace, id);
          if (id === "decision") {
            return (
              <div key={id} data-pn={id} className={"dvw-pg-dec" + (dim(id) ? " is-dim" : "")} style={{ gridRow: r + 1, order: r * 100 + 1 }}
                onMouseEnter={() => setHot(id)}>
                <div className="dvw-pg-dec-head"><b>{devTitle(b)}</b>{devCode(b) && <span className="mono dvw-muted">{devCode(b)}</span>}<StatusWord status={b.status} />
                  <span className="dvw-muted">Confidence and familiarity decide with the top class; the healthy-baseline checks add a caution only.</span></div>
                <div className="dvw-pg-cells">{DEC_KEYS.map(([k, name, ok, no]) => {
                  const j = checks.findIndex((x) => x.check.startsWith(k)), p = `${P}.output.checks.${j}`;
                  if (j < 0) return null;
                  const pass = valueOf(at(trace, `${p}.passed`)), base = k === "ae" || k === "b1s";
                  return <div key={k} data-cell={k} className={"dvw-pg-cell " + (base ? (pass ? "is-ok" : "is-info") : pass ? "is-ok" : "is-warn")}>
                    <span className="dvw-kick-s">{name}</span>
                    <span className="mono"><V trace={trace} p={`${p}.value`} f="check" />{` ${pass ? ok : no} `}<V trace={trace} p={`${p}.threshold`} f="check" /></span>
                    <span>{base ? (pass ? "Like healthy · caution only" : "Unlike healthy · caution only") : pass ? "Passes" : "Does not pass"}</span>
                  </div>;
                })}
                  <div className="dvw-pg-cell is-verdict"><span className="dvw-kick-s">Verdict</span><Verdict trace={trace} p={`${P}.output.display_label`} />
                    {valueOf(at(trace, `${P}.output.decision`)) === "known_fault" && <span className="dvw-muted">model prediction, according to the dataset label</span>}</div>
                </div>
              </div>);
          }
          const isOpen = !!open[id];
          const ins = edges.filter((e) => e.b === id), outs = edges.filter((e) => e.a === id);
          return (
            <div key={id} data-pn={id} className={"dvw-pg-node" + (dim(id) ? " is-dim" : "") + (hot === id ? " is-hot" : "")}
              style={{ gridRow: r + 1, gridColumn: c + 2, order: r * 100 + 1 + c }} onMouseEnter={() => setHot(id)}>
              <button type="button" aria-expanded={isOpen} onClick={() => setOpen((o) => ({ ...o, [id]: !o[id] }))}>
                <span className="dvw-pg-title"><b>{devTitle(b)}</b><span className="dvw-pg-caret" aria-hidden="true">▾</span></span>
                <span className="mono dvw-muted">{devCode(b) || id}</span>
                <span className="dvw-pg-st"><StatusWord status={b.status} />{b.timing_ms != null && <span className="mono dvw-muted">{msText(b.timing_ms)}</span>}</span>
                <span className="dvw-pg-sum"><Parts trace={trace} parts={caseLine(trace, id, "developer")} /></span>
                {b.derived && <span className="dvw-pg-chip">computed for this view</span>}
              </button>
              {isOpen && <div className="dvw-pg-open">
                <p>{b.operation_case || b.purpose}</p>
                <div className="dvw-pg-io">
                  {ins.map((e, k) => <span key={"i" + k}><em>in</em><i style={{ background: edgeColour(e.kind, trace.edge_kinds) }} /><b className="mono">{portShort(b, e.pb, "in")}</b>{`from ${devTitle(blockOf(trace, e.a))}`}</span>)}
                  {outs.map((e, k) => <span key={"o" + k}><em>out</em><i style={{ background: edgeColour(e.kind, trace.edge_kinds) }} /><b className="mono">{portShort(b, e.pa, "out")}</b>{`to ${devTitle(blockOf(trace, e.b))}`}</span>)}
                </div>
                {leaves(trace, id).length > 0 && <div className="dvw-pg-kv">{leaves(trace, id).map(([k, p]) => <div key={k}><span>{k.replace(/_/g, " ")}</span><V trace={trace} p={p} /></div>)}</div>}
              </div>}
            </div>);
        }))}
      </div>
    </div>
  );
}
