import { useEffect, useMemo, useRef, useState } from "react";
import { getTrace } from "../api.js";
import { TAG } from "../lib/theme.js";
import { LayoutError, layoutPipeline } from "../lib/pipelineLayout.js";
import { usePanelView } from "./Aside.jsx";
import PipelineCanvas, { MiniMap, edgeColour } from "./PipelineCanvas.jsx";
import { CaseHeading, CaseSection } from "./PipelineCase.jsx";
import { Logo } from "./TopBar.jsx";

// App-4.2: #/case/<case_id>/pipeline — the general architecture of a diagnosis, drawn from one GET of the case's trace.
// Structure only: the case id is used only to fetch a trace (it is not shown or linked); no score, threshold, probability,
// timing, status or hash is shown.
const KIND_WORDS = { signal: "measured signal", model: "model output", decision: "decision", explanation: "explanation" };
const kindWords = (kind, edgeKinds) => `${KIND_WORDS[kind] || kind} (${edgeKinds?.[kind] || "untagged"})`;

function EmptyState({ message }) {
  return (
    <div className="pl-root">
      <div className="pl-empty glass" role="status">
        <span className="mono kicker">Pipeline architecture</span>
        <p style={{ margin: "8px 0 4px", fontSize: 16 }}>{message}</p>
        <p className="pl-muted">Nothing is drawn unless the whole pipeline can be drawn.</p>
      </div>
    </div>
  );
}

// The route opens in case mode ("This diagnosis"); PipelineView on its own defaults to the general architecture.
export default function Pipeline({ caseId, initialTrace = null, fetchTrace = getTrace, initialMode = "case", ...rest }) {
  const [st, setSt] = useState(initialTrace ? { trace: initialTrace } : { loading: true });
  useEffect(() => {
    if (initialTrace) return undefined;
    const ac = new AbortController();
    fetchTrace(caseId, { signal: ac.signal })
      .then((trace) => setSt({ trace }))
      .catch((e) => { if (!ac.signal.aborted) setSt({ error: e?.response?.status === 404 ? "No such case." : "The pipeline could not be loaded." }); });
    return () => ac.abort();
  }, [caseId]);
  if (st.loading) return <EmptyState message="Loading the pipeline…" />;
  if (st.error) return <EmptyState message={st.error} />;
  return <PipelineView trace={st.trace} caseId={caseId} initialMode={initialMode} {...rest} />;
}

// a small on/off preference remembered in this browser (legend, overview, labels; App-4.2.1)
function useRemembered(key, initial) {
  const [v, setV] = useState(() => {
    try { const s = typeof window !== "undefined" && localStorage.getItem(key); return s === null || s === false ? initial : s === "1"; } catch { return initial; }
  });
  const set = (x) => { setV(x); try { localStorage.setItem(key, x ? "1" : "0"); } catch { /* browser-local only */ } };
  return [v, set];
}

export function PipelineView({ trace, caseId, initialLevel, initialSelected = null, initialSize, warn, initialMode = "architecture" }) {
  const layout = useMemo(() => {
    try { return layoutPipeline(trace, warn ? { warn } : undefined); } catch (e) { if (e instanceof LayoutError) return null; throw e; }
  }, [trace]);
  const [level, setLevel] = usePanelView(initialLevel);
  const [selected, setSelected] = useState(initialSelected);
  const [vs, setVs] = useState(null);
  const [mode, setMode] = useState(initialMode === "case" ? "case" : "architecture");
  const caseMode = mode === "case";
  const [legendOpen, setLegendOpen] = useRemembered("ms_pipeline_legend", true);
  const [mapOpen, setMapOpen] = useRemembered("ms_pipeline_overview", false);
  const [allLabels, setAllLabels] = useRemembered("ms_pipeline_all_labels", false);
  const canvas = useRef(null);
  useEffect(() => {
    const onKey = (e) => { if (e.key === "Escape") setSelected(null); };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);
  if (!layout) return <EmptyState message="The pipeline could not be drawn from this trace." />;

  const block = selected ? findBlock(trace.blocks, selected) : null;
  const kinds = [...new Set(Object.keys(trace.edge_kinds || {}))];
  return (
    <div className={"pl-root" + (caseMode ? " is-case" : "")}>
      <header className="pl-header glass">
        <div className="pl-brand"><Logo /><span style={{ fontSize: 17, fontWeight: 500 }}>MachineSense</span></div>
        <div className="pl-heading">
          <h1><span className={"mono kicker pl-mode-pill" + (caseMode ? " is-case" : "")}>{caseMode ? "This diagnosis" : "General architecture"}</span>
            {caseMode ? "How this diagnosis was made" : "How a diagnosis is made"}</h1>
          {caseMode ? <CaseHeading trace={trace} />
            : <p>General architecture view: the steps every diagnosis goes through, not the result of a specific diagnosis.</p>}
        </div>
        <div role="radiogroup" aria-label="View" className="seg pl-mode">
          {[["case", "This diagnosis"], ["architecture", "General architecture"]].map(([k, lab]) => (
            <button key={k} role="radio" aria-checked={mode === k} className={mode === k ? "on" : ""} onClick={() => setMode(k)}>{lab}</button>
          ))}
        </div>
        <div role="radiogroup" aria-label="Reading level" className="seg">
          {[["mechanic", "Mechanic"], ["developer", "Developer"]].map(([k, lab]) => (
            <button key={k} role="radio" aria-checked={level === k} className={level === k ? "on" : ""} onClick={() => setLevel(k)}>{lab}</button>
          ))}
        </div>
        {caseId && <a className="ctl pl-link" href={`#/case/${encodeURIComponent(caseId)}`}>Back to the case</a>}
      </header>

      {caseMode && trace.bundle_matches_record?.value === false && (
        <div className="pl-banner glass" role="alert">
          This case was produced by a different bundle than the one now loaded, so configuration and manifest values on this
          page describe the current bundle. {trace.bundle_note}
        </div>
      )}

      <PipelineCanvas ref={canvas} mode={mode} trace={caseMode ? trace : null} layout={layout} edgeKinds={trace.edge_kinds} level={level} selected={selected}
        onSelect={(id) => setSelected(id)} initialSize={initialSize} onViewChange={(view, size) => setVs({ view, size })}
        showAllLabels={allLabels} />

      <div className="pl-controls" role="toolbar" aria-label="Canvas controls">
        <button className="ctl" onClick={() => canvas.current?.zoomIn()} aria-label="Zoom in">+</button>
        <button className="ctl" onClick={() => canvas.current?.zoomOut()} aria-label="Zoom out">−</button>
        <button className="ctl" onClick={() => canvas.current?.fit()}>Fit to screen</button>
        <button className="ctl" onClick={() => canvas.current?.reset()}>Reset</button>
        <button className="ctl" aria-pressed={allLabels} onClick={() => setAllLabels(!allLabels)}
          title="Connection labels: by default only for the hovered or selected step">{allLabels ? "Hide other labels" : "Show all labels"}</button>
      </div>

      <div className={"pl-legend glass-soft" + (legendOpen ? "" : " is-closed")} aria-label="Edge colours">
        <button className="pl-panel-toggle mono kicker" aria-expanded={legendOpen} onClick={() => setLegendOpen(!legendOpen)}>
          {legendOpen ? "Connections carry ▾" : "Legend ▸"}
        </button>
        {legendOpen && kinds.map((k) => (
          <div key={k} className="pl-legend-row">
            <svg width="26" height="8" aria-hidden="true"><line x1="1" y1="4" x2="25" y2="4" stroke={edgeColour(k, trace.edge_kinds)} strokeWidth="1.5" /></svg>
            <span>{KIND_WORDS[k] || k}</span>
            <span className="tag" style={{ color: TAG[trace.edge_kinds[k]]?.c, background: TAG[trace.edge_kinds[k]]?.b }}>{trace.edge_kinds[k]}</span>
          </div>
        ))}
      </div>

      <div className={"pl-minimap-wrap glass-soft" + (mapOpen ? "" : " is-closed")}>
        <button className="pl-panel-toggle mono kicker" aria-expanded={mapOpen} onClick={() => setMapOpen(!mapOpen)}>
          {mapOpen ? "Overview ▾" : "Overview ▸"}
        </button>
        {mapOpen && vs && <MiniMap layout={layout} view={vs.view} size={vs.size} onCentre={(x, y) => canvas.current?.centreOn(x, y)} />}
      </div>

      {block && <Drawer block={block} trace={trace} layout={layout} level={level} caseMode={caseMode} onClose={() => setSelected(null)} />}
    </div>
  );
}

function findBlock(blocks, id) {
  for (const b of blocks) {
    if (b.id === id) return b;
    const c = b.children && findBlock(b.children, id);
    if (c) return c;
  }
  return null;
}

function Drawer({ block, trace, layout, level, caseMode, onClose }) {
  const t = layout.title(block, level);
  const dev = level === "developer";
  const name = (ref) => {
    const [bid, pid] = ref.split(".");
    const b = findBlock(trace.blocks, bid);
    const p = [...(b?.ports_in || []), ...(b?.ports_out || [])].find((x) => x.id === pid);
    return `${b ? layout.title(b, level) : bid} · ${p?.label || pid}`;
  };
  const ins = trace.edges.filter((e) => e.to.split(".")[0] === block.id);
  const outs = trace.edges.filter((e) => e.from.split(".")[0] === block.id);
  const members = block.children?.map((c) => layout.title(c, level));
  return (
    <aside className="pl-drawer glass" aria-label={`About: ${t}`}>
      <div className="pl-drawer-head">
        <span className="mono kicker">{block.children ? "Group" : "Step"}</span>
        <button className="icon-btn" onClick={onClose} aria-label="Close">×</button>
      </div>
      <h2>{t}</h2>
      {block.purpose && <p>{block.purpose}</p>}
      {members && <p className="pl-muted">Contains: {members.join(", ")}.</p>}
      {caseMode && <CaseSection trace={trace} block={block} level={level} />}
      {dev && (
        <>
          {block.operation && (<><h3 className="mono kicker">How it works</h3><p>{block.operation}</p></>)}
          {(block.ports_in?.length > 0 || block.ports_out?.length > 0) && (
            <>
              <h3 className="mono kicker">Ports</h3>
              <ul className="pl-list">
                {(block.ports_in || []).map((p) => <li key={"i" + p.id}><span className="pl-short">{p.short}</span> in · {p.label}</li>)}
                {(block.ports_out || []).map((p) => <li key={"o" + p.id}><span className="pl-short">{p.short}</span> out · {p.label}</li>)}
              </ul>
            </>
          )}
          {(ins.length > 0 || outs.length > 0) && (
            <>
              <h3 className="mono kicker">Connections</h3>
              <ul className="pl-list">
                {ins.map((e) => <li key={"i" + e.from + e.to}>From {name(e.from)} — carries a {kindWords(e.kind, trace.edge_kinds)}{e.label ? `: ${e.label}` : ""}</li>)}
                {outs.map((e) => <li key={"o" + e.from + e.to}>To {name(e.to)} — carries a {kindWords(e.kind, trace.edge_kinds)}{e.label ? `: ${e.label}` : ""}</li>)}
              </ul>
            </>
          )}
        </>
      )}
    </aside>
  );
}
