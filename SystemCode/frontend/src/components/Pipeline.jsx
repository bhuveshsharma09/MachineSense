import { Component, useEffect, useMemo, useRef, useState } from "react";
import { getCase, getSignals, getTrace } from "../api.js";
import { PIPE, PIPE_KIND } from "../lib/theme.js";
import { LayoutError, devCode, layoutPipeline } from "../lib/pipelineLayout.js";
import { usePanelView } from "./Aside.jsx";
import PipelineCanvas, { MiniMap, edgeCasing, edgeColour } from "./PipelineCanvas.jsx";
import { CaseHeading, CaseLine, CaseSection } from "./PipelineCase.jsx";
import { DEFAULT_CHANNEL, PREVIEW_BLOCKS, PreviewSection, THUMB_EXTRA, Thumb } from "./PipelinePreview.jsx";

// App-4.5: below this width the page is a stacked list of steps (same values, drawer content and labels), not a canvas
export const NARROW = 900;
const flatBlocks = (bs) => (bs || []).flatMap((b) => [b, ...flatBlocks(b.children)]);

// a response that is not a drawable trace (404 pages, HTML, truncated JSON, missing fields): the plain error state
export function traceProblem(t, caseMode) {
  if (!t || typeof t !== "object" || Array.isArray(t)) return "the response is not a pipeline trace";
  if (!Array.isArray(t.blocks) || !Array.isArray(t.edges) || !t.edge_kinds || typeof t.edge_kinds !== "object") return "the trace has no blocks, edges or edge kinds";
  const all = flatBlocks(t.blocks);
  if (!all.length || all.some((b) => !b || typeof b.id !== "string" || !Array.isArray(b.ports_in || []) || !Array.isArray(b.ports_out || [])))
    return "a block has no id or ports";
  if (t.edges.some((e) => !e || typeof e.from !== "string" || typeof e.to !== "string")) return "an edge has no ends";
  if (caseMode) {
    const d = all.find((b) => b.id === "decision");
    if (!t.case_id || !d?.output?.display_label || !d.output.decision) return "the trace records no decision for this case";
  }
  return null;
}

// a rendering failure the checks above did not foresee: the plain error state, never a half-drawn page
export class PipelineBoundary extends Component {
  constructor(props) { super(props); this.state = { failed: false }; }
  static getDerivedStateFromError() { return { failed: true }; }
  render() { return this.state.failed ? <EmptyState message="The pipeline could not be drawn from this trace." /> : this.props.children; }
}

// the trace's own statement that the stored signal cannot be read (presentation view /signals: available false + reason)
function signalsUnavailable(trace) {
  const pres = (Array.isArray(trace?.blocks) ? trace.blocks : []).find((b) => b?.id === "presentation");
  const v = (pres?.output?.views || []).find((x) => String(x.endpoint || "").endsWith("/signals"));
  return v && v.available?.value === false ? v.reason || "the stored signal is not available" : null;
}
import { Logo } from "./TopBar.jsx";
import { AskButton, useDock } from "./AssistantDock.jsx";

// App-4.2: #/case/<case_id>/pipeline — the general architecture of a diagnosis, drawn from one GET of the case's trace.
// Structure only: the case id is used only to fetch a trace (it is not shown or linked); no score, threshold, probability,
// timing, status or hash is shown.
const KIND_WORDS = { signal: "measured signal", model: "model output", decision: "decision", explanation: "explanation" };
const kindWords = (kind, edgeKinds) => `${KIND_WORDS[kind] || kind} (${edgeKinds?.[kind] || "untagged"})`;

function EmptyState({ message }) {
  return (
    <div className="pl-root">
      <a href="#/" className="ms-brand pl-empty-home" aria-label="MachineSense home"><Logo /><span>MachineSense</span></a>
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
  return <PipelineBoundary><PipelineView trace={st.trace} caseId={caseId} initialMode={initialMode} {...rest} /></PipelineBoundary>;
}

// a small on/off preference remembered in this browser (legend, overview, labels; App-4.2.1)
function useRemembered(key, initial) {
  const [v, setV] = useState(() => {
    try { const s = typeof window !== "undefined" && localStorage.getItem(key); return s === null || s === false ? initial : s === "1"; } catch { return initial; }
  });
  const set = (x) => { setV(x); try { localStorage.setItem(key, x ? "1" : "0"); } catch { /* browser-local only */ } };
  return [v, set];
}

export function PipelineView({ trace, caseId, initialLevel, initialSelected = null, initialSize, warn, initialMode = "architecture",
  fetchSignals = getSignals, fetchSaved = getCase, initialSignals = null, initialSaved = null }) {
  const [mode, setMode] = useState(initialMode === "case" ? "case" : "architecture");
  const caseMode = mode === "case";
  const problem = traceProblem(trace, caseMode);
  const [vw, setVw] = useState(initialSize?.w ?? (typeof window !== "undefined" ? window.innerWidth : 1440));
  useEffect(() => {
    if (initialSize) return undefined;
    const on = () => setVw(window.innerWidth);
    window.addEventListener("resize", on);
    return () => window.removeEventListener("resize", on);
  }, []);
  const narrow = vw < NARROW;
  const dock = useDock();
  const layout = useMemo(() => {
    if (problem) return null;
    // case mode: the three preview blocks get a thumbnail slot under their text (App-8: height; columns keep their width)
    const extraHeight = caseMode ? Object.fromEntries(Object.keys(PREVIEW_BLOCKS).map((k) => [k, THUMB_EXTRA])) : {};
    try { return layoutPipeline(trace, { ...(warn ? { warn } : {}), extraHeight }); } catch (e) { if (e instanceof LayoutError) return null; throw e; }
  }, [trace, caseMode]);
  // previews (App-4.4): fetched only by an explicit action in a drawer, once each, then cached
  const [sig, setSig] = useState(initialSignals ? { status: "ok", data: initialSignals } : { status: "idle" });
  const [saved, setSaved] = useState(initialSaved ? { status: "ok", data: initialSaved } : { status: "idle" });
  const [channel, setChannel] = useState(DEFAULT_CHANNEL);
  const failure = (e) => e?.response?.data?.detail || e?.message || "the request failed";
  const load = (kind) => {
    if (kind === "envelope") {
      if (saved.status !== "idle") return;
      setSaved({ status: "loading" });
      fetchSaved(trace.case_id).then((d) => setSaved({ status: "ok", data: d })).catch((e) => setSaved({ status: "error", error: failure(e) }));
    } else {
      if (sig.status !== "idle") return;
      setSig({ status: "loading" });
      fetchSignals(trace.case_id).then((d) => setSig({ status: "ok", data: d })).catch((e) => setSig({ status: "error", error: failure(e) }));
    }
  };
  const preview = caseMode ? { sig, saved, unavailable: signalsUnavailable(trace) } : null;
  const [level, setLevel] = usePanelView(initialLevel);
  const [selected, setSelected] = useState(initialSelected);
  const [vs, setVs] = useState(null);
  const [legendOpen, setLegendOpen] = useRemembered("ms_pipeline_legend", true);
  const [mapOpen, setMapOpen] = useRemembered("ms_pipeline_overview", false);
  const [allLabels, setAllLabels] = useRemembered("ms_pipeline_all_labels", false);
  const canvas = useRef(null);
  const drawerHead = useRef(null);
  const lastSelected = useRef(null);
  useEffect(() => {
    const onKey = (e) => { if (e.key === "Escape") setSelected(null); };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);
  // App-4.5 focus: into the drawer when it opens, back to its block when it closes
  useEffect(() => {
    if (narrow) return;
    if (selected) { lastSelected.current = selected; drawerHead.current?.focus(); }
    else if (lastSelected.current) {
      const el = document.querySelector(`[data-block="${lastSelected.current}"]`);
      lastSelected.current = null;
      el?.focus();
    }
  }, [selected, narrow]);
  if (problem || !layout) return <EmptyState message={`The pipeline could not be drawn from this trace (${problem || "its blocks and edges do not fit together"}).`} />;
  const previewFor = (b) => (caseMode && PREVIEW_BLOCKS[b.id] ? <PreviewSection kind={PREVIEW_BLOCKS[b.id]} st={preview} channel={channel}
    onChannel={setChannel} onLoad={() => load(PREVIEW_BLOCKS[b.id])} /> : null);

  const block = selected ? findBlock(trace.blocks, selected) : null;
  const mismatch = caseMode && trace.bundle_matches_record?.value === false;
  const kinds = [...new Set(Object.keys(trace.edge_kinds || {}))];
  return (
    <div className={"pl-root" + (caseMode ? " is-case" : "") + (narrow ? " is-narrow" : "") + (caseId && dock?.expanded(caseId) ? " has-assistant" : "")}>
      <header className="pl-header glass">
        <a className="pl-brand ms-brand" href="#/" aria-label="MachineSense home"><Logo /><span style={{ fontSize: 17, fontWeight: 500 }}>MachineSense</span></a>
        <div className="pl-heading">
          <h1><span className={"mono kicker pl-mode-pill" + (caseMode ? " is-case" : "")}>{caseMode ? "This diagnosis" : "General architecture"}</span>
            {caseMode ? "How this diagnosis was made" : "How a diagnosis is made"}</h1>
          {caseMode ? <CaseHeading trace={trace} />
            : <p>General architecture view: the steps every diagnosis goes through, not the result of a specific diagnosis.</p>}
          {mismatch && (
            <p className="pl-banner" role="alert">
              This case was produced by a different bundle than the one now loaded, so configuration and manifest values on this page
              describe the current bundle. <span className="pl-muted">Trace note: {trace.bundle_note}</span>
            </p>
          )}
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
        {caseId && <AskButton caseId={caseId} className="ctl pl-ask" />}
        {caseId && <a className="ctl pl-link" href={`#/case/${encodeURIComponent(caseId)}/dashboard`}>← Back to the case</a>}
      </header>


      {narrow ? (
        <StepList trace={trace} layout={layout} level={level} caseMode={caseMode} preview={preview} previewFor={previewFor} kinds={kinds}
          initialOpen={initialSelected} />
      ) : (<>
      <PipelineCanvas key={mode} ref={canvas} mode={mode} trace={caseMode ? trace : null} preview={preview} layout={layout} edgeKinds={trace.edge_kinds} level={level} selected={selected}
        onSelect={(id) => setSelected(id)} initialSize={initialSize} onViewChange={(view, size) => setVs({ view, size })}
        showAllLabels={allLabels} topInset={mismatch ? 140 : undefined} />

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
            <svg width="30" height="10" aria-hidden="true">
              {edgeCasing(k, trace.edge_kinds) && <line x1="4" y1="5" x2="26" y2="5" stroke={edgeCasing(k, trace.edge_kinds)} strokeWidth="3.75" />}
              <line x1="4" y1="5" x2="26" y2="5" stroke={edgeColour(k, trace.edge_kinds)} strokeWidth="1.75" />
              <circle cx="4.5" cy="5" r="3.5" fill={edgeColour(k, trace.edge_kinds)} /><circle cx="25.5" cy="5" r="3.5" fill={edgeColour(k, trace.edge_kinds)} /></svg>
            <span>{KIND_WORDS[k] || k}</span>
            <span className="tag pl-legend-tag" style={{ background: (PIPE_KIND[k] || PIPE[trace.edge_kinds[k]])?.b }}>{trace.edge_kinds[k]}</span>
          </div>
        ))}
      </div>

      <div className={"pl-minimap-wrap glass-soft" + (mapOpen ? "" : " is-closed")}>
        <button className="pl-panel-toggle mono kicker" aria-expanded={mapOpen} onClick={() => setMapOpen(!mapOpen)}>
          {mapOpen ? "Overview ▾" : "Overview ▸"}
        </button>
        {mapOpen && vs && <MiniMap layout={layout} view={vs.view} size={vs.size} onCentre={(x, y) => canvas.current?.centreOn(x, y)} />}
      </div>

      {block && <Drawer block={block} trace={trace} layout={layout} level={level} caseMode={caseMode} onClose={() => setSelected(null)}
        headRef={drawerHead} preview={previewFor(block)} />}
      </>)}
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

function Drawer({ block, trace, layout, level, caseMode, onClose, preview, headRef }) {
  const t = layout.title(block, level);
  return (
    <aside className="pl-drawer glass" aria-label={`About: ${t}`}>
      <div className="pl-drawer-head">
        <span className="mono kicker">{block.children ? "Group" : "Step"}</span>
        <button className="icon-btn" onClick={onClose} aria-label="Close">×</button>
      </div>
      <h2 ref={headRef} tabIndex={-1}>{t}</h2>
      {level === "developer" && devCode(block) && <span className="mono pl-code">{devCode(block)}</span>}
      <DrawerBody block={block} trace={trace} layout={layout} level={level} caseMode={caseMode} preview={preview} />
    </aside>
  );
}

// the drawer's content, shared by the desktop drawer and the phone step list (same evidence at every width)
function DrawerBody({ block, trace, layout, level, caseMode, preview }) {
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
    <>
      {block.purpose && <p>{block.purpose}</p>}
      {members && <p className="pl-muted">Contains: {members.join(", ")}.</p>}
      {preview}
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
    </>
  );
}

// App-4.5: the phone-width treatment. Every step in trace order as a card (groups hold their children), with the same
// status, case line, thumbnails and, when opened, the same content as the desktop drawer. No canvas and no pinch-zoom.
function StepList({ trace, layout, level, caseMode, preview, previewFor, kinds, initialOpen }) {
  const [open, setOpen] = useState(() => new Set(initialOpen ? [initialOpen] : []));
  const toggle = (id) => setOpen((s) => { const n = new Set(s); if (n.has(id)) n.delete(id); else n.add(id); return n; });
  const card = (b, depth) => {
    const t = layout.title(b, level);
    const isOpen = open.has(b.id);
    return (
      <section key={b.id} className={"pl-step" + (b.children ? " is-group" : "")} data-block={b.id} aria-label={t}>
        <button className="pl-step-head" aria-expanded={isOpen} aria-controls={`step-${b.id}`} onClick={() => toggle(b.id)}>
          <span className="pl-step-title">{t}</span>
          {caseMode ? <CaseLine trace={trace} id={b.id} level={level} /> : b.purpose && <span className="pl-sub">{b.purpose}</span>}
        </button>
        {caseMode && preview && PREVIEW_BLOCKS[b.id] && <Thumb kind={PREVIEW_BLOCKS[b.id]} st={preview} />}
        {isOpen && <div className="pl-step-body" id={`step-${b.id}`}>
          <DrawerBody block={b} trace={trace} layout={layout} level={level} caseMode={caseMode} preview={previewFor(b)} />
        </div>}
        {b.children && <div className="pl-step-children">{b.children.map((c) => card(c, depth + 1))}</div>}
      </section>
    );
  };
  return (
    <main className="pl-steps" aria-label="Pipeline steps, in order">
      <p className="pl-muted pl-steps-note">Steps in the order the diagnosis runs them. Open a step for its details; connections are listed in the Developer level.</p>
      <ul className="pl-steps-legend" aria-label="Connections carry">
        {kinds.map((k) => <li key={k}><span className="pl-swatch" style={{ background: edgeColour(k, trace.edge_kinds) }} />{KIND_WORDS[k] || k}</li>)}
      </ul>
      {trace.blocks.map((b) => card(b, 0))}
    </main>
  );
}
