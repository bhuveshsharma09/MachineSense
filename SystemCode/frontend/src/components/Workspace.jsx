import { useEffect, useState } from "react";
import { K, al } from "../lib/theme.js";
import { PART, cleanText, locations, nf, qualityFlags, recordingTitle, unusualSensors } from "../lib/model.js";
import { usePanelView } from "./Aside.jsx";
import CaseCard, { AssistantLink } from "./CaseCard.jsx";
import { factsFor } from "./EvidenceStrip.jsx";
import ReadingDetails from "./ReadingDetails.jsx";
import Inspector from "./Inspector.jsx";
import InvestigationDialog from "./InvestigationDialog.jsx";
import AgentChat from "./AgentChat.jsx";
import NameModal from "./NameModal.jsx";
import Scene, { markerFor } from "./Scene.jsx";
import TopBar from "./TopBar.jsx";
import MobileLayout from "./MobileLayout.jsx";

// One set of sizes for the panels (CSS variables) and the 3D rig (insets), from the viewport.
export function layoutFor(w, h) {
  const compact = h < 820 || w < 1400;
  const aside = Math.round(w * 0.3) - 16, strip = 44, top = 88;
  return { compact, aside, strip, top, insets: { top, right: aside + 32, bottom: strip + 52, left: 16 } };
}

export const MOBILE_MAX = 1024;  // below this width: the stacked small-screen layout (no 3D)

function useViewport(initial) {
  const get = () => (typeof window === "undefined" ? initial || { w: 1440, h: 900 } : { w: window.innerWidth, h: window.innerHeight });
  const [vp, setVp] = useState(() => initial || get());
  useEffect(() => {
    const on = () => setVp(get());
    window.addEventListener("resize", on);
    return () => window.removeEventListener("resize", on);
  }, []);
  return vp;
}

function Legend({ open, onToggle }) {
  const row = (dot, text) => (
    <div className="legend-item">
      <span style={{ width: 14, height: 14, flex: "none", display: "grid", placeItems: "center" }}>{dot}</span>{text}
    </div>
  );
  return (
    <div className="legend">
      <button onClick={onToggle} aria-expanded={open} className="ctl legend-toggle">Legend {open ? "Hide" : "Show"}</button>
      {open && (
        <div className="glass-soft legend-popover">
          <span className="mono kicker">Model and sensor highlights</span>
          <div className="legend-items">
            {row(<span style={{ width: 14, height: 14, borderRadius: "50%", border: "2px solid " + K.violet }} />, "Selected sensor")}
            {row(<span style={{ width: 14, height: 14, borderRadius: "50%", background: `radial-gradient(circle, ${K.orange} 0 35%, ${al(K.orange, 0.25)} 60%, transparent 72%)` }} />, "Model highlight")}
            {row(<span style={{ width: 10, height: 10, borderRadius: "50%", background: K.lime, border: "2px solid " + K.ink }} />, "Measured change")}
            {row(<span style={{ width: 14, height: 14, borderRadius: "50%", border: "1.5px dashed " + K.amberT, background: al(K.amber, 0.25) }} />, "Inspect this area")}
          </div>
          <p className="legend-note">Highlights suggest where to inspect; they do not confirm a fault.</p>
        </div>
      )}
    </div>
  );
}

export default function Workspace(props) {
  const { recordings, showcase, selection, upload, status, error, record, channels, signals, activity, report, limitations,
    userName, needsName, onSaveName, onChangeName, onPick, onFile, onClearUpload, onStart, onHome, forceStatic, initialUi,
    runningFor, initialViewport, explanation, onRegenerate, initialPanelView } = props;
  const vp = useViewport(initialViewport);
  const L = layoutFor(vp.w, vp.h);
  const [ui, setUi] = useState({ sel: null, hover: null, exploded: false, labels: true, evCh: null, evView: "spectrum", tab: "activity", resetTick: 0, legend: null, ...initialUi });
  const [dialogOpen, setDialogOpen] = useState(false);
  const [chatOpen, setChatOpen] = useState(false);
  const [view, setView] = usePanelView(initialPanelView);
  const [drawer, setDrawer] = useState(!!initialUi?.drawer);
  useEffect(() => {
    if (status === "running" || status === "error") setDialogOpen(true);
    if (status === "done" && record) setDialogOpen(false);
  }, [status, record?.case_id]);
  if (vp.w < MOBILE_MAX) return <MobileLayout {...props} view={view} onView={setView}
    chatOpen={chatOpen} onOpenChat={() => setChatOpen(true)} onCloseChat={() => setChatOpen(false)} />;
  const set = (p) => setUi((u) => ({ ...u, ...p }));
  const tab = props.tab ?? ui.tab;
  const onTab = props.onTab ?? ((t) => set({ tab: t }));
  const legendOpen = ui.legend ?? false;
  const reportPage = !!record && status === "done";
  const insets = reportPage ? L.insets : { top: L.top, right: 16, bottom: 78, left: Math.min(500, Math.round(vp.w * 0.29)) };

  const flags = qualityFlags(record);
  const loc = locations(record);
  const unusual = unusualSensors(channels, flags);
  const sceneState = { ...ui, fault: loc.fault, cand: loc.cand, unusual, flags, labelsAll: ui.labels };

  const part = ui.sel ? PART[ui.sel] : null;
  const chId = part ? (part.ch.includes(ui.evCh) ? ui.evCh : part.ch[0]) : ui.evCh;
  let inspector = null;
  if (part) {
    const m = markerFor(part, sceneState);
    const status2 = m.tag ? { text: m.tag, color: m.tagColor, dot: m.fill }
      : { text: record ? "Nothing flagged for this part" : "No recording investigated", color: K.ink3, dot: "oklch(0.23 0.006 260 / 0.25)" };
    inspector = <Inspector key={part.id} initialShowSignal={!!ui.signal} part={part} status={status2} chId={chId} onChannel={(c) => set({ evCh: c })} view={ui.evView} onView={(v) => set({ evView: v })}
      record={record} channels={channels} signals={signals} onClose={() => set({ sel: null })} maxHeight={vp.h - L.insets.top - L.insets.bottom - 16} />;
  }

  const rec = selection?.kind === "dataset" ? recordings?.find((r) => r.rec_id === selection.rec_id) : null;
  const fr = record?.signal_quality.shaft_speed_hz;
  const hero = rec
    ? { k: `Development recording · used for ${rec.used_for}`, t: rec.rec_id, label: `Dataset label: ${recordingTitle(rec)}`,
        s: `${(fr ?? rec.speed_hz).toFixed(1)} Hz · ${nf(60 * (fr ?? rec.speed_hz))} rpm · 5.00 s · 8 channels` }
    : upload ? { k: "Uploaded file", t: upload.name, s: record ? `${fr.toFixed(1)} Hz measured · 5.00 s · 8 channels` : error ? (error.status === 403 ? "Upload refused" : "File check failed") : "Not checked yet" }
    : { k: "Rotor fault test rig", t: "Understand what your machine is telling you", s: "Investigate a recording, inspect the evidence, and decide what needs attention." };

  const vars = { "--aside-w": L.aside + "px", "--strip-h": L.strip + "px", "--top": L.top + "px" };
  return (
    <div className={"ms-root " + (reportPage ? "report-page" : "home-page") + (L.compact ? " compact" : "")} style={vars}>
      <Scene state={sceneState} insets={insets} zoomOut={reportPage ? 1 : 1.14} onSelect={(id) => set({ sel: id, evCh: id && PART[id] && !PART[id].ch.includes(ui.evCh) ? PART[id].ch[0] : ui.evCh })}
        onHover={(id) => { if (id !== ui.hover) set({ hover: id }); }} inspector={inspector} forceStatic={forceStatic} />
      <div className="hero" style={{ opacity: part ? 0 : 1 }}>
        <span className="mono hero-k">{hero.k}</span>
        <span className="hero-t">{hero.t}</span>
        {hero.label && <span className="hero-label">{hero.label}</span>}
        <span className="hero-s">{hero.s}</span>
        {!reportPage && <span className="hero-scene-note">Illustrative 3D rig · drag to orbit gently · click a sensor or part</span>}
        {!reportPage && <button type="button" className="start-btn home-cta" onClick={() => setDialogOpen(true)}>Start investigation <span aria-hidden="true">→</span></button>}
      </div>
      <TopBar onHome={onHome} onInvestigate={() => setDialogOpen(true)} status={status}
        userName={userName} onChangeName={onChangeName} reportPage={reportPage} />
      <div className="controls">
        <button className="ctl" onClick={() => set({ sel: null, resetTick: ui.resetTick + 1 })}>Reset view</button>
        <button className="ctl" aria-pressed={ui.exploded} onClick={() => set({ exploded: !ui.exploded })}
          style={ui.exploded ? { background: K.white, borderColor: al(K.violet, 0.55), color: K.violet } : undefined}>Exploded view</button>
        <button type="button" className="ctl" aria-pressed={ui.labels} onClick={() => set({ labels: !ui.labels })}>
          {ui.labels ? "Hide sensor names" : "Show sensor names"}
        </button>
        {reportPage && <Legend open={legendOpen} onToggle={() => set({ legend: !legendOpen })} />}
      </div>
      {reportPage && <ReadingDetails open={drawer} onToggle={() => setDrawer(!drawer)} record={record} channels={channels} signals={signals}
        flags={flags} selCh={part ? chId : null} upload={upload} facts={factsFor(record, upload, error)}
        onTile={(c) => set({ sel: c.sensor, evCh: c.id })} />}
      {reportPage && <CaseCard record={record} report={report} onShowPart={(id) => set({ sel: id, evCh: PART[id]?.ch[0] })} />}
      <InvestigationDialog open={dialogOpen} recordings={recordings} showcase={showcase} selection={selection} upload={upload}
        onPick={onPick} onFile={onFile} onClearUpload={onClearUpload} onStart={onStart} onClose={() => setDialogOpen(false)}
        status={status} error={error} runningFor={runningFor} />
      {needsName && <NameModal initial={userName} onSave={onSaveName} />}
      {reportPage ? <AssistantLink caseId={record.case_id} />
        : <AgentChat caseId={record?.case_id} open={chatOpen} onOpen={() => setChatOpen(true)} onClose={() => setChatOpen(false)} />}
    </div>
  );
}
