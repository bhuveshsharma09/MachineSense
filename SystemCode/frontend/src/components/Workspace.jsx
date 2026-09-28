import { useEffect, useState } from "react";
import { K, al } from "../lib/theme.js";
import { PART, cleanText, locations, nf, qualityFlags, recordingTitle, unusualSensors } from "../lib/model.js";
import Aside from "./Aside.jsx";
import EvidenceStrip, { factsFor } from "./EvidenceStrip.jsx";
import Inspector from "./Inspector.jsx";
import LimitationsBanner from "./LimitationsBanner.jsx";
import NameModal from "./NameModal.jsx";
import Scene, { markerFor } from "./Scene.jsx";
import TopBar from "./TopBar.jsx";
import MobileLayout from "./MobileLayout.jsx";

// One set of sizes for the panels (CSS variables) and the 3D rig (insets), from the viewport.
export function layoutFor(w, h) {
  const compact = h < 820 || w < 1400;
  const aside = w < 1400 ? 340 : 380, strip = h < 820 ? 118 : 140, top = 124;
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
    <div style={{ display: "flex", alignItems: "center", gap: 10, fontSize: 12, color: "oklch(0.30 0.006 260)" }}>
      <span style={{ width: 14, height: 14, flex: "none", display: "grid", placeItems: "center" }}>{dot}</span>{text}
    </div>
  );
  return (
    <div className="glass-soft legend" style={{ position: "absolute", zIndex: 4, display: "flex", flexDirection: "column", gap: 8, padding: open ? "11px 14px" : "6px 12px", borderRadius: 14 }}>
      <button onClick={onToggle} aria-expanded={open} className="mono kicker" style={{ border: 0, background: "transparent", padding: 0, cursor: "pointer", textAlign: "left", display: "flex", justifyContent: "space-between", gap: 10 }}>
        <span>Legend</span><span>{open ? "Hide" : "Show"}</span>
      </button>
      {open && (
        <>
          {row(<span style={{ width: 14, height: 14, borderRadius: "50%", border: "2px solid " + K.violet }} />, "Selected part")}
          {row(<span style={{ width: 14, height: 14, borderRadius: "50%", background: `radial-gradient(circle, ${K.orange} 0 35%, ${al(K.orange, 0.25)} 60%, transparent 72%)` }} />, "Location according to the dataset label (model)")}
          {row(<span style={{ width: 10, height: 10, borderRadius: "50%", background: K.lime, border: "2px solid " + K.ink }} />, "Change beyond typical healthy variation · measured")}
          {row(<span style={{ width: 14, height: 14, borderRadius: "50%", border: "1.5px dashed " + K.amberT, background: al(K.amber, 0.25) }} />, "Candidate location or signal-quality flag")}
          <p style={{ margin: "2px 0 0", fontSize: 11, lineHeight: 1.4, color: "oklch(0.48 0.008 260)" }}>Measured change is descriptive, not a calibrated anomaly test: each recording is compared on ~20 channel × metric values, and a healthy recording often exceeds at least one 95th-percentile band by chance, so no change marker is shown on a healthy verdict. Orange glow is a visual highlight, not a measured temperature.</p>
        </>
      )}
    </div>
  );
}

export default function Workspace(props) {
  const { recordings, showcase, selection, upload, status, error, record, channels, signals, activity, report, limitations,
    userName, needsName, onSaveName, onChangeName, onPick, onFile, onClearUpload, onStart, forceStatic, initialUi,
    runningFor, initialViewport } = props;
  const vp = useViewport(initialViewport);
  const L = layoutFor(vp.w, vp.h);
  const [ui, setUi] = useState({ sel: null, hover: null, exploded: false, labels: false, evCh: null, evView: "spectrum", tab: "activity", resetTick: 0, legend: null, ...initialUi });
  if (vp.w < MOBILE_MAX) return <MobileLayout {...props} />;
  const set = (p) => setUi((u) => ({ ...u, ...p }));
  const tab = props.tab ?? ui.tab;
  const onTab = props.onTab ?? ((t) => set({ tab: t }));
  const legendOpen = ui.legend ?? !L.compact;

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
    inspector = <Inspector part={part} status={status2} chId={chId} onChannel={(c) => set({ evCh: c })} view={ui.evView} onView={(v) => set({ evView: v })}
      record={record} channels={channels} signals={signals} onClose={() => set({ sel: null })} maxHeight={vp.h - L.insets.top - L.insets.bottom - 16} />;
  }

  const rec = selection?.kind === "dataset" ? recordings?.find((r) => r.rec_id === selection.rec_id) : null;
  const fr = record?.signal_quality.shaft_speed_hz;
  const hero = rec
    ? { k: `Development recording · used for ${rec.used_for}`, t: rec.rec_id, label: `Dataset label: ${recordingTitle(rec)}`,
        s: `${(fr ?? rec.speed_hz).toFixed(1)} Hz · ${nf(60 * (fr ?? rec.speed_hz))} rpm · 5.00 s · 8 channels` }
    : upload ? { k: "Uploaded file", t: upload.name, s: record ? `${fr.toFixed(1)} Hz measured · 5.00 s · 8 channels` : error ? (error.status === 403 ? "Upload refused" : "File check failed") : "Not checked yet" }
    : { k: "Rotor fault test rig", t: "Select a recording", s: "Choose a development recording or upload a CSV file." };

  const vars = { "--aside-w": L.aside + "px", "--strip-h": L.strip + "px", "--top": L.top + "px" };
  return (
    <div className={"ms-root" + (L.compact ? " compact" : "")} style={vars}>
      <Scene state={sceneState} insets={L.insets} onSelect={(id) => set({ sel: id, evCh: id && PART[id] && !PART[id].ch.includes(ui.evCh) ? PART[id].ch[0] : ui.evCh })}
        onHover={(id) => { if (id !== ui.hover) set({ hover: id }); }} inspector={inspector} forceStatic={forceStatic} />
      <div className="hero" style={{ opacity: part ? 0 : 1 }}>
        <span className="mono hero-k">{hero.k}</span>
        <span className="hero-t">{hero.t}</span>
        {hero.label && <span className="hero-label">{hero.label}</span>}
        <span className="hero-s">{hero.s}</span>
      </div>
      <Legend open={legendOpen} onToggle={() => set({ legend: !legendOpen })} />
      <TopBar recordings={recordings} showcase={showcase} value={selection?.kind === "dataset" ? selection.rec_id : ""} onPick={onPick}
        onFile={onFile} upload={upload} onClearUpload={onClearUpload} onStart={onStart} status={status} hasRecord={!!record}
        userName={userName} onChangeName={onChangeName} runningFor={runningFor} />
      <LimitationsBanner items={limitations?.map(cleanText)} />
      <div className="controls">
        <button className="ctl" onClick={() => set({ sel: null, resetTick: ui.resetTick + 1 })}>Reset view</button>
        <button className="ctl" aria-pressed={ui.exploded} onClick={() => set({ exploded: !ui.exploded })}
          style={ui.exploded ? { background: K.white, borderColor: al(K.violet, 0.55), color: K.violet } : undefined}>Exploded view</button>
        <button className="ctl" onClick={() => set({ labels: !ui.labels })}>{ui.labels ? "Labels: all" : "Labels: minimal"}</button>
        <span className="controls-note">Illustrative 3D geometry, approximated from the reference photo.<br />Drag to orbit · scroll to zoom · click a part</span>
      </div>
      <EvidenceStrip record={record} channels={channels} signals={signals} flags={flags} selCh={part ? chId : null} upload={upload}
        facts={factsFor(record, upload, error)} onTile={(c) => set({ sel: c.sensor, evCh: c.id })} />
      <Aside record={record} status={status} error={error} selection={selection} runningFor={runningFor} activity={activity}
        report={report} tab={tab} onTab={onTab} />
      {needsName && <NameModal initial={userName} onSave={onSaveName} />}
    </div>
  );
}
