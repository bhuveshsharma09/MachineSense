import { useEffect, useRef, useState } from "react";
import { getActivity, getCase, getChannels, getExplanation, getReport, getSignals, postExplanation } from "../api.js";
import { K, TAG, al } from "../lib/theme.js";
import * as M from "../lib/mechanic.js";
import * as D from "../lib/dashboard.js";
import { decisionSummary } from "../lib/decisionSummary.js";
import {
  CH, IN_SAMPLE_NOTICE, PART, SCORES_NOTE, STEP_CODE, fmtScore, human, locations, shownWarnings, unusualSensors,
  verdict, verdictTitle,
} from "../lib/model.js";
import { Svg, bandPath, linePath, rangeMax } from "./Charts.jsx";
import { usePanelView } from "./Aside.jsx";
import Scene from "./Scene.jsx";
import { Logo } from "./TopBar.jsx";
import { AskButton } from "./AssistantDock.jsx";
import DeveloperView from "./DeveloperView.jsx";
import MechanicView from "./MechanicView.jsx";
import "../redesign.css";

// Stage App-8: #/case/<id>/dashboard — the whole diagnosis of one stored case on one page (layout "A · Rig stage").
// Every value comes from that case's API responses; nothing is re-run.
export const DASH_NARROW = 1024;  // below this width: one column; Developer omits the 3-D rig

export const VKIND = {
  fault: { color: K.orangeT, dot: K.orange, bg: al(K.orange, 0.12) },
  further: { color: K.amberT, dot: K.amber, bg: al(K.amber, 0.18) },
  caution: { color: K.amberT, dot: K.amber, bg: al(K.amber, 0.18) },
  healthy: { color: K.greenT, dot: K.green, bg: al(K.green, 0.14) },
};
const MECHANIC_FAULT_COLOR = "oklch(0.49 0.19 27)";

export function initials(name) {
  return (name || "Guest").trim().split(/\s+/).map((w) => w[0]).slice(0, 2).join("").toUpperCase() || "G";
}
const readName = () => { try { return localStorage.getItem("ms_display_name") || ""; } catch { return ""; } };

// Load one stored case and everything the dashboard shows about it (the main session's follow-up requests, read only;
// the explanation is generated on the server if the case has none yet, exactly as the report page does).
export function useCaseData(caseId, { withExplanation = true, skip = false } = {}) {
  const [st, setSt] = useState({ status: "loading" });
  const [regen, setRegen] = useState(0);
  useEffect(() => {
    if (skip) return undefined;
    const ac = new AbortController(), signal = ac.signal;
    const live = () => !signal.aborted;
    const patch = (p) => { if (live()) setSt((s) => ({ ...s, ...p })); };
    (async () => {
      try {
        const record = await getCase(caseId, { signal });
        patch({ status: "ok", record });
        const [report, activity] = await Promise.all([getReport(caseId, { signal }), getActivity(caseId, { signal })]);
        patch({ report, activity });
        Promise.all([getChannels(caseId, { signal }), getSignals(caseId, { signal })])
          .then(([channels, signals]) => patch({ channels, signals })).catch(() => {});
      } catch (e) {
        if (live()) setSt({ status: "error", message: e?.response?.status === 404 ? "No such case." : "The case could not be loaded." });
      }
    })();
    return () => ac.abort();
  }, [caseId]);
  useEffect(() => {
    if (st.status !== "ok" || !withExplanation) return undefined;
    const ac = new AbortController(), signal = ac.signal;
    setSt((s) => ({ ...s, explanation: { status: "loading" } }));
    (regen ? Promise.reject({ regen: true }) : getExplanation(caseId, { signal }))
      .catch((e) => { if (e?.regen || e?.response?.status === 404) return postExplanation(caseId, {}, { signal }); throw e; })
      .then((data) => { if (!signal.aborted) setSt((s) => ({ ...s, explanation: { status: "ready", data } })); })
      .catch((e) => { if (!signal.aborted) setSt((s) => ({ ...s, explanation: { status: "error", message: e?.message || "failed" } })); });
    return () => ac.abort();
  }, [caseId, st.status, regen]);
  return [st, () => setRegen((n) => n + 1)];
}

export default function Dashboard({ caseId, assistantOpen = false, focusRequest = null }) {
  const [st, regenerate] = useCaseData(caseId);
  if (st.status !== "ok") return <PageState message={st.status === "loading" ? "Loading the case…" : st.message} />;
  return <DashboardView {...st} caseId={caseId} onRegenerate={regenerate}
    assistantOpen={assistantOpen} focusRequest={focusRequest} />;
}

export function PageState({ message }) {
  return (
    <div className="rd-root"><div className="rd-state">
      <a href="#/" className="rd-brand"><Logo /><span>MachineSense</span></a>
      <div className="rd-card" role="status"><p style={{ margin: 0, fontSize: 16 }}>{message}</p></div>
    </div></div>
  );
}

function useWidth(initial) {
  const [w, setW] = useState(() => initial ?? (typeof window === "undefined" ? 1440 : window.innerWidth));
  useEffect(() => {
    if (initial) return undefined;
    const on = () => setW(window.innerWidth);
    window.addEventListener("resize", on);
    return () => window.removeEventListener("resize", on);
  }, []);
  return w;
}

export function Header({ caseId, record, view, onView, back, title, userName }) {
  const v = verdict(record), s = VKIND[v.kind];
  const mechanic = view === "mechanic";
  return (
    <header className="rd-header">
      <a href="#/" className="rd-brand" aria-label="MachineSense home"><Logo /><span>MachineSense</span></a>
      <a className="rd-back" href={back.href}><span aria-hidden="true">‹</span>{back.label}</a>
      <div className="rd-header-mid">
        {title || <div className="rd-case-pill">
          <span className="rd-dot" style={{ background: s.dot, boxShadow: "0 0 0 4px " + al(s.dot, 0.18) }} />
          {!mechanic && <span className="rd-case-id">Case {caseId.slice(0, 8)}</span>}
          {!mechanic && <span className="mono rd-case-file">{record.input?.rec_id || "uploaded recording"}</span>}
          <span className="rd-chip" style={{ background: s.bg, color: s.color }}>{mechanic && v.kind === "fault" ? "Faulty" : v.chip}</span>
        </div>}
      </div>
      {onView && <Seg label="Reading level" value={view} onChange={onView} options={[["mechanic", "Mechanic"], ["developer", "Developer"]]} />}
      {!title && <AskButton caseId={caseId}><span className="rd-dot" style={{ background: K.orange }} />Ask the assistant</AskButton>}
      <span className="rd-avatar" title={userName || "Guest"}>{initials(userName)}</span>
    </header>
  );
}

export function Seg({ label, value, onChange, options }) {
  return (
    <div role="radiogroup" aria-label={label} className="rd-seg">
      {options.map(([k, lab]) => <button key={k} role="radio" aria-checked={value === k} className={value === k ? "on" : ""} onClick={() => onChange(k)}>{lab}</button>)}
    </div>
  );
}

function Card({ id, area, title, kicker, kickerColor, children, label, className = "" }) {
  return (
    <section id={id} tabIndex={id ? -1 : undefined} className={"rd-card " + className} style={{ gridArea: area }} aria-label={label || title}>
      {(title || kicker) && <div className="rd-card-head">
        {title && <h2>{title}</h2>}
        {kicker && <span className="rd-kicker" style={kickerColor ? { color: kickerColor } : undefined}>{kicker}</span>}
      </div>}
      {children}
    </section>
  );
}

const Tick = ({ ok }) => <span className={"rd-tick" + (ok ? "" : " is-warn")} aria-hidden="true">{ok ? "✓" : "!"}</span>;

export function Caveat({ children }) { return <span className="rd-m-caveat"><span aria-hidden="true" className="rd-m-info">i</span>{children}</span>; }

function WhyStrip({ summary, warnings, caseId, developer = false }) {
  const value = (v, yes, no) => v == null ? "Not available" : v ? yes : no;
  const steps = [
    { title: "Signal usable", text: warnings.length ? "Recording-quality warning; review the signal" : "No signal-quality warnings detected", tone: warnings.length ? "warn" : "ok" },
    { title: "Confident match", text: summary.confidence, tone: summary.checks.msp == null ? "neutral" : summary.checks.msp ? "ok" : "warn" },
    { title: "Familiar pattern", text: value(summary.checks.maha, "Looks like recordings the model has seen", "Unlike recordings the model has seen"), tone: summary.checks.maha == null ? "neutral" : summary.checks.maha ? "ok" : "warn" },
    { title: "Healthy checks", text: summary.healthy, tone: "neutral" },
    { title: "Verdict", text: summary.outcome, tone: summary.outcome.startsWith("Known") ? "fault" : summary.outcome.startsWith("Further") || summary.outcome.includes("caution") ? "warn" : "ok" },
  ];
  return <section className="rd-card rd-m-why" aria-label="Why does it say this?" style={developer ? { gridArea: "sum" } : undefined}>
    <div className="rd-card-head"><h2>Why does it say this?</h2>{developer && <a className="rd-btn" href={`#/case/${encodeURIComponent(caseId)}/pipeline`}>See the full pipeline</a>}</div>
    <ol className="rd-m-why-steps">{steps.map((s) => <li key={s.title} className={`is-${s.tone}`}><strong>{s.title}</strong><span>{s.text}</span></li>)}</ol>
  </section>;
}

function FindingSummary({ area, record, report, explanation, loc, mloc, warnings, onShowPart }) {
  const d = record.diagnosis;
  const e = explanation?.status === "ready" ? explanation.data : null;
  const hasFaultArea = d.decision === "known_fault" && loc.fault.length > 0;
  const verdictLabel = d.decision === "known_fault" ? `Verdict · ${human(d.top_class)} (model prediction)`
    : d.decision === "unknown" ? "Verdict · Further inspection needed" : "Verdict · Healthy";
  const verdictColor = d.decision === "known_fault" ? MECHANIC_FAULT_COLOR
    : d.decision === "unknown" ? K.amberT : K.greenT;
  return <Card id="rd-finding" area={area} label="Finding" className="rd-m-finding">
    <div className="rd-m-finding-content"><div className="rd-m-answer">
      <span className="rd-m-category" style={{ color: verdictColor }}>{verdictLabel}</span>
      <h1 className="rd-m-title">{M.finding(record)}</h1>
      {d.caution && <span className="rd-sub">A healthy-baseline check disagrees; consider a recheck.</span>}
      {hasFaultArea && mloc.parts.length > 0 && <p className="rd-m-next-line">Start with the {mloc.parts.map((p) => M.PART_PLAIN[p].toLowerCase()).join(" and ")}.</p>}
      {d.decision === "unknown" && <p className="rd-m-next-line">The checks did not support a known condition; inspect the readings before choosing a fault area.</p>}
      <span className="rd-m-speed">Shaft speed {record.signal_quality.shaft_speed_hz.toFixed(1)} Hz ({Math.round(record.physics.shaft_speed_rpm).toLocaleString("en-US")} rpm)</span>
      {hasFaultArea && mloc.parts.length > 0 && <div className="rd-m-location">
        <strong>{mloc.parts.map((p) => M.PART_PLAIN[p]).join(" and ")}</strong>
        <Caveat>Mapped from the model’s predicted class on this test rig; the sensors do not pinpoint the fault.</Caveat>
        {onShowPart && <button className="rd-dark-btn" onClick={() => onShowPart(mloc.parts[0])}>Show {M.PART_PLAIN[mloc.parts[0]].toLowerCase()} on the rig</button>}
      </div>}
      <div className="rd-m-quality">
        {warnings.length ? <div className="rd-warn" role="alert">{M.warnings(record).map((w) => <div key={w}><Tick ok={false} />{w}</div>)}</div>
          : <span className="rd-ok"><Tick ok />No signal-quality warnings detected.</span>}
        {warnings.length > 0 && <details><summary>Signal quality details</summary><div className="rd-checklist">{D.qualityChecks(record).map((q) => <div key={q.text}><Tick ok={q.ok} />{q.text}</div>)}</div></details>}
      </div>
    </div><div className="rd-m-finding-evidence" aria-label="Evidence">
      <h2>Evidence</h2>
      {e ? <><Caveat>{e.badge}</Caveat><ul className="rd-bullets">{e.text.why_flagged.slice(0, 2).map((x) => <li key={x}>{x}</li>)}</ul></>
        : <p className="rd-muted">{explanation?.status === "error" ? "The explanation is unavailable. Review the measured channels below." : "Preparing the explanation…"}</p>}
      <div className="rd-box"><span className="rd-m-label">Healthy-baseline check</span><span>{M.supporting(record, report?.assessment) || "Checking…"}</span></div>
    </div></div>
  </Card>;
}

export function DashboardView({ caseId, record, report, channels, signals, activity, explanation, onRegenerate,
  initialView, initialWidth, initialSel = null, initialUi = {}, assistantOpen = false, focusRequest = null }) {
  const [view, setView] = usePanelView(initialView);
  const changeView = (next) => { setRigFocus(null); setView(next); };
  const dev = view === "developer";
  const width = useWidth(initialWidth);
  const narrow = width < DASH_NARROW || (assistantOpen && width < 1400);
  const [ui, setUi] = useState({ sel: initialSel, hover: null, exploded: false, labels: false, resetTick: 0, ch: null, sp: "order", allSteps: false, measure: "channels", zoomHint: false, ...initialUi });
  const set = (p) => setUi((u) => ({ ...u, ...p }));
  const zoomHintSeen = useRef(false);
  const zoomHintTimer = useRef(null);
  const rigSection = useRef(null);
  const [rigFocus, setRigFocus] = useState(null);
  const handledFocus = useRef(null);
  useEffect(() => () => clearTimeout(zoomHintTimer.current), []);
  useEffect(() => {
    if (!focusRequest || handledFocus.current === focusRequest) return undefined;
    const target = focusRequest?.target;
    if (!target) return undefined;
    if (view !== target.view) { setView(target.view); return undefined; }
    const el = document.getElementById(`rd-${target.card}`) || document.getElementById("rd-finding");
    if (!el) return undefined;
    handledFocus.current = focusRequest;
    el.classList.add("rd-focus-highlight");
    el.focus({ preventScroll: true });
    el.scrollIntoView({ behavior: "smooth", block: "center" });
    const timer = setTimeout(() => el.classList.remove("rd-focus-highlight"), 4200);
    return () => { clearTimeout(timer); el.classList.remove("rd-focus-highlight"); };
  }, [focusRequest, view]);
  const userName = useRef(readName()).current;
  // App-9 addendum A: the assistant panel is mounted by App (AssistantDock); assistantOpen only makes room for it here.
  const d = record.diagnosis;
  const flags = D.qualityFlags(record);
  const loc = locations(record);
  const unusual = unusualSensors(channels, flags);
  const rows = D.channelRows(channels);
  const partId = D.focusPart(ui.sel, loc, unusual);
  const part = PART[partId];
  const sceneState = { ...ui, fault: loc.fault, cand: loc.cand, unusual, flags, labelsAll: ui.labels };
  const chOptions = part.ch.filter((c) => signals?.channels?.[c]);
  const chId = chOptions.includes(ui.ch) ? ui.ch : chOptions[0] || part.ch[0];
  const select = (id) => set({ sel: id, ch: id && PART[id] && !PART[id].ch.includes(ui.ch) ? PART[id].ch[0] : ui.ch });
  const focusRigPart = (id) => {
    select(id);
    setRigFocus({ id });
    requestAnimationFrame(() => rigSection.current?.scrollIntoView({ behavior: "smooth", block: "start" }));
  };

  const mloc = M.location(record);
  const steps = D.guidanceSteps(M.guidanceItems(report));
  const shown = ui.allSteps ? steps : steps.slice(0, 3);
  const checks = report?.assessment?.checks;
  const warn = shownWarnings(record);
  const hasFaultArea = d.decision === "known_fault" && loc.fault.length > 0;
  const summary = decisionSummary({ topClass: d.top_class, decision: d.decision, caution: d.caution, checks });
  const href = (p) => `#/case/${encodeURIComponent(caseId)}/${p}`;

  if (!dev) return <MechanicView caseId={caseId} record={record} report={report} channels={channels} signals={signals}
    explanation={explanation} view={view} onView={changeView} userName={userName} assistantOpen={assistantOpen} narrow={narrow}
    initialSel={initialSel} initialUi={initialUi} />;

  if (!dev) {
    const showChecks = d.decision !== "healthy";
    const mechanicSceneState = hasFaultArea ? sceneState : { ...sceneState, cand: [] };
    const guidanceNotes = [...new Set(steps.map((s) => s.flag).filter(Boolean))];
    const evidenceChannel = rows.filter((r) => r.comparable && r.beyond && r.metric === "order_2x")
      .sort((a, b) => b.factor / b.band - a.factor / a.band)[0]?.id
      || rows.find((r) => r.comparable && r.beyond)?.id;
    const measuredChId = ui.ch || evidenceChannel || chId;
    const measuredPart = PART[CH[measuredChId]?.sensor] || part;
    const measuredOptions = measuredPart.ch.filter((c) => signals?.channels?.[c]);
    const showZoomHint = () => {
      if (zoomHintSeen.current) return;
      zoomHintSeen.current = true;
      set({ zoomHint: true });
      zoomHintTimer.current = setTimeout(() => set({ zoomHint: false }), 4500);
    };
    const showMeasurement = (target) => {
      set({ measure: "spectrum", sp: "order", ch: target.channel, sel: CH[target.channel].sensor });
      requestAnimationFrame(() => document.getElementById("rd-measurements")?.scrollIntoView({ behavior: "smooth", block: "start" }));
    };
    return (
      <div className={"rd-root rd-mechanic" + (narrow ? " is-narrow" : "") + (assistantOpen ? " has-assistant" : "")}>
        <Header caseId={caseId} record={record} view={view} onView={changeView} userName={userName}
          back={{ href: "#/", label: "Back to home" }} />
        <main className="rd-m-main">
          <FindingSummary record={record} report={report} explanation={explanation} loc={loc} mloc={mloc} warnings={warn} onShowPart={focusRigPart} />

          <section ref={rigSection} className="rd-rig rd-m-rig" aria-label="Interactive test rig" onMouseEnter={showZoomHint}>
            <Scene state={mechanicSceneState} insets={{ top: 66, right: 20, bottom: 46, left: 20 }} zoomOut={0.92} wheelZoomRequiresModifier focusOnSelect={false} focusRequest={rigFocus} onSelect={select}
              onHover={(id) => { if (id !== ui.hover) set({ hover: id }); }} />
            <div className="rd-rig-top"><div className="rd-rig-title"><Caveat>Test rig (illustrative)</Caveat><span>Select a part or sensor for a closer look</span></div>
              <div className="rd-rig-btns">
                <button className="rd-btn" onClick={() => set({ sel: null, resetTick: ui.resetTick + 1 })}>Reset view</button>
                <button className={"rd-btn" + (ui.exploded ? " is-on" : "")} aria-pressed={ui.exploded} onClick={() => set({ exploded: !ui.exploded })}>Exploded view</button>
                <button className="rd-btn" aria-pressed={ui.labels} onClick={() => set({ labels: !ui.labels })}>{ui.labels ? "Labels: all" : "Labels: minimal"}</button>
              </div>
            </div>
            <div className="rd-rig-legend">{hasFaultArea && <span><i className="lg-fault" />Model-mapped inspection area</span>}<span><i className="lg-change" />Measured change</span>
              <span><i className="lg-flag" />Signal-quality warning</span><Caveat>Highlights do not confirm a fault.</Caveat></div>
            {ui.zoomHint && <span className="rd-m-zoom-hint">Hold Ctrl (⌘) and scroll to zoom</span>}
          </section>

          {showChecks && <Card id="rd-suggested-checks" title="Suggested checks" label="Suggested checks" className="rd-m-next">
            <span className="rd-m-label">From curated guidance</span>
            {!report ? <p className="rd-muted">Suggested checks appear with the report.</p> : <>
              <ol className="rd-steps">{steps.map((s, i) => <li key={i} className={s.src ? "" : "is-plain"}>
                <span className="rd-step-n">{i + 1}</span>
                <div className="rd-col" style={{ gap: 6 }}><span className="rd-step-t">{s.text}</span>
                  {/\b(?:1×|2×|order)\b/i.test(s.text) && <span className="rd-m-gloss">1× = once per shaft turn; 2× = twice per turn.</span>}
                  {s.src && <span className="rd-m-citation">{s.title.split(" — ")[0]}</span>}
                  {D.showMeTarget(s, i) && <button className="rd-m-show" onClick={() => showMeasurement(D.showMeTarget(s, i))}>Show me</button>}
                </div>
              </li>)}</ol>
              {guidanceNotes.length > 0 && <Caveat>Guidance status: {guidanceNotes.join(" · ")}</Caveat>}
            </>}
          </Card>}

          <section id="rd-measurements" tabIndex={-1} className="rd-m-measure" aria-label="Explore measurements">
            <div className="rd-m-measure-head"><div><span className="rd-m-label">Measurements</span>
              <h2>{ui.measure === "channels" ? "Compare this recording with a healthy example" : ui.measure === "spectrum" ? "Inspect the frequency pattern" : "Inspect the signal over time"}</h2>
              <p>{ui.measure === "channels" ? (channels ? `${rows.filter((r) => r.beyond).length} of ${rows.length} vibration and sound channels differ from the typical healthy range. The tachometer measures shaft speed.` : "Loading measured comparisons…")
                : ui.measure === "spectrum" ? "This chart shows the selected channel from this recording. A matched healthy spectrum is not stored with this case."
                  : "This half-second excerpt can reveal repeated impacts, spikes or clipping, but cannot identify a fault on its own."}</p></div>
              <Seg label="Measurement view" value={ui.measure} onChange={(measure) => set({ measure })}
                options={[["channels", "Sensors"], ["spectrum", "Spectrum"], ["waveform", "Waveform"]]} />
            </div>
            {ui.measure === "channels" && <ChannelCard channels={channels} rows={rows} dev={false} sel={ui.sel} onSelect={(r) => set({ sel: r.sensor, ch: r.id })} />}
            {ui.measure === "spectrum" && <SpectrumCard record={record} signals={signals} part={measuredPart} chId={measuredChId} chOptions={measuredOptions} dev={false}
              mode={ui.sp} onMode={(sp) => set({ sp })} onChannel={(ch) => set({ ch })} />}
            {ui.measure === "waveform" && <WaveformCard signals={signals} chId={measuredChId} />}
          </section>

          <details className="rd-m-details"><summary>Recording details</summary>
            <div className="rd-kv">
              <div><span>Recording</span><span className="mono">{record.input?.rec_id || "uploaded recording"}</span></div>
              <div><span>Shaft speed</span><span>{Math.round(record.physics.shaft_speed_rpm).toLocaleString("en-US")} rpm</span></div>
              <div><span>Duration and channels</span><span>5.00 s · 8 channels</span></div>
              <div><span>Case</span><span className="mono">{caseId.slice(0, 12)}…</span></div>
            </div>
          </details>
        </main>
      </div>
    );
  }

  const areas = dev
    ? ["hero hero hero", "sum sum sum", "fd rig gg", "cd rig dq", "ch ch ch", "sp sp wf", "dc dc dc", "why nx nx", "xp rp rp", "rd ac ac", "lim lim lim"]
    : [];
  const narrowAreas = dev ? ["hero", "sum", "fd", "gg", "dq", "cd", "ch", "sp", "wf", "dc", "why", "nx", "xp", "rp", "rd", "ac", "lim"] : [];
  const gridStyle = narrow ? { gridTemplateColumns: "minmax(0,1fr)", gridTemplateAreas: narrowAreas.map((a) => `"${a}"`).join(" ") }
    : { gridTemplateColumns: "minmax(300px,340px) minmax(0,1fr) minmax(300px,340px)", gridTemplateAreas: areas.map((a) => `"${a}"`).join(" ") };

  if (dev) return <DeveloperView caseId={caseId} record={record} report={report} channels={channels} signals={signals} activity={activity}
    explanation={explanation} onRegenerate={onRegenerate} view={view} onView={changeView} userName={userName} assistantOpen={assistantOpen}
    narrow={narrow} initialSel={initialSel} />;

  return (
    <div className={"rd-root" + (narrow ? " is-narrow" : "") + (assistantOpen ? " has-assistant" : "")}>
      <Header caseId={caseId} record={record} view={view} onView={changeView} userName={userName}
        back={{ href: "#/", label: "Back to home" }} />
      <main className="rd-grid" style={gridStyle}>
        <FindingSummary area="hero" record={record} report={report} explanation={explanation} loc={loc} mloc={mloc} warnings={warn} onShowPart={narrow ? null : focusRigPart} />
        <WhyStrip summary={summary} warnings={warn} caseId={caseId} developer />
        {!narrow && <section ref={rigSection} className="rd-rig" style={{ gridArea: "rig" }} aria-label="3D rig">
          <Scene state={sceneState} insets={{ top: 84, right: 24, bottom: 56, left: 24 }} focusRequest={rigFocus} onSelect={select}
            onHover={(id) => { if (id !== ui.hover) set({ hover: id }); }} />
          <div className="rd-rig-top">
            <div className="rd-rig-title"><span className="rd-kicker">Rotor test rig · illustrative</span>
              <span>{mloc.parts.length ? `${verdictTitle(record)} · ${mloc.parts.map((p) => M.PART_PLAIN[p].toLowerCase()).join(" and ")}` : verdictTitle(record)}</span></div>
            <div className="rd-rig-btns">
              <button className="rd-btn" onClick={() => set({ sel: null, resetTick: ui.resetTick + 1 })}>Reset view</button>
              <button className={"rd-btn" + (ui.exploded ? " is-on" : "")} aria-pressed={ui.exploded} onClick={() => set({ exploded: !ui.exploded })}>Exploded view</button>
              <button className="rd-btn" aria-pressed={ui.labels} onClick={() => set({ labels: !ui.labels })}>{ui.labels ? "Labels: all" : "Labels: minimal"}</button>
            </div>
          </div>
          <div className="rd-rig-legend">
            <span><i className="lg-sel" />Selected</span><span><i className="lg-fault" />Predicted location · model</span>
            <span><i className="lg-change" />Measured change</span><span><i className="lg-flag" />Inspect · quality flag</span>
            <span className="rd-rig-note">Highlights suggest where to inspect; they do not confirm a fault.</span>
          </div>
        </section>}

        <Card id="rd-model-output" area="fd" title="Model output" kicker="Model">
          {mloc.parts.length > 0 && <div className="rd-loc" style={mloc.kind === "candidate" ? { background: al(K.amber, 0.14) } : undefined}>
            <span className="rd-kicker" style={{ color: mloc.kind === "candidate" ? K.amberT : K.orangeT }}>{mloc.kind === "candidate" ? "Candidate locations" : "Suggested location"}</span>
            <span className="rd-loc-name">{mloc.parts.map((p) => M.PART_PLAIN[p]).join(" and ")}</span>
            <span className="rd-sub-s">{mloc.kind === "candidate" ? "From the two most likely fault types (model prediction)" : "According to the dataset label (model prediction)"}</span>
          </div>}
          <div className="rd-models">
            <span className="mono rd-muted">CNN-6</span><span>{human(d.top_class)}</span><span className="mono">MSP {fmtScore(d.scores.msp)}</span>
            <span className="mono rd-muted">LR</span><span>{human(d.lr_secondary.top_class)}</span><span className="mono">{fmtScore(d.lr_secondary.probabilities[d.lr_secondary.top_class])}</span>
          </div>
          <span className="rd-note">{d.top_class === d.lr_secondary.top_class ? "Both rank the same class first. " : "They rank different classes first. "}{SCORES_NOTE}</span>
        </Card>

        <Gauge record={record} dev={dev} />

        <Card id="rd-signal-quality" area="dq" title="Signal quality" kicker="Measured">
          <div className="rd-big-line"><span className="rd-title-l">{record.signal_quality.warnings.length ? `${record.signal_quality.warnings.length} warning${record.signal_quality.warnings.length > 1 ? "s" : ""}` : "No warnings"}</span>
            <span className="rd-sub-s">{8 - (record.signal_quality.clipping_channels || []).length} of 8 channels without clipping</span></div>
          {dev ? <div className="rd-kv">{D.qualityRows(record).map(([k, x]) => <div key={k}><span>{k}</span><span className="mono">{x}</span></div>)}</div>
            : <div className="rd-checklist">{D.qualityChecks(record).map((q) => <div key={q.text}><Tick ok={q.ok} />{q.text}</div>)}</div>}
          {dev && record.signal_quality.warnings.length > 0 && <div className="rd-warn">{record.signal_quality.warnings.map((w) => <div key={w}><Tick ok={false} />{w}</div>)}</div>}
        </Card>

        <Card area="cd" title="Model’s ranked matches" kicker="Model" kickerColor="oklch(0.44 0.15 277)">
          <div className="rd-stack-8">
            {D.candidates(record).map((c) => (
              <div key={c.cls} className={"rd-cand" + (c.rank === 1 ? " is-top" : "")}>
                <span className="rd-rank">{c.rank}</span>
                <div className="rd-col"><span className="rd-strong">{c.name}</span><span className="rd-sub-s">{c.loc}</span></div>
                <span className="mono rd-right">{dev ? `${c.cnn} · ${c.lr}` : c.rank === 1 ? "Best match" : "Less likely"}</span>
              </div>
            ))}
          </div>
          <span className="rd-note">{dev ? "CNN-6 · secondary logistic regression. " : "Ranked by the main model. "}{SCORES_NOTE}</span>
        </Card>

        <ChannelCard channels={channels} rows={rows} dev={dev} sel={ui.sel} onSelect={(r) => set({ sel: r.sensor, ch: r.id })} />

        <SpectrumCard record={record} signals={signals} part={part} chId={chId} chOptions={chOptions} mode={ui.sp}
          onMode={(m) => set({ sp: m })} onChannel={(c) => set({ ch: c })} />

        <WaveformCard signals={signals} chId={chId} />

        {dev && <Card id="rd-decision-checks" area="dc" title="Decision checks">
          <span className="rd-rule" title={report?.assessment?.rule_tooltip}>{report?.assessment?.rule_line || "Decision rule: provisional research setting"}</span>
          {checks ? <div className="rd-checks">{checks.map((c) => (
            <div key={c.key} className="rd-check">
              <div className="rd-between"><span className="rd-strong">{c.name[0].toUpperCase() + c.name.slice(1)}</span>
                <span className="rd-status" style={{ color: c.tone === "good" ? K.greenT : c.tone === "warn" ? K.amberT : K.ink2,
                  background: c.tone === "good" ? al(K.green, 0.14) : c.tone === "warn" ? al(K.amber, 0.2) : "oklch(0.23 0.006 260 / 0.06)" }}>{c.passed ? "pass" : c.role === "healthy baseline" ? "above healthy limit" : "fail"}</span></div>
              <span className="mono">{D.checkExpr(c)}</span>
              <span className="mono rd-sub-s">margin {c.margin_abs >= 0 ? "+" : "−"}{Math.abs(c.margin_abs).toPrecision(3)}{c.narrow ? " · narrow" : ""}</span>
            </div>))}</div> : <p className="rd-muted">Loading the checks…</p>}
          <span className="rd-note">The verdict rests on the confidence and familiarity checks and the top class; the two healthy-baseline checks only add a caution to a healthy verdict.</span>
          {report?.assessment && <span className="rd-sub-s" title={report.assessment.variants.map((x) => `${x.name}: ${human(x.verdict)}`).join("\n")}>
            {report.assessment.variants_summary}{report.assessment.narrow_notes?.length ? " · " + report.assessment.narrow_notes.join(" ") : " · no narrow margins"}</span>}
        </Card>}

        <Card area="why" title="Why it was flagged" label="Why it was flagged">
          <ExplanationBody explanation={explanation} record={record} report={report} />
        </Card>

        <Card id="rd-developer-checks" area="nx" title="What to inspect next" kicker="Guidance" kickerColor="oklch(0.45 0.11 125)">
          {!report ? <p className="rd-muted">The inspection steps appear with the report.</p> : <>
            <ol className="rd-steps">{shown.map((s, i) => (
              <li key={i} className={s.src ? "" : "is-plain"}>
                {s.src && <span className="rd-step-n">{i + 1}</span>}
                <div className="rd-col" style={{ gap: 5 }}><span className="rd-step-t">{s.text}</span>
                  {s.src && <div className="rd-row-wrap"><span className="mono rd-src" title={s.title}>source: {s.src}</span>{s.flag && <span className="rd-flag">{s.flag}</span>}</div>}
                </div>
              </li>))}</ol>
            {steps.length > 3 && <button className="rd-ghost" onClick={() => set({ allSteps: !ui.allSteps })}>{ui.allSteps ? "Show fewer" : `Show ${steps.length - 3} more`}</button>}
          </>}
        </Card>

        {dev && <XpCard record={record} explanation={explanation} onRegenerate={onRegenerate} />}

        {dev && <Card id="rd-fixed-report" area="rp" title="Fixed evidence report">
          <details className="rd-report-details"><summary>View full report</summary>
          {report ? <div className="rd-report">{report.sections.map((sec) => (
            <div key={sec.n} className="rd-col" style={{ gap: 8 }}>
              <div className="rd-row-base"><span className="mono rd-muted">{sec.n}</span><h3>{sec.title}</h3></div>
              {sec.items.map((it, i) => <div key={i} className="rd-row-top" title={it.hover}>
                <span className="tag" style={{ color: TAG[it.tag].c, background: TAG[it.tag].b }}>{it.tag}</span><span className="rd-report-t">{it.text}</span></div>)}
            </div>))}</div> : <p className="rd-muted">Loading the report…</p>}
          </details>
          <span className="rd-note">Fixed template built from the evidence record; no language model.</span>
        </Card>}

        <Card area="rd" title="Recording details">
          <div className="rd-tiles">
            <div><span>Shaft speed</span><b>{Math.round(record.physics.shaft_speed_rpm).toLocaleString("en-US")} <small>rpm</small></b><span className="mono">{record.signal_quality.shaft_speed_hz.toFixed(2)} Hz</span></div>
            <div><span>Duration</span><b>5.00 <small>s</small></b><span className="mono">50 kHz · 8 ch</span></div>
          </div>
          <div className="rd-kv">
            <div><span>Recording</span><span className="mono">{record.input?.rec_id || "uploaded recording"}</span></div>
            <div><span>Case</span><span className="mono">{record.case_id.slice(0, 12)}…</span></div>
            <div><span>Source</span><span className="mono">{record.source === "dataset" ? "dataset (development pool)" : "upload"}</span></div>
            <div><span>Created</span><span className="mono">{record.created_at.slice(0, 16).replace("T", " ")} UTC</span></div>
          </div>
          {record.source === "dataset" && <span className="rd-in-sample">{IN_SAMPLE_NOTICE}</span>}
        </Card>

        <Card area="ac" title="Investigation activity" label="Investigation activity">
          {activity ? <ol className="rd-activity">{activity.steps.map((s) => (
            <li key={s.step}><span className="mono rd-code">{STEP_CODE[s.step]}</span>
              <div className="rd-col"><span className="rd-strong">{s.title}</span><span className="rd-sub-s rd-ellipsis" title={s.detail}>{dev ? s.detail : s.startup ? "at server startup" : "done"}</span></div>
              {dev && <span className="mono rd-muted">{s.ms == null ? "—" : s.ms < 1 ? "<1 ms" : `${Math.round(s.ms)} ms`}</span>}
            </li>))}</ol> : <p className="rd-muted">Loading…</p>}
        </Card>

        <footer className="rd-footer" style={{ gridArea: "lim" }}>
          <div className="rd-row-wrap" style={{ gap: 18 }}>
            <span className="rd-strong">{M.LIMITATION}</span><span>{SCORES_NOTE}</span>
          </div>
        </footer>
      </main>
    </div>
  );
}

function Gauge({ record, dev }) {
  const g = D.gauge(record), R = 96;
  const ticks = Array.from({ length: 21 }, (_, i) => i / 20);
  const [tx0, ty0] = D.arcPoint(g.t, R - 14), [tx1, ty1] = D.arcPoint(g.t, R + 14);
  return (
    <Card area="gg" title="Difference from healthy" kicker="Model" kickerColor="oklch(0.44 0.15 277)">
      <div className="rd-gauge">
        <svg viewBox="0 0 240 132" aria-hidden="true">
          {ticks.map((f) => { const [a, b] = D.arcPoint(f, R + 12), [c, e] = D.arcPoint(f, R + 16); return <line key={f} x1={a} y1={b} x2={c} y2={e} stroke="oklch(0.23 0.006 260 / 0.2)" strokeWidth="1" />; })}
          <path d={D.arcPath(0, 1, R)} fill="none" stroke="oklch(0.23 0.006 260 / 0.07)" strokeWidth="16" strokeLinecap="round" />
          <path d={D.arcPath(0, Math.min(g.t, g.f), R)} fill="none" stroke={al(K.green, 0.55)} strokeWidth="16" strokeLinecap="round" />
          {g.beyond && <path d={D.arcPath(g.t, g.f, R)} fill="none" stroke={K.orange} strokeWidth="16" strokeLinecap="round" />}
          <line x1={tx0} y1={ty0} x2={tx1} y2={ty1} stroke={K.ink} strokeWidth="2" />
        </svg>
        <div className="rd-gauge-val">
          <span style={{ fontSize: dev ? 34 : 24 }}>{dev ? D.fmtCheck(g.value) : g.beyond ? "Different from healthy" : "Like healthy"}</span>
          <span className="rd-sub-s">{dev ? `B1s speed-aware baseline · τ_b ${D.fmtCheck(g.threshold)}` : g.beyond ? "beyond the healthy limit" : "within the healthy limit"}</span>
        </div>
      </div>
      <div className="rd-between mono rd-sub-s" style={{ padding: "0 6px" }}><span>{dev ? "0" : "like healthy"}</span><span>{dev ? D.fmtCheck(g.max) : "further from healthy"}</span></div>
      <p className="rd-note">Compares this recording with healthy recordings at a similar speed. The marker shows the decision threshold. This is not a measure of damage severity.</p>
    </Card>
  );
}

// An isolated 3-D view of one part for the assistant context (rig/rigScene.js with `isolate`).
export function PartView({ partId, sceneState, height = 216, children }) {
  const ref = useRef(null), rig = useRef(null);
  const [failed, setFailed] = useState(false);
  useEffect(() => {
    let dead = false;
    import("../rig/rigScene.js").then((mod) => {
      if (dead || !ref.current) return;
      try {
        const rm = window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)");
        rig.current = mod.createRig(ref.current, { isolate: partId, autorotate: true, noPick: true, reducedMotion: !!(rm && rm.matches) });
      } catch { setFailed(true); }
    }).catch(() => setFailed(true));
    return () => { dead = true; if (rig.current) rig.current.dispose(); rig.current = null; };
  }, []);
  useEffect(() => {
    rig.current?.update({ isolate: partId, fault: sceneState.fault, cand: sceneState.cand, unusual: sceneState.unusual, unusable: Object.keys(sceneState.flags || {}) });
  });
  return (
    <div className="rd-iso" style={{ height }}>
      <div ref={ref} style={{ position: "absolute", inset: 0 }} />
      {failed && <span className="rd-iso-fail rd-sub-s">3D view unavailable</span>}
      {children}
    </div>
  );
}

export function ChannelCard({ channels, rows, dev, sel, onSelect }) {
  const n = rows.filter((r) => r.beyond).length;
  const ref = channels?.reference;
  return (
    <Card id="rd-sensor-channels" area="ch" title="Sensor channels vs a healthy recording" kicker={dev ? channels ? `${n} of ${rows.length} beyond typical variation` : "Measured" : undefined}>
      {!channels ? <p className="rd-muted">Loading the channel comparison…</p> : <>
        {!dev && <span className="rd-m-label">{n} of {rows.length} beyond typical variation</span>}
        <span className="rd-sub-s" style={{ marginTop: -8 }}>{dev ? <>Reference: healthy training recording <span className="mono">{ref.rec_id}</span> at {ref.speed_hz.toFixed(2)} Hz</> : <>Reference: a healthy recording at a similar speed ({ref.speed_hz.toFixed(1)} Hz)</>} · each channel's most changed measurement</span>
        <div className="rd-ch-rows">
          {rows.map((r) => {
            const on = sel === r.sensor;
            if (!r.comparable) return <div key={r.id} className="rd-ch-row is-off"><span className="rd-ch-name">{r.name}</span><span className="rd-sub-s">no comparable reading</span><span /></div>;
            const v = D.logPos(r.ratio), one = D.logPos(1), b0 = D.logPos(1 / r.band), b1 = D.logPos(r.band);
            const col = r.beyond ? K.limeT : K.ink2;
            return (
              <button key={r.id} className={"rd-ch-row" + (on ? " is-on" : "")} onClick={() => onSelect(r)}
                aria-label={`${r.name}: ${r.beyond ? "beyond" : "within"} typical healthy variation. Select the sensor on the rig.`}>
                <span className="rd-ch-name"><i style={{ background: r.beyond ? K.lime : "oklch(0.23 0.006 260 / 0.2)", border: r.beyond ? "1.5px solid " + K.ink : "none" }} />{r.name}</span>
                <span className="rd-ch-track">
                  <span className="rd-ch-axis" />
                  <span className="rd-ch-band" style={{ left: b0 + "%", width: b1 - b0 + "%" }} />
                  <span className="rd-ch-one" style={{ left: one + "%" }} />
                  <span className="rd-ch-bar" style={{ left: Math.min(one, v) + "%", width: Math.abs(v - one) + "%", background: r.beyond ? K.lime : "oklch(0.23 0.006 260 / 0.25)" }} />
                  <span className="rd-ch-knob" style={{ left: v + "%", background: r.beyond ? K.lime : K.white, boxShadow: "0 0 0 1px " + (r.beyond ? K.ink : "oklch(0.23 0.006 260 / 0.35)") }} />
                </span>
                <span className={dev ? "mono rd-ch-v" : "rd-ch-v"} style={{ color: col }}>{dev ? `${D.changeText(r)} ${D.METRIC_NAME[r.metric] || r.metric}` : r.beyond ? "Changed" : "Typical"}</span>
              </button>
            );
          })}
        </div>
        <div className="rd-ch-row rd-ch-foot"><span className="rd-sub-s"><i className="rd-band-key" />Typical healthy variation</span>
          <span className="rd-ch-track">{D.CH_TICKS.map(([x, l]) => <span key={l} className="mono rd-ch-tick" style={{ left: D.logPos(x) + "%" }}>{l}</span>)}</span><span /></div>
        <span className="rd-note">Measured differences from a healthy recording at a similar speed: descriptive, not a calibrated anomaly test. They show where the signal changed, not the fault location. Click a row to select the sensor on the rig.</span>
      </>}
    </Card>
  );
}

const MAXO = 8, ENV_MAXO = 12;
const f3 = (v) => (v == null || !Number.isFinite(v) ? "—" : Math.abs(v) >= 100 ? v.toFixed(0) : Math.abs(v) >= 1 ? v.toFixed(2) : v.toFixed(4));

export function SpectrumCard({ record, signals, part, chId, chOptions, mode, onMode, onChannel, dev = true }) {
  const fr = record.signal_quality.shaft_speed_hz;
  const env = record.physics.envelope_spectrum;
  const hasEnv = !!env?.channels?.[chId];
  const m = mode === "envelope" && hasEnv ? "envelope" : "order";
  let chart = null;
  if (m === "envelope") {
    const ys = env.channels[chId], ymax = rangeMax(env.orders, ys, 0.3, ENV_MAXO) * 1.15, p = linePath(env.orders, ys, 0, ENV_MAXO, ymax);
    const cur = record.physics.cursors_orders;
    const cs = [["ftf", "FTF", cur.ftf[0]], ["shaft", "1×", 1], ["bpfo", "BPFO", cur.bpfo[0]], ["bsf", "2×BSF", cur.bsf[1]], ["bpfi", "BPFI", cur.bpfi[1]]];
    chart = { p, ymax, max: ENV_MAXO, cursors: cs.map(([k, l, o]) => ({ k, l, left: (o / ENV_MAXO) * 100, color: k === "shaft" ? K.ink3 : K.orangeT })),
      ticks: [0, 3, 6, 9, 12], caption: `Envelope spectrum (${env.band_hz[0] / 1000}–${env.band_hz[1] / 1000} kHz band) · ${CH[chId].name} · orders of the measured speed (1× = ${fr.toFixed(2)} Hz)`,
      note: "Bearing cursors at the measured speed. Bearing evidence comes from the envelope spectrum." };
  } else if (signals?.channels?.[chId]) {
    const s = signals.channels[chId], ymax = rangeMax(signals.orders, s.spectrum, 0.5, MAXO) * 1.15, p = linePath(signals.orders, s.spectrum, 0, MAXO, ymax);
    chart = { p, ymax, max: MAXO, cursors: [1, 2, 3].map((o) => ({ k: o, l: o + "×", left: (o / MAXO) * 100, color: K.ink3 })), ticks: [0, 2, 4, 6, 8],
      caption: `Order spectrum · ${CH[chId].name} · multiples of the measured shaft speed (1× = ${fr.toFixed(2)} Hz) · V`,
      note: "Shaft orders are evidence for imbalance and misalignment." };
  }
  return (
    <Card area="sp" label="Frequency spectrum">
      <div className="rd-row-wrap" style={{ gap: 10 }}>
        <div className="rd-col" style={{ marginRight: "auto" }}><h2>Frequency spectrum</h2><span className="rd-sub-s">{chart ? chart.caption : "Loading the stored signal…"}</span></div>
        {chOptions.length > 1 && <div className="rd-row-wrap" style={{ gap: 6 }}>{chOptions.map((c) => (
          <button key={c} className={"rd-chipbtn" + (c === chId ? " is-on" : "")} onClick={() => onChannel(c)}>{part.id.endsWith("_brg") || part.id.startsWith("s_") ? CH[c].dir || CH[c].name : CH[c].short}</button>))}</div>}
        <Seg label="Spectrum" value={m} onChange={(x) => (x === "order" || hasEnv) && onMode(x)} options={[["order", "Order"], ...(hasEnv ? [["envelope", "Envelope"]] : [])]} />
      </div>
      {chart ? <>
        <div className="rd-plot" style={{ marginTop: m === "envelope" ? 22 : 18 }}>
          <Svg paths={[{ d: chart.p.area, fill: al(K.orange, 0.1) }, { d: chart.p.line, stroke: K.orange, w: 1.4 }]} />
          {chart.cursors.map((c, i) => <span key={c.k}><span className="rd-cursor" style={{ left: c.left + "%", borderColor: c.color }} />
            <span className="mono rd-cursor-l" style={{ left: c.left + "%", color: c.color, top: m === "envelope" ? -15 - (i % 2) * 11 : -15 }}>{c.l}</span></span>)}
          {chart.ticks.map((t) => <span key={t} className="mono rd-xtick" style={{ left: (t / chart.max) * 100 + "%" }}>{t ? t + "×" : "0"}</span>)}
          <span className="mono rd-ytick" style={{ top: -6 }}>{f3(chart.ymax)}</span><span className="mono rd-ytick" style={{ bottom: -6 }}>0</span>
        </div>
        <div className="rd-row-wrap rd-sub-s" style={{ gap: 16 }}><span className="rd-legend-line"><i style={{ background: K.orange }} />This recording</span>
          {dev ? <span>No healthy-reference spectrum is stored with the case.</span> : <Caveat>No healthy-reference spectrum is stored with the case.</Caveat>}<span style={{ marginLeft: "auto" }}>{chart.note}</span></div>
        {!dev && <Caveat>Without a healthy spectrum, compare the 2× peak with the 1× peak in this recording.</Caveat>}
      </> : <p className="rd-muted">{part.ch.includes(chId) ? "The stored signal is not available for this channel." : ""}</p>}
    </Card>
  );
}

export function WaveformCard({ signals, chId }) {
  const s = signals?.channels?.[chId];
  let el = null;
  if (s) {
    const w = s.waveform, m = Math.max(...w.max.map(Math.abs), ...w.min.map(Math.abs)) * 1.15 || 1, dur = w.dt_s * w.max.length;
    el = { m, dur, w, path: bandPath(w.min, w.max, m) };
  }
  return (
    <Card area="wf" label="Time waveform">
      <div className="rd-col"><h2>Time waveform</h2><span className="rd-sub-s">{el ? `${CH[chId].name} · ${el.dur.toFixed(2)} s excerpt of the cleaned recording (V)` : "Loading the stored signal…"}</span></div>
      {el && <div className="rd-plot" style={{ marginTop: 4, backgroundSize: "100% 50%" }}>
        <Svg paths={[{ d: el.path, fill: al(K.violet, 0.5), stroke: K.violet, w: 0.6 }]} />
        {[0, el.dur / 2, el.dur].map((t) => <span key={t} className="mono rd-xtick" style={{ left: (t / el.dur) * 100 + "%" }}>{(el.w.t0_s + t).toFixed(2)} s</span>)}
        <span className="mono rd-ytick" style={{ top: -6 }}>+{f3(el.m)}</span><span className="mono rd-ytick" style={{ bottom: -6 }}>−{f3(el.m)}</span>
      </div>}
      <span className="rd-note">Minimum and maximum per time bucket of the stored, cleaned signal. Repeated pulses or isolated spikes may suggest impacts or signal quality problems; this trace alone cannot name a fault.</span>
    </Card>
  );
}

function ExplanationBody({ explanation, record, report }) {
  const e = explanation?.status === "ready" ? explanation.data : null;
  return <>
    {e && <span className="rd-chip" style={{ alignSelf: "flex-start", height: 24, fontWeight: 400, fontSize: 11.5, marginTop: -4,  color: e.source === "llm" ? "oklch(0.44 0.15 277)" : K.ink2, background: e.source === "llm" ? al(K.violet, 0.09) : "oklch(0.23 0.006 260 / 0.06)" }}>{e.badge}</span>}
    {!explanation || explanation.status === "loading" ? <p className="rd-muted">Preparing explanation…</p>
      : explanation.status === "error" ? <p className="rd-muted">The explanation could not be prepared; the report below still applies.</p> : <>
        <ul className="rd-bullets">{e.text.why_flagged.map((x) => <li key={x}>{x}</li>)}</ul>
        {e.text.uncertainty?.length > 0 && <div className="rd-unc"><span className="rd-kicker" style={{ color: K.amberT }}>Uncertainty</span>
          {e.text.uncertainty.map((x) => <div key={x}><i />{x}</div>)}</div>}
      </>}
    <div className="rd-box"><span className="rd-kicker">Supporting checks</span><span>{M.supporting(record, report?.assessment) || "Checking…"}</span></div>
  </>;
}

function XpCard({ record, explanation, onRegenerate }) {
  const lr = record.xai?.lr, cnn = record.xai?.cnn, e = explanation?.status === "ready" ? explanation.data : null;
  const feats = (lr?.top_features || []).slice(0, 5), mx = Math.max(1e-9, ...feats.map((f) => Math.abs(f.contribution)));
  return (
    <Card area="xp" title="Explanations">
      {lr && <><span className="rd-sub-s">Secondary logistic regression · feature contributions ({human(lr.class)}) · top 5</span>
        <div className="rd-stack-6">{feats.map((f) => {
          const w = (Math.abs(f.contribution) / mx) * 50;
          return <div key={f.feature} className="rd-feat" title={f.definition || f.feature}>
            <span className="mono rd-ellipsis">{f.feature}</span>
            <span className="rd-div"><i /><span style={{ left: f.contribution >= 0 ? "50%" : 50 - w + "%", width: w + "%", background: f.contribution >= 0 ? K.orange : K.violet }} /></span>
            <span className="mono" style={{ textAlign: "right" }}>{f.contribution >= 0 ? "+" : "−"}{Math.abs(f.contribution).toFixed(2)}</span></div>;
        })}</div></>}
      {cnn && <div className="rd-dashed"><span className="rd-kicker" style={{ color: K.amberT }}>CNN attribution · exploratory</span>
        <span className="rd-p-s">{cnn.label}. Top channels: {cnn.top_channels.map((c) => c.channel).join(", ")}. Order bands {cnn.top_order_bands.map((b) => `${b.orders[0].toFixed(1)}–${b.orders[1].toFixed(1)}×`).join(", ")}. Not evidence of location.</span></div>}
      {e ? <div className="rd-kv">
        <div><span>Source</span><span>{e.source === "llm" ? "Language model" : "Template"}</span></div>
        <div><span>Prompt version</span><span className="mono">{e.prompt_version}</span></div>
        <div><span>Latency · attempts</span><span className="mono">{Math.round(e.latency_ms)} ms · {e.attempts}</span></div>
        <div><span>Fallback reason</span><span>{e.fallback_reason || "None"}</span></div>
      </div> : <p className="rd-muted">Preparing explanation…</p>}
      {onRegenerate && <button className="rd-ghost" onClick={onRegenerate}>Regenerate explanation</button>}
    </Card>
  );
}
