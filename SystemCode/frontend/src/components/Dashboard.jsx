import { useEffect, useRef, useState } from "react";
import { getActivity, getCase, getChannels, getExplanation, getRecent, getReport, getSignals, postExplanation } from "../api.js";
import { K, TAG, al } from "../lib/theme.js";
import * as M from "../lib/mechanic.js";
import * as D from "../lib/dashboard.js";
import {
  CH, IN_SAMPLE_NOTICE, PART, SCORES_NOTE, STEP_CODE, cleanText, fmtScore, human, locations, shownWarnings, unusualSensors,
  verdict, verdictTitle,
} from "../lib/model.js";
import { Svg, bandPath, linePath, rangeMax } from "./Charts.jsx";
import { AssistantIcon } from "./AgentChat.jsx";
import { usePanelView } from "./Aside.jsx";
import Scene, { markerFor } from "./Scene.jsx";
import { Logo } from "./TopBar.jsx";
import "../redesign.css";

// Stage App-8: #/case/<id>/dashboard — the whole diagnosis of one stored case on one page (layout "A · Rig stage").
// Every value comes from that case's API responses; nothing is re-run.
export const DASH_NARROW = 1024;  // below this width: one column, no 3-D views
const RECENT = 5;

export const VKIND = {
  fault: { color: K.orangeT, dot: K.orange, bg: al(K.orange, 0.12) },
  further: { color: K.amberT, dot: K.amber, bg: al(K.amber, 0.18) },
  caution: { color: K.amberT, dot: K.amber, bg: al(K.amber, 0.18) },
  healthy: { color: K.greenT, dot: K.green, bg: al(K.green, 0.14) },
};

export function initials(name) {
  return (name || "Guest").trim().split(/\s+/).map((w) => w[0]).slice(0, 2).join("").toUpperCase() || "G";
}
const readName = () => { try { return localStorage.getItem("ms_display_name") || ""; } catch { return ""; } };

// Load one stored case and everything the dashboard shows about it (the main session's follow-up requests, read only;
// the explanation is generated on the server if the case has none yet, exactly as the report page does).
export function useCaseData(caseId, { withRecent = false, withExplanation = true } = {}) {
  const [st, setSt] = useState({ status: "loading" });
  const [regen, setRegen] = useState(0);
  useEffect(() => {
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
        if (withRecent) {
          getRecent(RECENT + 1, { signal }).then((cs) => Promise.all(cs.filter((c) => c.case_id !== caseId).slice(0, RECENT - 1)
            .map((c) => getCase(c.case_id, { signal }).catch(() => null))))
            .then((xs) => patch({ recent: [...xs.filter(Boolean).reverse(), record] })).catch(() => patch({ recent: [record] }));
        }
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

export default function Dashboard({ caseId }) {
  const [st, regenerate] = useCaseData(caseId, { withRecent: true });
  if (st.status !== "ok") return <PageState message={st.status === "loading" ? "Loading the case…" : st.message} />;
  return <DashboardView {...st} caseId={caseId} onRegenerate={regenerate} />;
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
  return (
    <header className="rd-header">
      <a href="#/" className="rd-brand" aria-label="MachineSense home"><Logo /><span>MachineSense</span></a>
      <span className="rd-pill-outline">Research demo</span>
      <a className="rd-back" href={back.href}><span aria-hidden="true">‹</span>{back.label}</a>
      <div className="rd-header-mid">
        {title || <div className="rd-case-pill">
          <span className="rd-dot" style={{ background: s.dot, boxShadow: "0 0 0 4px " + al(s.dot, 0.18) }} />
          <span className="rd-case-id">Case {caseId.slice(0, 8)}</span>
          <span className="mono rd-case-file">{record.input?.rec_id || "uploaded recording"}</span>
          <span className="rd-chip" style={{ background: s.bg, color: s.color }}>{v.chip}</span>
        </div>}
      </div>
      {onView && <Seg label="Reading level" value={view} onChange={onView} options={[["mechanic", "Mechanic"], ["developer", "Developer"]]} />}
      {!title && <a className="rd-ask" href={`#/case/${encodeURIComponent(caseId)}/assistant`}><span className="rd-dot" style={{ background: K.orange }} />Ask the assistant</a>}
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

function Card({ area, title, kicker, kickerColor, children, label, className = "" }) {
  return (
    <section className={"rd-card " + className} style={{ gridArea: area }} aria-label={label || title}>
      {(title || kicker) && <div className="rd-card-head">
        {title && <h2>{title}</h2>}
        {kicker && <span className="rd-kicker" style={kickerColor ? { color: kickerColor } : undefined}>{kicker}</span>}
      </div>}
      {children}
    </section>
  );
}

const Tick = ({ ok }) => <span className={"rd-tick" + (ok ? "" : " is-warn")} aria-hidden="true">{ok ? "✓" : "!"}</span>;

export function DashboardView({ caseId, record, report, channels, signals, activity, explanation, recent, onRegenerate,
  initialView, initialWidth, initialSel = null }) {
  const [view, setView] = usePanelView(initialView);
  const dev = view === "developer";
  const width = useWidth(initialWidth);
  const narrow = width < DASH_NARROW;
  const [ui, setUi] = useState({ sel: initialSel, hover: null, exploded: false, labels: false, resetTick: 0, ch: null, sp: "order", allSteps: false, lim: false });
  const set = (p) => setUi((u) => ({ ...u, ...p }));
  const userName = useRef(readName()).current;

  const d = record.diagnosis, v = verdict(record), vs = VKIND[v.kind];
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

  const mloc = M.location(record);
  const steps = D.guidanceSteps(M.guidanceItems(report));
  const shown = ui.allSteps ? steps : steps.slice(0, 3);
  const checks = report?.assessment?.checks;
  const warn = shownWarnings(record);
  const limits = (record.limitations || []).map(cleanText);
  const href = (p) => `#/case/${encodeURIComponent(caseId)}/${p}`;

  const areas = dev
    ? ["fd rig gg", "pt rig dq", "cd ch ch", "sp sp wf", "mo dc dc", "why nx nx", "xp rp rp", "rd tr ac", "tc tc tc", "lim lim lim"]
    : ["fd rig gg", "pt rig dq", "cd ch ch", "sp sp wf", "why nx nx", "rd tr ac", "lim lim lim"];
  const narrowAreas = (dev ? ["fd", "gg", "dq", "cd", "ch", "sp", "wf", "mo", "dc", "why", "nx", "xp", "rp", "rd", "tr", "ac", "tc", "lim"]
    : ["fd", "gg", "dq", "cd", "ch", "sp", "wf", "why", "nx", "rd", "tr", "ac", "lim"]);
  const gridStyle = narrow ? { gridTemplateColumns: "minmax(0,1fr)", gridTemplateAreas: narrowAreas.map((a) => `"${a}"`).join(" ") }
    : { gridTemplateColumns: "minmax(300px,340px) minmax(0,1fr) minmax(300px,340px)", gridTemplateAreas: areas.map((a) => `"${a}"`).join(" ") };

  return (
    <div className={"rd-root" + (narrow ? " is-narrow" : "")}>
      <Header caseId={caseId} record={record} view={view} onView={setView} userName={userName}
        back={{ href: `#/case/${encodeURIComponent(caseId)}`, label: "Back to rig view" }} />
      <main className="rd-grid" style={gridStyle}>
        {!narrow && <section className="rd-rig" style={{ gridArea: "rig" }} aria-label="3D rig">
          <Scene state={sceneState} insets={{ top: 84, right: 24, bottom: 56, left: 24 }} onSelect={select}
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

        <Card area="fd" label="Finding">
          <div className="rd-card-head">
            <span className="rd-verdict" style={{ color: vs.color }}><span className="rd-dot" style={{ background: vs.dot, boxShadow: "0 0 0 4px " + al(vs.dot, 0.18) }} />{v.chip}</span>
            <span className="rd-kicker">Finding</span>
          </div>
          <div className="rd-stack-4">
            <span className="rd-title-xl">{dev ? verdictTitle(record) : M.finding(record)}</span>
            {d.caution && <span className="rd-sub">A healthy-baseline check disagrees; consider a recheck.</span>}
          </div>
          {mloc.parts.length > 0 && <div className="rd-loc" style={mloc.kind === "candidate" ? { background: al(K.amber, 0.14) } : undefined}>
            <span className="rd-kicker" style={{ color: mloc.kind === "candidate" ? K.amberT : K.orangeT }}>{mloc.kind === "candidate" ? "Candidate locations" : "Suggested location"}</span>
            <span className="rd-loc-name">{mloc.parts.map((p) => M.PART_PLAIN[p]).join(" and ")}</span>
            <span className="rd-sub-s">{mloc.kind === "candidate" ? "From the two most likely fault types (model prediction)" : "According to the dataset label (model prediction)"}</span>
          </div>}
          {mloc.parts.length > 0 && <button className="rd-dark-btn" onClick={() => select(mloc.parts[0])}>Show {M.PART_PLAIN[mloc.parts[0]].toLowerCase()} on the rig</button>}
          {dev ? <>
            <div className="rd-models">
              <span className="mono rd-muted">CNN-6</span><span>{human(d.top_class)}</span><span className="mono">MSP {fmtScore(d.scores.msp)}</span>
              <span className="mono rd-muted">LR</span><span>{human(d.lr_secondary.top_class)}</span><span className="mono">{fmtScore(d.lr_secondary.probabilities[d.lr_secondary.top_class])}</span>
            </div>
            <span className="rd-note">{d.top_class === d.lr_secondary.top_class ? "Both rank the same class first. " : "They rank different classes first. "}{SCORES_NOTE}</span>
          </> : <p className="rd-p">{M.supporting(record, report?.assessment) || "Checking…"}</p>}
          {record.source === "dataset" && <span className="rd-in-sample">{IN_SAMPLE_NOTICE}</span>}
          {warn.length ? <div className="rd-warn" role="alert">{(dev ? warn : M.warnings(record)).map((w) => <div key={w}><Tick ok={false} />{w}</div>)}</div>
            : <div className="rd-ok"><Tick ok />No warnings affect this result.</div>}
        </Card>

        <Gauge record={record} dev={dev} />

        {!narrow && <PartCard partId={partId} record={record} rows={rows} sceneState={sceneState} dev={dev} />}

        <Card area="dq" title="Signal quality" kicker="Measured">
          <div className="rd-big-line"><span className="rd-title-l">{record.signal_quality.warnings.length ? `${record.signal_quality.warnings.length} warning${record.signal_quality.warnings.length > 1 ? "s" : ""}` : "No warnings"}</span>
            <span className="rd-sub-s">{8 - (record.signal_quality.clipping_channels || []).length} of 8 channels without clipping</span></div>
          {dev ? <div className="rd-kv">{D.qualityRows(record).map(([k, x]) => <div key={k}><span>{k}</span><span className="mono">{x}</span></div>)}</div>
            : <div className="rd-checklist">{D.qualityChecks(record).map((q) => <div key={q.text}><Tick ok={q.ok} />{q.text}</div>)}</div>}
          {dev && record.signal_quality.warnings.length > 0 && <div className="rd-warn">{record.signal_quality.warnings.map((w) => <div key={w}><Tick ok={false} />{w}</div>)}</div>}
        </Card>

        <Card area="cd" title="Most likely conditions" kicker="Model" kickerColor="oklch(0.44 0.15 277)">
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

        {dev && <Card area="mo" title="Model outputs" kicker="Model" kickerColor="oklch(0.44 0.15 277)">
          <div className="rd-stack-8">{D.candidates(record).map((c) => (
            <div key={c.cls} className="rd-col" style={{ gap: 4 }}>
              <div className="rd-between"><span>{c.name}</span><span className="mono">{c.cnn} · {c.lr}</span></div>
              <span className="rd-track"><span style={{ width: (d.class10_probabilities[c.cls] * 100).toFixed(1) + "%", background: c.rank === 1 ? K.orange : "oklch(0.23 0.006 260 / 0.25)" }} /></span>
            </div>))}</div>
          <span className="rd-note">Top-3 class scores · CNN-6 · secondary logistic regression. {d.top_class === d.lr_secondary.top_class ? "Both rank the same class first." : "They rank different classes first."}</span>
          <span className="rd-note">{SCORES_NOTE}</span>
        </Card>}

        {dev && <Card area="dc" title="Decision checks">
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

        <Card area="nx" title="What to inspect next" kicker="Guidance" kickerColor="oklch(0.45 0.11 125)">
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

        {dev && <Card area="rp" title="Report">
          {report ? <div className="rd-report">{report.sections.map((sec) => (
            <div key={sec.n} className="rd-col" style={{ gap: 8 }}>
              <div className="rd-row-base"><span className="mono rd-muted">{sec.n}</span><h3>{sec.title}</h3></div>
              {sec.items.map((it, i) => <div key={i} className="rd-row-top" title={it.hover}>
                <span className="tag" style={{ color: TAG[it.tag].c, background: TAG[it.tag].b }}>{it.tag}</span><span className="rd-report-t">{it.text}</span></div>)}
            </div>))}</div> : <p className="rd-muted">Loading the report…</p>}
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

        <TrendCard recent={recent} record={record} dev={dev} />

        <Card area="ac" title="Investigation activity" label="Investigation activity">
          <a className="rd-link rd-card-link" href={href("pipeline")}>View pipeline ›</a>
          {activity ? <ol className="rd-activity">{activity.steps.map((s) => (
            <li key={s.step}><span className="mono rd-code">{STEP_CODE[s.step]}</span>
              <div className="rd-col"><span className="rd-strong">{s.title}</span><span className="rd-sub-s rd-ellipsis" title={s.detail}>{dev ? s.detail : s.startup ? "at server startup" : "done"}</span></div>
              {dev && <span className="mono rd-muted">{s.ms == null ? "—" : s.ms < 1 ? "<1 ms" : `${Math.round(s.ms)} ms`}</span>}
            </li>))}</ol> : <p className="rd-muted">Loading…</p>}
        </Card>

        {dev && <Card area="tc" title="Traceability">
          <div className="rd-kv rd-kv-3">{[
            ["Recording", record.input?.rec_id || "uploaded recording"], ["Case", record.case_id], ["Created", record.created_at],
            ["Bundle manifest sha256", record.provenance.manifest_sha256.slice(0, 16) + "…"], ["Bundle git sha", String(record.provenance.bundle_git_sha).slice(0, 12)],
            ["Preprocess config sha256", record.provenance.preprocess_config_sha256.slice(0, 16) + "…"], ["Split hash", record.provenance.split_hash.slice(0, 16) + "…"],
            ["Schema version", record.provenance.schema_version], ["Diagnosis time", `${Math.round(record.timing_ms?.total || 0)} ms`],
          ].map(([k, x]) => <div key={k}><span>{k}</span><span className="mono" title={x}>{x}</span></div>)}</div>
        </Card>}

        <footer className="rd-footer" style={{ gridArea: "lim" }}>
          <div className="rd-row-wrap" style={{ gap: 18 }}>
            <span className="rd-strong">{M.LIMITATION}</span><span>{SCORES_NOTE}</span>
            {limits.length > 0 && <button className="rd-ghost" style={{ marginLeft: "auto" }} aria-expanded={ui.lim} onClick={() => set({ lim: !ui.lim })}>{ui.lim ? "Hide research limitations" : "Research limitations"}</button>}
          </div>
          {ui.lim && <ul>{limits.map((l) => <li key={l}>{l}</li>)}</ul>}
        </footer>
      </main>
      <a className="rd-fab" href={href("assistant")} aria-label="Open the investigation assistant" title="Investigation assistant"><AssistantIcon size={58} /></a>
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

// An isolated, slowly turning 3-D view of one part (rig/rigScene.js with `isolate`); nothing to pick.
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

function PartCard({ partId, record, rows, sceneState, dev }) {
  const part = PART[partId], m = markerFor(part, sceneState);
  const tag = m.tag ? { text: m.tag, color: m.tagColor, bg: al(m.fill, 0.14) }
    : { text: "Nothing flagged", color: K.ink3, bg: "oklch(0.23 0.006 260 / 0.06)" };
  const facts = D.partFacts(partId, record, rows, dev);
  return (
    <Card area="pt" label="Part view">
      <div className="rd-col" style={{ gap: 6, alignItems: "flex-start" }}><h2>{part.full || part.label}</h2>
        <span className="rd-chip" style={{ color: tag.color, background: tag.bg, height: 24, fontWeight: 400, fontSize: 11.5, whiteSpace: "normal" }}>{tag.text}</span></div>
      <PartView partId={partId} sceneState={sceneState}>
        <div className="rd-iso-facts">{facts.map((f, i) => <div key={f.k} style={{ textAlign: i ? "right" : "left" }}><span>{f.k}</span><b>{f.v}</b></div>)}</div>
        <span className="rd-iso-hint rd-kicker">Drag to rotate</span>
      </PartView>
      <p className="rd-p-s">{part.desc}</p>
      <span className="rd-sub-s">Select a part or sensor on the rig to show it here.</span>
    </Card>
  );
}

function ChannelCard({ channels, rows, dev, sel, onSelect }) {
  const n = rows.filter((r) => r.beyond).length;
  const ref = channels?.reference;
  return (
    <Card area="ch" title="Sensor channels vs a healthy recording" kicker={channels ? `${n} of ${rows.length} beyond typical variation` : "Measured"}>
      {!channels ? <p className="rd-muted">Loading the channel comparison…</p> : <>
        <span className="rd-sub-s" style={{ marginTop: -8 }}>Reference: healthy training recording <span className="mono">{ref.rec_id}</span> at {ref.speed_hz.toFixed(2)} Hz · each channel's most changed measurement</span>
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

function SpectrumCard({ record, signals, part, chId, chOptions, mode, onMode, onChannel }) {
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
          <span>No healthy-reference spectrum is stored with the case.</span><span style={{ marginLeft: "auto" }}>{chart.note}</span></div>
      </> : <p className="rd-muted">{part.ch.includes(chId) ? "The stored signal is not available for this channel." : ""}</p>}
    </Card>
  );
}

function WaveformCard({ signals, chId }) {
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
      <span className="rd-note">Minimum and maximum per time bucket of the stored, cleaned signal.</span>
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

function TrendCard({ recent, record, dev }) {
  const t = recent?.length ? D.trendBars(recent, record) : null;
  return (
    <Card area="tr" title="Compared with earlier cases" kicker={t ? `${t.bars.length} case${t.bars.length > 1 ? "s" : ""}` : "Model"}>
      <span className="rd-sub-s" style={{ marginTop: -8 }}>Speed-aware healthy check (B1s) of the most recent stored cases{dev ? ` · line: τ_b ${D.fmtCheck(t?.threshold ?? record.diagnosis.thresholds.tau_b)}` : " · line: the healthy limit"}</span>
      {!t ? <p className="rd-muted">Loading earlier cases…</p> : <div className="rd-trend" style={{ gridTemplateColumns: `repeat(${t.bars.length}, minmax(0,1fr))` }}>
        <span className="rd-trend-line" style={{ bottom: `calc(44px + (100% - 44px) * ${t.line / 100})` }} />
        {t.bars.map((b) => (
          <div key={b.id} className="rd-trend-col" title={`${b.label} UTC`}>
            {dev && <span className="mono rd-sub-s">{D.fmtCheck(b.value)}</span>}
            <span className="rd-trend-bar" style={{ height: b.h + "%", background: b.current ? K.orange : b.beyond ? al(K.orange, 0.35) : al(K.green, 0.45),
              outline: b.current ? "2px solid " + al(K.orange, 0.3) : "none" }} />
            <div className="rd-trend-lab"><b style={{ fontWeight: b.current ? 600 : 400 }}>{b.current ? "This case" : b.label.slice(0, 5)}</b>
              <span>{D.verdictWord(b.decision, b.caution)}</span></div>
          </div>))}
      </div>}
      <span className="rd-note">Earlier cases are separate dataset recordings, not repeated readings of one machine.</span>
    </Card>
  );
}
