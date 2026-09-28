import { K, TAG, al } from "../lib/theme.js";
import {
  ASK_DISABLED, IN_SAMPLE_NOTICE, PART_NAME, PIPELINE, PLANNED_AGENTS, SCORES_NOTE, STEP_CODE, cap, fmtScore, human,
  locations, shownWarnings, verdict, verdictTitle,
} from "../lib/model.js";

const STYLE = {
  healthy: { color: K.greenT, dot: K.green, halo: al(K.green, 0.18) },
  caution: { color: K.amberT, dot: K.amber, halo: al(K.amber, 0.25) },
  fault: { color: K.orangeT, dot: K.orange, halo: al(K.orange, 0.2) },
  further: { color: K.amberT, dot: K.amber, halo: al(K.amber, 0.25) },
  neutral: { color: K.ink2, dot: K.ink2, halo: "oklch(0.23 0.006 260 / 0.08)" },
  running: { color: K.ink, dot: K.ink, halo: "oklch(0.23 0.006 260 / 0.1)" },
  refused: { color: K.amberT, dot: K.amber, halo: al(K.amber, 0.2) },
};

function Chip({ kind, text }) {
  const s = STYLE[kind];
  return (
    <span style={{ display: "inline-flex", alignItems: "center", gap: 8, fontSize: 13, fontWeight: 500, color: s.color }}>
      <span style={{ width: 8, height: 8, borderRadius: "50%", background: s.dot, boxShadow: "0 0 0 4px " + s.halo }} />{text}
    </span>
  );
}

function Warnings({ title, items }) {
  if (!items.length) return null;
  return (
    <div role="alert" style={{ display: "flex", flexDirection: "column", gap: 6, padding: "10px 12px", borderRadius: 12, background: al(K.amber, 0.12), border: "1px solid oklch(0.70 0.13 70 / 0.35)" }}>
      <span className="mono" style={{ fontSize: 9.5, letterSpacing: "0.08em", textTransform: "uppercase", color: "oklch(0.48 0.1 62)" }}>{title}</span>
      {items.map((w) => (
        <div key={w} style={{ display: "flex", gap: 8, alignItems: "baseline", fontSize: 12.5, lineHeight: 1.4, color: "oklch(0.32 0.03 60)" }}>
          <span style={{ width: 5, height: 5, flex: "none", background: "oklch(0.72 0.14 70)", transform: "translateY(-2px) rotate(45deg)" }} /><span>{w}</span>
        </div>
      ))}
    </div>
  );
}

const TONE = { good: K.greenT, warn: K.amberT, neutral: K.ink2 };
const pct = (x) => `${x >= 0 ? "+" : "−"}${Math.abs(x * 100) < 0.01 ? Math.abs(x * 100).toFixed(4) : Math.abs(x * 100).toFixed(2)} %`;
const abs = (x) => { const a = Math.abs(x); const t = a >= 100 ? a.toFixed(1) : a >= 1 ? a.toFixed(3) : a.toPrecision(3); return (x >= 0 ? "+" : "−") + t; };

function Checks({ a }) {
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 5 }}>
      <span className="mono kicker">Checks against calibrated thresholds</span>
      {a.checks.map((c) => (
        <div key={c.key} className="check-row" title={`${c.name}: needs ${c.required} ${c.threshold_text}; margin ${c.margin_scale}`}>
          <span className="check-name">{cap(c.name)}</span>
          <span className="mono">{c.value_text} {c.relation} {c.threshold_text}</span>
          <span className="check-status"><b style={{ color: TONE[c.tone], fontWeight: 500 }}>{c.status}</b> · margin {abs(c.margin_abs)} ({pct(c.margin_rel)}{c.key === "msp" ? " of 1 − τ" : ""}){c.narrow ? " · narrow" : ""}</span>
        </div>
      ))}
    </div>
  );
}

export function Diagnosis({ record, status, error, selection, assessment, runningFor }) {
  if (status === "running") {
    return (
      <>
        <Chip kind="running" text={`Diagnosing ${runningFor || selection?.label || "this recording"}`} />
        <div className="dx-title">Running the deterministic pipeline</div>
        <p className="dx-sub">Signal quality, CNN-6, the healthy-baseline checks, the decision rule and the report are computed on the server for <b style={{ overflowWrap: "anywhere" }}>{runningFor || selection?.label || "this recording"}</b>.</p>
        <div className="progress"><div /></div>
      </>
    );
  }
  if (error) {
    const refused = error.status === 403;
    return (
      <>
        <Chip kind="refused" text={refused ? "Upload refused" : "Input not accepted"} />
        <div className="dx-title">{refused ? "This appears to be a reserved test recording" : "The file doesn’t match the expected format"}</div>
        <p className="dx-sub">{refused
          ? "Reserved test recordings are kept sealed for the final evaluation, so the app does not diagnose them. The check compares fingerprints of known reserved recordings; it is a guard, not a guarantee."
          : "Expected a headerless CSV with 8 numeric columns (tachometer, 6 accelerometer axes, microphone) and 250,000 rows: 5 s at 50 kHz."}</p>
        <Warnings title={refused ? "Server response · 403" : "File check · server response 422"} items={[error.message]} />
      </>
    );
  }
  if (!record) {
    return (
      <>
        <div className="dx-title">{selection ? "Ready to investigate" : "No recording selected"}</div>
        <p className="dx-sub">{selection ? `Start the investigation to check signal quality and run the diagnostic models on ${selection.label}.` : "Choose a development recording or upload a CSV file to begin."}</p>
      </>
    );
  }
  const d = record.diagnosis, v = verdict(record), loc = locations(record), a = assessment;
  const warn = shownWarnings(record);
  const locText = loc.fault.length ? loc.fault.map((p) => PART_NAME[p]).join(" and ")
    : loc.cand.length ? loc.cand.map((p) => PART_NAME[p]).join(", ") : "";
  const lr = d.lr_secondary, st = STYLE[v.kind];
  return (
    <>
      {a?.disagree && (
        <div role="alert" style={{ display: "flex", flexDirection: "column", gap: 4, padding: "10px 12px", borderRadius: 12, background: al(K.amber, 0.2), border: "1px solid oklch(0.70 0.13 70 / 0.55)", color: "oklch(0.40 0.09 60)" }}>
          <div style={{ display: "flex", gap: 10, alignItems: "center", fontSize: 14.5, fontWeight: 500, lineHeight: 1.3 }}>
            <span style={{ width: 9, height: 9, flex: "none", background: "oklch(0.72 0.14 70)", transform: "rotate(45deg)" }} />Checks disagree — recheck recommended.
          </div>
          <span style={{ fontSize: 12, lineHeight: 1.4 }}>{cap(a.disagree_reasons.join("; "))}.</span>
        </div>
      )}
      {d.decision === "known_fault" && <Chip kind={v.kind} text={v.chip} />}
      <div className="dx-title" style={{ display: "flex", alignItems: "center", gap: 12 }}>
        {d.decision !== "known_fault" && <span style={{ width: 10, height: 10, flex: "none", borderRadius: "50%", background: st.dot, boxShadow: "0 0 0 4px " + st.halo }} />}
        {verdictTitle(record)}
      </div>
      {d.caution && <p className="dx-sub">A healthy-baseline check disagrees ({d.caution.replace(/_/g, " ")}); consider a recheck.</p>}
      {a?.narrow_notes?.length > 0 && (
        <div style={{ fontSize: 12.5, lineHeight: 1.45, padding: "7px 11px", borderRadius: 10, background: "oklch(0.23 0.006 260 / 0.05)", color: K.ink2 }}>
          {a.narrow_notes.map((n) => <div key={n}>{n}</div>)}
        </div>
      )}
      {locText && (
        <div style={{ fontSize: 13, color: K.ink2 }}>
          {loc.fault.length ? "Location according to the dataset label (model)" : "Candidate locations · top-2 classes (model)"} · <span style={{ color: st.color }}>{locText}</span>
        </div>
      )}
      {record.source === "dataset" && (
        <div className="in-sample">{IN_SAMPLE_NOTICE}</div>
      )}
      <Warnings title="Warnings · measured speed and signal quality" items={warn} />
      <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
        <div className="model-row"><span className="mono lbl">CNN-6 · main model</span><span className="cls">{human(d.top_class)}</span><span className="mono sc">MSP {fmtScore(d.scores.msp)}</span></div>
        <div className="model-row"><span className="mono lbl">Secondary logistic regression</span><span className="cls">{human(lr.top_class)}</span><span className="mono sc">{fmtScore(lr.probabilities[lr.top_class])}</span></div>
        <span style={{ fontSize: 11.5, lineHeight: 1.4, color: "oklch(0.48 0.008 260)" }}>
          {d.top_class === lr.top_class ? "Both rank the same class first. " : "They rank different classes first. "}{SCORES_NOTE}
        </span>
      </div>
      {a && <Checks a={a} />}
      {a && (
        <div style={{ display: "flex", flexDirection: "column", gap: 3, fontSize: 12, color: K.ink2 }}>
          <span title={a.variants.map((x) => `${x.name}: ${human(x.verdict)}`).join("\n")}>{a.variants_summary}</span>
          <span title={a.rule_tooltip} style={{ color: "oklch(0.48 0.008 260)", textDecoration: "underline dotted", textUnderlineOffset: 3, cursor: "help", alignSelf: "flex-start" }}>{a.rule_line}</span>
        </div>
      )}
    </>
  );
}

function Activity({ status, activity }) {
  return (
    <div>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 12 }}>
        <span style={{ fontSize: 14, fontWeight: 500 }}>Investigation activity</span>
        <span className="badge">Deterministic pipeline</span>
      </div>
      {status === "running" && (
        <ol className="timeline">
          {PIPELINE.map((t, i) => (
            <li key={t}><span className={"code" + (i === 0 ? " active" : "")}>{Object.values(STEP_CODE)[i]}</span>
              <div><div className="row"><span className="act">{t}</span><span className="mono t">…</span></div><span className="agent">deterministic step · pending (computed on the server)</span></div></li>
          ))}
        </ol>
      )}
      {status !== "running" && activity && (
        <ol className="timeline">
          {activity.steps.map((s) => (
            <li key={s.step}>
              <span className="code">{STEP_CODE[s.step]}</span>
              <div>
                <div className="row"><span className="act">{s.title}</span><span className="mono t">{s.ms == null ? "—" : s.ms < 1 ? "<1 ms" : `${s.ms.toFixed(0)} ms`}</span></div>
                <span className="agent">{s.kind}{s.startup ? " · at server startup" : ""}</span>
                <span className="res">{s.detail}</span>
              </div>
            </li>
          ))}
        </ol>
      )}
      {status !== "running" && !activity && <p style={{ margin: "0 0 12px", fontSize: 12.5, lineHeight: 1.5, color: "oklch(0.45 0.008 260)" }}>The pipeline steps and their measured timings appear here when you start an investigation.</p>}
      <div style={{ marginTop: 14, display: "flex", flexDirection: "column", gap: 10, opacity: 0.7 }}>
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
          <span style={{ fontSize: 13, fontWeight: 500 }}>Investigation agents</span>
          <span className="badge">Planned</span>
        </div>
        <p style={{ margin: 0, fontSize: 12, lineHeight: 1.5, color: "oklch(0.45 0.008 260)" }}>Five agents are planned; none is connected. The steps above are fixed code, not agents.</p>
        {PLANNED_AGENTS.map((a) => (
          <div key={a.code} style={{ display: "grid", gridTemplateColumns: "28px minmax(0,1fr)", gap: 12, alignItems: "start" }}>
            <span className="code">{a.code}</span>
            <div style={{ display: "flex", flexDirection: "column" }}><span style={{ fontSize: 13, fontWeight: 500 }}>{a.name}</span><span style={{ fontSize: 12.5, color: "oklch(0.45 0.008 260)" }}>{a.role}</span></div>
          </div>
        ))}
      </div>
    </div>
  );
}

export function Report({ status, report }) {
  if (!report) {
    return <p style={{ margin: 0, fontSize: 13, lineHeight: 1.5, color: "oklch(0.45 0.008 260)" }}>
      {status === "running" ? "The report is written after the decision rule." : "Select a recording and start the investigation to produce a report."}</p>;
  }
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 20 }}>
      {report.sections.map((sec) => (
        <div key={sec.n} style={{ display: "flex", flexDirection: "column", gap: 10 }}>
          <div style={{ display: "flex", gap: 9, alignItems: "baseline" }}><span className="mono" style={{ fontSize: 10.5, color: K.ink3 }}>{sec.n}</span><h3 style={{ margin: 0, fontSize: 15, fontWeight: 500 }}>{sec.title}</h3></div>
          {sec.items.map((it, i) => (
            <div key={i} title={it.hover} style={{ display: "flex", flexDirection: "column", gap: 4 }}>
              <span className="tag" style={{ alignSelf: "flex-start", color: TAG[it.tag].c, background: TAG[it.tag].b }}>{it.tag}</span>
              <span style={{ fontSize: 13, lineHeight: 1.5, color: "oklch(0.30 0.007 260)" }}>{it.text}</span>
            </div>
          ))}
        </div>
      ))}
      <p style={{ margin: 0, fontSize: 11.5, lineHeight: 1.5, color: "oklch(0.48 0.008 260)" }}>Fixed template built from the evidence record; no language model. Guidance items are placeholders: the guidance library is not yet connected.</p>
    </div>
  );
}

export default function Aside({ record, status, error, selection, activity, report, tab, onTab, runningFor }) {
  return (
    <aside aria-label="Diagnosis and investigation" className="glass aside" style={{ position: "absolute", right: 16, bottom: 16, zIndex: 5, display: "flex", flexDirection: "column", borderRadius: 20, overflow: "hidden" }}>
      <div style={{ flex: 1, minHeight: 0, overflow: "auto" }}>
        <div className="dx-block" style={{ display: "flex", flexDirection: "column", borderBottom: "1px solid oklch(0.23 0.006 260 / 0.08)" }}>
          <span className="mono kicker">Diagnosis</span>
          <Diagnosis record={record} status={status} error={error} selection={selection} assessment={report?.assessment}
            runningFor={runningFor} />
        </div>
        <div role="tablist" className="seg" style={{ margin: "12px 16px 0", display: "flex", padding: 3, borderRadius: 11 }}>
          {[["activity", "Activity"], ["report", "Report"]].map(([k, lab]) => (
            <button key={k} role="tab" aria-selected={tab === k} onClick={() => onTab(k)} className={tab === k ? "on" : ""} style={{ flex: 1, height: 32, fontSize: 13 }}>{lab}</button>
          ))}
        </div>
        <div style={{ padding: "14px 22px 16px" }}>
          {tab === "activity" ? <Activity status={status} activity={activity} /> : <Report status={status} report={report} />}
        </div>
      </div>
      <div style={{ flex: "none", padding: "8px 14px 10px", borderTop: "1px solid oklch(0.23 0.006 260 / 0.08)" }}>
        <button disabled className="ghost-btn" title={ASK_DISABLED} style={{ width: "100%", minHeight: 34, padding: "4px 10px", fontSize: 12, lineHeight: 1.3 }}>
          <span style={{ color: K.ink2 }}>Ask about this recording</span> · <span>{ASK_DISABLED}</span>
        </button>
      </div>
    </aside>
  );
}
