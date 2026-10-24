import { K, al } from "../lib/theme.js";
import * as M from "../lib/mechanic.js";
import { IN_SAMPLE_NOTICE, shownWarnings, verdict, verdictTitle } from "../lib/model.js";
import { AssistantIcon } from "./AgentChat.jsx";
import { PipelineButton } from "./TopBar.jsx";

// Stage App-8: on the report page the right-hand panel is this card; the full diagnosis lives on the dashboard
// (#/case/<id>/dashboard) and the assistant on its own page (#/case/<id>/assistant).
const DOT = { fault: K.orange, further: K.amber, caution: K.amber, healthy: K.green };
const COLOR = { fault: K.orangeT, further: K.amberT, caution: K.amberT, healthy: K.greenT };
const page = (id, p) => `#/case/${encodeURIComponent(id)}/${p}`;

export default function CaseCard({ record, onShowPart }) {
  const v = verdict(record), loc = M.location(record), warn = shownWarnings(record);
  return (
    <aside aria-label="Diagnosis summary" className="glass aside case-card">
      <span className="case-card-chip" style={{ color: COLOR[v.kind] }}>
        <span style={{ width: 8, height: 8, borderRadius: "50%", background: DOT[v.kind], boxShadow: "0 0 0 4px " + al(DOT[v.kind], 0.2) }} />{v.chip}</span>
      <div className="case-card-title">{verdictTitle(record)}</div>
      {loc.text && <div className="case-card-loc">{loc.text}</div>}
      {loc.parts.length > 0 && <div className="mv-parts">{loc.parts.map((p) => <button key={p} className="chip" onClick={() => onShowPart?.(p)}>Show {M.PART_PLAIN[p].toLowerCase()} on the rig</button>)}</div>}
      {warn.length > 0 && <div className="case-card-warn" role="alert">{warn.length} signal-quality warning{warn.length > 1 ? "s" : ""} affect this result — see the dashboard.</div>}
      {record.source === "dataset" && <div className="in-sample">{IN_SAMPLE_NOTICE}</div>}
      <a className="case-card-open" href={page(record.case_id, "dashboard")}>Open the dashboard <span aria-hidden="true">→</span></a>
      <div className="case-card-links">
        <a href={page(record.case_id, "assistant")}>Ask the assistant</a>
        <PipelineButton caseId={record.case_id} />
      </div>
      <span className="case-card-note">{M.LIMITATION}</span>
    </aside>
  );
}

// The assistant launcher once a case exists: a link to its page instead of the chat panel.
export function AssistantLink({ caseId }) {
  return (
    <a className="agent-launcher" href={page(caseId, "assistant")} aria-label="Open the investigation assistant" title="Open the investigation assistant">
      <AssistantIcon size={76} spin /><span className="agent-launcher-tip">Ask MachineSense</span>
    </a>
  );
}

// Phone report page: the same two links above the stacked panel.
export function CaseLinks({ caseId }) {
  return (
    <section className="m-card case-links">
      <a className="case-card-open" href={page(caseId, "dashboard")}>Open the dashboard <span aria-hidden="true">→</span></a>
      <a href={page(caseId, "assistant")}>Ask the assistant</a>
    </section>
  );
}
