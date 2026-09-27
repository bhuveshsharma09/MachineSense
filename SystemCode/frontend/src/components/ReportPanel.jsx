import { useEffect, useState } from "react";
import { errorText, getReport } from "../api.js";

// Minimal renderer for the fixed report template (headings, bullets, bold); no markdown library.
function inline(t) {
  return t.split(/(\*\*[^*]+\*\*)/).map((part, i) =>
    part.startsWith("**") ? <strong key={i}>{part.slice(2, -2)}</strong> : <span key={i}>{part}</span>);
}

function Markdown({ text }) {
  const out = [];
  let list = [];
  const flush = () => { if (list.length) { out.push(<ul key={out.length} className="list-disc ml-6 my-1">{list}</ul>); list = []; } };
  text.split("\n").forEach((line, i) => {
    if (line.startsWith("- ")) { list.push(<li key={i}>{inline(line.slice(2))}</li>); return; }
    flush();
    if (line.startsWith("# ")) out.push(<h3 key={i} className="text-lg font-semibold mt-2">{line.slice(2)}</h3>);
    else if (line.startsWith("## ")) out.push(<h4 key={i} className="font-semibold mt-3">{line.slice(3)}</h4>);
    else if (line.trim()) out.push(<p key={i} className="my-1">{inline(line)}</p>);
  });
  flush();
  return <div className="text-sm leading-relaxed">{out}</div>;
}

export default function ReportPanel({ caseId }) {
  const [report, setReport] = useState(null);
  useEffect(() => { getReport(caseId).then(setReport).catch((e) => setReport({ report: errorText(e) })); }, [caseId]);
  return (
    <section className="card">
      <h2 className="card-title">Report</h2>
      <p className="text-xs text-slate-500 mb-2">{report?.generated_by}</p>
      {report?.report ? <Markdown text={report.report} /> : <p className="text-sm">Loading…</p>}
    </section>
  );
}
