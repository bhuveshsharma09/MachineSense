import VerdictCard from "./VerdictCard.jsx";
import SignalQualityPanel from "./SignalQualityPanel.jsx";
import ProbabilityChart from "./ProbabilityChart.jsx";
import EnvelopeChart from "./EnvelopeChart.jsx";
import ExplanationPanel from "./ExplanationPanel.jsx";
import ReportPanel from "./ReportPanel.jsx";

export default function DiagnosisScreen({ record }) {
  const p = record.provenance;
  return (
    <div className="space-y-4">
      <div className="text-sm text-slate-600">
        <a href="#/" className="underline">← choose another recording</a>
        <span className="ml-3">{record.input.rec_id || "uploaded recording"} · case {record.case_id.slice(0, 8)}</span>
      </div>
      <div className="grid gap-4 md:grid-cols-3">
        <div className="md:col-span-2"><VerdictCard d={record.diagnosis} /></div>
        <SignalQualityPanel q={record.signal_quality} />
      </div>
      <ProbabilityChart d={record.diagnosis} />
      <EnvelopeChart physics={record.physics} />
      <ExplanationPanel xai={record.xai} />
      <ReportPanel caseId={record.case_id} />
      <footer className="text-xs text-slate-500">
        Bundle manifest {p.manifest_sha256.slice(0, 12)} · schema {p.schema_version} · split hash {p.split_hash.slice(0, 12)}
        · bundle revision {p.bundle_git_sha.slice(0, 8)}
      </footer>
    </div>
  );
}
