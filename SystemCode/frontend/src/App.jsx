import { useEffect, useState } from "react";
import { diagnoseRecording, errorText, getCase, getHealth } from "./api.js";
import LimitationsBanner from "./components/LimitationsBanner.jsx";
import SelectScreen from "./components/SelectScreen.jsx";
import DiagnosisScreen from "./components/DiagnosisScreen.jsx";

// Hash routes: #/ (select), #/case/<case_id>, #/diagnose/<rec_id> (diagnoses on load; useful for links).
function useHash() {
  const [hash, setHash] = useState(window.location.hash || "#/");
  useEffect(() => {
    const on = () => setHash(window.location.hash || "#/");
    window.addEventListener("hashchange", on);
    return () => window.removeEventListener("hashchange", on);
  }, []);
  return hash;
}

export default function App() {
  const hash = useHash();
  const [health, setHealth] = useState(null);
  const [record, setRecord] = useState(null);
  const [error, setError] = useState(null);
  const [busy, setBusy] = useState(false);

  useEffect(() => { getHealth().then(setHealth).catch((e) => setError(errorText(e))); }, []);

  useEffect(() => {
    const [, kind, id] = hash.replace(/^#/, "").split("/");
    if (kind === "case" && id && record?.case_id !== id) {
      getCase(id).then(setRecord).catch((e) => setError(errorText(e)));
    } else if (kind === "diagnose" && id) {
      setBusy(true); setError(null);
      diagnoseRecording(decodeURIComponent(id))
        .then((r) => { setRecord(r); window.location.hash = `#/case/${r.case_id}`; })
        .catch((e) => setError(errorText(e)))
        .finally(() => setBusy(false));
    }
  }, [hash]);

  const onResult = (r) => { setRecord(r); window.location.hash = `#/case/${r.case_id}`; };
  const onCase = hash.startsWith("#/case/") && record;

  return (
    <div className="min-h-screen">
      <header className="bg-slate-900 text-white">
        <div className="max-w-6xl mx-auto px-4 py-3 flex items-center justify-between">
          <a href="#/" className="text-lg font-semibold">MachineSense</a>
          <span className="text-xs text-slate-300">
            {health ? `bundle ${health.manifest_sha256.slice(0, 12)} · verified` : "connecting…"}
          </span>
        </div>
      </header>
      <LimitationsBanner items={record?.limitations || health?.limitations} />
      <main className="max-w-6xl mx-auto px-4 py-6">
        {error && <div className="mb-4 rounded-md border border-red-300 bg-red-50 p-3 text-red-800">{error}</div>}
        {busy && <div className="mb-4 text-slate-600">Diagnosing…</div>}
        {onCase ? <DiagnosisScreen record={record} /> : <SelectScreen onResult={onResult} onError={setError} />}
      </main>
    </div>
  );
}
