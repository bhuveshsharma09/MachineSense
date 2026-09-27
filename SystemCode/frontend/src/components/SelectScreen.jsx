import { useEffect, useMemo, useState } from "react";
import { diagnoseRecording, diagnoseUpload, errorText, getRecordings } from "../api.js";

export default function SelectScreen({ onResult, onError }) {
  const [recs, setRecs] = useState([]);
  const [recId, setRecId] = useState("");
  const [file, setFile] = useState(null);
  const [nominal, setNominal] = useState("");
  const [busy, setBusy] = useState(false);

  useEffect(() => { getRecordings().then(setRecs).catch((e) => onError(errorText(e))); }, []);
  const groups = useMemo(() => {
    const g = {};
    for (const r of recs) (g[r.class10] ||= []).push(r);
    return g;
  }, [recs]);

  const run = (p) => { setBusy(true); onError(null); p.then(onResult).catch((e) => onError(errorText(e))).finally(() => setBusy(false)); };

  return (
    <div className="grid gap-6 md:grid-cols-2">
      <section className="card">
        <h2 className="card-title">Prepared recording</h2>
        <p className="text-sm text-slate-600 mb-3">
          Development-pool recordings only ({recs.length}); reserved test recordings are never offered. Each entry says
          whether it was used for model training or threshold calibration, so a result on it is not an independent test.
        </p>
        <label className="block text-sm font-medium mb-1" htmlFor="rec">Recording</label>
        <select id="rec" className="w-full rounded-md border border-slate-300 p-2 text-sm" value={recId}
                onChange={(e) => setRecId(e.target.value)}>
          <option value="">Choose a recording…</option>
          {Object.entries(groups).map(([cls, rows]) => (
            <optgroup key={cls} label={cls.replaceAll("_", " ")}>
              {rows.map((r) => (
                <option key={r.rec_id} value={r.rec_id}>
                  {r.rec_id} — {r.severity}{r.severity_unit} · {r.speed_hz.toFixed(1)} Hz · {r.used_for}
                </option>
              ))}
            </optgroup>
          ))}
        </select>
        <button className="mt-3 rounded-md bg-slate-900 px-4 py-2 text-white disabled:opacity-50" disabled={!recId || busy}
                onClick={() => run(diagnoseRecording(recId))}>
          {busy ? "Diagnosing…" : "Diagnose"}
        </button>
      </section>
      <section className="card">
        <h2 className="card-title">Upload a recording</h2>
        <p className="text-sm text-slate-600 mb-3">
          Headerless CSV, 8 columns (tacho, underhang axial/radial/tangential, overhang axial/radial/tangential,
          microphone), 250,000 rows = 5 s at 50 kHz. The shaft speed is measured from the recording.
        </p>
        <input type="file" accept=".csv,text/csv" onChange={(e) => setFile(e.target.files?.[0] || null)}
               className="block w-full text-sm" />
        <label className="block text-sm font-medium mt-3 mb-1" htmlFor="nom">Nominal speed (Hz, optional)</label>
        <input id="nom" type="number" step="0.1" className="w-40 rounded-md border border-slate-300 p-2 text-sm"
               value={nominal} onChange={(e) => setNominal(e.target.value)} />
        <div>
          <button className="mt-3 rounded-md bg-slate-900 px-4 py-2 text-white disabled:opacity-50" disabled={!file || busy}
                  onClick={() => run(diagnoseUpload(file, nominal))}>
            {busy ? "Diagnosing…" : "Upload and diagnose"}
          </button>
        </div>
      </section>
    </div>
  );
}
