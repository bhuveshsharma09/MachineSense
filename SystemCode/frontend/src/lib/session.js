// Diagnosis session controller (Stage App-2a.3): every request is tied to the selection that started it.
// - Each run gets a fresh request id; the diagnose request carries it and the input id (rec_id / upload id / stored case),
//   and the server echoes both. A response is applied only if it is the latest request, its echoed ids match, and the
//   selection has not changed; otherwise it is dropped.
// - Changing the selection aborts in-flight requests (AbortController) and clears the old result at once.
// - The follow-up fetches (/activity, /report, /channels, /signals) are applied only if their case_id is still the
//   displayed case and the run is still current.
// Plain JavaScript with an injected `api`, so it runs in Node tests without a browser.

export const EMPTY_EVIDENCE = { activity: null, report: null, channels: null, signals: null, explanation: null };

export const inputKey = (input) =>
  !input ? null : input.kind === "dataset" ? input.rec_id : input.kind === "upload" ? input.uploadId : `case:${input.caseId}`;

export const isAbort = (e) => e?.name === "CanceledError" || e?.name === "AbortError" || e?.code === "ERR_CANCELED";

export const errorInfo = (e) => ({ status: e?.response?.status ?? 0, message: e?.response?.data?.detail || e?.message || "request failed" });

let counter = 0;
const newRequestId = () => `req-${Date.now().toString(36)}-${(++counter).toString(36)}`;

export function createSession(api, onChange = () => {}) {
  let state = { input: null, status: "idle", record: null, error: null, ev: { ...EMPTY_EVIDENCE }, runningFor: null, requestId: null };
  let seq = 0;
  let controller = null;
  const set = (patch) => { state = { ...state, ...patch }; onChange(state); };
  const abort = () => { if (controller) controller.abort(); controller = null; };

  function select(input) {
    abort();
    seq += 1;
    set({ input, status: "idle", record: null, error: null, ev: { ...EMPTY_EVIDENCE }, runningFor: null, requestId: null });
  }

  async function follow(record, my, signal) {
    const id = record.case_id;
    const live = () => my === seq && !signal.aborted && state.record?.case_id === id;
    const [activity, report] = await Promise.all([api.getActivity(id, { signal }), api.getReport(id, { signal })]);
    if (!live() || activity.case_id !== id || report.case_id !== id) return;
    set({ ev: { ...state.ev, activity, report, explanation: { status: "loading" } }, status: "done", runningFor: null });
    explain(id, my, signal, false);  // in parallel: a slow model never delays the numbers
    const [channels, signals] = await Promise.all([api.getChannels(id, { signal }), api.getSignals(id, { signal })]);
    if (!live() || channels.case_id !== id || signals.case_id !== id) return;
    set({ ev: { ...state.ev, channels, signals } });
  }

  // The explanation (App-3): the stored one if any, else generated on the server (language model or template). Tied
  // to its case_id like every other fetch; its failure never affects the verdict, which comes from the record.
  async function explain(id, my, signal, regenerate) {
    const live = () => my === seq && !signal.aborted && state.record?.case_id === id;
    if (!api.getExplanation) return;
    const userFields = state.input?.kind === "upload" ? { upload_filename: state.input.file?.name } : {};
    let e = null;
    try {
      if (!regenerate) {
        try { e = await api.getExplanation(id, { signal }); } catch (err) { if (isAbort(err) || err?.response?.status !== 404) throw err; }
      }
      if (!e) e = await api.postExplanation(id, userFields, { signal });
      if (!live() || e.case_id !== id) return;
      set({ ev: { ...state.ev, explanation: { status: "ready", data: e } } });
    } catch (err) {
      if (!live() || isAbort(err)) return;
      set({ ev: { ...state.ev, explanation: { status: "error", message: errorInfo(err).message } } });
    }
  }

  function regenerate() {
    const id = state.record?.case_id;
    if (!id || !controller) return;
    set({ ev: { ...state.ev, explanation: { status: "loading" } } });
    explain(id, seq, controller.signal, true);
  }

  async function run() {
    const input = state.input;
    if (!input || (input.kind === "stored" && input.canRerun === false)) return;
    abort();
    const my = ++seq;
    controller = new AbortController();
    const { signal } = controller;
    const requestId = newRequestId();
    set({ status: "running", runningFor: input.label, record: null, error: null, ev: { ...EMPTY_EVIDENCE }, requestId });
    const live = () => my === seq && !signal.aborted;
    try {
      const opts = { signal, requestId };
      const res = input.kind === "dataset" ? await api.diagnoseRecording(input.rec_id, opts)
        : input.kind === "upload" ? await api.diagnoseUpload(input.file, { ...opts, uploadId: input.uploadId })
        : await api.rerunCase(input.caseId, opts);
      if (!live() || res.requestId !== requestId || res.inputId !== inputKey(input)) return;  // stale: drop it
      // The app navigates straight to the case dashboard; it loads its own stored evidence.
      // Controllers without a navigation callback retain the report-page flow.
      if (api.onCase?.(res.record.case_id) === true) return;
      set({ record: res.record });
      await follow(res.record, my, signal);
    } catch (e) {
      if (!live() || isAbort(e)) return;
      set({ error: errorInfo(e), status: "error", runningFor: null });
    }
  }

  // Reopen a stored case: an uploaded case becomes a "stored" input that "Run again" re-runs on the server
  // (disabled if the stored array is gone); a dataset case becomes its recording.
  async function open(caseId) {
    abort();
    const my = ++seq;
    controller = new AbortController();
    const { signal } = controller;
    try {
      const [record, meta] = await Promise.all([api.getCase(caseId, { signal }), api.getCaseMeta(caseId, { signal })]);
      if (my !== seq || signal.aborted || record.case_id !== caseId) return;
      const input = record.source === "dataset"
        ? { kind: "dataset", rec_id: record.input.rec_id, label: record.input.rec_id }
        : { kind: "stored", caseId, label: `uploaded recording (case ${caseId.slice(0, 8)})`, canRerun: meta.can_rerun,
            note: meta.rerun_note };
      set({ input, record, status: "running", error: null, ev: { ...EMPTY_EVIDENCE } });
      await follow(record, my, signal);
    } catch (e) {
      if (my !== seq || isAbort(e)) return;
      set({ error: errorInfo(e), status: "error" });
    }
  }

  return { get: () => state, select, run, open, abort, regenerate };
}
