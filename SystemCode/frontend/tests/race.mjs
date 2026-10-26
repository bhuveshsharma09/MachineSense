// App-2a.3 race test: delayed mock responses arriving in reversed order must never put a result under the wrong
// recording. Runs the real session controller (src/lib/session.js) with a mock API; no browser.
import { createSession, inputKey } from "../src/lib/session.js";

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
let failures = 0;
const check = (cond, msg) => { if (!cond) { failures++; console.error("✗ " + msg); } };

// Mock API with a per-input delay; honours AbortSignal like axios (CanceledError) and echoes the request / input ids.
function mockApi(delays, log) {
  const wait = (ms, signal) => new Promise((resolve, reject) => {
    const t = setTimeout(resolve, ms);
    signal?.addEventListener("abort", () => { clearTimeout(t); const e = new Error("canceled"); e.name = "CanceledError"; reject(e); });
  });
  const cases = {};
  const diag = async (key, opts) => {
    log.push(`start ${key}`);
    await wait(delays[key] ?? 10, opts.signal);
    const case_id = `case-${key}-${Math.random().toString(36).slice(2, 8)}`;
    cases[case_id] = key;
    log.push(`resolve ${key}`);
    return { record: { case_id, source: "dataset", input: { rec_id: key }, limitations: [] }, requestId: opts.requestId, inputId: key };
  };
  const tied = (kind) => async (id, { signal } = {}) => { await wait(delays[`${kind}:${cases[id]}`] ?? 5, signal); return { case_id: id, kind }; };
  return {
    diagnoseRecording: (rec_id, opts) => diag(rec_id, opts),
    diagnoseUpload: (file, opts) => diag(opts.uploadId, opts).then((r) => ({ ...r, inputId: opts.uploadId })),
    rerunCase: async (caseId, opts) => ({ ...(await diag(`case:${caseId}`, opts)), inputId: `case:${caseId}` }),
    getActivity: tied("activity"), getReport: tied("report"), getChannels: tied("channels"), getSignals: tied("signals"),
    getCase: async (id) => ({ case_id: id, source: "upload", input: { rec_id: null }, limitations: [] }),
    getCaseMeta: async (id) => ({ case_id: id, can_rerun: false, rerun_note: "re-upload the file to run again" }),
    getExplanation: async (id, { signal } = {}) => { await wait(delays[`explanation:${cases[id]}`] ?? 5, signal); const e = new Error("404"); e.response = { status: 404 }; throw e; },
    postExplanation: async (id, uf, { signal } = {}) => { await wait(delays[`explain:${cases[id]}`] ?? 5, signal); return { case_id: id, source: "template", text: {} }; },
  };
}

// Invariant on every state the UI would render: a shown record (and every piece of its evidence) belongs to the
// currently selected input.
function watch(name) {
  const states = [];
  const onChange = (s) => {
    states.push(s);
    if (s.record) {
      const want = inputKey(s.input);
      const got = s.input?.kind === "stored" ? `case:${s.input.caseId}` : s.record.input.rec_id;
      check(s.input?.kind === "stored" || got === want, `${name}: record ${got} shown under selection ${want}`);
      for (const [k, v] of Object.entries(s.ev)) {
        const cid = k === "explanation" ? v?.data?.case_id : v?.case_id;
        check(!v || (k === "explanation" && !v.data) || cid === s.record.case_id, `${name}: ${k} of another case shown`);
      }
    } else {
      check(Object.values(s.ev).every((v) => v === null), `${name}: evidence shown without a record`);
    }
  };
  return { states, onChange };
}

const A = { kind: "dataset", rec_id: "A", label: "A" };
const B = { kind: "dataset", rec_id: "B", label: "B" };

// 1. Run A (slow), switch to B and run it (fast): A's response arrives last and must be dropped.
{
  const log = [], w = watch("reversed order");
  const s = createSession(mockApi({ A: 120, B: 20 }, log), w.onChange);
  s.select(A); s.run(); await sleep(10);
  s.select(B); s.run(); await sleep(250);
  check(s.get().input.rec_id === "B" && s.get().record?.input.rec_id === "B", "reversed order: final result is B under B");
  check(s.get().status === "done" && s.get().ev.report?.case_id === s.get().record.case_id, "reversed order: report tied to B's case");
  check(!log.includes("resolve A"), "reversed order: A's request was aborted, not resolved");
  console.log("✓ reversed order: B shown, A aborted", `(${w.states.length} states checked)`);
}

// 2. Run A, change the selection to B WITHOUT running: B shows "no result yet", A never appears.
{
  const log = [], w = watch("switch without run");
  const s = createSession(mockApi({ A: 80 }, log), w.onChange);
  s.select(A); s.run(); await sleep(10);
  s.select(B);
  check(s.get().record === null && s.get().status === "idle", "switch: old result cleared immediately");
  await sleep(150);
  check(s.get().record === null && s.get().input.rec_id === "B", "switch: A's late response is not shown under B");
  console.log("✓ switch without run: B stays 'no result yet'");
}

// 3. A response that arrives although the request was not aborted (e.g. a proxy ignored the abort) is still dropped
//    because its request id / input id no longer match.
{
  const w = watch("stale echo");
  let release;
  const gate = new Promise((r) => { release = r; });
  const api = { ...mockApi({}, []),
    diagnoseRecording: async (rec_id, opts) => {
      if (rec_id === "A") { await gate; return { record: { case_id: "case-A", source: "dataset", input: { rec_id: "A" } }, requestId: opts.requestId, inputId: "A" }; }
      return { record: { case_id: "case-B", source: "dataset", input: { rec_id: "B" } }, requestId: opts.requestId, inputId: "B" };
    } };
  const s = createSession(api, w.onChange);
  s.select(A); s.run(); await sleep(5);
  s.select(B); s.run(); await sleep(30);
  release(); await sleep(30);
  check(s.get().record?.case_id === "case-B", "stale echo: B still shown after A's late non-aborted response");
  console.log("✓ stale response without abort: dropped by the request / input id check");
}

// 4. Follow-up fetches in reversed order: A's slow /activity and /report never land on B's case.
{
  const log = [], w = watch("follow-ups");
  const s = createSession(mockApi({ A: 5, "activity:A": 150, "report:A": 150, B: 40 }, log), w.onChange);
  s.select(A); s.run(); await sleep(30);   // A's record is shown; its activity / report are still in flight
  s.select(B); s.run(); await sleep(250);
  const st = s.get();
  check(st.record?.input.rec_id === "B" && st.ev.activity?.case_id === st.record.case_id, "follow-ups: only B's activity shown");
  console.log("✓ follow-up fetches: tied to their case_id");
}

// 5. Reopened upload whose stored array is gone: run() is a no-op (the button is disabled).
{
  const w = watch("stored missing");
  const s = createSession(mockApi({}, []), w.onChange);
  await s.open("abcdef123456");
  const st = s.get();
  check(st.input.kind === "stored" && st.input.canRerun === false && /re-upload/.test(st.input.note), "stored: marked as not re-runnable");
  await s.run();
  check(st.record?.case_id === "abcdef123456" && s.get().status !== "running", "stored: run() does nothing without the array");
  console.log("✓ reopened upload without its stored array: not re-runnable");
}

// 6. The app navigates as soon as diagnosis is stored, without rendering the intermediate report page.
{
  const log = [];
  const api = mockApi({ A: 5 }, log);
  api.onCase = (id) => { log.push(`dashboard ${id}`); return true; };
  api.getActivity = () => { log.push("intermediate activity"); throw new Error("intermediate page loaded"); };
  const s = createSession(api);
  s.select(A);
  await s.run();
  check(log.some((x) => x.startsWith("dashboard case-A-")) && !log.includes("intermediate activity") && s.get().record === null,
    "start investigation: dashboard navigation precedes any intermediate report load");
  console.log("✓ start investigation: opens the dashboard directly");
}

if (failures) { console.error(`${failures} race check(s) failed`); process.exit(1); }
console.log("all race checks passed");
