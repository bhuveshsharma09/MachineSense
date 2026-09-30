// App-4.5: audit the pipeline page against REAL saved cases (not fixtures). Point it at a directory of GET
// /api/cases/{id}/trace responses (read-only; the page and this audit never create or change a case):
//   npm run audit:cases -- <directory of trace json files>
// Every case renders in both modes and both reading levels, with the drawer opened on every block; each page is checked
// with the value walker (every value equals its trace field), the null audit (a null reads "not recorded" or a
// field-specific "none" label, never 0, blank, "null", "NaN" or a dash), the same verdict at both levels, no case value in
// architecture mode, and no timing shown where the trace has none.
import { readdirSync, readFileSync } from "node:fs";
import { join } from "node:path";
import { renderToString } from "react-dom/server";
import { PipelineView } from "../src/components/Pipeline.jsx";
import { at, missing, valueOf } from "../src/lib/pipelineCase.js";
import { stripPreviews, walkPage } from "./pipeline_walk.js";

const dir = process.argv[2];
if (!dir) { console.error("usage: node audit_real_cases.js <directory of trace json files>"); process.exit(2); }
const NONE_LABELS = new Set(["none", "not a re-run", "none (uploaded file)"]);
const BAD_TEXT = /^(|null|undefined|NaN|—|-|–)$/;
const SIZE = { w: 1440, h: 900 };
const flat = (bs) => bs.flatMap((b) => [b, ...flat(b.children || [])]);
const text = (h) => h.replace(/<!-- -->/g, "").replace(/<[^>]+>/g, " ").replace(/&amp;/g, "&").replace(/\s+/g, " ");

function nullAudit(html, trace) {
  const issues = [];
  const re = /<span[^>]*?data-p="([^"]+)"[^>]*>([\s\S]*?)<\/span>/g;
  let m;
  while ((m = re.exec(html))) {
    const node = at(trace, m[1]);
    const shown = m[2].replace(/<!-- -->/g, "").replace(/<[^>]+>/g, "").trim();
    if (BAD_TEXT.test(shown)) issues.push(`${m[1]}: shows "${shown}"`);
    const isNull = missing(node) || valueOf(node) === null || valueOf(node) === undefined;
    if (isNull && !(shown.startsWith("not recorded") || NONE_LABELS.has(shown))) issues.push(`${m[1]}: null shown as "${shown}"`);
  }
  if (/\b(null|undefined|NaN)\b/.test(text(html).replace(/not recorded/g, ""))) issues.push("the word null / undefined / NaN in the page text");
  return issues;
}

const files = readdirSync(dir).filter((f) => f.endsWith(".json")).sort();
const report = [];
let pages = 0, values = 0;
const t0 = Date.now();
for (const f of files) {
  const trace = JSON.parse(readFileSync(join(dir, f), "utf8"));
  const issues = [];
  const render = (props) => {
    pages++;
    try { return renderToString(<PipelineView trace={trace} initialSize={SIZE} {...props} />); }  // props may override the size
    catch (e) { issues.push(`render threw (${JSON.stringify(props)}): ${e.message}`); return ""; }
  };
  const verdicts = {};
  for (const level of ["mechanic", "developer"]) {
    for (const sel of [null, ...flat(trace.blocks).map((b) => b.id)]) {
      const h = render({ initialMode: "case", initialLevel: level, initialSelected: sel });
      if (!h) continue;
      const w = walkPage(stripPreviews(h), trace);
      values += w.checked;
      issues.push(...w.errors.map((e) => `${level}/${sel || "page"}: ${e}`));
      issues.push(...nullAudit(h, trace).map((e) => `${level}/${sel || "page"}: ${e}`));
      if (!sel) verdicts[level] = [...h.matchAll(/data-verdict="1"[^>]*>([^<]*)</g)].map((x) => x[1]);
      // a block whose trace timing is null never shows a number as its step time
      if (sel && level === "developer") {
        const b = flat(trace.blocks).find((x) => x.id === sel);
        if (b.timing_ms === null && !/recorded step time<\/span><span class="pl-fv"><span[^>]*data-nr="1"[^>]*>not recorded/.test(h))
          issues.push(`${sel}: timing is null in the trace but the drawer does not say "not recorded"`);
      }
    }
  }
  // the phone-width step list (App-4.5): same checks, every step opened
  for (const level of ["mechanic", "developer"]) for (const sel of flat(trace.blocks).map((b) => b.id)) {
    const h = render({ initialMode: "case", initialLevel: level, initialSelected: sel, initialSize: { w: 390, h: 844 } });
    if (!h) continue;
    if (!h.includes('class="pl-steps"')) issues.push(`phone ${level}/${sel}: not the step list`);
    const w = walkPage(stripPreviews(h), trace);
    values += w.checked;
    issues.push(...w.errors.map((e) => `phone ${level}/${sel}: ${e}`), ...nullAudit(h, trace).map((e) => `phone ${level}/${sel}: ${e}`));
  }
  const label = trace.blocks.find((b) => b.id === "decision").output.display_label.value;
  const vm = new Set(verdicts.mechanic), vd = new Set(verdicts.developer);
  if (!(vm.size === 1 && vd.size === 1 && vm.has(label) && vd.has(label))) issues.push(`verdict differs between levels: ${[...vm]} / ${[...vd]} (trace: ${label})`);
  const arch = render({ initialMode: "architecture", initialLevel: "developer" });
  if (arch.includes("data-p=") || arch.includes(trace.case_id)) issues.push("architecture mode shows a case value");
  const banner = text(render({ initialMode: "case", initialLevel: "mechanic" })).includes("produced by a different bundle than the one now loaded");
  if (banner !== (trace.bundle_matches_record?.value === false)) issues.push(`bundle banner ${banner ? "shown" : "missing"} but bundle_matches_record is ${trace.bundle_matches_record?.value}`);
  report.push({ file: f, case_id: trace.case_id, verdict: label, issues });
}
const bad = report.filter((r) => r.issues.length);
for (const r of bad) console.log(`✗ ${r.case_id} (${r.verdict}): ${r.issues.length} issue(s)\n   ${r.issues.slice(0, 6).join("\n   ")}`);
console.log(`${files.length} real cases, ${pages} pages rendered, ${values} displayed values checked, ${bad.length} case(s) with issues, ${((Date.now() - t0) / 1000).toFixed(0)} s`);
process.exit(bad.length ? 1 : 0);
