// App-4.3: the case-mode page walker, shared by the smoke test. Every value the page shows is a
// <span data-p="<trace path>" data-fmt="<formatter>">; the walker resolves the path in the FIXTURE trace, formats it
// with the same named helper and requires the displayed text to match exactly. Text outside those spans may contain a
// digit only if it is a string or key of the fixture trace (titles, port names, class names), never a stray number.
import { FMT, at, missing, valueOf } from "../src/lib/pipelineCase.js";

const unescape = (s) => s.replace(/&lt;/g, "<").replace(/&gt;/g, ">").replace(/&quot;/g, '"').replace(/&#x27;/g, "'").replace(/&amp;/g, "&");

export function traceStrings(trace) {
  const out = new Set();
  const walk = (x) => {
    if (typeof x === "string") out.add(x);
    else if (Array.isArray(x)) x.forEach(walk);
    else if (x && typeof x === "object") for (const [k, v] of Object.entries(x)) { out.add(k); out.add(k.replace(/_/g, " ")); walk(v); }
  };
  walk(trace);
  return [...out];
}

export function walkPage(html, fixture) {
  const errors = [];
  let checked = 0;
  // value spans: <span ... data-p="..." data-fmt="...">text</span> (or data-nr for "not recorded")
  const re = /<span[^>]*?data-p="([^"]+)"([^>]*)>([\s\S]*?)<\/span>/g;
  let m;
  while ((m = re.exec(html))) {
    const [, p, attrs, inner] = m;
    const text = unescape(inner.replace(/<!-- -->/g, "").replace(/<[^>]+>/g, ""));
    const node = at(fixture, p);
    checked++;
    if (/data-nr="1"/.test(attrs)) {
      const ok = missing(node) ? text === `not recorded: ${node.reason}` : (valueOf(node) === null || valueOf(node) === undefined) && text === "not recorded";
      if (!ok) errors.push(`${p}: shows "${text}" but the fixture has ${JSON.stringify(node)}`);
      continue;
    }
    const f = (/data-fmt="([^"]+)"/.exec(attrs) || [])[1];
    if (!f || !FMT[f]) { errors.push(`${p}: unknown formatter ${f}`); continue; }
    if (node === undefined) { errors.push(`${p}: not in the fixture`); continue; }
    const want = FMT[f](valueOf(node));
    if (text !== want) errors.push(`${p}: shows "${text}" but the fixture gives "${want}"`);
  }
  // everything else: no digit unless the text is a fixture string or key
  const strings = traceStrings(fixture);
  const rest = html.replace(re, " ").replace(/<(style|script)[\s\S]*?<\/\1>/g, " ");
  for (const raw of rest.split(/<[^>]*>/)) {
    const t = unescape(raw).trim();
    if (!/\d/.test(t)) continue;
    // structure labels may join trace strings (e.g. "<block title> · <port label>"); each piece must be a trace string
    const pieces = t.split(/ · |, |: |^: /).map((x) => x.trim()).filter((x) => /\d/.test(x));
    if (!pieces.every((x) => strings.some((s) => s.includes(x)))) errors.push(`stray number in text: "${t.slice(0, 80)}"`);
  }
  return { checked, errors };
}
