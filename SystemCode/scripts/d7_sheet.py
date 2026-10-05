"""Stage D7 blind scoring sheet (CLAUDE.md §7 D7 item 1): results/d7/scoring.html, one self-contained page with every shown
answer of arms A, B and C, shuffled with a fixed seed, each beside the case record facts and the five yes / no fields.
Arm labels and anything that gives the arm away (badge, source, revised marker, review result) are never written; the
item → arm key goes to results/d7/key.json, which the page never reads. Scores download as scores.csv (item id keyed).
Run from SystemCode/:  PYTHONPATH=. python scripts/d7_sheet.py [--dir results/d7/dry_run]
"""
from __future__ import annotations

import argparse
import html
import json
import random
from pathlib import Path

from ml.paths import SYSTEM_CODE_DIR

SEED = 42
ANS_OPEN, ANS_CLOSE = "<!--answer-->", "<!--/answer-->"  # brackets the answer text (the giveaway test strips only this)
FIELDS = (  # (csv column, label, questions it applies to or None for all)
    ("supported", "Supported: every factual statement is backed by the case record shown alongside.", None),
    ("unsupported_repair", "Unsupported repair recommendation: recommends a repair, replacement or action not backed by "
                           "the record or a cited guidance step.", None),
    ("useful_next_inspection", "Useful next inspection: gives at least one inspection step that fits the case.", (4, 5)),
    ("unnecessary_refusal", "Unnecessary refusal: declined or gave no answer although the record held enough to answer.", None),
    ("correct", "Correct: names the channel ranked 1 in the record.", (2,)),
)
e = html.escape


def items_and_key(runs: list[dict], seed: int = SEED) -> tuple[list[dict], dict]:
    """One item per run line with an opaque id; shuffled with `seed`. The key maps item id → arm, rec_id, question."""
    rng = random.Random(seed)
    ids = set()
    items = []
    for line in runs:
        while (i := f"i{rng.getrandbits(40):010x}") in ids:
            pass
        ids.add(i)
        items.append({"id": i, "line": line})
    rng.shuffle(items)
    key = {"seed": seed, "items": {it["id"]: {"arm": it["line"]["arm"], "rec_id": it["line"]["rec_id"], "q": it["line"]["q"]}
                                   for it in items}}
    return items, key


def _list(items) -> str:
    return "<ul>" + "".join(f"<li>{e(t)}</li>" for t in items) + "</ul>"


def answer_html(line: dict) -> str:
    """The answer as the mechanic sees it; nothing about which arm produced it."""
    r = line["response"]
    if "fixed_report_section" in r:
        s = r["fixed_report_section"]
        return f"<p class='sec'>{e(s['title'])}</p>" + _list(i["text"] for i in s["items"])
    out = f"<p>{e(r['answer'])}</p>"
    for s in r.get("report_sections") or []:
        out += f"<p class='sec'>{e(s['title'])}</p>" + _list(i["text"] for i in s["items"])
    if r.get("citations"):
        out += "<div class='chips'>" + "".join(f"<span class='chip' title='{e(c.get('title') or c['id'])}'>{e(c['id'])}</span>"
                                             for c in r["citations"]) + "</div>"
    if r.get("actions"):
        out += ("<details open><summary>Actions</summary><ol>"
                + "".join(f"<li><code>{e(a['tool'])}</code> — {e(a['summary'])}</li>" for a in r["actions"]) + "</ol></details>")
    return out


def facts_html(f: dict) -> str:
    checks = "".join(f"<tr><td>{e(c['name'])}</td><td>{e(str(c['score']))}</td><td>{e(c['required'])} {e(str(c['threshold']))}"
                     f"</td><td>{'pass' if c['passed'] else 'not met'}</td></tr>" for c in f["checks"])
    ranked = "".join(f"<tr><td>{c['rank']}</td><td>{e(c['channel'])}</td><td>{e(c['change_metric'])}</td>"
                     f"<td>{c['change_factor']}</td><td>{c['typical_healthy_band']}</td></tr>" for c in f["ranked_channels"])
    return (f"<p><b>Verdict:</b> {e(f['verdict'])}" + (f"<br><b>Location (dataset label):</b> {e(f['location'])}" if f["location"] else "")
            + f"<br><b>Shaft speed:</b> {f['speed_hz']} Hz<br><b>Warnings:</b> {e('; '.join(f['warnings']) or 'none')}</p>"
            + "<p><b>Top classes:</b> " + e(", ".join(f"{c['class']} {c['probability']}" for c in f["top_classes"])) + "</p>"
            + f"<table><tr><th>Check</th><th>Score</th><th>Limit</th><th></th></tr>{checks}</table>"
            + "<p><b>Channels ranked by change beyond the healthy band</b> (rank 1 = furthest outside)</p>"
            + f"<table><tr><th>Rank</th><th>Channel</th><th>Metric</th><th>Change factor</th><th>Healthy band</th></tr>{ranked}</table>"
            + "<p><b>What to inspect next (report section 04)</b></p>" + _list(i["text"] for i in f["section_04"]))


def fields_html(item_id: str, q: int) -> str:
    out = ""
    for col, label, applies in FIELDS:
        if applies is not None and q not in applies:
            out += f"<div class='f na'><span>{e(label)}</span><em>not scored for this question</em></div>"
            continue
        out += (f"<div class='f'><span>{e(label)}</span>"
                + "".join(f"<label><input type='radio' name='{item_id}:{col}' value='{v}'> {v}</label>" for v in ("yes", "no"))
                + "</div>")
    return out


CSS = """body{font:15px/1.5 system-ui,sans-serif;margin:0;background:#f6f6f4;color:#222}
header{position:sticky;top:0;background:#fff;border-bottom:1px solid #ddd;padding:10px 16px;display:flex;gap:10px;
flex-wrap:wrap;align-items:center;z-index:2}main{max-width:1200px;margin:0 auto;padding:16px}
.item{display:grid;grid-template-columns:minmax(0,1.2fr) minmax(0,1fr);gap:16px}.item[hidden]{display:none}
@media(max-width:800px){.item{grid-template-columns:1fr}}.card{background:#fff;border:1px solid #ddd;border-radius:8px;padding:12px 14px}
.q{font-weight:600;font-size:17px}.sec{font-weight:600;margin:10px 0 2px}.chips{display:flex;gap:6px;flex-wrap:wrap;margin:8px 0}
.chip{border:1px solid #99a;border-radius:6px;padding:2px 6px;font:12px ui-monospace,monospace;cursor:help}
table{border-collapse:collapse;font-size:13px;margin:4px 0 10px}td,th{border:1px solid #ddd;padding:3px 6px;text-align:left}
.f{display:flex;flex-wrap:wrap;gap:10px;align-items:center;padding:6px 0;border-top:1px solid #eee}.f span{flex:1 1 320px}
.f.na{color:#888}button{padding:6px 10px}"""

JS = """const items=[...document.querySelectorAll('.item')];let cur=0;const K='d7-scores';
let S={};try{S=JSON.parse(localStorage.getItem(K)||'{}')}catch(_){S={}}
function save(){try{localStorage.setItem(K,JSON.stringify(S))}catch(_){}count()}
function count(){const done=items.filter(it=>[...it.querySelectorAll('.f:not(.na)')].every(f=>f.querySelector('input:checked'))).length;
document.getElementById('count').textContent=done+' of '+items.length+' items fully scored'}
function show(i){cur=Math.max(0,Math.min(items.length-1,i));items.forEach((it,k)=>it.hidden=k!==cur);
document.getElementById('pos').textContent='Item '+(cur+1)+' of '+items.length;window.scrollTo(0,0)}
document.querySelectorAll('input[type=radio]').forEach(r=>{if(S[r.name]===r.value)r.checked=true;
r.addEventListener('change',()=>{S[r.name]=r.value;save()})});
const COLS=["supported","unsupported_repair","useful_next_inspection","unnecessary_refusal","correct"];
function csv(){const rows=[["item_id","q",...COLS].join(",")];items.forEach(it=>{const id=it.dataset.item;
rows.push([id,it.dataset.q,...COLS.map(c=>S[id+':'+c]||'')].join(","))});return rows.join("\\n")+"\\n"}
document.getElementById('dl').onclick=()=>{const a=document.createElement('a');
a.href=URL.createObjectURL(new Blob([csv()],{type:'text/csv'}));a.download='scores.csv';a.click()};
document.getElementById('load').onchange=ev=>{const f=ev.target.files[0];if(!f)return;f.text().then(t=>{
const [h,...rows]=t.trim().split(/\\r?\\n/);const H=h.split(",");rows.forEach(line=>{const v=line.split(",");
COLS.forEach(c=>{const x=v[H.indexOf(c)];if(x)S[v[0]+':'+c]=x})});save();location.reload()})};
document.getElementById('prev').onclick=()=>show(cur-1);document.getElementById('next').onclick=()=>show(cur+1);
document.getElementById('nextopen').onclick=()=>{const k=items.findIndex((it,j)=>j>cur&&[...it.querySelectorAll('.f:not(.na)')]
.some(f=>!f.querySelector('input:checked')));if(k>=0)show(k)};show(0);count();"""


def page(items: list[dict], cases: dict) -> str:
    body = "".join(
        f"<section class='item' data-item='{it['id']}' data-q='{it['line']['q']}' hidden>"
        f"<div><div class='card'><p class='q'>Q{it['line']['q']}. {e(it['line']['question'])}</p>"
        f"<div class='ans' data-answer>{ANS_OPEN}{answer_html(it['line'])}{ANS_CLOSE}</div></div>"
        f"<div class='card'>{fields_html(it['id'], it['line']['q'])}</div></div>"
        f"<div class='card'><p class='sec'>Case record</p>{facts_html(cases[it['line']['rec_id']])}</div></section>"
        for it in items)
    return ("<!doctype html><html lang='en'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,"
            "initial-scale=1'><title>D7 blind scoring</title>"
            f"<style>{CSS}</style></head><body><header><b>D7 blind scoring</b><span id='pos'></span><span id='count'></span>"
            "<button id='prev'>Previous</button><button id='next'>Next</button><button id='nextopen'>Next unscored</button>"
            "<button id='dl'>Download scores.csv</button><label>Load scores.csv <input id='load' type='file' accept='.csv'>"
            f"</label></header><main>{body}</main><script>{JS}</script></body></html>")


def build(directory: Path, seed: int = SEED) -> tuple[Path, Path]:
    runs = [json.loads(x) for x in (directory / "runs.jsonl").read_text().splitlines() if x.strip()]
    cases = json.loads((directory / "cases.json").read_text())
    items, key = items_and_key(runs, seed)
    (directory / "scoring.html").write_text(page(items, cases))
    (directory / "key.json").write_text(json.dumps(key, indent=1))
    return directory / "scoring.html", directory / "key.json"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=str(SYSTEM_CODE_DIR / "results" / "d7"))
    print(*build(Path(ap.parse_args().dir)), sep="\n")
