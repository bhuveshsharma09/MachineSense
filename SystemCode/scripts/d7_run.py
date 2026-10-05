"""Stage D7 runner (CLAUDE.md §7 D7 item 1): asks the 5 pilot questions about each committed case under arm A (the fixed
report), then B (agent without review) and C (agent with review), through the app itself (FastAPI TestClient, a dedicated
case store), and writes one JSON line per answer to results/d7/runs.jsonl plus the case facts the scorer needs to
results/d7/cases.json. Each arm runs once; a crashed arm is re-run only with --rerun-arm and a reason (logged).
Run from SystemCode/:
    PYTHONPATH=. python scripts/d7_run.py --dry-run          # 1 case, writes results/d7/dry_run/ only
    PYTHONPATH=. caffeinate -i python scripts/d7_run.py      # the pilot (≈ 90 min; ask Bhuvesh first)
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from ml.paths import SYSTEM_CODE_DIR

FROZEN_PROMPT_VERSION = "8fb7865754455d65"  # App-7 addendum 12
OUT = SYSTEM_CODE_DIR / "results" / "d7"
ARMS = ("A", "B", "C")
QUESTIONS = {1: "Why was this recording flagged?", 2: "Which sensor shows the largest change?",
             3: "How sure is the system, and what could make it wrong?", 4: "What should I inspect next?",
             5: "Should I replace the bearing now?"}
HEALTHY_Q1 = "Why is this recording considered healthy?"
SECTION_FOR = {1: "01", 2: "02", 3: "03", 4: "04", 5: "04"}  # arm A: the fixed report's answer to each question


class Refused(RuntimeError):
    pass


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=SYSTEM_CODE_DIR, capture_output=True, text=True, check=True).stdout


def question(q: int, decision: str) -> str:
    return HEALTHY_Q1 if q == 1 and decision == "healthy" else QUESTIONS[q]


def preconditions(prompt_version: str, status: str) -> None:
    """Refuse a dirty working tree (tracked or untracked changes) or an assistant that is not the frozen D7 version."""
    if status.strip():
        raise Refused("the working tree is not clean; commit first so every run line names the code it ran on:\n" + status)
    if prompt_version != FROZEN_PROMPT_VERSION:
        raise Refused(f"AGENT_PROMPT_VERSION is {prompt_version}, not the frozen D7 version {FROZEN_PROMPT_VERSION}")


def model_reachable() -> None:
    import httpx

    from backend.settings import llm_base_url, llm_enabled, llm_model
    if not llm_enabled():
        raise Refused("the language model is disabled (MACHINESENSE_LLM_ENABLED); arms B and C need it")
    try:
        ids = [m["id"] for m in httpx.get(f"{llm_base_url()}/models", timeout=10).json()["data"]]
    except Exception as e:  # noqa: BLE001
        raise Refused(f"the language model endpoint is not reachable ({e})") from e
    if llm_model() not in ids:
        raise Refused(f"model {llm_model()} is not served at {llm_base_url()} ({ids})")


def case_facts(engine, case_id: str, rec: dict, stratum: str, report: dict) -> dict:
    """What the scorer sees beside every answer about this case (from the tools, rounded as the report shows them), and the
    case's section 04 entry ids for the citation-validity measure."""
    from backend.agent.tools import Tools
    t = Tools(engine)
    dg, sig, ev, gd = (t.diagnose(case_id), t.check_signal(case_id), t.get_evidence(case_id), t.lookup_guidance(case_id))
    ranked = [{"channel": c["channel"], "rank": c["rank"]["value"], "change_metric": c["change_metric"],
               "change_factor": c["change_factor"]["value"], "typical_healthy_band": c["typical_healthy_band"]["value"]}
              for c in ev.get("channels", []) if isinstance(c.get("rank"), dict)]
    return {"rec_id": rec["input"]["rec_id"], "case_id": case_id, "stratum": stratum,
            "verdict": rec["diagnosis"]["display_label"], "decision": rec["diagnosis"]["decision"],
            "location": dg.get("location", {}).get("value"),
            "checks": [{"name": c["name"], "score": c["score"]["value"], "required": c["required"],
                        "threshold": c["threshold"]["value"], "passed": c["passed"]["value"], "status": c["status"]}
                       for c in dg["checks"]],
            "top_classes": [{"class": c["class"], "probability": c["probability"]["value"]} for c in dg["top_classes"]],
            "speed_hz": sig["shaft_speed_hz"]["value"], "warnings": [w["value"] for w in sig["warnings"]],
            "ranked_channels": ranked, "rank_1_channel": ranked[0]["channel"] if ranked else None,
            "section_04": [{"tag": i["tag"], "text": i["text"]} for i in gd["items"]],
            "guidance_ids": sorted({i["cite"] for i in gd["items"] if i.get("cite")}),
            "report_sections": {s["n"]: s for s in report["sections"]}}


def run(arms=ARMS, dry_run: bool = False, rerun: str | None = None, reason: str | None = None) -> Path:
    from fastapi.testclient import TestClient

    from backend.agent.prompt import AGENT_PROMPT_VERSION
    from backend.app import create_app
    preconditions(AGENT_PROMPT_VERSION, git("status", "--porcelain"))
    out = OUT / "dry_run" if dry_run else OUT
    out.mkdir(parents=True, exist_ok=True)
    runs_path = out / "runs.jsonl"
    done = {json.loads(x)["arm"] for x in runs_path.read_text().splitlines()} if runs_path.exists() and not dry_run else set()
    if rerun:
        if not reason:
            raise Refused("--rerun-arm needs --reason (the crash that was fixed)")
        arms = (rerun,)
        lines = [x for x in runs_path.read_text().splitlines() if json.loads(x)["arm"] != rerun]
        runs_path.write_text("".join(x + "\n" for x in lines))
        with open(out / "reruns.jsonl", "a") as fh:
            fh.write(json.dumps({"arm": rerun, "reason": reason, "at": datetime.now(timezone.utc).isoformat()}) + "\n")
    elif done & set(arms):
        raise Refused(f"arm(s) {sorted(done & set(arms))} already ran; each arm runs once (use --rerun-arm after a crash)")
    if dry_run and runs_path.exists():
        runs_path.unlink()
    if {"B", "C"} & set(arms):
        model_reachable()
    cases = pd.read_csv(OUT / "cases.csv")
    if dry_run:
        cases = cases.head(1)
    revision = git("rev-parse", "HEAD").strip()
    started = datetime.now(timezone.utc).isoformat(timespec="seconds")
    store = (Path(tempfile.mkdtemp()) if dry_run else SYSTEM_CODE_DIR / "backend" / "data") / "d7_cases.db"
    with TestClient(create_app(cases_path=store)) as client:
        eng = client.app.state.engine
        facts = {}
        for row in cases.itertuples():
            rec = client.post("/api/diagnose", json={"rec_id": row.rec_id}).json()
            report = client.get(f"/api/cases/{rec['case_id']}/report").json()
            facts[row.rec_id] = case_facts(eng, rec["case_id"], rec, row.stratum, report)
        (out / "cases.json").write_text(json.dumps(facts, indent=1, ensure_ascii=False))
        n_total, n, t0 = len(cases) * 5 * len([a for a in arms if a != "A"]), 0, time.perf_counter()
        with open(runs_path, "a") as fh:
            for arm in arms:
                if arm != "A":
                    os.environ["MACHINESENSE_AGENT_REVIEW"] = "off" if arm == "B" else "on"
                for f in facts.values():
                    for q in QUESTIONS:
                        line = {"arm": arm, "rec_id": f["rec_id"], "case_id": f["case_id"], "stratum": f["stratum"],
                                "q": q, "question": question(q, f["decision"]), "prompt_version": AGENT_PROMPT_VERSION,
                                "code_revision": revision, "run_started": started}
                        if arm == "A":
                            line["response"] = {"fixed_report_section": f["report_sections"][SECTION_FOR[q]]}
                        else:
                            r = client.post(f"/api/cases/{f['case_id']}/ask", json={"question": line["question"]})
                            if r.status_code != 200:
                                raise RuntimeError(f"arm {arm} {f['rec_id']} Q{q}: HTTP {r.status_code} {r.text[:300]}")
                            line["response"] = r.json()
                            n += 1
                            el = time.perf_counter() - t0
                            print(f"{arm} {f['rec_id']} Q{q}: {line['response']['review']['status']} "
                                  f"{line['response']['latency_ms'] / 1000:.0f} s  ({n}/{n_total}, "
                                  f"~{el / n * (n_total - n) / 60:.0f} min left)", flush=True)
                        fh.write(json.dumps(line, ensure_ascii=False) + "\n")
                        fh.flush()
    return runs_path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="1 case into results/d7/dry_run/ (not part of the pilot)")
    ap.add_argument("--rerun-arm", choices=ARMS, help="after a crash only: drop that arm's lines and run it again")
    ap.add_argument("--reason", help="the crash that was fixed (required with --rerun-arm)")
    a = ap.parse_args()
    try:
        print(run(dry_run=a.dry_run, rerun=a.rerun_arm, reason=a.reason))
    except Refused as e:
        print(f"REFUSED: {e}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
