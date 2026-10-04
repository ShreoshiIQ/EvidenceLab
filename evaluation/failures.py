"""python -m evaluation.failures   -> writes failures.txt (UTF-8): per-question failures for every pipeline + failure counts by qtype,
and for failed agentic runs the stop reason and the orchestrator/tool trace. Read-only; spends no API quota."""
import json, sys
from collections import Counter, defaultdict
from pathlib import Path

def load(path):
    d = {}
    if Path(path).exists():
        for l in open(path, encoding="utf-8"):
            if l.strip():
                r = json.loads(l); d[r["qid"]] = r        # later records replace earlier (retries)
    return d

def short(x, n=240): return json.dumps(x, ensure_ascii=False)[:n]

def main():
    gold = {}
    for l in open("data/eval_public.jsonl", encoding="utf-8"):
        if l.strip(): d = json.loads(l); gold[d["qid"]] = d
    P = {p: load(f"runs/public_{p}.jsonl") for p in ("rag", "graphrag", "agentic")}
    ok = lambda r: r.get("verdict") == "PASS"
    out, tally, total = [], defaultdict(Counter), defaultdict(Counter)
    for q, g in gold.items():
        for p, runs in P.items():
            if q in runs:
                total[g.get("qtype")][p] += 1
                if not ok(runs[q]): tally[g.get("qtype")][p] += 1
    out.append("FAILURES BY QUESTION TYPE  (failed/answered)")
    for qt in sorted(total, key=str):
        out.append(f"  {qt}: " + "  ".join(f"{p} {tally[qt][p]}/{total[qt][p]}" for p in P if total[qt][p]))
    out.append("")
    only = set(sys.argv[1:])   # optional: python -m evaluation.failures pub-006 pub-007
    for q, g in gold.items():
        if only and q not in only: continue
        bad = [p for p, runs in P.items() if q in runs and not ok(runs[q])]
        if not bad: continue
        out += [f"== {q} [{g.get('qtype')}] FAILED: {bad}", f"Q: {g['question']}", f"GOLD: {g['answer']}"]
        for p in bad:
            r = P[p][q]; m = r.get("meta") or {}
            out.append(f"  {p} -> {r.get('answer') or r.get('answer_list')!r}" + (f"  [ERROR {r['error'][:120]}]" if r.get("error") else ""))
            if p == "agentic":
                out.append(f"     stop={r.get('stop_reason') or m.get('stop_reason')}  detail={r.get('stop_detail') or m.get('stop_detail')}  tokens={r['tokens'].get('total')}")
                for e in r.get("evidence") or []:
                    if e.get("kind") == "aggregate": out.append("     AGGREGATE EVIDENCE: " + (e.get("content") or "")[:1500].replace("\n", "\n       "))
                for e in r.get("evidence") or []:
                    if e.get("kind") == "vertex" and (e.get("content") or "").startswith("Events ("): out.append("     EVENTS: " + (e["content"] or "")[:420].replace("\n", " / "))
                for s in r.get("trace") or []:
                    if s.get("agent") == "answer_generation": continue
                    out.append(f"     - {s.get('agent')}/{s.get('tool')} args={short(s.get('args') if s.get('args') is not None else s.get('plan'))} new={s.get('new_evidence')} {s.get('error', '')}")
        out.append("")
    Path("failures.txt").write_text("\n".join(out), encoding="utf-8")
    print("wrote failures.txt  (", sum(1 for _ in out), "lines )")

if __name__ == "__main__": main()
