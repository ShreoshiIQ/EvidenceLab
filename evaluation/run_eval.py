"""python -m evaluation.run_eval --file data/eval_public.jsonl --split public
   python -m evaluation.run_eval --file data/eval_hidden.jsonl --split hidden   (run ONCE, after freezing)
Resumable: finished (qid, pipeline) pairs are skipped."""
import argparse, hashlib, json
from pathlib import Path
import config as C
from evaluation.judge import judge
from evaluation.completeness import completeness, semantic_sim
from gateway.llm import QuotaExhausted
from pipelines import PIPES, run
from tgdb import get_tg

def manifest(out):
    h = hashlib.sha256()
    for p in sorted(list(Path(".").glob("*.py")) + list(Path(".").glob("*/*.py")) + list(Path("tgdb").glob("*.gsql"))):
        h.update(p.read_bytes())
    m = {"code_sha256": h.hexdigest(), "model": C.GEN_MODEL, "rag_topk": C.RAG_TOPK, "agent_max_steps": C.AGENT_MAX_STEPS,
         "agent_max_tokens": C.AGENT_MAX_TOKENS, "tg_backend": C.TG_BACKEND}
    (out / "hidden_manifest.json").write_text(json.dumps(m, indent=2))

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--file", required=True)
    ap.add_argument("--split", default="public", choices=["public", "hidden"])
    ap.add_argument("--pipelines", nargs="+", default=list(PIPES)); ap.add_argument("--limit", type=int); ap.add_argument("--qtypes", nargs="+", help="only these question types, e.g. aggregation multi_hop superlative")
    ap.add_argument("--no-judge", action="store_true"); ap.add_argument("--no-semantic", action="store_true"); ap.add_argument("--out", default="runs")
    a = ap.parse_args(); out = Path(a.out); out.mkdir(exist_ok=True)
    qs = [json.loads(l) for l in open(a.file, encoding="utf-8") if l.strip()]
    if a.qtypes: qs = [q for q in qs if q.get("qtype") in a.qtypes]
    qs = qs[:a.limit]
    if a.split == "hidden": manifest(out)
    tg = get_tg(); tg.ensure_up()
    try: _loop(a, out, qs, tg)
    except QuotaExhausted as e:
        print(f"\nSTOPPED (progress is saved; finished questions are not repeated): {e}\nRerun the same command to continue.", flush=True)

def _loop(a, out, qs, tg):
    for name in a.pipelines:
        path = out / f"{a.split}_{name}.jsonl"
        done = {r["qid"] for r in (json.loads(l) for l in open(path, encoding="utf-8")) if not r.get("error")} if path.exists() else set()   # failed runs are retried
        with open(path, "a", encoding="utf-8") as f:
            for q in qs:
                if q["qid"] in done: continue
                try: rec = run(name, q["question"], tg)
                except QuotaExhausted: raise   # stop cleanly; do not write a failed row for every remaining question
                except Exception as e: rec = {"pipeline": name, "question": q["question"], "answer": "", "error": str(e)[:300],
                                              "tokens": {"context": 0, "input": 0, "output": 0, "total": 0}, "trace": [], "retrieved_doc_ids": []}
                rec.update(qid=q["qid"], qtype=q.get("qtype"))
                gold = q.get("answer")
                if gold is not None:
                    rec["gold"] = gold; rec["guess_baseline"] = q.get("guess_baseline")
                    gd = set(q.get("gold_doc_ids", []))
                    rec["doc_recall"] = round(len(gd & set(rec["retrieved_doc_ids"])) / len(gd), 3) if gd else None
                    if not a.no_judge and rec.get("answer"): rec["verdict"], rec["judge_method"] = judge(q["question"], gold, rec.get("answer_list") or rec["answer"])
                    else: rec["verdict"] = "FAIL"
                    pred = rec.get("answer_list") or rec.get("answer") or ""
                    rec["completeness"] = completeness(gold, pred)
                    if not a.no_semantic: rec["semantic_sim"] = semantic_sim(gold, pred)
                f.write(json.dumps(rec, ensure_ascii=False) + "\n"); f.flush()
                print(name, q["qid"], rec.get("verdict", "-"), rec["tokens"]["total"], "tok", rec.get("stop_reason", ""))
                if rec.get("error"): print("   ERROR:", rec["error"][:300], flush=True)

if __name__ == "__main__": main()
