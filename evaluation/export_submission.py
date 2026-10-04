"""python -m evaluation.export_submission [--hidden data/eval_hidden.jsonl]

Writes, for the hidden questions:
  submission/hidden_answers.jsonl            <- SAME FORMAT as eval_public.jsonl: {"qid","question","qtype","answer":[...]}  (agentic = primary system)
  submission/hidden_answers_<pipeline>.jsonl <- same format for rag / graphrag / agentic
  submission/hidden_results.json(.jsonl)     <- raw outputs: answers, tokens, latency, steps and the full trace per pipeline
The organizers' exact submission schema is not known to this repo: adjust `answers_row()` / `shape()` if they publish one."""
import argparse, json
from pathlib import Path

PIPES = ("rag", "graphrag", "agentic")
NOT_FOUND = ["Not found in the corpus"]
KEEP = ("answer", "answer_list", "citations", "tokens", "latency_s", "latency_adj_s", "retrieval_steps", "reasoning_steps", "retrieval_methods",
        "n_chunks", "n_citations", "citation_valid_rate", "answer_supported_by_citations", "models_used", "investigation_path", "trace", "evidence", "mcp_tool_calls", "error")
AGENTIC_ONLY = ("rounds", "stop_reason", "stop_detail", "strategy_changes", "agents_used", "tools_used")

def shape(rec):
    o = {k: rec[k] for k in KEEP if k in rec}
    if rec.get("pipeline") == "agentic": o.update({k: rec[k] for k in AGENTIC_ONLY if k in rec})
    return o

def answers_row(q, rec):
    """Same fields as the public eval file, with 'answer' = list of strings."""
    items = (rec or {}).get("answer_list") or ([rec["answer"]] if (rec or {}).get("answer") else [])
    return {"qid": q["qid"], "question": q["question"], "qtype": q.get("qtype"), "answer": items or NOT_FOUND}

def export(runs_dir="runs", out_dir="submission", hidden_file=None):
    runs_dir, out = Path(runs_dir), Path(out_dir); out.mkdir(exist_ok=True)
    rows = [json.loads(l) for p in sorted(runs_dir.glob("hidden_*.jsonl")) for l in open(p) if l.strip()]
    if not rows: raise SystemExit(f"No {runs_dir}/hidden_*.jsonl found. Run evaluation.run_eval --split hidden first.")
    by = {}
    for r in rows:
        q = by.setdefault(r["qid"], {"qid": r["qid"], "question": r["question"], "qtype": r.get("qtype"), "pipelines": {}})
        q["pipelines"][r["pipeline"]] = r
    order = [json.loads(l)["qid"] for l in open(hidden_file) if l.strip()] if hidden_file and Path(hidden_file).exists() else sorted(by)
    qs = [by[i] for i in order if i in by]
    for name in PIPES:
        with open(out / (f"hidden_answers_{name}.jsonl"), "w") as f:
            for q in qs: f.write(json.dumps(answers_row(q, q["pipelines"].get(name)), ensure_ascii=False) + "\n")
    with open(out / "hidden_answers.jsonl", "w") as f:       # primary submission = the agentic system
        for q in qs: f.write(json.dumps(answers_row(q, q["pipelines"].get("agentic")), ensure_ascii=False) + "\n")
    man = runs_dir / "hidden_manifest.json"
    detail = [{"qid": q["qid"], "question": q["question"], "qtype": q["qtype"], "pipelines": {k: shape(v) for k, v in q["pipelines"].items()}} for q in qs]
    (out / "hidden_results.json").write_text(json.dumps({"n_questions": len(qs), "manifest": json.loads(man.read_text()) if man.exists() else None, "results": detail}, ensure_ascii=False, indent=1))
    with open(out / "hidden_results.jsonl", "w") as f:
        for d in detail: f.write(json.dumps(d, ensure_ascii=False) + "\n")
    missing = [q["qid"] for q in qs if set(q["pipelines"]) != set(PIPES)]
    unanswered = [q["qid"] for q in qs if not (q["pipelines"].get("agentic") or {}).get("answer")]
    print(f"{len(qs)} questions exported to {out}/ | missing a pipeline: {missing or 'none'} | agentic unanswered: {unanswered or 'none'}")
    return qs

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--hidden", default="data/eval_hidden.jsonl"); ap.add_argument("--runs", default="runs"); ap.add_argument("--out", default="submission")
    a = ap.parse_args(); export(a.runs, a.out, a.hidden)

if __name__ == "__main__": main()
