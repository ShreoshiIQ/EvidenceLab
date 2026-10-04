"""python -m evaluation.report --split public|hidden  ->  reports/scorecard_<split>.{md,json}, reports/per_question_<split>.csv"""
import argparse, json
from pathlib import Path
from evaluation.metrics import load_runs, scorecard, to_markdown

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--split", default="public", choices=["public", "hidden"]); ap.add_argument("--runs", default="runs")
    a = ap.parse_args(); df = load_runs(a.split, a.runs)
    if df.empty: raise SystemExit(f"No runs/{a.split}_*.jsonl found")
    sc = scorecard(df); out = Path("reports"); out.mkdir(exist_ok=True)
    (out / f"scorecard_{a.split}.json").write_text(json.dumps(sc, indent=2))
    md = to_markdown(sc, f"Scorecard: {a.split} questions"); (out / f"scorecard_{a.split}.md").write_text(md)
    cols = [c for c in ("qid", "qtype", "pipeline", "answer", "verdict", "total_tokens", "context_tokens", "input_tokens", "output_tokens", "llm_calls",
                        "latency", "retrieval_steps", "reasoning_steps", "n_chunks", "n_citations", "stop_reason", "strategy_changes", "doc_recall") if c in df]
    df[cols].sort_values(["qid", "pipeline"]).to_csv(out / f"per_question_{a.split}.csv", index=False)
    print(md); print(f"written to {out}/")

if __name__ == "__main__": main()
