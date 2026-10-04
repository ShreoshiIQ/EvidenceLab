"""Single source of truth for every number reported (scorecard.md/json, dashboard, API /metrics)."""
import json
from collections import Counter
from pathlib import Path
import pandas as pd

ORDER = ["rag", "graphrag", "agentic"]

def load_runs(split="public", d="runs"):
    rows = []
    for p in sorted(Path(d).glob(f"{split}_*.jsonl")): rows += [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]
    df = pd.DataFrame(rows)
    if df.empty: return df
    df = df.drop_duplicates(subset=["qid", "pipeline"], keep="last").reset_index(drop=True)   # a retried question replaces its failed record
    tk = lambda k: df["tokens"].apply(lambda t: (t or {}).get(k, 0))
    for k in ("context", "input", "output", "total"): df[f"{k}_tokens"] = tk(k)
    df["llm_calls"] = tk("llm_calls")
    for col in ("retrieval_steps", "reasoning_steps", "n_chunks", "n_citations"):
        df[col] = pd.to_numeric(df.get(col), errors="coerce").fillna(0)
    lat = df["latency_adj_s"] if "latency_adj_s" in df else pd.Series([None] * len(df))
    df["latency"] = pd.to_numeric(lat, errors="coerce").fillna(pd.to_numeric(df["latency_s"], errors="coerce"))
    if "verdict" in df: df["correct"] = df["verdict"].map({"PASS": 1.0, "FAIL": 0.0})
    return df

def _r(x, n=2): return None if pd.isna(x) else round(float(x), n)

def _models(g):
    s = set()
    for ms in g.get("models_used", []):
        if isinstance(ms, list): s |= set(ms)
    return s

def _method_share(g):
    c = Counter(m for ms in g.get("retrieval_methods", []) if isinstance(ms, list) for m in set(ms))
    return {k: round(v / len(g), 3) for k, v in sorted(c.items(), key=lambda kv: -kv[1])}

def _rate(g, col):
    if col not in g: return None
    s = g[col].dropna()
    return _r(s.astype(float).mean(), 3) if len(s) else None

def scorecard(df):
    if df.empty: return {}
    out = {"n_questions": int(df["qid"].nunique()), "pipelines": {}}; models = set()
    for name in [p for p in ORDER if p in set(df.pipeline)]:
        g = df[df.pipeline == name]; m = _models(g); models |= m
        p = {"n": int(len(g)), "errors": int(g["error"].notna().sum()) if "error" in g else 0, "models": sorted(m),
             "tokens_per_question": {k: _r(g[f"{k}_tokens"].mean(), 1) for k in ("context", "input", "output", "total")},
             "median_total_tokens": _r(g["total_tokens"].median(), 1),
             "latency_s": {"avg": _r(g["latency"].mean(), 2), "median": _r(g["latency"].median(), 2), "p95": _r(g["latency"].quantile(0.95), 2)},
             "avg_retrieval_steps": _r(g["retrieval_steps"].mean()), "avg_reasoning_steps": _r(g["reasoning_steps"].mean()),
             "avg_llm_calls": _r(g["llm_calls"].mean()), "retrieval_methods_share": _method_share(g),
             "avg_chunks": _r(g["n_chunks"].mean()), "avg_citations": _r(g["n_citations"].mean()),
             "citation_valid_rate": _rate(g, "citation_valid_rate"), "answer_supported_rate": _rate(g, "answer_supported_by_citations")}
        if "correct" in g and g["correct"].notna().any():
            p["accuracy"] = _r(g["correct"].mean(), 3)
            p["completeness"] = _rate(g, "completeness"); p["semantic_sim"] = _rate(g, "semantic_sim")
            p["accuracy_per_1k_tokens"] = _r(p["accuracy"] / max(p["tokens_per_question"]["total"], 1) * 1000, 3)
            p["n_pass"] = int(g["correct"].sum())
            if "guess_baseline" in g and g["guess_baseline"].notna().any():
                p["avg_guess_baseline"] = _r(pd.to_numeric(g["guess_baseline"], errors="coerce").mean(), 3); p["lift_over_guess"] = _r(p["accuracy"] - p["avg_guess_baseline"], 3)
            if "doc_recall" in g: p["avg_doc_recall"] = _r(pd.to_numeric(g["doc_recall"], errors="coerce").mean(), 3)
        out["pipelines"][name] = p
    out["models"] = sorted(models); out["same_model_across_pipelines"] = len(models) == 1
    if "correct" in df and "qtype" in df:
        bt = {}
        for (qt, pl), g in df.groupby(["qtype", "pipeline"]):
            bt.setdefault(str(qt), {})[pl] = {"n": int(len(g)), "accuracy": _r(g["correct"].mean(), 3),
                                             "completeness": _rate(g, "completeness"), "avg_tokens": _r(g["total_tokens"].mean(), 1), "avg_latency_s": _r(g["latency"].mean(), 2)}
        out["by_qtype"] = bt
    a = df[df.pipeline == "agentic"]
    if len(a):
        out["agentic"] = {
            "avg_tokens_per_question": _r(a["total_tokens"].mean(), 1), "avg_time_per_question_s": _r(a["latency"].mean(), 2),
            "avg_retrieval_steps": _r(a["retrieval_steps"].mean()), "avg_reasoning_steps": _r(a["reasoning_steps"].mean()),
            "avg_total_steps": _r((a["retrieval_steps"] + a["reasoning_steps"]).mean()),
            "avg_rounds": _r(pd.to_numeric(a.get("rounds"), errors="coerce").mean()) if "rounds" in a else None,
            "strategy_change_rate": _r((pd.to_numeric(a.get("strategy_changes"), errors="coerce").fillna(0) > 0).mean(), 3),
            "stop_reasons": dict(Counter(a["stop_reason"].dropna())) if "stop_reason" in a else {},
            "tools": dict(Counter(t for ts in a.get("tools_used", []) if isinstance(ts, list) for t in ts)),
            "agents": dict(Counter(t for ts in a.get("agents_used", []) if isinstance(ts, list) for t in ts))}
    return out

summarize = scorecard   # backwards-compatible name

def to_markdown(sc, title):
    P = sc["pipelines"]; names = list(P)
    L = [f"# {title}", "", f"Questions: **{sc['n_questions']}** | LLM used by ALL pipelines: **{', '.join(sc['models'])}** "
         f"({'PASS: single model' if sc['same_model_across_pipelines'] else 'FAIL: multiple models!'})", "",
         "## Per-pipeline scorecard", "", "| Metric | " + " | ".join(names) + " |", "|---|" + "---|" * len(names)]
    row = lambda label, f: L.append(f"| {label} | " + " | ".join(str(f(P[n])) for n in names) + " |")
    row("Accuracy (PASS rate)", lambda p: p.get("accuracy", "n/a (hidden)"))
    row("Completeness (gold items found)", lambda p: p.get("completeness", "n/a"))
    row("Semantic similarity (embedding)", lambda p: p.get("semantic_sim", "n/a"))
    row("Accuracy per 1k tokens", lambda p: p.get("accuracy_per_1k_tokens", "n/a"))
    row("Avg tokens / question (total)", lambda p: p["tokens_per_question"]["total"])
    row("&nbsp;&nbsp;context tokens", lambda p: p["tokens_per_question"]["context"])
    row("&nbsp;&nbsp;LLM input tokens", lambda p: p["tokens_per_question"]["input"])
    row("&nbsp;&nbsp;LLM output tokens", lambda p: p["tokens_per_question"]["output"])
    row("Avg latency (s)", lambda p: p["latency_s"]["avg"]); row("Median / p95 latency (s)", lambda p: f"{p['latency_s']['median']} / {p['latency_s']['p95']}")
    row("Avg retrieval steps", lambda p: p["avg_retrieval_steps"]); row("Avg reasoning steps", lambda p: p["avg_reasoning_steps"])
    row("Avg LLM calls", lambda p: p["avg_llm_calls"]); row("Avg chunks / citations", lambda p: f"{p['avg_chunks']} / {p['avg_citations']}")
    row("Citation validity", lambda p: p["citation_valid_rate"]); row("Answer supported by cited evidence", lambda p: p["answer_supported_rate"])
    row("Lift over blind-guess baseline", lambda p: p.get("lift_over_guess", "n/a"))
    row("Avg doc recall vs gold docs", lambda p: p.get("avg_doc_recall", "n/a"))
    L += ["", "## Retrieval methods used (share of questions using each)", ""]
    for n in names: L.append(f"- **{n}**: " + (", ".join(f"{k} {int(v*100)}%" for k, v in P[n]["retrieval_methods_share"].items()) or "none"))
    if "agentic" in sc:
        a = sc["agentic"]
        L += ["", "## Agentic GraphRAG", "", f"- Avg tokens / question: **{a['avg_tokens_per_question']}**", f"- Avg time / question: **{a['avg_time_per_question_s']} s**",
              f"- Avg steps: {a['avg_total_steps']} ({a['avg_retrieval_steps']} retrieval + {a['avg_reasoning_steps']} reasoning); avg orchestrator rounds: {a['avg_rounds']}",
              f"- Strategy-change rate: {a['strategy_change_rate']}", f"- Stop reasons: {a['stop_reasons']}", f"- Tools used: {a['tools']}", f"- Agents invoked: {a['agents']}"]
    if "by_qtype" in sc:
        L += ["", "## By question type", "", "| qtype | " + " | ".join(f"{n} acc | {n} tokens | {n} s" for n in names) + " |", "|---|" + "---|" * (3 * len(names))]
        for qt, d in sorted(sc["by_qtype"].items()):
            L.append(f"| {qt} | " + " | ".join(f"{d[n]['accuracy']} | {d[n]['avg_tokens']} | {d[n]['avg_latency_s']}" if n in d else "- | - | -" for n in names) + " |")
    return "\n".join(L) + "\n"
