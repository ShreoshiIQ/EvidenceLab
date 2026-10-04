"""EvidenceLab dashboard.  streamlit run dashboard/app.py
Tab 1 asks one question to all three pipelines side by side; the other tabs read the saved evaluation runs (runs/*.jsonl)."""
import json, os, sys
from pathlib import Path
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
for _n in ("tigergraph-mcp", "tigergraph-mcp.exe"):   # on a cloud host the venv's bin folder may not be on PATH: use the full path when it exists
    _c = Path(sys.executable).parent / _n
    if _c.exists(): os.environ.setdefault("MCP_COMMAND", str(_c)); break
import pandas as pd, streamlit as st
from evaluation.metrics import load_runs, scorecard

st.set_page_config(page_title="EvidenceLab", layout="wide")
MAX_LIVE = int(os.getenv("DEMO_MAX_QUERIES", "10"))          # per browser session, protects the free-tier quota on a public demo
NAMES = {"rag": "RAG", "graphrag": "GraphRAG", "agentic": "Agentic GraphRAG"}
FALLBACK = {"aggregation": "According to the provided corpus, how many biathlon events at the 2018 Winter Olympics had more than 73 competitors?",
            "superlative": "According to the provided corpus, which alpine skiing event at the 1988 Winter Olympics had the highest number of competitors?",
            "multi_hop": "Who won the gold medal in the event held at Richmond Olympic Oval on 14 February 2010?",
            "temporal": "Who won the gold medal in the men's 80 kg taekwondo event at the Summer Olympics held immediately before 2016?"}

@st.cache_data
def examples():
    """One public question per type (from the public file when it is shipped, else a built-in list)."""
    out = dict(FALLBACK); p = ROOT / "data" / "eval_public.jsonl"
    if p.exists():
        seen = {}
        for l in open(p, encoding="utf-8"):
            if l.strip():
                q = json.loads(l); seen.setdefault(q.get("qtype"), q["question"])
        out = {k: seen[k] for k in ("lookup", "temporal", "multi_hop", "aggregation", "superlative") if k in seen} or out
    return out

@st.cache_resource
def _tg():
    from tgdb import get_tg
    tg = get_tg(); tg.ensure_up(); return tg

def fmt(x, nd=1): return "-" if x is None or (isinstance(x, float) and pd.isna(x)) else (f"{x:,.{nd}f}" if isinstance(x, float) else f"{x:,}")

# ------------------------------------------------------------------ sidebar
st.sidebar.title("EvidenceLab")
st.sidebar.caption("One question, three pipelines, one model, one TigerGraph graph.")
split = st.sidebar.selectbox("Evaluation split shown in the other tabs", ["public", "hidden"])
df = load_runs(split, str(ROOT / "runs")); sc = scorecard(df) if not df.empty else {}
P = sc.get("pipelines", {})
if sc:
    st.sidebar.write("**Model (all pipelines):**", ", ".join(sc["models"]))
    st.sidebar.write("Same model everywhere:", "yes" if sc["same_model_across_pipelines"] else "NO")
    st.sidebar.write(f"Questions: {sc['n_questions']}")
st.sidebar.markdown("Stack: TigerGraph Savanna, TigerVector, TigerGraph MCP server, Gemini.")

st.title("RAG vs GraphRAG vs Agentic GraphRAG")
st.caption("Which questions need an agent, and which do not? Ask one below, or open the scorecard tabs for the full evaluation.")
t_ask, t_need, t_score, t_eff, t_drill = st.tabs(["Ask", "Which questions need an agent", "Scorecard", "Efficiency and agent behaviour", "Drill-down"])

# ------------------------------------------------------------------ 1. live ask
with t_ask:
    st.subheader("Ask all three pipelines")
    ex = examples()
    st.write("Try an example, or type your own question about Olympic events:")
    cols = st.columns(len(ex))
    for c, (qt, q) in zip(cols, ex.items()):
        if c.button(qt.replace("_", " "), help=q): st.session_state["q"] = q
    q = st.text_area("Question", key="q", height=80)
    chosen = st.multiselect("Pipelines", list(NAMES), default=list(NAMES), format_func=NAMES.get)
    used = st.session_state.get("used", 0)
    go = st.button("Run", type="primary", disabled=used >= MAX_LIVE or not q or not chosen)
    cap = st.empty()
    if go:
        import pipelines
        from gateway.llm import QuotaExhausted
        res = {}
        try:
            tg = _tg()
            for p in chosen:
                with st.spinner(f"Running {NAMES[p]}..."): res[p] = pipelines.run(p, q, tg)
            st.session_state["used"] = used + 1
        except QuotaExhausted:
            st.error("The shared LLM quota is used up for now. The saved results in the other tabs are still available.")
        except Exception as e:
            st.error(f"Could not run the pipelines: {str(e)[:300]}")
        st.session_state["res"] = (q, res)
    cap.caption(f"Live questions this session: {st.session_state.get('used', 0)}/{MAX_LIVE} (the limit protects the shared API quota).")
    q_done, res = st.session_state.get("res", (None, {}))
    if res:
        st.markdown(f"**Question:** {q_done}")
        cols = st.columns(len(res))
        for c, (p, r) in zip(cols, res.items()):
            with c:
                st.markdown(f"### {NAMES[p]}"); st.success(r["answer"] or "(no answer)")
                a, b = st.columns(2)
                a.metric("Tokens", fmt(r["tokens"]["total"], 0)); b.metric("Latency (s)", fmt(r.get("latency_adj_s", r["latency_s"]), 1), help=f"Wall clock {r['latency_s']:.1f}s. Cached LLM replies are counted at their original call time, as in the scorecard.")
                a.metric("Retrieval steps", fmt(r.get("retrieval_steps"), 0)); b.metric("Reasoning steps", fmt(r.get("reasoning_steps"), 0))
                if r.get("stop_reason"): st.caption(f"Stopped because: {r['stop_reason']}")
                st.caption("Methods: " + ", ".join(r.get("retrieval_methods") or []))
        st.markdown("#### How each pipeline got there")
        for p, r in res.items():
            with st.expander(f"{NAMES[p]}: investigation path and evidence", expanded=p == "agentic"):
                for i, s in enumerate(r.get("investigation_path") or []): st.markdown(f"{i+1}. {s}")
                st.write("**Citations:**", r.get("citations"), "| valid:", r.get("citation_valid_rate"), "| supported by evidence:", r.get("answer_supported_by_citations"))
                if r.get("reasoning"): st.write("**Reasoning:**", r["reasoning"])
                with st.expander("Evidence ledger"): st.json(r.get("evidence") or [])
                with st.expander("Raw trace"): st.json(r.get("trace") or [])

# ------------------------------------------------------------------ 2. which questions need an agent
with t_need:
    if "by_qtype" not in sc: st.info("Needs a split with answers (public). Run evaluation.run_eval on the public file first.")
    else:
        bt = sc["by_qtype"]; order = [t for t in ("lookup", "temporal", "multi_hop", "aggregation", "superlative") if t in bt] + [t for t in bt if t not in ("lookup", "temporal", "multi_hop", "aggregation", "superlative")]
        acc = pd.DataFrame({NAMES[n]: {t: bt[t].get(n, {}).get("accuracy") for t in order} for n in P}); acc.index.name = "question type"
        tok = pd.DataFrame({NAMES[n]: {t: bt[t].get(n, {}).get("avg_tokens") for t in order} for n in P}); tok.index.name = "question type"
        st.subheader("Accuracy by question type")
        st.bar_chart(acc)
        flat = {t: ("Agent not needed: a cheaper pipeline ties it" if max(v for v in acc.loc[t].dropna()) - acc.loc[t].dropna().drop("Agentic GraphRAG", errors="ignore").max() <= 0.05 else "Agent helps") if "Agentic GraphRAG" in acc.columns and len(acc.loc[t].dropna()) > 1 else "-" for t in order}
        n_q = {t: max((bt[t].get(n, {}).get("n", 0) for n in P), default=0) for t in order}
        tab = acc.copy(); tab.insert(0, "questions", pd.Series(n_q)); tab["verdict"] = pd.Series(flat)
        st.dataframe(tab)
        st.subheader("Average tokens per question, by type"); st.dataframe(tok.round(0))
        st.caption("Rule used for the verdict: if no cheaper pipeline is within 5 points of the agent's accuracy, the question type needs an agent.")

# ------------------------------------------------------------------ 3. scorecard
with t_score:
    if not P: st.warning("No saved runs for this split yet. Run evaluation.run_eval first.")
    else:
        if any("accuracy" in p for p in P.values()):
            c = st.columns(len(P))
            for col, (n, p) in zip(c, P.items()):
                col.subheader(NAMES[n]); col.metric("Accuracy (PASS)", p.get("accuracy")); col.metric("Completeness", p.get("completeness"))
                col.metric("Avg tokens / question", p["tokens_per_question"]["total"]); col.metric("Avg latency (s)", p["latency_s"]["avg"])
            st.subheader("Accuracy, completeness, semantic similarity")
            st.bar_chart(pd.DataFrame({NAMES[n]: {"Accuracy": p.get("accuracy"), "Completeness": p.get("completeness"), "Semantic sim": p.get("semantic_sim")} for n, p in P.items()}))
        else: st.info("The hidden split has no ground truth here: tokens, latency and steps are shown; the organizers grade accuracy.")
        rows = {"Accuracy": {n: p.get("accuracy") for n, p in P.items()},
                "Completeness": {n: p.get("completeness") for n, p in P.items()},
                "Semantic similarity": {n: p.get("semantic_sim") for n, p in P.items()},
                "Accuracy per 1k tokens": {n: p.get("accuracy_per_1k_tokens") for n, p in P.items()},
                "Avg tokens / question": {n: p["tokens_per_question"]["total"] for n, p in P.items()},
                "  context tokens": {n: p["tokens_per_question"]["context"] for n, p in P.items()},
                "  LLM input tokens": {n: p["tokens_per_question"]["input"] for n, p in P.items()},
                "  LLM output tokens": {n: p["tokens_per_question"]["output"] for n, p in P.items()},
                "Avg latency (s)": {n: p["latency_s"]["avg"] for n, p in P.items()},
                "Median / p95 latency (s)": {n: f"{p['latency_s']['median']} / {p['latency_s']['p95']}" for n, p in P.items()},
                "Avg retrieval steps": {n: p["avg_retrieval_steps"] for n, p in P.items()},
                "Avg reasoning steps": {n: p["avg_reasoning_steps"] for n, p in P.items()},
                "Avg LLM calls": {n: p["avg_llm_calls"] for n, p in P.items()},
                "Avg chunks": {n: p["avg_chunks"] for n, p in P.items()},
                "Citation validity": {n: p["citation_valid_rate"] for n, p in P.items()},
                "Answer supported by citations": {n: p["answer_supported_rate"] for n, p in P.items()}}
        st.dataframe(pd.DataFrame(rows).T.astype(str))
        st.subheader("Retrieval methods used (share of questions)")
        st.dataframe(pd.DataFrame({NAMES[n]: p["retrieval_methods_share"] for n, p in P.items()}).fillna(0))

# ------------------------------------------------------------------ 4. efficiency + agent behaviour
with t_eff:
    if not P: st.warning("No saved runs for this split yet.")
    else:
        if any("accuracy" in p for p in P.values()):
            ov = pd.DataFrame([{"pipeline": NAMES[n], "avg_total_tokens": p["tokens_per_question"]["total"], "accuracy": p.get("accuracy")} for n, p in P.items()])
            st.subheader("Accuracy vs tokens"); st.scatter_chart(ov, x="avg_total_tokens", y="accuracy", color="pipeline", size=200)
        st.subheader("Token breakdown (avg per question)")
        st.bar_chart(df.groupby("pipeline")[["context_tokens", "input_tokens", "output_tokens"]].mean())
        st.subheader("Latency (s), avg"); st.bar_chart(pd.Series({NAMES[n]: p["latency_s"]["avg"] for n, p in P.items()}))
        a = sc.get("agentic")
        if a:
            st.subheader("Agentic GraphRAG")
            c = st.columns(4); c[0].metric("Avg tokens / question", a["avg_tokens_per_question"]); c[1].metric("Avg time / question (s)", a["avg_time_per_question_s"])
            c[2].metric("Avg steps", a["avg_total_steps"]); c[3].metric("Avg orchestrator rounds", a.get("avg_rounds"))
            ag = df[df.pipeline == "agentic"]
            l, r = st.columns(2)
            l.markdown("**Steps per question**"); l.bar_chart((ag["retrieval_steps"] + ag["reasoning_steps"]).value_counts().sort_index())
            r.markdown("**Why it stopped**"); r.bar_chart(pd.Series(a["stop_reasons"]))
            l, r = st.columns(2)
            l.markdown("**Tools used**"); l.bar_chart(pd.Series(a["tools"])); r.markdown("**Agents invoked**"); r.bar_chart(pd.Series(a["agents"]))

# ------------------------------------------------------------------ 5. drill-down
with t_drill:
    if df.empty: st.warning("No saved runs for this split yet.")
    else:
        fails = st.checkbox("Only questions at least one pipeline got wrong") if "correct" in df else False
        ids = sorted(df["qid"].unique())
        if fails: ids = sorted(df[df["correct"] == 0.0]["qid"].unique())
        if not ids: st.info("No failures.")
        else:
            qid = st.selectbox("Question", ids); sub = df[df.qid == qid]
            st.write(sub.iloc[0]["question"]);
            if "gold" in sub: st.write("**Gold:**", sub.iloc[0].get("gold"))
            for _, r in sub.iterrows():
                with st.expander(f"{NAMES.get(r['pipeline'], r['pipeline'])} -> {r['answer']}  [{r.get('verdict', '-')}]  {r['total_tokens']} tokens, {r['latency']:.1f}s", expanded=r["pipeline"] == "agentic"):
                    st.write("Citations:", r["citations"], "| valid:", r.get("citation_valid_rate"), "| supported:", r.get("answer_supported_by_citations"),
                             "| doc_recall:", r.get("doc_recall"), "| stop:", r.get("stop_reason"))
                    st.markdown("**Investigation path**")
                    for i, s in enumerate(r.get("investigation_path") or []): st.markdown(f"{i+1}. {s}")
                    with st.expander("Raw trace"): st.json(r["trace"])
                    if isinstance(r.get("evidence"), list):
                        with st.expander("Evidence ledger"): st.json(r["evidence"])
