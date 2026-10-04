import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import pandas as pd, streamlit as st
from evaluation.metrics import load_runs, scorecard

st.set_page_config(page_title="GraphRAG Bench", layout="wide")
st.title("RAG vs GraphRAG vs Agentic GraphRAG")
split = st.sidebar.selectbox("Split", ["public", "hidden"]); df = load_runs(split)
if df.empty: st.warning("No runs yet. Run evaluation.run_eval first."); st.stop()
sc = scorecard(df); P = sc["pipelines"]
st.sidebar.write("**Model (all pipelines):**", ", ".join(sc["models"])); st.sidebar.write("Single model across pipelines:", "yes" if sc["same_model_across_pipelines"] else "NO")
t1, t2, t3, t4, t5 = st.tabs(["Scorecard", "Accuracy vs tokens", "Agentic behaviour", "Drill-down", "Live ask"])
with t1:
    if any("accuracy" in p for p in P.values()):
        c = st.columns(len(P))
        for col, (n, p) in zip(c, P.items()):
            col.subheader(n); col.metric("Accuracy (PASS)", p.get("accuracy")); col.metric("Completeness", p.get("completeness"))
            col.metric("Avg tokens / question", p["tokens_per_question"]["total"]); col.metric("Avg latency (s)", p["latency_s"]["avg"])
        st.subheader("Accuracy vs completeness")
        st.bar_chart(pd.DataFrame({n: {"Accuracy": p.get("accuracy"), "Completeness": p.get("completeness"), "Semantic sim": p.get("semantic_sim")} for n, p in P.items()}))
    else: st.info("Hidden split has no ground truth here: tokens, latency and steps are shown; organizers grade accuracy.")
    rows = {"Accuracy": {n: p.get("accuracy") for n, p in P.items()},
            "Completeness": {n: p.get("completeness") for n, p in P.items()},
            "Semantic similarity": {n: p.get("semantic_sim") for n, p in P.items()},
            "Accuracy per 1k tokens": {n: p.get("accuracy_per_1k_tokens") for n, p in P.items()},
            "Avg tokens / question": {n: p["tokens_per_question"]["total"] for n, p in P.items()},
            "  context tokens": {n: p["tokens_per_question"]["context"] for n, p in P.items()},
            "  LLM input tokens": {n: p["tokens_per_question"]["input"] for n, p in P.items()},
            "  LLM output tokens": {n: p["tokens_per_question"]["output"] for n, p in P.items()},
            "Avg latency (s)": {n: p["latency_s"]["avg"] for n, p in P.items()},
            "Avg retrieval steps": {n: p["avg_retrieval_steps"] for n, p in P.items()},
            "Avg reasoning steps": {n: p["avg_reasoning_steps"] for n, p in P.items()},
            "Avg LLM calls": {n: p["avg_llm_calls"] for n, p in P.items()},
            "Avg chunks": {n: p["avg_chunks"] for n, p in P.items()},
            "Citation validity": {n: p["citation_valid_rate"] for n, p in P.items()},
            "Answer supported by citations": {n: p["answer_supported_rate"] for n, p in P.items()}}
    st.dataframe(pd.DataFrame(rows).T, use_container_width=True)
    st.subheader("Retrieval methods used (share of questions)")
    st.dataframe(pd.DataFrame({n: p["retrieval_methods_share"] for n, p in P.items()}).fillna(0), use_container_width=True)
    if "by_qtype" in sc:
        st.subheader("Accuracy by question type")
        st.bar_chart(pd.DataFrame({qt: {n: d.get(n, {}).get("accuracy") for n in P} for qt, d in sc["by_qtype"].items()}).T)
    st.subheader("Latency (s), avg"); st.bar_chart(pd.Series({n: p["latency_s"]["avg"] for n, p in P.items()}))
with t2:
    if any("accuracy" in p for p in P.values()):
        ov = pd.DataFrame([{"pipeline": n, "avg_total_tokens": p["tokens_per_question"]["total"], "accuracy": p.get("accuracy")} for n, p in P.items()])
        st.scatter_chart(ov, x="avg_total_tokens", y="accuracy", color="pipeline", size=200)
    st.subheader("Token breakdown (avg per question)"); st.bar_chart(df.groupby("pipeline")[["context_tokens", "input_tokens", "output_tokens", "total_tokens"]].mean())
with t3:
    a = sc.get("agentic")
    if not a: st.info("No agentic runs.")
    else:
        c = st.columns(4); c[0].metric("Avg tokens/question", a["avg_tokens_per_question"]); c[1].metric("Avg time/question (s)", a["avg_time_per_question_s"])
        c[2].metric("Avg steps", a["avg_total_steps"]); c[3].metric("Strategy-change rate", a["strategy_change_rate"])
        ag = df[df.pipeline == "agentic"]
        st.subheader("Steps histogram"); st.bar_chart((ag["retrieval_steps"] + ag["reasoning_steps"]).value_counts().sort_index())
        st.subheader("Tool usage"); st.bar_chart(pd.Series(a["tools"])); st.subheader("Agents invoked"); st.bar_chart(pd.Series(a["agents"]))
        st.subheader("Why it stopped"); st.bar_chart(pd.Series(a["stop_reasons"]))
with t4:
    qid = st.selectbox("Question", sorted(df["qid"].unique())); sub = df[df.qid == qid]
    st.write(sub.iloc[0]["question"]); st.write("**Gold:**", sub.iloc[0].get("gold"))
    for _, r in sub.iterrows():
        with st.expander(f"{r['pipeline']} -> {r['answer']}  [{r.get('verdict')}]  {r['total_tokens']} tokens, {r['latency']:.1f}s", expanded=r["pipeline"] == "agentic"):
            st.write("Citations:", r["citations"], "| valid:", r.get("citation_valid_rate"), "| supported:", r.get("answer_supported_by_citations"),
                     "| doc_recall:", r.get("doc_recall"), "| stop:", r.get("stop_reason"))
            st.markdown("**Investigation path**"); [st.markdown(f"{i+1}. {s}") for i, s in enumerate(r.get("investigation_path") or [])]
            with st.expander("Raw trace"): st.json(r["trace"])
            if isinstance(r.get("evidence"), list):
                with st.expander("Evidence ledger"): st.json(r["evidence"])
with t5:
    import pipelines
    from tgdb import get_tg
    q = st.text_input("Ask a question"); p = st.selectbox("Pipeline", list(pipelines.PIPES))
    if st.button("Run") and q:
        r = pipelines.run(p, q, get_tg()); st.success(r["answer"])
        st.markdown("**Investigation path**"); [st.markdown(f"{i+1}. {s}") for i, s in enumerate(r["investigation_path"])]
        st.json({k: r[k] for k in ("tokens", "latency_s", "citations", "retrieval_methods", "retrieval_steps", "reasoning_steps")})
