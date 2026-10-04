import time
import config as C
from common import grounding, investigation_path, step_stats
from gateway.llm import RunMeter
from pipelines import rag, graphrag
from agentic import harness

PIPES = {"rag": rag.run, "graphrag": graphrag.run, "agentic": harness.run}

def run(name, question, tg):
    """Uniform wrapper: same metering and output schema for every pipeline."""
    getattr(tg, "drain_log", lambda: [])()
    t0 = time.time()
    with RunMeter() as m:
        out = PIPES[name](question, tg)
    wall = time.time() - t0
    models = sorted({c.get("model", "?") for c in m.calls})
    if set(models) - {C.GEN_MODEL}:      # hard guarantee: every pipeline LLM call uses the SAME pinned model
        raise RuntimeError(f"Pipeline {name} used models {models}, expected only {C.GEN_MODEL}")
    tot = m.totals(); ev = out["evidence"]
    llm_wall = sum(c["latency_s"] for c in m.calls); llm_orig = sum(c.get("orig_latency_s", c["latency_s"]) for c in m.calls)
    return {"pipeline": name, "question": question, "answer": out["answer"], "answer_list": out.get("answer_list", [out["answer"]]), "citations": out["citations"],
            "reasoning": out.get("reasoning", ""), "models_used": models,
            "tokens": {"context": out["context_tokens"], "input": tot["in"], "output": tot["out"], "total": tot["total"],
                       "llm_calls": tot["calls"]},
            # latency_s = wall clock; latency_adj_s replaces cache-hit LLM time with the ORIGINAL call time so reruns stay honest
            "latency_s": round(wall, 3), "latency_adj_s": round(max(0.0, wall - llm_wall + llm_orig), 3),
            "n_chunks": sum(e.kind == "chunk" for e in ev), "n_citations": len(out["citations"]),
            **grounding(out["answer"], out["citations"], ev), **step_stats(out["trace"]),
            "investigation_path": investigation_path(out["trace"]),
            "retrieved_doc_ids": sorted({d for e in ev for d in e.doc_ids()}),
            "mcp_tool_calls": getattr(tg, "drain_log", lambda: [])(),
            "evidence": [e.to_dict() for e in ev], "trace": out["trace"], **out.get("meta", {})}
