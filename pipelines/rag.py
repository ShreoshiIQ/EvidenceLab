import config as C
from answer import generate_answer
from common import Ledger, Trace
from gateway.embed import embed

def run(question, tg):
    tr, led = Trace(), Ledger()
    with tr.op("embedding", "bge"): qv = embed([question], query=True)[0]
    with tr.op("similarity_search", "vec_chunks", k=C.RAG_TOPK) as r:
        rows = tg.vec_chunks(qv, C.RAG_TOPK); r["n_chunks"] = len(rows)
    for x in rows: led.add("chunk", x["text"], {"chunk_id": x["id"], "doc_id": x["doc_id"]}, "similarity_search", 1)
    with tr.op("answer_generation", "llm"): ans = generate_answer(question, led.items)
    return {**ans, "trace": tr.steps, "evidence": led.items, "meta": {"steps": 1, "stop_reason": "fixed_pipeline"}}
