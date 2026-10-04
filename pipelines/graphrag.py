import config as C
from answer import generate_answer
from common import Ledger, Trace, events_text
from gateway.embed import embed
from agentic.linking import link

def run(question, tg):
    """Fixed strategy decided once in code: link -> vector seeds -> 1-hop expand -> entity-neighbourhood facts."""
    tr, led = Trace(), Ledger()
    with tr.op("embedding", "bge"): qv = embed([question], query=True)[0]
    with tr.op("entity_linking", "alias+vector") as r:
        ents = [e for e in link(tg, question) if e["conf"] >= 0.9][:3]; r["entities"] = [e["id"] for e in ents]
    with tr.op("similarity_search", "vec_chunks", k=C.GRAPH_SEEDS) as r:
        seeds = tg.vec_chunks(qv, C.GRAPH_SEEDS); r["n_chunks"] = len(seeds)
    with tr.op("graph_traversal", "expand_chunks") as r:
        rel = tg.expand_chunks([s["id"] for s in seeds], C.GRAPH_EXPAND_LIMIT)["chunks"]; r["n_chunks"] = len(rel)
    events = []
    with tr.op("graph_traversal", "entity_events") as r:
        sets = [{e["id"]: e for e in tg.entity_events(x["id"])} for x in ents]
        if sets:
            keys = set.intersection(*map(set, sets)) or set().union(*sets)
            events = [next(s[k] for s in sets if k in s) for k in sorted(keys)][:C.GRAPH_MAX_EVENTS]
        r["n_events"] = len(events)
    if events:
        led.add("vertex", events_text(events, "linked to question entities"), {"doc_ids": [e["id"] for e in events]}, "graph_traversal", 1)
    for x in seeds + rel: led.add("chunk", x["text"], {"chunk_id": x["id"], "doc_id": x["doc_id"]}, "graph_traversal", 1)
    with tr.op("answer_generation", "llm"): ans = generate_answer(question, led.items)
    return {**ans, "trace": tr.steps, "evidence": led.items, "meta": {"steps": 4, "stop_reason": "fixed_pipeline"}}
