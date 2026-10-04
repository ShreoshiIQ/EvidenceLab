from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import config as C
from evaluation.metrics import load_runs, scorecard
from pipelines import PIPES, run
from tgdb import get_tg

app = FastAPI(title="TigerGraph GraphRAG Bench")
_tg = None
def tg():
    global _tg
    _tg = _tg or get_tg(); return _tg

class Q(BaseModel):
    question: str
    pipeline: str = "agentic"

@app.get("/health")
def health():
    return {"tigergraph_up": tg().ensure_up(wait=20), "backend": C.TG_BACKEND, "model": C.GEN_MODEL,
            "keys": len(C.API_KEYS), "free_tier_enforced": C.ENFORCE_FREE_TIER}

@app.post("/ask")
def ask(q: Q):
    if q.pipeline not in PIPES: raise HTTPException(400, f"pipeline must be one of {list(PIPES)}")
    if not tg().ensure_up(wait=60): raise HTTPException(503, "TigerGraph Savanna workspace unreachable (suspended?). Resume it and retry.")
    return run(q.pipeline, q.question, tg())

@app.get("/metrics")
def metrics(split: str = "public"): return scorecard(load_runs(split))
