import hashlib, re, time, unicodedata
from contextlib import contextmanager
from dataclasses import dataclass
from gateway.llm import current_meter

def norm(s):
    s = unicodedata.normalize("NFKD", s); s = "".join(c for c in s if not unicodedata.combining(c))
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s'-]", " ", s.lower())).strip()

def ent_id(etype, name): return f"{etype.lower()}:{norm(name)}"
def est_tokens(s): return max(1, len(s) // 4)   # approx; context_tokens is an estimate

@dataclass
class Evidence:
    id: str; kind: str; content: str; provenance: dict; retrieved_by: str; step: int
    def doc_ids(self):
        p = self.provenance
        return [p[k] for k in ("doc_id", "event_id") if p.get(k)] + list(p.get("doc_ids", []))
    def to_dict(self, n=400):
        return {"id": self.id, "kind": self.kind, "content": self.content[:n], "provenance": self.provenance,
                "retrieved_by": self.retrieved_by, "step": self.step}

class Ledger:
    """Evidence ledger shared by all three pipelines (deduped, stable IDs E1, E2, ...)."""
    def __init__(self): self.items, self._k = [], set()
    def add(self, kind, content, provenance, by, step):
        k = hashlib.md5((kind + content[:300]).encode()).hexdigest()
        if k in self._k: return None
        ev = Evidence(f"E{len(self.items)+1}", kind, content, provenance, by, step)
        self.items.append(ev); self._k.add(k); return ev

class Trace:
    """Per-operation timing and token deltas."""
    def __init__(self): self.steps = []
    @contextmanager
    def op(self, agent, tool, **meta):
        m = current_meter(); b = m.totals() if m else {"in": 0, "out": 0, "total": 0, "calls": 0}
        rec = {"step": len(self.steps) + 1, "agent": agent, "tool": tool, **meta}; t = time.time()
        try: yield rec
        finally:
            a = m.totals() if m else b
            rec["latency_s"] = round(time.time() - t, 3)
            rec["tokens"] = {"in": a["in"] - b["in"], "out": a["out"] - b["out"], "total": a["total"] - b["total"], "llm_calls": a["calls"] - b["calls"]}
            self.steps.append(rec)

def events_text(events, label):
    def line(e):
        s = f"{e['id']} | {e['title']} | competitors={e.get('competitors')} | nations={e.get('nations')}"
        for k in ("date", "venue", "gold", "silver", "bronze"):
            if e.get(k): s += f" | {k}={e[k]}"
        return s
    return f"Events ({len(events)}) {label}:\n" + "\n".join(line(e) for e in events)

# ---------------- per-question metrics shared by all pipelines ----------------
RETRIEVAL_AGENTS = {"entity_linking", "similarity_search", "graph_traversal", "document_retrieval"}
REASONING_AGENTS = {"orchestrator", "multi_hop_reasoning", "evidence_evaluation", "aggregation", "answer_generation"}

def method_label(step):
    a, t = step["agent"], step["tool"]
    if a == "similarity_search": return "vector_search"
    if a == "graph_traversal": return f"graph_traversal:{t}"
    return a

def step_stats(trace):
    """Retrieval steps = fetching evidence; reasoning steps = planning/evaluating/computing/answering.
    The query-embedding op is preparation and is not counted in either."""
    retr = [s for s in trace if s["agent"] in RETRIEVAL_AGENTS]
    reas = [s for s in trace if s["agent"] in REASONING_AGENTS]
    return {"retrieval_steps": len(retr), "reasoning_steps": len(reas),
            "retrieval_methods": sorted({method_label(s) for s in retr}),
            "reasoning_methods": sorted({s["agent"] for s in reas})}

def investigation_path(trace):
    """Human-readable investigation path for the dashboard / presentation."""
    out = []
    for s in trace:
        a = s["agent"]
        if a == "embedding": continue
        if a == "orchestrator":
            plan = s.get("plan") or []
            out.append(f"Plan (round {s.get('round', '?')}): " + (" -> ".join(str(p.get("agent", "?")) for p in plan) or "no valid plan"))
        elif a == "answer_generation": out.append("Generate the answer from the cited evidence")
        else:
            new = s.get("new_evidence"); why = s.get("rationale") or ""
            out.append(f"{a}[{s['tool']}]" + (f": {why}" if why else "") + (f" (+{len(new)} evidence)" if isinstance(new, list) else "")
                       + (f"  [strategy change: {s['strategy_change_reason']}]" if s.get("strategy_change") else ""))
    return out

def grounding(answer, citations, evidence):
    """Deterministic grounding checks: are cited ids real, and does the answer appear in the cited evidence?"""
    ids = {e.id for e in evidence}; valid = [x for x in citations if x in ids]
    rate = round(len(valid) / len(citations), 3) if citations else 0.0
    a = norm(answer or ""); supported = None
    if a and "not found" not in a:
        text = norm(" ".join(e.content for e in evidence if e.id in valid))
        toks = re.findall(r"[\w']+", a)
        supported = bool(valid) and all(t in text for t in toks)
    return {"citation_valid_rate": rate, "answer_supported_by_citations": supported}


# ---------------- date matching ("15 to 22 August 2004" vs "15–22 August 2004" / "1992-08-06 (qualification) 1992-08-08 (final)") ----------------
_MONTHS = ["january", "february", "march", "april", "may", "june", "july", "august", "september", "october", "november", "december"]

def date_sig(s):
    import re as _re
    t = (s or "").lower()
    t = _re.sub(r"(\d{4})-(\d{2})-(\d{2})", lambda m: f"{int(m[3])} {_MONTHS[int(m[2]) - 1] if 1 <= int(m[2]) <= 12 else ''}", t)
    months = {m for m in _MONTHS if m in t} | {m for m in _MONTHS if _re.search(r"\b" + m[:3] + r"\b", t)}
    t = _re.sub(r"\b(?:19|20)\d{2}\b", " ", t)
    days = {int(x) for x in _re.findall(r"(?<!\d)(\d{1,2})(?!\d)", t) if 1 <= int(x) <= 31}
    ranges = set()
    for a, b in _re.findall(r"(?<!\d)(\d{1,2})\s*(?:[–-]|to)\s*(\d{1,2})(?!\d)", t):   # expand ranges
        if int(a) <= int(b) <= 31: days |= set(range(int(a), int(b) + 1)); ranges.add((int(a), int(b)))
    return days, months, ranges

def date_match(query, event_date):
    qd, qm, qr = date_sig(query); ed, em, er = date_sig(event_date)
    if not ed or not qd: return False
    if qm and em and not (qm & em): return False
    if qr: return qr <= er                       # a question range ("15 to 22 August") must equal the event's range
    return qd <= ed
