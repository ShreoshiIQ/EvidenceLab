"""Shared answer generation: IDENTICAL model, prompt, schema for all three pipelines."""
import config as C
from common import est_tokens
from gateway.llm import llm, jparse

SYSTEM = """You answer questions using ONLY the numbered evidence provided. Never use outside knowledge.
- For counting/aggregation, use AGGREGATE evidence if present; otherwise enumerate the items you can see.
- If the evidence does not contain the answer, the answer is ["Not found in the corpus"].
- Cite the evidence IDs you used, e.g. "E3".
- "answer" is a JSON list with one short string per final answer item (a single-item list for a single answer: a number, a name or a short phrase).
  Use bare names/values: no country codes or parentheses unless the question asks for the country.
  Medalist names: copy them EXACTLY as written in the evidence. Team medalists are often written run together ("Dani KingLaura TrottJoanna Rowsell"): keep that exact string as ONE item, with no separators added.
  When asked which event/competition, answer with its full title exactly as in the evidence (keep the en dash), e.g. "Sailing at the 2000 Summer Olympics – Soling".
Return JSON: {"answer": ["<item>"], "citations": ["E1"], "reasoning": "<max 2 sentences>"}"""

ORDER = {"aggregate": 0, "vertex": 1, "chunk": 2}

def build_context(evs, max_chars=None):
    max_chars = max_chars or C.MAX_CONTEXT_CHARS
    out, used = [], 0
    for e in sorted(evs, key=lambda e: (ORDER.get(e.kind, 3), int(e.id[1:]))):
        s = f"[{e.id}] ({e.kind}) {e.content[:C.PER_ITEM_CHARS]}"
        if used + len(s) > max_chars: break
        out.append(s); used += len(s)
    return "\n\n".join(out)

def generate_answer(question, evs):
    ctx = build_context(evs)
    text, rec = llm("answer", f"Question: {question}\n\nEvidence:\n{ctx}", SYSTEM, json_mode=True, max_out=400)
    j = jparse(text, {}) or {}
    raw = j.get("answer", text.strip())
    items = [str(x).strip() for x in raw] if isinstance(raw, list) else [str(raw).strip()]
    items = [x for x in items if x]
    return {"answer": ", ".join(items), "answer_list": items, "citations": [c for c in j.get("citations", []) if isinstance(c, str)],
            "reasoning": j.get("reasoning", ""), "context_tokens": est_tokens(ctx)}
