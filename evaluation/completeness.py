"""Free, deterministic partial-credit metrics (no LLM calls).
completeness: share of ground-truth items present in the answer (1.0 = every item found).
semantic_sim: cosine similarity of local bge embeddings (cheap BERTScore-style proxy), optional."""
import re
from common import norm

def _toks(s): return set(re.findall(r"[a-z0-9]+", norm(s)))

def completeness(gold, pred):
    gold = [str(x) for x in (gold if isinstance(gold, list) else [gold])]
    ptxt = ", ".join(map(str, pred)) if isinstance(pred, list) else str(pred or "")
    pn, pt = norm(ptxt), _toks(ptxt)
    if not gold or not pn or "not found in the corpus" in pn: return 0.0
    hit = 0.0
    for g in gold:
        gn = norm(g)
        if re.fullmatch(r"-?\d+(\.\d+)?", gn.replace(",", "")):
            hit += 1.0 if gn.replace(",", "") in {t.replace(",", "") for t in re.findall(r"-?\d[\d,]*\.?\d*", ptxt)} else 0.0
        elif gn and (gn in pn or re.sub(r'\W','',gn) in re.sub(r'\W','',pn)): hit += 1.0
        else:
            gt = _toks(g); hit += (len(gt & pt) / len(gt)) if gt else 0.0   # partial token recall (fused names, dashes)
    return round(hit / len(gold), 3)

def semantic_sim(gold, pred):
    try:
        from gateway.embed import embed
        g = " ".join(map(str, gold)) if isinstance(gold, list) else str(gold)
        p = ", ".join(map(str, pred)) if isinstance(pred, list) else str(pred or "")
        if not p.strip(): return 0.0
        a, b = embed([g, p])[:2]
        import math
        d = sum(x * y for x, y in zip(a, b)); n = math.sqrt(sum(x*x for x in a)) * math.sqrt(sum(y*y for y in b))
        return round(d / n, 3) if n else 0.0
    except Exception:
        return None
