import difflib, re
from common import norm
from gateway.embed import embed

ETYPES = ["games", "sport", "venue", "country", "athlete"]

def ngrams(text, nmax=6):
    w = re.findall(r"[\w'-]+", norm(text))
    for n in range(1, nmax + 1):
        for i in range(len(w) - n + 1): yield " ".join(w[i:i + n])

def link(tg, text, mentions=None, k=6):
    """Alias index (n-gram -> exact entity id, conf 1.0) + vector fallback (low confidence)."""
    found = {}
    for s in (mentions or [text]):
        ids = {f"{t}:{g}" for g in set(ngrams(s)) for t in ETYPES}
        for y in re.findall(r"\b(?:19|20)\d{2}\b", s):      # "2004" -> games:2004 summer/winter olympics/paralympics
            ids |= {f"games:{y} {se} {k}" for se in ("summer", "winter") for k in ("olympics", "paralympics")}
        for e in tg.entity_by_ids(sorted(ids)): found[e["id"]] = {**e, "conf": 1.0, "method": "alias"}
        for e in tg.vec_entities(embed([s], query=True)[0], k):
            if e["id"] in found: continue
            r = difflib.SequenceMatcher(None, norm(e["name"]), norm(s)).ratio() if mentions else 0.4
            found[e["id"]] = {**e, "conf": round(r, 2), "method": "vector"}
    return sorted(found.values(), key=lambda e: -e["conf"])
