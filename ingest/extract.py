"""Optional LLM extraction over PROSE only (tables/infobox are parsed deterministically).
Entity resolution = normalised-key merge (accent/case-insensitive) via common.ent_id."""
from common import ent_id
from gateway.llm import llm, jparse

PROMPT = """Extract explicitly named entities from each numbered passage.
Types: Athlete, Country, Venue, Organisation.
Return JSON {"items":[{"i":<index>,"entities":[{"name":"...","type":"..."}]}]}"""

def extract_entities(chunks, batch=6):
    out = {}
    for s in range(0, len(chunks), batch):
        part = chunks[s:s + batch]
        passages = "\n\n".join(f"[{i}] " + "\n".join(l for l in c["text"].split("\n") if "|" not in l)[:1200] for i, c in enumerate(part))
        text, _ = llm("extract", PROMPT + "\n\n" + passages, json_mode=True, max_out=1200)
        for it in (jparse(text, {}) or {}).get("items", []):
            try: c = part[int(it["i"])]
            except Exception: continue
            for e in it.get("entities", []):
                if e.get("name") and e.get("type") in ("Athlete", "Country", "Venue", "Organisation"):
                    out.setdefault(c["id"], []).append({"id": ent_id(e["type"], e["name"]), "name": e["name"], "etype": e["type"], "role": "mentioned"})
    return out
