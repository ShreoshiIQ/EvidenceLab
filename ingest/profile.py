"""python -m ingest.profile data/corpus.jsonl   -- no network, no API quota.
Counts how many docs parse (title pattern, infobox, medals, date, venue) and prints examples of failures."""
import json, sys, random
from collections import Counter
from ingest.parse import parse_doc

def main(path, n_examples=3):
    docs = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    c, ex = Counter(), {}
    def flag(k, d):
        c[k] += 1; ex.setdefault(k, []).append(d["doc_id"])
    keys = Counter()
    for d in docs:
        keys.update(d.keys()); c["docs"] += 1
        if not d["text"].startswith("[Infobox"): flag("no_infobox", d); continue
        try: p = parse_doc(d)
        except Exception as e: flag("parse_error", d); continue
        ev = p["event"]; types = {e["etype"] for e in p["entities"]}
        c["TOTAL_CHUNKS_to_embed"] += len(p["chunks"])
        c["chunks_in_olympic_pages" if "Games" in types else "chunks_in_other_pages"] += len(p["chunks"])
        if "Games" not in types: flag("no_games_in_title", d)
        if "Sport" not in types: flag("no_sport_in_title", d)
        if not ev["date"]: flag("no_date", d)
        if not ev["venue"]: flag("no_venue", d)
        if ev["nations"] is None: flag("no_nations", d)
        if ev["competitors"] is None: flag("no_competitors", d)
        if not ev["gold"]: flag("no_gold", d)
        if len(ev["gold"]) > 60 and ", " not in ev["gold"]: flag("gold_looks_fused", d)
    print("fields in docs:", dict(keys)); print()
    for k, v in sorted(c.items()): print(f"{k:22s} {v:7d}  {100*v/c['docs']:.1f}%")
    print("\nexample doc_ids per problem:")
    for k, v in ex.items(): print(" ", k, random.sample(v, min(n_examples, len(v))))

if __name__ == "__main__": main(sys.argv[1])
