"""python -m ingest.peek data/corpus.jsonl   -- writes peek.txt (UTF-8). No network."""
import json, sys, re, random, io
from collections import Counter
from ingest.parse import parse_doc, TITLE_RE

random.seed(1)
OUT = io.open("peek.txt", "w", encoding="utf-8")
def P(*a): print(*a, file=OUT)
docs = [json.loads(l) for l in open(sys.argv[1], encoding="utf-8") if l.strip()]
keys, groups = Counter(), {"no_title_match": [], "title_ok_no_date": [], "title_ok_no_gold": [], "title_ok_no_venue": []}
for d in docs:
    t = d["text"]
    if t.startswith("[Infobox"):
        for l in t.partition("\n\n")[0].split("\n")[1:]: keys[l.strip().partition(":")[0].strip()] += 1
    if not TITLE_RE.match(d["title"]): groups["no_title_match"].append(d); continue
    try: ev = parse_doc(d)["event"]
    except Exception: continue
    if not ev["date"]: groups["title_ok_no_date"].append(d)
    if not ev["gold"]: groups["title_ok_no_gold"].append(d)
    if not ev["venue"]: groups["title_ok_no_venue"].append(d)
P("INFOBOX KEYS:", dict(keys.most_common(40)), "\n")
P("TITLE SHAPES (digits->#):", dict(Counter(re.sub(r"\d+", "#", d["title"])[:40] for d in docs).most_common(25)), "\n")
for g, ds in groups.items():
    P(f"=== {g}: {len(ds)} docs; 3 samples ===")
    for d in random.sample(ds, min(3, len(ds))): P("TITLE:", d["title"], "| id:", d["doc_id"], "\n", d["text"][:450].replace("\n", " | "), "\n")

OUT.close(); print("wrote peek.txt")
