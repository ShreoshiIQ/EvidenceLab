"""python -m ingest.verify data/corpus.jsonl  -- no re-upload. Searches with the stored text's own embedding; each chunk should come back as its own nearest neighbour."""
import json, random, sys, time
from gateway.embed import embed
from ingest.parse import parse_doc
from tgdb import get_tg

docs = [json.loads(l) for l in open(sys.argv[1], encoding="utf-8") if l.strip()]
chunks = {c["id"]: c["text"] for d in docs for c in parse_doc(d)["chunks"]}
ids = list(chunks); sample = random.Random(7).sample(ids[1500:], 10)
vecs = dict(zip(sample, embed([chunks[i] for i in sample], progress=False)))
tg = get_tg(); tg.ensure_up()
hits = 0
for cid in sample:
    t = time.time(); got = [r.get("id") for r in tg._vq("vec_chunks", vecs[cid], 3, "Top")]
    ok = cid in got; hits += ok; print(("OK  " if ok else "MISS"), cid, "->", got, f"{time.time() - t:.1f}s")
print(f"\n{hits}/{len(sample)} chunks found by their own vector")
print("VECTORS STORED" if hits >= len(sample) - 1 else "VECTORS NOT STORED (or index still building: wait a minute and run again)")
