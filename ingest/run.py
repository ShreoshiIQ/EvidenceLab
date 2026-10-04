"""python -m ingest.run --corpus data/corpus.jsonl [--llm-extract] [--limit N]
Vertices/edges: bulk upsert (pyTigerGraph REST). Embeddings: TigerGraph MCP tool upsert_vectors (when TG_BACKEND=mcp)."""
import argparse, json, time
import config as C
from gateway.embed import embed
from ingest.parse import parse_doc
from tgdb import get_tg

def vtx(vt, rows):
    return {"vertices": {vt: {i: {k: {"value": v} for k, v in a.items()} for i, a in rows.items()}}}

def edg(items):
    out = {"edges": {}}
    for st, sid, et, tt, tid, at in items:
        out["edges"].setdefault(st, {}).setdefault(sid, {}).setdefault(et, {}).setdefault(tt, {})[tid] = {k: {"value": v} for k, v in (at or {}).items()}
    return out

def batches(seq, n):
    for i in range(0, len(seq), n): yield seq[i:i + n]

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--corpus", required=True)
    ap.add_argument("--llm-extract", action="store_true"); ap.add_argument("--mcp-vectors", action="store_true", help="slow: one vector per call via the MCP tool"); ap.add_argument("--limit", type=int)
    a = ap.parse_args()
    docs = [json.loads(l) for l in open(a.corpus) if l.strip()][:a.limit]
    events, ents, chunks, e_edges, c_edges = {}, {}, {}, [], []
    for d in docs:
        p = parse_doc(d); ev = p["event"]; events[ev["id"]] = {k: v for k, v in ev.items() if k != "id"}
        for e in p["entities"]:
            ents[e["id"]] = {"name": e["name"], "etype": e["etype"]}
            e_edges.append(("Event", ev["id"], "INVOLVES", "Entity", e["id"], {"role": e["role"]}))
        extra = {}
        if a.llm_extract:
            from ingest.extract import extract_entities
            extra = extract_entities(p["chunks"])
        for c in p["chunks"]:
            chunks[c["id"]] = {"text": c["text"], "doc_id": ev["id"], "title": p["title"]}
            e_edges.append(("Event", ev["id"], "HAS_CHUNK", "Chunk", c["id"], None))
            ids = set(c["mentions"])
            for e in extra.get(c["id"], []):
                ents.setdefault(e["id"], {"name": e["name"], "etype": e["etype"]}); ids.add(e["id"])
            for eid in ids: c_edges.append(("Chunk", c["id"], "MENTIONS", "Entity", eid, None))
    print(f"{len(events)} events, {len(chunks)} chunks, {len(ents)} entities")
    t0 = time.time(); cids, eids = list(chunks), list(ents)
    cvec = dict(zip(cids, embed([chunks[i]["text"] for i in cids])))
    evec = dict(zip(eids, embed([f"{ents[i]['name']} ({ents[i]['etype']})" for i in eids])))
    print(f"embeddings done in {time.time() - t0:.0f}s"); tg = get_tg(); tg.ensure_up()
    vecs = {"Entity": evec, "Chunk": cvec}
    # Bulk path (default): vector attribute `emb` goes in the SAME REST upsert as the vertex (50 per request).
    # The MCP tool upsert_vectors writes ONE vector per HTTP call (hours for 18k vectors), so it is only used with --mcp-vectors.
    for vt, rows in (("Event", events), ("Entity", ents), ("Chunk", chunks)):
        keys, size = list(rows), (50 if vt in vecs and not a.mcp_vectors else 100)
        for n, b in enumerate(batches(keys, size)):
            payload = {k: dict(rows[k], emb=vecs[vt][k]) if (vt in vecs and not a.mcp_vectors) else rows[k] for k in b}
            tg.upsert(vtx(vt, payload))
            if n % 20 == 0: print(f"  {vt}: {min((n + 1) * size, len(keys))}/{len(keys)}  [{time.time() - t0:.0f}s]", flush=True)
        print("loaded", vt, len(keys), "(with vectors)" if vt in vecs and not a.mcp_vectors else "")
    if a.mcp_vectors:
        for vt, vv in (("Entity", evec), ("Chunk", cvec)):
            keys = list(vv)
            for n, b in enumerate(batches(keys, 100)):
                tg.upsert_vectors(vt, {k: vv[k] for k in b})
                if n % 10 == 0: print(f"  vectors {vt}: {min((n + 1) * 100, len(keys))}/{len(keys)}", flush=True)
            print("loaded vectors", vt, len(keys))
    for b in batches(e_edges + c_edges, 500): tg.upsert(edg(b))
    print("loaded edges", len(e_edges) + len(c_edges))
    if hasattr(tg, "_vq"):                                           # functional check: each chunk must be its own nearest neighbour
        import random
        ids = list(cvec); sample = random.Random(1).sample(ids[1500:] or ids, 8)        # late chunks: not touched by the slow first run
        for attempt in range(6):                                     # the vector index builds in the background
            hits = 0
            for cid in sample:
                try: hits += cid in [r.get("id") for r in tg._vq("vec_chunks", cvec[cid], 3, "Top")]
                except Exception as e: print("search error:", str(e)[:200]); break
            print(f"check: {hits}/{len(sample)} chunks found by searching with their own vector")
            if hits == len(sample): break
            if attempt < 5: print("index may still be building; waiting 20s..."); time.sleep(20)
        else: print("!! vectors do not look stored. Send me this output. (Slow fallback: python -m ingest.run --corpus data/corpus.jsonl --mcp-vectors)")

if __name__ == "__main__": main()
