"""python -m tgdb.mcp_probe  -> verifies MCP + Savanna + vector search and prints RAW response shapes.
If a parser in tgdb/mcp_client.py (_ids/_find) does not match what you see here, adjust it."""
import json, random
import config as C
from tgdb import get_tg

tg = get_tg()
print("up:", tg.ensure_up(30))
if C.TG_BACKEND == "mcp":
    for name in ("list_vector_attributes", "get_vector_index_status", "get_vertex_count"):
        try: print(f"\n== {name}\n", json.dumps(tg.tool(name), indent=1)[:900])
        except Exception as e: print(f"\n== {name} FAILED:", e)
    qv = [random.random() for _ in range(C.EMBED_DIM)]
    try:
        raw = tg.tool("search_top_k_similarity", vertex_type="Chunk", vector_attribute="emb", query_vector=qv, top_k=3, return_vectors=False)
        print("\n== search_top_k_similarity RAW\n", json.dumps(raw, indent=1)[:1500])
        from tgdb.mcp_client import _ids; print("parsed ids:", _ids(raw.get("data")))
    except Exception as e: print("\nsearch FAILED:", e)
    import time
    print("\n== FAST path: installed query vec_chunks via run_installed_query")
    try:
        t = time.time(); rows = tg._vq("vec_chunks", qv, 3, "Top")
        print(f"returned {len(rows)} rows in {time.time() - t:.1f}s; ids:", [r.get("id") for r in rows])
        print("first row keys:", sorted(rows[0].keys()) if rows else None)
        raw = tg.tool("run_installed_query", query_name="vec_chunks", params={"qvec": qv, "k": 1}).get("data")
        print("raw data (trimmed):", json.dumps(raw)[:700])
        rows2 = tg.get_chunks([rows[0]["id"]]) if rows else []
        print("get_chunks round trip:", [(r.get("id"), (r.get("text") or "")[:50]) for r in rows2])
    except Exception as e: print("FAILED:", e, "\n-> install the queries: python -m tgdb.setup print queries  (paste into the Savanna Query Editor)")
    print("\nMCP calls:", tg.drain_log())
else: print("TG_BACKEND=rest; set TG_BACKEND=mcp to probe MCP")
