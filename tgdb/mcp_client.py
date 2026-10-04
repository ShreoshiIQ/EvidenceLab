"""TigerGraph access THROUGH the TigerGraph MCP server (tigergraph-mcp, stdio) -- https://github.com/tigergraph/tigergraph-mcp
  vector search : tigergraph__search_top_k_similarity   graph queries : tigergraph__run_installed_query
  vectors       : tigergraph__upsert_vectors            schema        : tigergraph__add_vector_attribute / tigergraph__gsql
Bulk vertex/edge upserts still use pyTigerGraph (REST): they are batch loads, not agent tools."""
import asyncio, json, re, threading, time
from contextlib import AsyncExitStack
import config as C
from tgdb.client import TG, per_event_chunks, flat_rows

_BLOCK = re.compile(r"```json\s*(.*?)\s*```", re.S)   # MCP replies = a ```json block followed by markdown commentary

def _jwt_from_secret():
    """Secret -> JWT (POST /gsql/v1/tokens, lasts ~7 days). Works for SSO accounts: no password needed."""
    import requests
    r = requests.post(C.TG_HOST.rstrip("/") + "/gsql/v1/tokens", json={"secret": C.TG_SECRET, "graph": C.TG_GRAPH}, timeout=30)
    tok = (r.json().get("token") if r.ok else None) if r.headers.get("content-type", "").startswith("application/json") else None
    if not tok: raise RuntimeError(f"Could not get a token from the secret (HTTP {r.status_code}). Run: python -m tgdb.auth_check")
    return tok

def parse_reply(txt):
    """MCP reply text -> dict, or None when unparseable."""
    mm = _BLOCK.search(txt)
    try: j = json.loads(mm.group(1) if mm else txt.strip())
    except Exception: return None
    return j if isinstance(j, dict) else None

def _find(o, key):
    if isinstance(o, dict):
        if key in o: return o[key]
        for v in o.values():
            r = _find(v, key)
            if r is not None: return r
    elif isinstance(o, list):
        for v in o:
            r = _find(v, key)
            if r is not None: return r
    return None

def _ids(o):
    """Ordered vertex ids from a similarity-search payload (shape-tolerant; verify with `python -m tgdb.mcp_probe`)."""
    out = []
    def walk(x):
        if isinstance(x, dict):
            for k in ("vertex_id", "v_id", "id"):
                if isinstance(x.get(k), (str, int)): out.append(str(x[k])); return
            for v in x.values(): walk(v)
        elif isinstance(x, list):
            for v in x: walk(v)
    walk(o); return out

def _ordered(rows, ids):
    m = {str(r["id"]): r for r in rows}; return [m[i] for i in ids if i in m]

class MCPTG:
    def __init__(self):
        self.loop = asyncio.new_event_loop()
        threading.Thread(target=self.loop.run_forever, daemon=True).start()
        self.log, self._rest = [], None
        self._run(self._start())

    def _run(self, coro, timeout=300): return asyncio.run_coroutine_threadsafe(coro, self.loop).result(timeout)

    async def _start(self):
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client
        env = {"TG_HOST": C.TG_HOST, "TG_GRAPHNAME": C.TG_GRAPH, "TG_TGCLOUD": "true"}
        if C.TG_SECRET:
            env["TG_SECRET"] = C.TG_SECRET
            env["TG_JWT_TOKEN"] = _jwt_from_secret()   # fetched here so the MCP server never falls back to the default tigergraph/tigergraph login
        else: env.update(TG_USERNAME=C.TG_USER, TG_PASSWORD=C.TG_PASSWORD)
        self.stack = AsyncExitStack()
        r, w = await self.stack.enter_async_context(stdio_client(StdioServerParameters(command=C.MCP_COMMAND, args=C.MCP_ARGS, env=env)))
        self.session = await self.stack.enter_async_context(ClientSession(r, w)); await self.session.initialize()

    def tool(self, name, **args):
        t = time.time()
        res = self._run(self.session.call_tool("tigergraph__" + name, arguments={k: v for k, v in args.items() if v is not None}))
        txt = "".join(getattr(c, "text", "") or "" for c in res.content)
        j = parse_reply(txt)
        ok = (not res.isError) and isinstance(j, dict) and j.get("success") is True
        self.log.append({"tool": name, "latency_s": round(time.time() - t, 3), "ok": ok})
        if not ok:   # fail loudly: never treat an unparseable or error reply as an empty result
            why = (j or {}).get("error") or (j or {}).get("summary") or txt[:300]
            raise RuntimeError(f"MCP tigergraph__{name} failed: {why}")
        return j

    def drain_log(self): l, self.log = self.log, []; return l

    def ensure_up(self, wait=120):
        t0 = time.time()
        while True:
            try: self.tool("get_vertex_count"); return True   # graph-scoped call (a graph token is not allowed to list_graphs)
            except Exception:
                if time.time() - t0 > wait: return False
                time.sleep(5)

    def _iq(self, name, params, key):
        return flat_rows(_find(self.tool("run_installed_query", query_name=name, params=params).get("data"), key) or [])

    def _vsearch(self, vtype, qvec, k):
        """SLOW fallback: the MCP search tool creates + installs a temporary query on EVERY call (~30 s on Savanna)."""
        j = self.tool("search_top_k_similarity", vertex_type=vtype, vector_attribute="emb", query_vector=qvec, top_k=k, return_vectors=False)
        return _ids(j.get("data"))

    def _vq(self, name, qvec, k, key):
        """FAST path: the installed query (vectorSearch = TigerVector) run through the MCP tool run_installed_query. Returns rows nearest-first."""
        data = self.tool("run_installed_query", query_name=name, params={"qvec": list(qvec), "k": int(k)}).get("data")
        rows = flat_rows(_find(data, key) or [])
        d = _find(data, "dist")
        if isinstance(d, dict) and d:
            def dist(r):
                for kk in (str(r.get("id")), "Chunk:" + str(r.get("id")), "Entity:" + str(r.get("id"))):
                    if kk in d: return d[kk]
                return float("inf")
            if any(dist(r) != float("inf") for r in rows): rows = sorted(rows, key=dist)
        return rows

    # ---- same interface as tgdb.client.TG ----
    def _vec(self, vtype, qname, key, getter, qvec, k):
        if not getattr(self, "_slow_vec", False):
            try: return self._vq(qname, qvec, k, key)
            except Exception as e:
                self._slow_vec = True
                print(f"[tgdb] installed query {qname} failed ({str(e)[:160]}); falling back to the slow MCP search tool. Run: python -m tgdb.setup print queries", flush=True)
        ids = self._vsearch(vtype, qvec, k); return _ordered(getter(ids), ids)
    def vec_chunks(self, qvec, k): return self._vec("Chunk", "vec_chunks", "Top", self.get_chunks, qvec, k)
    def vec_entities(self, qvec, k): return self._vec("Entity", "vec_entities", "Top", self.entity_by_ids, qvec, k)
    def get_chunks(self, ids): return self._iq("get_chunks", {"ids": list(ids)}, "C") if ids else []
    def entity_by_ids(self, ids): return self._iq("entity_by_ids", {"ids": list(ids)}, "E") if ids else []
    def entities_by_type(self, etype): return self._iq("entities_by_type", {"t": etype}, "E")
    def entity_events(self, eid): return self._iq("entity_events", {"eid": eid}, "E")
    def event_chunks(self, eids, per_event=1):
        return per_event_chunks(self._iq("event_chunks", {"eids": list(eids)}, "C"), per_event) if eids else []
    def expand_chunks(self, ids, lim):
        if not ids: return {"chunks": [], "entities": []}
        d = self.tool("run_installed_query", query_name="expand_chunks", params={"ids": list(ids), "lim": lim}).get("data")
        return {"chunks": flat_rows(_find(d, "Rel") or []), "entities": flat_rows(_find(d, "Ents") or [])}

    # ---- setup / ingestion helpers ----
    def gsql(self, text, graph=None): return self.tool("gsql", command=text, graph_name=graph)
    def add_vector_attribute(self, vtype, name, dim, metric="COSINE"):
        return self.tool("add_vector_attribute", vertex_type=vtype, vector_name=name, dimension=dim, metric=metric)
    def upsert_vectors(self, vtype, vecs):
        return self.tool("upsert_vectors", vertex_type=vtype, vector_attribute="emb",
                         vectors=[{"vertex_id": k, "vector": v} for k, v in vecs.items()])
    def upsert(self, payload):
        self._rest = self._rest or TG(); return self._rest.upsert(payload)
