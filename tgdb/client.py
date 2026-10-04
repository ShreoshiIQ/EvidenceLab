"""REST backend (pyTigerGraph). Used for bulk upserts always, and for everything when TG_BACKEND=rest."""
import time
import config as C

def _dist(res):
    d = res.get("dist", {}); out = {}
    if isinstance(d, dict): return {str(k): v for k, v in d.items()}
    for it in d or []:
        k = it.get("key"); k = k.get("v_id") if isinstance(k, dict) else k
        out[str(k)] = it.get("value")
    return out

def flat(r):
    """Installed-query rows may arrive flat ({"id":..}) or wrapped ({"v_id":..,"attributes":{..}}); always return the flat form."""
    if not isinstance(r, dict): return r
    base = dict(r["attributes"]) if isinstance(r.get("attributes"), dict) else dict(r)
    if "id" not in base and "v_id" in r: base["id"] = r["v_id"]
    return base

def flat_rows(rows): return [flat(x) for x in rows] if isinstance(rows, list) else rows

def per_event_chunks(rows, per_event=1):
    key = lambda r: int(str(r["id"]).split("#")[-1])
    rows = sorted(rows, key=lambda r: (key(r), r["id"])); out, seen = [], {}
    for r in rows:
        if seen.get(r["doc_id"], 0) < per_event: out.append(r); seen[r["doc_id"]] = seen.get(r["doc_id"], 0) + 1
    return out

class TG:
    def __init__(self):
        import pyTigerGraph as tgpy
        self.conn = tgpy.TigerGraphConnection(host=C.TG_HOST, graphname=C.TG_GRAPH, username=C.TG_USER,
                                              password=C.TG_PASSWORD, gsqlSecret=C.TG_SECRET or None, tgCloud=True)
        if C.TG_SECRET: self.conn.getToken(C.TG_SECRET)

    def ensure_up(self, wait=120):
        """Savanna workspaces can auto-suspend: poll until reachable."""
        t0 = time.time()
        while True:
            try: self.conn.echo(); return True
            except Exception:
                if time.time() - t0 > wait: return False
                time.sleep(5)

    def _q(self, name, params):
        try: res = self.conn.runInstalledQuery(name, params, usePost=True)
        except TypeError: res = self.conn.runInstalledQuery(name, params)
        out = {}
        for d in res: out.update(d)
        return {k: flat_rows(v) for k, v in out.items()}

    def _ranked(self, name, qvec, k):
        res = self._q(name, {"qvec": qvec, "k": k}); d = _dist(res)
        rows = list(res.get("Top", []))
        rows.sort(key=lambda r: d.get(str(r.get("id")), 1e9))
        return rows

    def vec_chunks(self, qvec, k): return self._ranked("vec_chunks", qvec, k)
    def vec_entities(self, qvec, k): return self._ranked("vec_entities", qvec, k)
    def entity_by_ids(self, ids): return self._q("entity_by_ids", {"ids": list(ids)}).get("E", []) if ids else []
    def entities_by_type(self, etype): return self._q("entities_by_type", {"t": etype}).get("E", [])
    def entity_events(self, eid): return self._q("entity_events", {"eid": eid}).get("E", [])
    def get_chunks(self, ids): return self._q("get_chunks", {"ids": list(ids)}).get("C", []) if ids else []

    def event_chunks(self, eids, per_event=1):
        rows = self._q("event_chunks", {"eids": list(eids)}).get("C", []) if eids else []
        return per_event_chunks(rows, per_event)

    def expand_chunks(self, ids, lim):
        if not ids: return {"chunks": [], "entities": []}
        res = self._q("expand_chunks", {"ids": list(ids), "lim": lim})
        return {"chunks": res.get("Rel", []), "entities": res.get("Ents", [])}

    def upsert(self, payload): return self.conn.upsertData(payload)
    def gsql(self, text, graph=None): return self.conn.gsql(text)
    def add_vector_attribute(self, vtype, name, dim, metric="COSINE"):
        job = f"vec_{vtype}_{name}"
        return self.conn.gsql(f'USE GRAPH {C.TG_GRAPH}\nCREATE SCHEMA_CHANGE JOB {job} FOR GRAPH {C.TG_GRAPH} {{ ALTER VERTEX {vtype} ADD VECTOR ATTRIBUTE {name}(DIMENSION={dim}, METRIC="{metric}"); }}\nRUN SCHEMA_CHANGE JOB {job}')
    def upsert_vectors(self, vtype, vecs):
        return self.conn.upsertData({"vertices": {vtype: {k: {"emb": {"value": v}} for k, v in vecs.items()}}})
    def drain_log(self): return []
