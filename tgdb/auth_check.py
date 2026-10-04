"""python -m tgdb.auth_check   -- finds out why 'User authentication failed'. Never prints the secret."""
import requests
import config as C

def show(label, r):
    body = r.text[:300].replace(C.TG_SECRET or "\0", "***")
    print(f"{label}: HTTP {r.status_code} {body}")

def main():
    host = (C.TG_HOST or "").rstrip("/")
    print("host        :", host)
    print("graph       :", C.TG_GRAPH)
    s = C.TG_SECRET or ""
    print("secret      : length", len(s), "| has spaces/quotes:", any(c in s for c in " \"'\t\r\n"), "| starts/ends:", repr(s[:2]), repr(s[-2:]))
    if not host.startswith("https://") or ":" in host.split("//", 1)[1]: print("!! TG_HOST should be https://<name>.i.tgcloud.io with no port and no path")
    if not s: print("!! TG_SECRET is empty"); return
    tok = None
    for label, url, kw in [
        ("POST /gsql/v1/tokens (graph)", f"{host}/gsql/v1/tokens", {"json": {"secret": s, "graph": C.TG_GRAPH}}),
        ("POST /gsql/v1/tokens (no graph)", f"{host}/gsql/v1/tokens", {"json": {"secret": s}}),
        ("GET  /restpp/requesttoken", f"{host}/restpp/requesttoken", {"params": {"secret": s}}),
    ]:
        try:
            r = (requests.post if label.startswith("POST") else requests.get)(url, timeout=30, **kw); show(label, r)
            if r.ok:
                j = r.json(); tok = j.get("token") or (j.get("results") or {}).get("token")
                if tok: print("   -> token received"); break
        except Exception as e: print(label, "ERROR", type(e).__name__, str(e)[:200])
    if tok:
        h = {"Authorization": f"Bearer {tok}"}
        for label, url in [("restpp echo", f"{host}/restpp/echo"), ("list vertices", f"{host}/restpp/graph/{C.TG_GRAPH}/vertices/Chunk?limit=1")]:
            try: show(label, requests.get(url, headers=h, timeout=30))
            except Exception as e: print(label, "ERROR", str(e)[:200])
    else:
        print("\nNo token. Likely causes: wrong/expired secret, alias pasted instead of the secret value, wrong TG_HOST, or the secret belongs to a different database.")

if __name__ == "__main__": main()
