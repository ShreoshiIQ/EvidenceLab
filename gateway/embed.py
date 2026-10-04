"""Local embeddings (free). Disk cache: a text is embedded once per model, so re-running ingest or eval is fast."""
import hashlib, sqlite3, struct, sys
from functools import lru_cache
from pathlib import Path
import config as C

@lru_cache(1)
def _model():
    import os, torch
    torch.set_num_threads(os.cpu_count() or 4)            # use every core
    from sentence_transformers import SentenceTransformer
    m = SentenceTransformer(C.EMBED_MODEL); m.max_seq_length = 256   # chunks are <=~180 words; 512 only costs time
    return m

@lru_cache(1)
def _db():
    p = Path(getattr(C, "CACHE_PATH", "cache/llm.sqlite")).with_name("embeddings.sqlite"); p.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(str(p), check_same_thread=False); db.execute("CREATE TABLE IF NOT EXISTS e (k TEXT PRIMARY KEY, v BLOB)"); return db

def _key(t): return hashlib.sha1((C.EMBED_MODEL + "\0" + t).encode("utf-8")).hexdigest()
def _pack(v): return struct.pack(f"{len(v)}f", *v)
def _unpack(b): return list(struct.unpack(f"{len(b) // 4}f", b))

def embed(texts, query=False, progress=None):
    if query and "bge" in C.EMBED_MODEL.lower():
        texts = ["Represent this sentence for searching relevant passages: " + t for t in texts]
    db, keys = _db(), [_key(t) for t in texts]
    have = {}
    for i in range(0, len(keys), 500):
        part = keys[i:i + 500]
        for k, v in db.execute(f"SELECT k, v FROM e WHERE k IN ({','.join('?' * len(part))})", part): have[k] = _unpack(v)
    todo = [i for i, k in enumerate(keys) if k not in have]
    show = progress if progress is not None else len(todo) > 64
    if show: print(f"embedding {len(todo)} new texts ({len(texts) - len(todo)} cached)...", file=sys.stderr, flush=True)
    for s in range(0, len(todo), 256):                       # chunks of 256 so progress is visible and work is saved as we go
        idx = todo[s:s + 256]
        vecs = _model().encode([texts[i] for i in idx], normalize_embeddings=True, batch_size=32).tolist()
        for i, v in zip(idx, vecs): have[keys[i]] = v; db.execute("INSERT OR REPLACE INTO e VALUES (?,?)", (keys[i], _pack(v)))
        db.commit()
        if show: print(f"  {min(s + 256, len(todo))}/{len(todo)}", file=sys.stderr, flush=True)
    return [have[k] for k in keys]
