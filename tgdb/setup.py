"""python -m tgdb.setup schema | queries     (needs the DB user password for DDL)\n   python -m tgdb.setup print schema | queries  (prints GSQL to paste into the Savanna Query Editor; no password)"""
import sys
from pathlib import Path
import config as C
from tgdb import get_tg

def render(name):
    return (Path(__file__).parent / name).read_text().replace("{{GRAPH}}", C.TG_GRAPH).replace("{{DIM}}", str(C.EMBED_DIM))

if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "schema"
    if what == "print":   # no connection, no password: paste the output into the Savanna Query Editor
        print(render({"schema": "schema_existing_graph.gsql", "queries": "queries.gsql"}[sys.argv[2]])); sys.exit(0)
    tg = get_tg(); tg.ensure_up()
    if what == "schema":
        print(tg.gsql(render("schema.gsql")))                          # graph + vertex/edge types
        for vt in ("Chunk", "Entity"):                                    # vector attributes (MCP tool add_vector_attribute)
            print(tg.add_vector_attribute(vt, "emb", C.EMBED_DIM, "COSINE"))
    else:
        print(tg.gsql(render("queries.gsql"), graph=C.TG_GRAPH))
