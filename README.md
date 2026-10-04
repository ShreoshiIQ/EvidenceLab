# TigerGraph GraphRAG Bench: RAG vs GraphRAG vs Agentic GraphRAG

## Links
| What | Link |
|---|---|
| TigerGraph MCP server (GitHub) | https://github.com/tigergraph/tigergraph-mcp |
| TigerGraph MCP on PyPI | https://pypi.org/project/pyTigerGraph-mcp/ (command: `tigergraph-mcp`) |
| TigerGraph Savanna (cloud) | https://savanna.tgcloud.io/  (tgcloud.io) |
| Savanna: connect AI tools with MCP | https://www.tigergraph.com/docs/savanna/main/get-started/connect-agent-mcp |
| TigerGraph docs home | https://www.tigergraph.com/docs/home/ |


## Hackathon criteria -> where it lives
| Criterion | What the repo does |
|---|---|
| Investigation accuracy (30%) | Deterministic infobox parsing (competitors, nations, date, venue, medalists) -> typed `Event` vertices; aggregation/comparison computed in code, not by the LLM; entity linking incl. bare years; date/title filters for multi-hop (venue + date -> winner). |
| Evidence quality & explainability (15%) | Every answer cites ledger ids (`E1`..); `citation_valid_rate` and `answer_supported_by_citations` are computed per question; each record has a readable `investigation_path` plus the raw trace; the dashboard drill-down shows path, trace and evidence. |
| Agentic effectiveness & efficiency (15%) | Orchestrator plans up to 3 chained actions per LLM call, re-plans when a step yields nothing, rule-based stopping (0 tokens) before any LLM sufficiency check, token/step/round budgets, strategy-change and stop-reason logging. |
| Design, engineering, code quality (15%) | Harness / orchestrator / specialised agents separation, single LLM gateway (key failover, cache, metering), MCP + REST backends behind one interface, `tests/` (offline, incl. a real `tigergraph-mcp` round trip), `Makefile`, Docker. |
| Innovation (15%) | Hybrid structured + text graph from messy flattened tables; code-executed aggregation with a completeness check (`unknown` counts) that lets the agent stop without an LLM; investigation-path view. |
| Presentation (10%) | Dashboard scorecard, accuracy-vs-tokens scatter, agentic behaviour tab, per-question drill-down, live "ask" panel. |

## Required metrics -> command
| Required item | Where |
|---|---|
| Same LLM in all 3 pipelines | `pipelines.run` raises if any call used a model other than `GEN_MODEL`; scorecard prints `PASS: single model`. (The judge may use `JUDGE_MODEL`; it is not a pipeline.) |
| Tokens per query (context / LLM input / LLM output / total) | scorecard, per pipeline |
| Latency (avg response seconds) | scorecard (`latency_adj_s`: cache hits are replaced by the original call time, so reruns stay honest; for the hidden run also use a fresh cache) |
| Avg retrieval and reasoning steps per question | scorecard (`retrieval_steps`, `reasoning_steps`; query embedding is not counted) |
| Retrieval methods used | scorecard (share of questions using vector_search, entity_linking, graph_traversal:*, document_retrieval) |
| Agentic: avg tokens and avg time per question | scorecard "Agentic GraphRAG" section |
| 50 hidden question results | `make hidden` -> `submission/hidden_results.json(.jsonl)` (answers, tokens, latency, steps, full trace per pipeline) |

```bash
make public            # 100 questions x 3 pipelines + reports/scorecard_public.{md,json} + per_question CSV
CACHE_PATH=runs/hidden_cache.sqlite make hidden   # fresh cache; run ONCE after freezing; writes submission/
make dashboard
make test
```

## Hidden-question output format
`make hidden` writes `submission/hidden_answers.jsonl` in the **same format as `eval_public.jsonl`** (agentic pipeline = primary system):
```json
{"qid": "eval-001", "question": "Who won the gold medal ...?", "qtype": "multi_hop", "answer": ["Chen Ding"]}
```
`answer` is always a list of strings (the model is asked for a JSON list). The same format is written per pipeline (`hidden_answers_rag|graphrag|agentic.jsonl`), and `hidden_results.json(.jsonl)` holds the raw outputs (tokens, latency, steps, full agentic trace). Unanswerable or failed questions get `["Not found in the corpus"]`.

## Constraints enforced in code
| Requirement | Where |
|---|---|
| Same model for all answer generation | `gateway/llm.py`: one pinned `GEN_MODEL`, allowlist `FREE_TIER_MODELS`, refuses anything else, no silent fallback model. `answer.py` is the single answer function used by all 3 pipelines. |
| Key handling | `API_KEYS=primary,backup2,...`: always uses the first key that still has quota; on a daily-quota 429 it marks that key dead until midnight Pacific (persisted in the cache DB) and moves to the next. |
| Vector DB = TigerGraph | `Chunk.emb` and `Entity.emb` are TigerGraph vector attributes (created with MCP tool `add_vector_attribute`, loaded with `upsert_vectors`). Search = MCP tool `search_top_k_similarity`. No FAISS/Chroma. RAG (P1) uses the same index. |
| TigerGraph Savanna | `TG_HOST` = your workspace URL, `TG_SECRET` = database secret. |
| TigerGraph MCP | `TG_BACKEND=mcp` (default): `tgdb/mcp_client.py` launches `tigergraph-mcp` and calls `tigergraph__search_top_k_similarity`, `tigergraph__run_installed_query`, `tigergraph__upsert_vectors`, `tigergraph__add_vector_attribute`, `tigergraph__gsql`. Every MCP call is logged per question in `mcp_tool_calls`. Bulk vertex/edge upserts use pyTigerGraph REST (batch loads, not agent tools). `TG_BACKEND=rest` bypasses MCP. |
| Deployment | `Dockerfile`, `docker-compose.yml`, FastAPI (`api/`), Streamlit (`dashboard/`). |
| Metrics | Every pipeline: context/input/output/total tokens, latency, chunks, citations. Agentic: per-step trace (agent, tool, rationale, latency, tokens), rounds, strategy changes, stop reason. |

## Setup of the MCP server (also used by Cursor / VS Code for development)
1. Savanna console: open your workspace, copy its URL (`TG_HOST`), and create a **database secret** (`TG_SECRET`).
2. `pip install -r requirements.txt` (installs `pyTigerGraph-mcp`; it needs `mcp<2`, which is pinned).
3. IDE: copy `editor_mcp/cursor.mcp.json` (Cursor: `~/.cursor/mcp.json`) or `editor_mcp/vscode.mcp.json` (`.vscode/mcp.json`), fill the three values, restart, then try the prompts in `editor_mcp/PROMPTS.md`. (Uses `uvx`; or set `"command": "tigergraph-mcp"` after pip install.)
4. The benchmark itself launches the same server on its own through `tgdb/mcp_client.py`; you only fill `.env`.

## Run order
```bash
python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt
cp .env.example .env                     # keys + Savanna values
python -m tgdb.setup schema              # graph + vertex/edge types + vector attributes (via MCP)
python -m tgdb.setup queries             # install GSQL queries
python -m ingest.run --corpus data/corpus.jsonl --limit 20   # small test, then without --limit
python -m tgdb.mcp_probe                 # PRINTS the raw MCP response shapes; confirms vector search works
python -m evaluation.run_eval --file data/eval_public.jsonl --split public --limit 5
python -m evaluation.run_eval --file data/eval_public.jsonl --split public
streamlit run dashboard/app.py
# freeze code/prompts, then ONCE:
python -m evaluation.run_eval --file data/eval_hidden.jsonl --split hidden
```

## Fairness design
Same chunks, embeddings, TigerGraph index, model, temperature 0, answer prompt, context cap and evidence-ID citations. P1 gets 10 chunks; P2 gets 6 seed + up to 4 expanded chunks (also 10) plus graph facts; P3 gets what its orchestrator retrieves under a token/step budget. Orchestrator, evaluator and reasoner tokens **are counted** in the agentic total. Judge tokens are excluded. Cached calls keep their original token counts.

## Things to verify on your Savanna workspace (not testable offline)
1. `python -m tgdb.mcp_probe`: if the parsed ids line is empty, adjust `_ids()` in `tgdb/mcp_client.py` to the raw shape printed above it.
2. Vector attribute creation and `search_top_k_similarity` need TigerGraph 4.2+ (TigerVector).
3. Vertex/edge upsert through `upsertData` (`ingest/run.py`). If rejected, switch to a loading job.
4. The GSQL in `tgdb/schema.gsql` / `queries.gsql` (`CREATE GRAPH`, `vectorSearch`). Adjust if your version complains.
5. The `run_installed_query` result shape: `_find()` looks for the printed key (`C`, `E`, `Rel`, `Ents`) anywhere in the payload.

## Deploy
Render / Cloud Run / Fly.io for the API (`docker build .`); the API container starts `tigergraph-mcp` as a subprocess. Streamlit Cloud or a second container for the dashboard (reads `runs/*.jsonl`). Set secrets as env vars. Resume the Savanna workspace before demos; `/health` and `/ask` report if it is unreachable.

## Accuracy, completeness and the dashboard
Public split (ground truth available) is scored three ways, all in `evaluation/run_eval.py`:
- **Accuracy**: PASS/FAIL. Exact/numeric match first (free), LLM judge only when needed (judge calls are not counted in pipeline tokens).
- **Completeness**: share of gold items found in the answer (deterministic, partial credit; handles fused team names).
- **Semantic similarity**: local bge embedding cosine, a BERTScore-style proxy (`--no-semantic` to skip).

`streamlit run dashboard/app.py` compares the three pipelines on accuracy, completeness, tokens, latency, steps and retrieval methods. The hidden split shows everything except accuracy, which the organizers grade.
