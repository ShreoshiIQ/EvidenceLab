import os
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()
E = os.getenv
ROOT = Path(__file__).parent

# ---- LLM (single pinned model) ----
PROVIDER = E("PROVIDER", "gemini")   # gemini | openai
GEN_MODEL = E("GEN_MODEL", "gpt-4o-mini" if PROVIDER == "openai" else "gemini-3.5-flash-lite")
JUDGE_MODEL = E("JUDGE_MODEL", GEN_MODEL)
FREE_TIER_MODELS = {m.strip() for m in E("FREE_TIER_MODELS", "gpt-4o-mini,gpt-4.1-mini,gpt-4.1-nano" if PROVIDER == "openai" else "gemini-3.5-flash-lite,gemini-3.5-flash,gemini-3.6-flash,gemini-3.7-flash,gemini-3.8-flash,gemini-2.5-flash-lite,gemini-2.5-flash").split(",")}
GEMINI_THINKING = E("GEMINI_THINKING", "minimal")   # Gemini 3.x thinking level (minimal|low|medium|high) or "off" to send nothing
ENFORCE_FREE_TIER = E("ENFORCE_FREE_TIER", "1") == "1"
API_KEYS = [k.strip() for k in (E("API_KEYS") or E("GEMINI_API_KEYS") or E("OPENAI_API_KEYS") or "").split(",") if k.strip()]
GEMINI_API_KEYS = API_KEYS  # backwards-compatible alias
RPM_PER_KEY = int(E("RPM_PER_KEY", "10"))
WAIT_FOR_RESET = E("WAIT_FOR_RESET", "0") == "1"   # 1 = when every key is out for the day, sleep until quota resets instead of stopping
CACHE_PATH = E("CACHE_PATH", str(ROOT / "runs" / "llm_cache.sqlite"))

# ---- Embeddings ----
EMBED_MODEL = E("EMBED_MODEL", "BAAI/bge-small-en-v1.5")
EMBED_DIM = int(E("EMBED_DIM", "384"))

# ---- TigerGraph ----
TG_HOST, TG_GRAPH = E("TG_HOST", ""), E("TG_GRAPH", "OlympicsKG")
TG_BACKEND = E("TG_BACKEND", "mcp")   # mcp (TigerGraph MCP server) | rest (pyTigerGraph)
MCP_COMMAND = E("MCP_COMMAND", "tigergraph-mcp")
MCP_ARGS = [a for a in E("MCP_ARGS", "").split() if a]
TG_USER, TG_PASSWORD, TG_SECRET = E("TG_USER", "tigergraph"), E("TG_PASSWORD", ""), E("TG_SECRET", "")

# ---- Retrieval budgets (equal chunk budget for P1 and P2 => fair comparison) ----
RAG_TOPK = 10
GRAPH_SEEDS, GRAPH_EXPAND_LIMIT, GRAPH_MAX_EVENTS = 6, 4, 40
MAX_CONTEXT_CHARS, PER_ITEM_CHARS = 24000, 6000

# ---- Agent harness limits ----
AGENT_MAX_STEPS, AGENT_MAX_TOKENS, AGENT_STALL_LIMIT = 8, 40000, 2   # tool steps / tokens / stalled rounds
AGENT_MAX_ROUNDS, AGENT_BATCH = 5, 3   # orchestrator calls / actions planned per orchestrator call

# ---- ingest size caps (CPU embedding is the slow part). The infobox + lead paragraph are in the first chunks.
MAX_CHUNKS_EVENT = int(E("MAX_CHUNKS_EVENT", "6"))    # Olympic event pages
MAX_CHUNKS_OTHER = int(E("MAX_CHUNKS_OTHER", "2"))    # distractor pages (films, albums...): kept as retrieval noise, but cheap
