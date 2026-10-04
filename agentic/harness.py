"""Agent harness: owns state, tool registry, metering, evidence ledger, budgets, stopping rules.
The orchestrator LLM plans 1-3 chained actions per call; the harness runs them, and returns control to the
orchestrator immediately when a step yields nothing new (so strategy still adapts to evidence)."""
import json
from dataclasses import dataclass, field
import config as C
from answer import generate_answer
from common import Ledger, Trace
from gateway.llm import llm, jparse, current_meter
from agentic.agents import Tools, rule_sufficient

TOOLS = ["entity_linking", "similarity_search", "graph_traversal", "document_retrieval",
         "aggregation", "multi_hop_reasoning", "evidence_evaluation"]

ORCH_SYSTEM = """You are the orchestrator of a graph-RAG investigation. Decide what to do next from the current state.
Tools (agent: args):
- entity_linking: {"mentions": [optional strings]}  links question text to graph entities; ids look like "games:2018 winter olympics", "sport:biathlon".
- similarity_search: {"query": str, "target": "chunk"|"entity", "k": int}  use when no entity is named or linking found nothing.
- graph_traversal: ALWAYS include "op". {"op":"entity_events","entity_ids":[ids] or "$linked","mode":"intersect"|"union","date_contains":"15 to 22 August","title_contains":"singles"}
                   lists ALL events attached to the entities; each event line carries competitors, nations, date, venue and gold/silver/bronze,
                   so "who won / which country / when / where" questions can be answered from it directly. date_contains ignores the year.
                   Multi-hop example (venue + date -> winner): entity_linking, then graph_traversal intersect of the linked venue and games with date_contains.
                   A bare year in the question (e.g. 2004) is auto-linked to games:YYYY summer/winter olympics.
                   {"op":"adjacent_games","year":2016,"direction":"before"|"after","season":"summer"|"winter"} resolves TEMPORAL references
                   ("the Games held immediately before 2016"). The year named in such a question is NOT the edition you need: after adjacent_games,
                   "$linked" / "$games" refer to the resolved edition instead of the year-linked one.
                   Temporal example plan: entity_linking -> adjacent_games(year 2016, before, summer) -> entity_events(entity_ids "$linked", title_contains "men's 20 kilometres walk").
                   If the question names several dates for one event (e.g. "21 September 2000 (slow)22 September 2000 (fast)"), pass the WHOLE date text
                   in ONE date_contains string, not one call per date. For "event at <venue> on <date>" use entity_ids ["venue:<name>","games:<year> <season> olympics"],
                   mode intersect; use the venue name exactly as the question writes it.
                   title_contains matches whole words ("men's" does not match "women's").
                   {"op":"expand_chunks","chunk_ids":[ids],"limit":4} related chunks via shared entities.
- document_retrieval: {"event_ids":[..] | "from_state_events":true | "chunk_ids":[..], "per_event":1, "limit":8}  fetch supporting text.
- aggregation: {"field":"competitors"|"nations","op":"count|sum|avg|max|min|list","cmp":">|>=|<|<=|==","value":number,"title_contains":str}
               computed in code over events already fetched by graph_traversal. NEVER count or compare numbers yourself.
- multi_hop_reasoning: {}  (LLM call) decompose into remaining sub-questions. Use only for genuinely multi-hop questions.
- evidence_evaluation: {}  checks sufficiency (rules first, LLM only if needed).
- ANSWER: {}  stop and answer (also when the corpus cannot answer).
You may return up to 3 actions that chain without needing to inspect intermediate output, e.g.
entity_linking -> graph_traversal(entity_ids "$linked") -> aggregation. The harness runs them in order and hands control back
to you as soon as a step returns nothing new, so plan only what you are confident about. The harness stops automatically once an
aggregate over fully-known events exists. If a method returned nothing, switch method. Do not repeat actions.
Return JSON: {"actions":[{"agent":"...","args":{...},"addresses":"<info need>","rationale":"<1 sentence>"}]}
To stop: {"actions":[{"agent":"ANSWER","args":{},"rationale":"..."}]}"""

# the orchestrator sometimes names a graph_traversal operation as if it were an agent
OP_ALIASES = {"adjacent_games": "graph_traversal", "entity_events": "graph_traversal", "expand_chunks": "graph_traversal"}

def action_sig(agent, args, st):
    """Duplicate-detection key. aggregation / evidence_evaluation depend on what is already in state, so the same call is NOT a
    duplicate once more events or evidence have been fetched since it last ran."""
    sig = (agent, json.dumps(args, sort_keys=True))
    return sig + (len(st.events), len(st.ledger.items)) if agent in ("aggregation", "evidence_evaluation") else sig

@dataclass
class State:
    question: str
    ledger: Ledger = field(default_factory=Ledger)
    linked: dict = field(default_factory=dict)
    events: dict = field(default_factory=dict)
    open_q: list = field(default_factory=list)
    history: list = field(default_factory=list)
    last_eval: dict = None
    focus_games: str = None   # edition resolved by adjacent_games

def summarize(st, rnd, used, steps):
    L = [f"QUESTION: {st.question}",
         "LINKED: " + ("; ".join(f"{k} ({v['etype']}, conf {v['conf']})" for k, v in st.linked.items()) or "none"),
         f"EVENTS FETCHED INTO STATE: {len(st.events)}", "LEDGER:"]
    L += [f" [{e.id}] {e.kind} via {e.retrieved_by}: {e.content[:160]!r}" for e in st.ledger.items] or [" (empty)"]
    L.append("OPEN: " + ("; ".join(st.open_q) or "unknown/none"))
    if st.last_eval: L.append("LAST EVAL: " + json.dumps(st.last_eval))
    L.append("HISTORY: " + (" | ".join(f"{h['agent']}{json.dumps(h['args'])[:80]} -> +{h['new_evidence']}" + (f" ERROR: {h['error']}" if h.get("error") else "") for h in st.history) or "none"))
    L.append(f"BUDGET LEFT: rounds={C.AGENT_MAX_ROUNDS - rnd + 1}, tool steps={C.AGENT_MAX_STEPS - steps}, tokens={C.AGENT_MAX_TOKENS - used}")
    return "\n".join(L)

def run(question, tg):
    st, tr = State(question), Trace(); tools = Tools(tg, st)
    stop, detail, stall, changes, seen, nsteps = None, "", 0, 0, set(), 0
    for rnd in range(1, C.AGENT_MAX_ROUNDS + 1):
        used = current_meter().totals()["total"]
        if used > C.AGENT_MAX_TOKENS: stop, detail = "budget_tokens", f"{used} tokens"; break
        with tr.op("orchestrator", "llm", round=rnd) as r:
            t, _ = llm("orchestrator", summarize(st, rnd, used, nsteps), ORCH_SYSTEM, json_mode=True, max_out=500)
            plan = jparse(t, {}) or {}
            acts = (plan.get("actions") or ([plan] if plan.get("agent") else []))[:C.AGENT_BATCH]; r["plan"] = acts
        if not acts:
            stall += 1
            if stall >= C.AGENT_STALL_LIMIT: stop, detail = "no_progress", "orchestrator returned no valid plan"; break
            continue
        round_new = 0
        for act in acts:
            agent, args = act.get("agent"), dict(act.get("args") or {})
            if agent in OP_ALIASES: args.setdefault("op", agent); agent = OP_ALIASES[agent]
            if agent == "ANSWER": stop, detail = "orchestrator_answer", act.get("rationale", ""); break
            sig = action_sig(agent, args, st)
            if agent not in TOOLS or sig in seen:
                why = f"unknown agent '{agent}'; valid agents: {', '.join(TOOLS)}" if agent not in TOOLS else "duplicate action; choose a different one"
                st.history.append({"agent": str(agent), "args": args, "new_evidence": 0, "error": why}); break
            if nsteps >= C.AGENT_MAX_STEPS: stop, detail = "budget_steps", f"{nsteps} tool steps"; break
            seen.add(sig); nsteps += 1; prev = st.history[-1] if st.history else None; op = args.get("op", "")
            with tr.op(agent, op or agent, addresses=act.get("addresses"), rationale=act.get("rationale"), args=args) as r:
                err = None
                try: outs = getattr(tools, agent)(args)
                except Exception as e: outs, err = [], str(e)[:200]; r["error"] = err
                new = [e for e in (st.ledger.add(k, c, p, agent, rnd) for k, c, p in outs) if e]
                r["new_evidence"] = [e.id for e in new]; r["n_chunks"] = sum(e.kind == "chunk" for e in new)
                r["strategy_change"] = bool(prev and prev["new_evidence"] == 0 and prev["agent"] not in ("multi_hop_reasoning",)
                                            and (prev["agent"], prev.get("op", "")) != (agent, op))
                if r["strategy_change"]: changes += 1; r["strategy_change_reason"] = act.get("rationale", "")
            st.history.append({"agent": agent, "op": op, "args": args, "new_evidence": len(new), "error": err}); round_new += len(new)
            if agent == "evidence_evaluation" and (st.last_eval or {}).get("sufficient"):
                stop, detail = "evaluator_sufficient", f"method={st.last_eval.get('method', 'llm')}"; break
            if rule_sufficient(st): break   # evidence already settles it: skip the rest of the batch (the check after this loop records the stop)
            if not new and agent not in ("multi_hop_reasoning", "evidence_evaluation"): break   # yielded nothing: re-plan now
        if stop: break
        if rule_sufficient(st): stop, detail = "rule_sufficient", "evidence settled by rules (aggregate over known events, or a filtered lookup resolved to 1-3 events)"; break
        stall = 0 if round_new else stall + 1
        if stall >= C.AGENT_STALL_LIMIT: stop, detail = "no_progress", f"{stall} rounds without new evidence"; break
    stop = stop or "budget_rounds"
    with tr.op("answer_generation", "llm"): ans = generate_answer(question, st.ledger.items)
    ops = [s for s in tr.steps if s["agent"] not in ("orchestrator", "answer_generation")]
    meta = {"steps": len(ops), "rounds": sum(s["agent"] == "orchestrator" for s in tr.steps), "stop_reason": stop, "stop_detail": detail,
            "strategy_changes": changes, "agents_used": sorted({s["agent"] for s in ops}), "tools_used": sorted({s["tool"] for s in ops})}
    return {**ans, "trace": tr.steps, "evidence": st.ledger.items, "meta": meta}
