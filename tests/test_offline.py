"""Offline tests (no Gemini, no Savanna): python -m pytest -q tests"""
import json, shutil
import pytest, requests
import config as C
import gateway.llm as G
import agentic.agents as A, agentic.harness as H, agentic.linking as L, answer as ANS
from common import Evidence, grounding, investigation_path, step_stats

L.embed = A.embed = lambda t, query=False: [[0.0] * 384 for _ in t]   # no model download

# ---------------------------------------------------------------- parsing
def test_parser_event_attributes():
    from ingest.parse import parse_doc
    d = {"doc_id": "Q1", "title": "Tennis at the 2004 Summer Olympics – Men's singles",
         "text": "[Infobox Olympic event]\n  venue: Olympic Tennis Centre\n  date: 15 to 22 August\n  competitors: 64\n  nations: 30\n"
                 "  gold: Nicolás MassúAndré Agassi\n  goldNOC: CHI\n\nBody text."}
    e = parse_doc(d)["event"]
    assert e["venue"] == "Olympic Tennis Centre" and e["date"] == "15 to 22 August 2004" and e["competitors"] == 64
    assert e["gold"] == "Nicolás MassúAndré Agassi"   # stored exactly as in the source (fused), as the ground truth is

# ---------------------------------------------------------------- aggregation
class FakeState:
    def __init__(self, comps):
        self.events = {f"Q{i}": {"id": f"Q{i}", "title": f"Biathlon {i}", "competitors": c, "nations": 9} for i, c in enumerate(comps)}
def test_aggregation_count_in_code():
    txt = A.Tools(None, FakeState([60, 74, 87, 75, 70, 90, 73, 80])).aggregation({"field": "competitors", "op": "count", "cmp": ">", "value": 73})[0][1]
    assert "result=5" in txt and "0 unknown" in txt

def test_judge_numeric():
    from evaluation.judge import judge
    assert judge("q", ["5"], "5")[0] == "PASS" and judge("q", ["5"], "6")[0] == "FAIL"

# ---------------------------------------------------------------- gateway: primary-first failover
def test_primary_first_failover(monkeypatch, tmp_path):
    monkeypatch.setattr(C, "API_KEYS", ["k1", "k2", "k3"]); monkeypatch.setattr(C, "CACHE_PATH", str(tmp_path / "c.sqlite"))
    monkeypatch.setattr(C, "RPM_PER_KEY", 100); monkeypatch.setattr(C, "PROVIDER", "gemini"); monkeypatch.setattr(C, "ENFORCE_FREE_TIER", False)
    calls = []
    class R:
        def __init__(s, c, b): s.status_code, s._b, s.text = c, b, json.dumps(b)
        def json(s): return s._b
    def post(url, headers=None, json=None, timeout=0):
        k = headers["x-goog-api-key"]; calls.append(k)
        if k == "k1" and len(calls) > 3: return R(429, {"error": {"message": "GenerateRequestsPerDayPerProjectPerModel"}})
        if k == "k2" and calls.count("k2") > 2: return R(429, {"error": {"message": "quota PerDay"}})
        return R(200, {"candidates": [{"content": {"parts": [{"text": "{}"}]}}], "usageMetadata": {"promptTokenCount": 10, "candidatesTokenCount": 2}})
    monkeypatch.setattr(requests, "post", post)
    gw = G.Gateway()
    for i in range(10): gw.call("t", f"p{i}", json_mode=True)
    firsts = [k for i, k in enumerate(calls) if i == 0 or calls[i - 1] != k]
    assert firsts == ["k1", "k2", "k3"]                      # strict priority order, never back to a dead key
    assert Gateway_dead_persisted(tmp_path)

def Gateway_dead_persisted(tmp_path):
    import sqlite3; return sqlite3.connect(str(tmp_path / "c.sqlite")).execute("select count(*) from dead").fetchone()[0] == 2

# ---------------------------------------------------------------- agent loop (scripted LLM + fake graph)
def fake_llm_factory(script):
    def f(role, prompt, system="", json_mode=False, model=None, max_out=1024):
        rec = {"role": role, "model": C.GEN_MODEL, "in": 100, "out": 20, "cached": False, "latency_s": 0, "orig_latency_s": 0}
        G.current_meter().calls.append(rec); return script.pop(0), rec
    return f

EVENTS = {
    "sport:biathlon": [{"id": f"B{i}", "title": f"Biathlon event {i} at the 2018 Winter Olympics", "competitors": c, "nations": 20, "date": "", "venue": "", "gold": "", "silver": "", "bronze": ""}
                       for i, c in enumerate([60, 74, 87, 75, 70, 90, 73, 80, 88, 50, 68])],
}
EVENTS["games:2018 winter olympics"] = EVENTS["sport:biathlon"]
TENNIS = [{"id": "T1", "title": "Tennis at the 2004 Summer Olympics – Men's singles", "competitors": 64, "nations": 30, "date": "15 to 22 August", "venue": "Olympic Tennis Centre", "gold": "Nicolás Massú (CHI)", "silver": "Mardy Fish (USA)", "bronze": "Fernando González (CHI)"},
          {"id": "T2", "title": "Tennis at the 2004 Summer Olympics – Women's singles", "competitors": 64, "nations": 30, "date": "16 to 21 August", "venue": "Olympic Tennis Centre", "gold": "Justine Henin (BEL)", "silver": "", "bronze": ""}]
EVENTS["venue:olympic tennis centre"] = TENNIS; EVENTS["games:2004 summer olympics"] = TENNIS

class FakeTG:
    known = {"games:2018 winter olympics", "sport:biathlon", "venue:olympic tennis centre", "games:2004 summer olympics"}
    def entity_by_ids(s, ids): return [{"id": i, "name": i.split(":")[1], "etype": i.split(":")[0].title()} for i in ids if i in s.known]
    def vec_entities(s, q, k): return []
    def entity_events(s, eid): return EVENTS.get(eid, [])
    def drain_log(s): return []

def test_agent_biathlon_count(monkeypatch):
    from pipelines import run
    script = [json.dumps({"actions": [{"agent": "entity_linking", "args": {}, "rationale": "link"},
                                      {"agent": "graph_traversal", "args": {"op": "entity_events", "entity_ids": "$linked", "mode": "intersect"}, "rationale": "enumerate"},
                                      {"agent": "aggregation", "args": {"field": "competitors", "op": "count", "cmp": ">", "value": 73}, "rationale": "count"}]}),
              json.dumps({"answer": "6", "citations": ["E3"], "reasoning": "x"})]
    f = fake_llm_factory(script)
    for m in (H, ANS, A): monkeypatch.setattr(m, "llm", f)
    r = run("agentic", "How many biathlon events at the 2018 Winter Olympics had more than 73 competitors?", FakeTG())
    assert r["answer"] == "6" and r["stop_reason"] == "rule_sufficient" and r["tokens"]["llm_calls"] == 2
    assert r["retrieval_steps"] == 2 and r["reasoning_steps"] == 3          # link+traverse | orchestrator+aggregation+answer
    assert r["retrieval_methods"] == ["entity_linking", "graph_traversal:entity_events"]
    assert r["citation_valid_rate"] == 1.0 and r["answer_supported_by_citations"] is True
    assert r["models_used"] == [C.GEN_MODEL] and r["investigation_path"]

def test_multi_hop_venue_date_to_winner(monkeypatch):
    from pipelines import run
    q = "Who won the gold medal in the event held at Olympic Tennis Centre on 15 to 22 August 2004?"
    script = [json.dumps({"actions": [{"agent": "entity_linking", "args": {}, "rationale": "link venue + year"},
                                      {"agent": "graph_traversal", "args": {"op": "entity_events", "entity_ids": "$linked", "mode": "intersect", "date_contains": "15 to 22 August 2004"}, "rationale": "filter by date"}]}),
              json.dumps({"answer": "Nicolás Massú", "citations": ["E3"], "reasoning": "x"})]
    f = fake_llm_factory(script)
    for m in (H, ANS, A): monkeypatch.setattr(m, "llm", f)
    r = run("agentic", q, FakeTG())
    ev2 = [e for e in r["evidence"] if e["content"].startswith("Events (")][0]["content"]   # E1,E2 = the two linked entities (venue, 2004 games)
    assert "Events (1)" in ev2 and "Nicolás Massú" in ev2 and "Justine Henin" not in ev2   # date filter isolated one event
    assert r["answer"] == "Nicolás Massú" and r["stop_reason"] == "rule_sufficient"   # filtered lookup resolved to 1 event: no extra orchestrator call
    assert r["retrieval_steps"] == 2 and r["answer_supported_by_citations"] is True

# ---------------------------------------------------------------- shared helpers
def test_step_stats_and_grounding():
    tr = [{"agent": "embedding", "tool": "bge"}, {"agent": "similarity_search", "tool": "vec_chunks"}, {"agent": "answer_generation", "tool": "llm"}]
    s = step_stats(tr); assert s["retrieval_steps"] == 1 and s["reasoning_steps"] == 1 and s["retrieval_methods"] == ["vector_search"]
    ev = [Evidence("E1", "chunk", "Gold: Rudolf Dombi", {}, "x", 1)]
    assert grounding("Rudolf Dombi", ["E1"], ev) == {"citation_valid_rate": 1.0, "answer_supported_by_citations": True}
    assert grounding("Someone Else", ["E1", "E9"], ev) == {"citation_valid_rate": 0.5, "answer_supported_by_citations": False}
    assert investigation_path(tr)

# ---------------------------------------------------------------- MCP reply parsing + real server round trip
def test_mcp_reply_parsing():
    from tgdb.mcp_client import parse_reply, _find, _ids
    txt = '```json\n' + json.dumps({"success": True, "data": {"results": [{"C": [{"id": "Q1#0"}]}]}}) + '\n```\n\n**done** extra markdown'
    j = parse_reply(txt); assert j["success"] is True and _find(j["data"], "C") == [{"id": "Q1#0"}]
    assert parse_reply("not json at all") is None
    assert _ids({"results": [{"vertex_id": "a"}, {"vertex_id": "b"}]}) == ["a", "b"]

@pytest.mark.skipif(shutil.which("tigergraph-mcp") is None, reason="tigergraph-mcp not installed")
def test_mcp_server_roundtrip_errors_are_loud(monkeypatch):
    monkeypatch.setattr(C, "TG_HOST", "https://example.invalid"); monkeypatch.setattr(C, "TG_SECRET", "x"); monkeypatch.setattr(C, "TG_GRAPH", "G")
    import tgdb.mcp_client as mc
    monkeypatch.setattr(mc, "_jwt_from_secret", lambda: "x")   # no network: the JWT fetch would fail on the fake host before the test starts
    from tgdb.mcp_client import MCPTG
    tg = MCPTG()
    names = {t.name for t in tg._run(tg.session.list_tools()).tools}
    for n in ("list_graphs", "gsql", "add_vector_attribute", "upsert_vectors", "search_top_k_similarity", "run_installed_query"):
        assert "tigergraph__" + n in names
    assert tg.ensure_up(wait=1) is False
    with pytest.raises(RuntimeError): tg.vec_chunks([0.1] * 384, 3)       # an unreachable DB must raise, never look like "no results"

# ---------------------------------------------------------------- scorecard
def test_scorecard_all_required_metrics(tmp_path):
    from evaluation.metrics import load_runs, scorecard, to_markdown
    def rec(p, qid, tot, lat, verdict, **kw):
        return {"pipeline": p, "qid": qid, "qtype": "multi_hop", "answer": "a", "verdict": verdict, "models_used": [C.GEN_MODEL],
                "tokens": {"context": 100, "input": tot - 10, "output": 10, "total": tot, "llm_calls": 1}, "latency_s": lat, "latency_adj_s": lat,
                "retrieval_steps": 1, "reasoning_steps": 1, "retrieval_methods": ["vector_search"], "n_chunks": 5, "n_citations": 1,
                "citation_valid_rate": 1.0, "answer_supported_by_citations": True, **kw}
    for p, tot in (("rag", 1000), ("graphrag", 1500), ("agentic", 4000)):
        with open(tmp_path / f"public_{p}.jsonl", "w") as f:
            for i in range(4):
                extra = {"rounds": 2, "strategy_changes": i % 2, "stop_reason": "rule_sufficient", "tools_used": ["graph_traversal"], "agents_used": ["aggregation"]} if p == "agentic" else {}
                f.write(json.dumps(rec(p, f"q{i}", tot, 2.0, "PASS" if i < 3 else "FAIL", **extra)) + "\n")
    sc = scorecard(load_runs("public", str(tmp_path)))
    assert sc["same_model_across_pipelines"] and sc["pipelines"]["rag"]["accuracy"] == 0.75
    assert sc["pipelines"]["agentic"]["tokens_per_question"]["total"] == 4000 and sc["agentic"]["strategy_change_rate"] == 0.5
    md = to_markdown(sc, "t"); assert "Avg retrieval steps" in md and "PASS: single model" in md


# ---------------------------------------------------------------- temporal question (pub-002) + word-boundary title filter
def _ev(i, title, gold): return {"id": i, "title": title, "competitors": 60, "nations": 30, "date": "", "venue": "", "gold": gold, "silver": "", "bronze": ""}
M12, W12 = _ev("Q1050909", "Athletics at the 2012 Summer Olympics – Men's 20 kilometres walk", "Chen Ding (CHN)"), _ev("Q2", "Athletics at the 2012 Summer Olympics – Women's 20 kilometres walk", "Elena Lashmanova (RUS)")
M16, W16 = _ev("Q3", "Athletics at the 2016 Summer Olympics – Men's 20 kilometres walk", "Wang Zhen (CHN)"), _ev("Q4", "Athletics at the 2016 Summer Olympics – Women's 20 kilometres walk", "Liu Hong (CHN)")

class TemporalTG(FakeTG):
    known = {"sport:athletics", "games:2016 summer olympics"}
    def entity_events(s, eid):
        return {"sport:athletics": [M12, W12, M16, W16], "games:2012 summer olympics": [M12, W12], "games:2016 summer olympics": [M16, W16]}.get(eid, [])
    def entities_by_type(s, t):
        return [{"id": f"games:{y} {se} olympics", "name": f"{y} {se} olympics", "etype": "Games"} for y, se in ((2008, "summer"), (2010, "winter"), (2012, "summer"), (2014, "winter"), (2016, "summer"))]

def test_temporal_edition_before_year(monkeypatch):
    from pipelines import run
    q = "Who won the gold medal in the men's 20 kilometres walk athletics event at the Summer Olympics held immediately before 2016?"
    script = [json.dumps({"actions": [{"agent": "entity_linking", "args": {}, "rationale": "link sport and year"},
                                      {"agent": "graph_traversal", "args": {"op": "adjacent_games", "year": 2016, "direction": "before", "season": "summer"}, "rationale": "resolve 'immediately before 2016'"},
                                      {"agent": "graph_traversal", "args": {"op": "entity_events", "entity_ids": "$linked", "mode": "intersect", "title_contains": "men's 20 kilometres walk"}, "rationale": "find the event"}]}),
              json.dumps({"answer": ["Chen Ding"], "citations": ["E4"], "reasoning": "x"})]
    f = fake_llm_factory(script)
    for m in (H, ANS, A): monkeypatch.setattr(m, "llm", f)
    r = run("agentic", q, TemporalTG())
    timeline = [e["content"] for e in r["evidence"] if e["content"].startswith("Timeline")][0]
    assert "games:2012 summer olympics" in timeline                  # 2012, not the year named in the question
    ev = [e["content"] for e in r["evidence"] if e["content"].startswith("Events (")][0]
    assert "Events (1)" in ev and "Chen Ding" in ev and "Wang Zhen" not in ev and "Women" not in ev    # right edition, men's only (word-boundary filter)
    assert r["answer_list"] == ["Chen Ding"] and r["answer_supported_by_citations"] is True
    assert r["retrieval_methods"] == ["entity_linking", "graph_traversal:adjacent_games", "graph_traversal:entity_events"]

def test_adjacent_games_after_and_missing():
    st = FakeState([]); st.linked = {}; st.focus_games = None
    t = A.Tools(TemporalTG(), st)
    t.graph_traversal({"op": "adjacent_games", "year": 2012, "direction": "after", "season": "summer"}); assert st.focus_games == "games:2016 summer olympics"
    out = A.Tools(TemporalTG(), FakeState([])).graph_traversal({"op": "adjacent_games", "year": 2008, "direction": "before", "season": "summer"})
    assert "No summer" in out[0][1]

# ---------------------------------------------------------------- list answers + submission format
def test_judge_list_answers(monkeypatch):
    import evaluation.judge as J
    calls = []
    monkeypatch.setattr(J, "llm", lambda *a, **k: (calls.append(1) or json.dumps({"verdict": "FAIL"}), {}))
    assert J.judge("q", ["Chen Ding"], ["Chen Ding"]) == ("PASS", "exact") and J.judge("q", ["A", "B"], ["b", "a"])[0] == "PASS" and J.judge("q", ["5"], ["5"])[0] == "PASS"
    assert not calls                                              # exact/numeric matches never spend an LLM call
    assert J.judge("q", ["A", "B"], ["A"]) == ("FAIL", "llm") and calls   # partial answer -> LLM fallback decides

def test_answer_generation_list_and_string(monkeypatch):
    f = fake_llm_factory([json.dumps({"answer": ["Chen Ding"], "citations": ["E1"]}), json.dumps({"answer": "6", "citations": []})])
    monkeypatch.setattr(ANS, "llm", f)
    with G.RunMeter():
        a1 = ANS.generate_answer("q", []); a2 = ANS.generate_answer("q", [])
    assert a1["answer_list"] == ["Chen Ding"] and a1["answer"] == "Chen Ding" and a2["answer_list"] == ["6"]

def test_hidden_answers_use_public_format(tmp_path):
    from evaluation.export_submission import export
    runs, out = tmp_path / "runs", tmp_path / "sub"; runs.mkdir()
    hidden = tmp_path / "hidden.jsonl"
    hidden.write_text("\n".join(json.dumps({"qid": f"eval-00{i}", "question": f"Q{i}?", "qtype": "multi_hop"}) for i in (1, 2)) + "\n")
    for p in ("rag", "graphrag", "agentic"):
        with open(runs / f"hidden_{p}.jsonl", "w") as f:
            f.write(json.dumps({"pipeline": p, "qid": "eval-001", "question": "Q1?", "qtype": "multi_hop", "answer": "Chen Ding", "answer_list": ["Chen Ding"], "tokens": {"total": 1}, "trace": []}) + "\n")
            f.write(json.dumps({"pipeline": p, "qid": "eval-002", "question": "Q2?", "qtype": "multi_hop", "answer": "", "error": "boom", "tokens": {"total": 0}, "trace": []}) + "\n")
    export(str(runs), str(out), str(hidden))
    rows = [json.loads(l) for l in open(out / "hidden_answers.jsonl")]
    assert rows[0] == {"qid": "eval-001", "question": "Q1?", "qtype": "multi_hop", "answer": ["Chen Ding"]}      # exactly the public-file fields
    assert rows[1]["answer"] == ["Not found in the corpus"] and set(rows[1]) == {"qid", "question", "qtype", "answer"}
    assert (out / "hidden_answers_rag.jsonl").exists() and (out / "hidden_results.json").exists()


def test_completeness():
    from evaluation.completeness import completeness
    assert completeness(["7"], ["7"]) == 1.0
    assert completeness(["a b", "c d"], ["a b"]) == 0.5
    assert completeness(["Dani KingLaura Trott"], ["Dani King, Laura Trott"]) == 1.0
    assert completeness(["x"], ["Not found in the corpus"]) == 0.0


def test_date_match_formats():
    from common import date_match
    assert date_match("15 to 22 August 2004", "15–22 August 2004")
    assert not date_match("15 to 22 August 2004", "15–25 August 2004")
    assert date_match("17 August 2008", "15 August 2008 (heats) 17 August 2008 (final)")
    assert date_match("8 August 1992", "1992-08-06 (qualification) 1992-08-08 (final)")
    assert not date_match("8 September 1992", "1992-08-06 (qualification) 1992-08-08 (final)")


def test_parse_real_shapes():
    from ingest.parse import parse_doc
    d = {"doc_id": "Q1", "title": "Archery at the 2008 Summer Olympics – Women's team", "text":
         "[Infobox Olympic event]\n  event: Women's team\n  games: 2008 Summer\n  venue: Olympic Green Archery Field\n  dates: 9–10 August\n  nations: 10\n  competitors: 30\n"
         "  gold: Park Sung-hyunYun Ok-HeeJoo Hyun-Jung\n  goldNOC: KOR\n  silver: Chen LingGuo Dan\n  silverNOC: CHN\n  bronze: A B\n  bronze2: C D\n\nBody."}
    ev = parse_doc(d)["event"]
    assert ev["date"] == "9–10 August 2008" and ev["gold"] == "Park Sung-hyunYun Ok-HeeJoo Hyun-Jung" and ev["bronze"] == "A B, C D"
    f = {"doc_id": "Q2", "title": "Football at the 2004 Summer Olympics – Women's tournament", "text":
         "[Infobox international football competition]\n  dates: 11–26 August\n  venues: 5\n  num_teams: 10\n  champion_other: USA\n  second_other: BRA\n  third_other: GER\n\nBody"}
    e2 = parse_doc(f)["event"]
    assert e2["venue"] == "" and e2["gold"] == "USA" and e2["nations"] == 10 and e2["date"].endswith("2004")


def test_judge_fused_names():
    import evaluation.judge as J
    J.llm = lambda *a, **k: (_ for _ in ()).throw(AssertionError("LLM must not be called"))
    assert J.judge("q", ["Dani KingLaura TrottJoanna Rowsell"], ["Dani King, Laura Trott, Joanna Rowsell"])[0] == "PASS"
    assert J.judge("q", ["Dani KingLaura TrottJoanna Rowsell"], ["Dani KingLaura TrottJoanna Rowsell"])[0] == "PASS"


def test_gateway_429_handling(monkeypatch, tmp_path):
    """Per-minute 429s must not consume retries; daily 429 fails over to key 2; bad key is skipped."""
    import json as _j, config as C
    from gateway import llm as L
    monkeypatch.setattr(C, "API_KEYS", ["k1", "k2", "k3"]); monkeypatch.setattr(C, "CACHE_PATH", str(tmp_path / "c.sqlite"))
    monkeypatch.setattr(C, "RPM_PER_KEY", 100); monkeypatch.setattr(C, "WAIT_FOR_RESET", False)
    monkeypatch.setattr(L.time, "sleep", lambda s: None)
    class R:
        def __init__(s, code, body): s.status_code, s._b, s.text = code, body, _j.dumps(body)
        def json(s): return s._b
    daily = {"error": {"details": [{"violations": [{"quotaId": "GenerateRequestsPerDayPerProjectPerModel-FreeTier"}]}]}}
    minute = {"error": {"details": [{"violations": [{"quotaId": "GenerateRequestsPerMinutePerProjectPerModel"}]}, {"retryDelay": "7s"}]}}
    ok = {"candidates": [{"content": {"parts": [{"text": "hi"}]}}], "usageMetadata": {"promptTokenCount": 3, "candidatesTokenCount": 1}}
    seq = {"k1": [R(429, daily)], "k2": [R(403, {"error": {"message": "API_KEY_INVALID"}})], "k3": [R(200, ok)]}
    def post(url, headers, json, timeout): return seq[headers["x-goog-api-key"]].pop(0)
    monkeypatch.setattr(L.requests, "post", post)
    g = L.Gateway(); t, rec = g.call("t", "p"); assert t == "hi" and rec["key_idx"] == 3
    # 20 consecutive per-minute 429s on the only usable key still succeed (they do not burn the 8-retry budget)
    g2 = L.Gateway(); g2.dead = {}; g2.keys = ["k3"]; g2.hist = {"k3": []}
    n = {"i": 0}
    def post2(url, headers, json, timeout):
        n["i"] += 1; return R(429, minute) if n["i"] <= 20 else R(200, ok)
    monkeypatch.setattr(L.requests, "post", post2); g2.dead = {}
    monkeypatch.setattr(L.Gateway, "_acquire", lambda self: ("k3", 0))
    assert g2.call("t", "p2")[0] == "hi"


def test_graph_traversal_missing_op_and_noisy_intersect():
    from agentic.agents import Tools
    from agentic.harness import State
    ev = lambda i, d: {"id": i, "title": i, "date": d, "competitors": 10, "nations": 5, "venue": "v"}
    A, B, C_ = ev("e1", "14 February 2010"), ev("e2", "20 February 2010"), ev("e3", "14 February 2010")
    members = {"venue:x": [A, B], "games:2010 winter olympics": [A, B, C_], "noise:1": [C_], "noise:2": [B]}
    class TG:
        def entity_events(self, i): return members[i]
    st = State("q"); t = Tools(TG(), st)
    out = t.graph_traversal({"entity_ids": ["venue:x", "games:2010 winter olympics"], "date_contains": "14 February"})   # no "op"
    assert "e1" in out[0][1] and "e2" not in out[0][1]
    out = t.graph_traversal({"op": "entity_events", "mode": "intersect", "date_contains": "14 February",
                             "entity_ids": ["venue:x", "games:2010 winter olympics", "noise:1", "noise:2"]})   # strict intersection is empty
    assert st.events and "e1" in st.events


def test_fuzzy_venue_and_exact_day_preference():
    from agentic.agents import Tools
    from agentic.harness import State
    ev = lambda i, d: {"id": i, "title": i, "date": d, "competitors": 10, "nations": 5, "venue": "v"}
    weights, boxing = ev("weight", "23 September 2000"), ev("boxing", "18–23 September 2000")
    class TG:
        def entities_by_type(self, t): return [{"id": "venue:beijing science and technologyuniversity gymnasium"}, {"id": "venue:sydney cec"}]
        def entity_events(self, i): return {"venue:beijing science and technologyuniversity gymnasium": [weights], "venue:sydney cec": [weights, boxing]}.get(i, [])
    st = State("q"); t = Tools(TG(), st)
    out = t.graph_traversal({"op": "entity_events", "entity_ids": ["venue:beijing science and technology university gymnasium"], "date_contains": "23 September"})
    assert "weight" in out[0][1]                               # fuzzy id resolved the fused venue name
    out = t.graph_traversal({"op": "entity_events", "entity_ids": ["venue:sydney cec"], "date_contains": "23 September"})
    assert "weight" in out[0][1] and "boxing" not in out[0][1]   # exact single-day match wins over the range
    out = t.graph_traversal({"op": "entity_events", "entity_ids": ["venue:sydney cec"], "date_contains": "18 to 23 September"})
    assert "boxing" in out[0][1]                               # a question range still matches the range event


def test_no_partial_overlap_without_date_and_agent_alias():
    from agentic.agents import Tools
    from agentic.harness import State, OP_ALIASES
    ev = lambda i: {"id": i, "title": "Sailing " + i, "date": "1 June 2000", "competitors": 10, "nations": 5, "venue": "v"}
    members = {"games:x": [ev("a"), ev("b"), ev("c")], "sport:sailing": [ev("a"), ev("b"), ev("c")], "noise:1": [ev("a")], "noise:2": [ev("b")]}
    class TG:
        def entity_events(self, i): return members[i]
    st = State("q"); t = Tools(TG(), st)
    t.graph_traversal({"op": "entity_events", "mode": "intersect", "title_contains": "sailing", "entity_ids": list(members)})
    assert not st.events                       # count/max questions must NOT silently get a partial event set
    assert OP_ALIASES["adjacent_games"] == "graph_traversal"


def test_repeat_aggregation_allowed_after_new_events():
    from agentic.harness import State, action_sig
    st = State("q"); a = {"op": "count", "field": "competitors"}
    s1 = action_sig("aggregation", a, st); st.events["e1"] = {"id": "e1"}
    assert action_sig("aggregation", a, st) != s1                       # events were fetched since: recompute is legitimate
    g = {"op": "entity_events", "entity_ids": ["x"]}
    assert action_sig("graph_traversal", g, st) == action_sig("graph_traversal", g, st)   # other tools still dedupe


def test_aggregation_scope_core_linked_and_containment():
    from agentic.agents import Tools
    from agentic.harness import State
    ev = lambda i, t, c: {"id": i, "title": t, "date": "1 June 1996", "competitors": c, "nations": 5, "venue": "v"}
    g96 = [ev("a", "Athletics at the 1996 Summer Olympics – A", 60), ev("b", "Athletics at the 1996 Summer Olympics – B", 40)]
    g92 = [ev("c", "Athletics at the 1992 Summer Olympics – C", 90)]
    members = {"games:1996 summer olympics": g96, "games:1992 summer olympics": g92}
    class TG:
        def entity_events(self, i): return members.get(i, [])
        def entities_by_type(self, t): return [{"id": "venue:kvitfjell and hafjell"}]
    st = State("q"); t = Tools(TG(), st)
    t.graph_traversal({"op": "entity_events", "mode": "union", "entity_ids": ["games:1992 summer olympics"]})      # noisy earlier fetch
    t.graph_traversal({"op": "entity_events", "mode": "union", "entity_ids": ["games:1996 summer olympics"]})
    out = t.aggregation({"op": "count", "field": "competitors", "cmp": ">", "value": 49, "title_contains": "athletics"})
    assert "result=1" in out[0][1] and "1992" not in out[0][1]                  # only the latest traversal is counted
    assert t._fix_id("venue:kvitfjell") == "venue:kvitfjell and hafjell"        # typed name contained in the real venue name


def test_early_stop_rule():
    from agentic.agents import rule_sufficient
    from agentic.harness import State
    def st_with(question, resolved):
        st = State(question); st.ledger.add("vertex", "events text", {"resolved": resolved}, "graph_traversal", 1); return st
    assert rule_sufficient(st_with("Who won the gold medal in the event held at X on 14 February 2010?", True))
    assert not rule_sufficient(st_with("Who won the gold medal in the event held at X on 14 February 2010?", False))
    assert not rule_sufficient(st_with("How many biathlon events had more than 73 competitors?", True))      # counts must go through aggregation
    assert not rule_sufficient(st_with("Which sailing event had the highest number of competitors?", True))  # extremes too


def test_early_stop_named_event_lookup():
    from agentic.agents import Tools, rule_sufficient
    from agentic.harness import State
    ev = lambda i, t: {"id": i, "title": t, "date": "", "competitors": 10, "nations": 26, "venue": "v"}
    T = "Sailing at the 2016 Summer Olympics – Women's RS:X"
    members = {"games:2016 summer olympics": [ev("a", T), ev("b", "Sailing at the 2016 Summer Olympics – Men's RS:X"), ev("c", "Sailing at the 2016 Summer Olympics – Laser")]}
    class TG:
        def entity_events(self, i): return members[i]
    q = f"How many nations competed in {T}?"
    st = State(q); t = Tools(TG(), st)
    out = t.graph_traversal({"op": "entity_events", "entity_ids": ["games:2016 summer olympics"]})     # NO filter: the question quotes one event's title
    st.ledger.add(*out[0], "graph_traversal", 1)
    assert rule_sufficient(st)
    st2 = State("How many sailing events at the 2016 Summer Olympics had more than 20 competitors?"); Tools(TG(), st2)
    st2.ledger.add("vertex", "x", {"resolved": True}, "graph_traversal", 1)
    assert not rule_sufficient(st2)                                                                       # counting events is still aggregation-only


def test_quoted_venue_overrides_orchestrator_and_accents():
    from agentic.agents import Tools
    from agentic.harness import State
    ev = lambda i, d: {"id": i, "title": i, "date": d, "competitors": 10, "nations": 5, "venue": "v"}
    combined, men = ev("combined", "20–21 February 1994"), ev("men", "20 February 1994")
    members = {"venue:kvitfjell and hafjell": [combined], "venue:kvitfjell": [men], "venue:val-d'isere": [ev("val", "16 February 1992")],
               "games:1994 winter olympics": [combined, men], "games:1992 winter olympics": [ev("val", "16 February 1992")]}
    class TG:
        def entities_by_type(self, t): return [{"id": k} for k in members if k.startswith("venue:")]
        def entity_events(self, i): return members.get(i, [])
    st = State("Who won the gold medal in the event held at Kvitfjell and Hafjell on February 20–21, 1994?"); t = Tools(TG(), st)
    out = t.graph_traversal({"op": "entity_events", "mode": "intersect", "date_contains": "20–21 February",
                             "entity_ids": ["venue:Kvitfjell", "games:1994 winter olympics"]})                 # orchestrator typed the wrong venue
    assert "combined" in out[0][1] and "men" not in out[0][1].replace("Events", "")
    st2 = State("Who won the gold medal in the event held at Val-d'Isère on 16 February 1992?"); t2 = Tools(TG(), st2)
    assert t2._fix_id("venue:Val-d'Isère") == "venue:val-d'isere"                                              # accent/case-insensitive
