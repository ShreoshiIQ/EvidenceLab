"""Specialised agents. Deterministic wherever possible; LLM only for reasoning/evaluation.
Each returns a list of (kind, content, provenance) tuples; the harness owns the ledger."""
import difflib, operator, re
from common import events_text, norm, date_match, date_sig
from gateway.embed import embed
from gateway.llm import llm, jparse
from agentic.linking import link
from answer import build_context

OPS = {">": operator.gt, ">=": operator.ge, "<": operator.lt, "<=": operator.le, "==": operator.eq}

def _nodate(s): return re.sub(r"\s+", " ", re.sub(r"\b(?:19|20)\d{2}\b", "", norm(s))).strip()   # compare dates ignoring the year

# questions that need a computed count / extreme over MANY events (never settled by an early stop).
# "How many nations competed in <one named event>?" is a lookup of one event's attribute and is NOT matched.
AGG_Q = re.compile(r"\bhow many (?:[\w'-]+ ){0,4}events\b|\bnumber of (?:[\w'-]+ ){0,3}events\b|"
                   r"\b(?:highest|lowest|most|fewest|least|largest|smallest|average|total|more than|fewer than|less than)\b", re.I)

def rule_sufficient(st):
    """Stop without an LLM check when (a) an aggregate over a non-empty, fully-known event set exists, or
    (b) a FILTERED lookup (date and/or title filter) resolved to 1-3 events and the question does not ask for a count/extreme
    (those must always be computed by the aggregation tool)."""
    if any(e.kind == "aggregate" and e.provenance.get("unknown", 1) == 0 and e.provenance.get("n_events", 0) > 0 for e in st.ledger.items): return True
    if AGG_Q.search(st.question): return False
    return any(e.kind == "vertex" and e.provenance.get("resolved") for e in st.ledger.items)

def _contains(haystack, needle):
    """Word-boundary match so "men's 20 kilometres walk" does NOT match "women's 20 kilometres walk"."""
    n = norm(needle)
    return bool(n) and re.search(r"(?<![\w'])" + re.escape(n) + r"(?![\w])", norm(haystack)) is not None

GAMES_ID = re.compile(r"games:(\d{4}) (summer|winter) (olympics|paralympics)$")

class Tools:
    def __init__(self, tg, st): self.tg, self.st, self._ents = tg, st, {}

    def _quoted_venue(self):
        """The venue whose full name appears in the question (longest match). The question text is the ground truth for the venue:
        "Kvitfjell and Hafjell" is ONE venue, "Estadi Olimpic Lluis Companys, Barcelona" includes the city; the orchestrator often shortens or splits them."""
        if "Venue" not in self._ents:
            try: self._ents["Venue"] = [e["id"] for e in self.tg.entities_by_type("Venue")]
            except Exception: self._ents["Venue"] = []   # venue list unavailable: keep the orchestrator's ids
        qn = norm(self.st.question); best = None
        for i in self._ents["Venue"]:
            name = i.split(":", 1)[1]
            if name and re.search(r"(?<![\w'-])" + re.escape(name) + r"(?![\w'-])", qn) and (best is None or len(name) > len(best.split(":", 1)[1])): best = i
        return best

    def _fix_id(self, eid):
        """An id the orchestrator typed that does not exist (e.g. venue with a different word split than the corpus, "Technology University"
        vs the corpus's fused "TechnologyUniversity"): map it to the closest real entity of the same type, ignoring spaces/punctuation."""
        if ":" not in eid: return eid
        etype, name = eid.split(":", 1); etype = etype.capitalize()
        if etype not in self._ents: self._ents[etype] = [e["id"] for e in self.tg.entities_by_type(etype)]
        sq = lambda x: re.sub(r"[\W_]+", "", norm(x)); ids = self._ents[etype]; key = sq(name)
        bysq = {sq(i.split(":", 1)[1]): i for i in ids}
        if key in bysq: return bysq[key]
        near = difflib.get_close_matches(key, list(bysq), n=1, cutoff=0.9)
        if near: return bysq[near[0]]
        if len(key) >= 6:   # "venue:kvitfjell" -> the corpus entity "kvitfjell and hafjell": accept the shortest entity that contains the typed name
            cont = sorted((k for k in bysq if key in k), key=len)
            if cont: return bysq[cont[0]]
        return eid

    def _resolve_ids(self, ids):
        """"$linked" = confidently linked entities; once an edition was resolved (adjacent_games) it replaces any year-linked Games.
        "$games" = that resolved edition."""
        focus = getattr(self.st, "focus_games", None)
        linked = [k for k, v in self.st.linked.items() if v["conf"] >= 0.9]
        if focus: linked = [k for k in linked if not k.startswith("games:")] + [focus]
        if ids == "$linked": return linked
        out = []
        for i in (ids or []):
            if i == "$linked": out += linked
            elif i == "$games":
                if focus: out.append(focus)
            else: out.append(i)
        return list(dict.fromkeys(out))

    def entity_linking(self, a):
        out = []
        for e in link(self.tg, self.st.question, a.get("mentions")):
            self.st.linked[e["id"]] = e
            out.append(("vertex", f"Entity {e['id']} ({e['etype']}) conf={e['conf']} via {e['method']}", {"entity_id": e["id"]}))
        return out

    def similarity_search(self, a):
        qv = embed([a.get("query") or self.st.question], query=True)[0]; k = int(a.get("k", 5))
        if a.get("target") == "entity":
            return [("vertex", f"Entity {r['id']} ({r['etype']}) name='{r['name']}'", {"entity_id": r["id"]}) for r in self.tg.vec_entities(qv, k)]
        return [("chunk", r["text"], {"chunk_id": r["id"], "doc_id": r["doc_id"]}) for r in self.tg.vec_chunks(qv, k)]

    def graph_traversal(self, a):
        op = a.get("op")
        if not op:   # the orchestrator often omits "op": infer it from the arguments instead of failing
            op = ("entity_events" if "entity_ids" in a else "adjacent_games" if "year" in a else "expand_chunks" if "chunk_ids" in a else None)
        if op == "entity_events":
            ids = self._resolve_ids(a.get("entity_ids", []))      # placeholders let one plan chain link -> resolve -> traverse
            if any(i.startswith("venue:") for i in ids):   # use the venue exactly as the question names it, whatever the orchestrator typed
                qv = self._quoted_venue()
                if qv: ids = list(dict.fromkeys([i for i in ids if not i.startswith("venue:")] + [qv]))
            if a.get("mode", "intersect") == "intersect" and len(ids) > 2:   # venue+date lookups: noisy linked entities (other venues/editions) must not widen the match
                core = [i for i in ids if i.split(":")[0] in ("games", "venue")]
                if len(core) >= 2 and any(i.startswith("venue:") for i in core): ids = core
            sets = []
            for i in ids:
                evs0 = self.tg.entity_events(i)
                if not evs0:   # unknown id: try the closest real entity before giving up
                    j = self._fix_id(i)
                    if j != i: evs0 = self.tg.entity_events(j); self.st.linked.setdefault(j, {"id": j, "etype": j.split(":")[0].capitalize(), "conf": 1.0, "method": "fuzzy-id"})
                sets.append({e["id"]: e for e in evs0})
            if not sets: return []
            mode = a.get("mode", "intersect")
            def pick(keys):
                evs = [next(s[k] for s in sets if k in s) for k in sorted(keys)]
                if a.get("date_contains"): evs = [e for e in evs if date_match(a["date_contains"], e.get("date", ""))]
                if a.get("title_contains"): evs = [e for e in evs if _contains(e.get("title", ""), a["title_contains"])]
                if a.get("date_contains"):   # a single named day: an event held exactly that day beats a multi-day event that merely spans it
                    qd, _, qr = date_sig(a["date_contains"])
                    exact = [e for e in evs if not qr and date_sig(e.get("date", ""))[0] == qd]
                    if exact: evs = exact
                return evs
            if mode == "intersect":
                evs = pick(set.intersection(*map(set, sets)))
                if not evs and len(sets) > 2 and a.get("date_contains"):   # venue+date lookups only: a noisy entity empties the strict intersection, the date keeps the partial overlap safe.
                    # NOT for counts/maxima: a partial event set would be silently treated as complete
                    from collections import Counter
                    cnt = Counter(k for s in sets for k in s)
                    for need in range(len(sets) - 1, 1, -1):
                        evs = pick({k for k, c in cnt.items() if c >= need})
                        if evs: mode = f"best-overlap (>= {need} of {len(sets)} entities)"; break
            else: evs = pick(set().union(*sets))
            self.st.events.update({e["id"]: e for e in evs}); self.st.last_events = {e["id"]: e for e in evs}
            filtered = bool(a.get("date_contains") or a.get("title_contains")); partial = str(mode).startswith("best-overlap")
            qn = norm(self.st.question); named = [e for e in evs if norm(e.get("title", "")) and norm(e["title"]) in qn]   # the question quotes exactly one fetched event's full title
            resolved = not partial and ((filtered and 1 <= len(evs) <= 3) or len(named) == 1)
            return [("vertex", events_text(evs, f"{mode} of {ids}"), {"doc_ids": [e["id"] for e in evs], "resolved": resolved})]
        if op == "adjacent_games":
            year, direction = int(a["year"]), a.get("direction", "before")
            season, kind = (a.get("season") or "").lower(), (a.get("kind") or "olympics").lower()
            cand = []
            for e in self.tg.entities_by_type("Games"):
                m = GAMES_ID.match(e["id"])
                if m and (not season or m[2] == season) and m[3] == kind: cand.append((int(m[1]), e))
            pool = [c for c in cand if (c[0] < year if direction == "before" else c[0] > year)]
            pick = (max(pool, key=lambda c: c[0]) if direction == "before" else min(pool, key=lambda c: c[0])) if pool else None
            years = sorted(c[0] for c in cand)
            if not pick:
                return [("vertex", f"No {season or 'any'} {kind} edition {direction} {year} exists in the corpus (editions present: {years})", {})]
            self.st.focus_games = pick[1]["id"]
            self.st.linked[pick[1]["id"]] = {**pick[1], "conf": 1.0, "method": "temporal"}
            return [("vertex", f"Timeline: the {season or ''} {kind} edition immediately {direction} {year} is {pick[1]['id']} (editions in corpus: {years})",
                     {"entity_id": pick[1]["id"]})]
        if op == "expand_chunks":
            r = self.tg.expand_chunks(a.get("chunk_ids", []), int(a.get("limit", 4)))
            return [("chunk", c["text"], {"chunk_id": c["id"], "doc_id": c["doc_id"]}) for c in r["chunks"]]
        raise ValueError(f"unknown graph op {op}")

    def document_retrieval(self, a):
        ids = list(self.st.events) if a.get("from_state_events") else a.get("event_ids", [])
        lim = int(a.get("limit", 8))
        rows = self.tg.event_chunks(ids, int(a.get("per_event", 1)))[:lim] if ids else self.tg.get_chunks(a.get("chunk_ids", []))[:lim]
        return [("chunk", r["text"], {"chunk_id": r["id"], "doc_id": r["doc_id"]}) for r in rows]

    def aggregation(self, a):
        """Computed in code, never by the LLM."""
        # count over the events of the MOST RECENT traversal, not everything ever fetched (an earlier noisy fetch of other editions must not leak in)
        ev = list((getattr(self.st, "last_events", None) or self.st.events).values()); f = a.get("field", "competitors")
        if a.get("title_contains"): ev = [e for e in ev if a["title_contains"].lower() in e["title"].lower()]
        if a.get("cmp") and a.get("value") is None: raise ValueError('aggregation with "cmp" needs a numeric "value"')
        known = [e for e in ev if e.get(f)]
        sel = [e for e in known if OPS[a["cmp"]](e[f], float(a["value"]))] if a.get("cmp") else known
        op = a.get("op", "count")
        if op == "count": res = len(sel)
        elif op in ("max", "min") and sel:
            res = (max if op == "max" else min)(e[f] for e in sel); sel = [e for e in sel if e[f] == res]
        elif op == "sum": res = sum(e[f] for e in sel)
        elif op == "avg" and sel: res = round(sum(e[f] for e in sel) / len(sel), 2)
        else: res = [e["id"] for e in sel]
        txt = (f"AGGREGATE {op}({f}) cmp={a.get('cmp')} {a.get('value')} over {len(ev)} events "
               f"({len(known)} with known {f}, {len(ev) - len(known)} unknown): result={res}\n"
               + "\n".join(f"{e['id']} | {e['title']} | {f}={e[f]}" for e in sel))
        return [("aggregate", txt, {"doc_ids": [e["id"] for e in sel], "unknown": len(ev) - len(known), "n_events": len(ev)})]

    def multi_hop_reasoning(self, a):
        p = (f"Question: {self.st.question}\nEvidence so far:\n{build_context(self.st.ledger.items, 6000)}\n"
             'Decompose into the minimal sub-questions still needed. JSON: {"sub_questions":["..."]}')
        t, _ = llm("reasoner", p, json_mode=True, max_out=300)
        self.st.open_q = (jparse(t, {}) or {}).get("sub_questions", []); return []

    def evidence_evaluation(self, a):
        # Deterministic checks first (0 LLM tokens); the LLM judge runs only when the rules are inconclusive.
        if not self.st.ledger.items:
            self.st.last_eval = {"sufficient": False, "missing": ["no evidence yet"], "conflicts": [], "method": "rule"}
            self.st.open_q = ["no evidence yet"]; return []
        if rule_sufficient(self.st):
            self.st.last_eval = {"sufficient": True, "missing": [], "conflicts": [], "method": "rule"}; self.st.open_q = []; return []
        p = (f"Question: {self.st.question}\nEvidence:\n{build_context(self.st.ledger.items, 12000)}\n"
             'Is this evidence sufficient to answer fully and correctly? For counts, is the enumeration complete? '
             'JSON: {"sufficient":true|false,"missing":["..."],"conflicts":["..."]}')
        t, _ = llm("evaluator", p, json_mode=True, max_out=300)
        self.st.last_eval = jparse(t, {"sufficient": False, "missing": ["unparseable evaluation"], "conflicts": []})
        self.st.open_q = self.st.last_eval.get("missing", []); return []
