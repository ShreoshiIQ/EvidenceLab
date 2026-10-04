"""Deterministic parsing (no LLM): infobox -> typed Event attributes, title -> Games/Sport,
infobox names/NOCs/venue -> entities, text -> title-prefixed chunks."""
import re
import config as C
from common import ent_id

TITLE_RE = re.compile(r"^(?P<sport>.+?) at the (?P<games>\d{4} (?:Summer|Winter) (?:Olympics|Paralympics))")
SPLIT = re.compile(r"(?<=[a-zà-ÿ])(?=[A-ZÀ-Þ])")   # un-fuse "Rudolf DombiRoland Kökény"

def _int(v):
    m = re.search(r"\d[\d,]*", v or "")
    return int(m.group(0).replace(",", "")) if m else 0

def _medal(info, k):
    """Gold/silver/bronze exactly as written in the source. Team events are stored fused ("Dani KingLaura TrottJoanna Rowsell")
    because that is the form the ground-truth answers use; do NOT split or add country codes here."""
    v = re.sub(r"\s+", " ", info.get(k, "")).strip()
    if k == "bronze" and info.get("bronze2"): v = (v + ", " if v else "") + re.sub(r"\s+", " ", info["bronze2"]).strip()
    return v

MONTH_YEAR = re.compile(r"\b(?:19|20)\d{2}\b")

def _clean_info(info, games_year):
    """Alias fixes found by profiling: date/dates, venue/venues (football uses venues: 5 = a COUNT), football medal keys, missing years."""
    info = dict(info)
    if not info.get("date") and info.get("dates"): info["date"] = info["dates"]
    if not info.get("venue") and info.get("venues") and not re.fullmatch(r"\d+", info["venues"].strip()): info["venue"] = info["venues"]
    for src, dst in (("champion_other", "gold"), ("second_other", "silver"), ("third_other", "bronze")):
        if not info.get(dst) and info.get(src): info[dst] = info[src]
    if not info.get("nations") and info.get("num_teams"): info["nations"] = info["num_teams"]
    d = re.sub(r"\)(?=\S)", ") ", info.get("date", "")).strip()          # "(heats)17 August" -> "(heats) 17 August"
    if d and games_year and not MONTH_YEAR.search(d): d = f"{d} {games_year}"
    if d: info["date"] = d
    return info

def make_chunks(title, info, body, max_words=180):
    head = title + "\n" + "; ".join(f"{k}: {v}" for k, v in info.items())
    lines = [l.strip() for l in body.split("\n") if l.strip()]
    chunks, cur, n = [], [head], len(head.split())
    for l in lines:
        w = len(l.split())
        if n + w > max_words and len(cur) > 1:
            chunks.append("\n".join(cur)); cur, n = [title], len(title.split())
        cur.append(l); n += w
    if len(cur) > 1 or not chunks: chunks.append("\n".join(cur))
    return chunks

def parse_doc(d):
    text, info, body = d["text"], {}, d["text"]
    if text.startswith("[Infobox"):
        head, _, body = text.partition("\n\n")
        for l in head.split("\n")[1:]:
            k, _, v = l.strip().partition(":"); info[k.strip()] = v.strip()
    m = TITLE_RE.match(d["title"])
    info = _clean_info(info, m["games"][:4] if m else None)
    ents = {}
    def add(etype, name, role):
        name = name.strip()
        if name: ents[ent_id(etype, name)] = {"id": ent_id(etype, name), "name": name, "etype": etype, "role": role}
    if m: add("Games", m["games"], "games"); add("Sport", m["sport"], "sport")
    if info.get("venue"): add("Venue", info["venue"], "venue")
    for medal in ("gold", "silver", "bronze"):
        for nm in SPLIT.split(info.get(medal, "")): add("Athlete", nm, f"medal_{medal}")
        if info.get(medal + "NOC"): add("Country", info[medal + "NOC"], f"medal_{medal}")
    event = {"id": d["doc_id"], "title": d["title"], "competitors": _int(info.get("competitors")),
             "nations": _int(info.get("nations")), "date": info.get("date", ""),
             "venue": info.get("venue", ""), "gold": _medal(info, "gold"), "silver": _medal(info, "silver"), "bronze": _medal(info, "bronze")}
    chunks = []
    cap = C.MAX_CHUNKS_EVENT if (m or "games" in info) else C.MAX_CHUNKS_OTHER
    for i, t in enumerate(make_chunks(d["title"], info, body)[:cap]):
        low = t.lower()
        ment = [e["id"] for e in ents.values() if e["etype"] in ("Games", "Sport") or e["name"].lower() in low]
        chunks.append({"id": f"{d['doc_id']}#{i}", "text": t, "mentions": ment})
    return {"event": event, "entities": list(ents.values()), "chunks": chunks, "title": d["title"]}
