"""PASS/FAIL grading: exact/numeric match first (free), LLM judge only if needed.
Judge calls run OUTSIDE the RunMeter so they never count toward pipeline tokens."""
import re
import config as C
from common import norm
from gateway.llm import llm, jparse

def _num(s):
    m = re.fullmatch(r"-?\d+(?:\.\d+)?", s.replace(",", "").strip()); return float(m.group(0)) if m else None

def judge(question, gold, pred):
    """gold: list of strings. pred: list of strings (preferred) or a string."""
    g = [norm(str(x)) for x in gold]
    ptext = ", ".join(map(str, pred)) if isinstance(pred, list) else str(pred or "")
    p = norm(ptext)
    if not p or "not found in the corpus" in p: return "FAIL", "rule"
    pset = {norm(str(x)) for x in pred} if isinstance(pred, list) else {norm(x) for x in re.split(r",|;| and ", ptext) if x.strip()}
    squash = lambda x: re.sub(r"[\W_]+", "", x)
    if squash("".join(g)) == squash("".join(norm(str(x)) for x in pred)) if isinstance(pred, list) else squash("".join(g)) == squash(p):
        return "PASS", "exact"                      # same names, only separators differ
    if len(g) == 1:
        gn, pn = _num(g[0]), _num(p)
        if gn is not None and pn is not None: return ("PASS" if gn == pn else "FAIL"), "numeric"
        if g[0] == p or pset == {g[0]}: return "PASS", "exact"
    elif set(g) == pset: return "PASS", "exact"
    gt, pt = set(re.findall(r"\w+", " ".join(g))), set(re.findall(r"\w+", p))   # nothing in common with the gold -> wrong, no LLM call needed
    if gt and pt and len(gt & pt) / len(gt | pt) < 0.15: return "FAIL", "rule_disjoint"
    prompt = (f"Question: {question}\nGround truth: {gold}\nSystem answer: {ptext}\n"
              'Is the system answer correct and complete vs the ground truth (same meaning, all items)? JSON {"verdict":"PASS"|"FAIL"}')
    t, _ = llm("judge", prompt, json_mode=True, model=C.JUDGE_MODEL, max_out=60)
    return (jparse(t, {}) or {}).get("verdict", "FAIL"), "llm"
