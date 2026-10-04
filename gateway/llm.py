"""Single LLM gateway. Every LLM call (answers, orchestrator, evaluator, extraction, judge) goes through here:
one pinned model, PRIMARY-FIRST key failover, per-key RPM limiter, disk cache, exact token metering."""
import contextvars, hashlib, json, re, sqlite3, sys, threading, time
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo
import requests
import config as C

URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
_cur = contextvars.ContextVar("meter", default=None)
def current_meter(): return _cur.get()

class QuotaExhausted(RuntimeError): pass

def _next_reset():
    """Gemini daily quotas reset at midnight Pacific time."""
    now = datetime.now(ZoneInfo("America/Los_Angeles"))
    return (now + timedelta(days=1)).replace(hour=0, minute=0, second=5, microsecond=0).timestamp()

class RunMeter:
    def __init__(self): self.calls = []
    def __enter__(self): self._t = _cur.set(self); return self
    def __exit__(self, *a): _cur.reset(self._t)
    def totals(self):
        i = sum(c["in"] for c in self.calls); o = sum(c["out"] for c in self.calls)
        return {"calls": len(self.calls), "in": i, "out": o, "total": i + o}

class Gateway:
    def __init__(self):
        if not C.API_KEYS: raise RuntimeError("Set API_KEYS in .env (comma separated, primary key first)")
        self.keys, self.lock, self.active = C.API_KEYS, threading.Lock(), 0
        self.hist = {k: [] for k in self.keys}; self.dead = {}; self._no_think = set()
        Path(C.CACHE_PATH).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(C.CACHE_PATH, check_same_thread=False)
        self.db.execute("create table if not exists c(k text primary key, v text)")
        self.db.execute("create table if not exists dead(kh text primary key, until real)")
        for k in self.keys:   # remember daily-exhausted keys across restarts (no wasted request on a dead primary)
            row = self.db.execute("select until from dead where kh=?", (self._kh(k),)).fetchone()
            if row and row[0] > time.time(): self.dead[k] = row[0]

    @staticmethod
    def _kh(k): return hashlib.sha256(k.encode()).hexdigest()[:16]

    def _mark_daily_dead(self, key):
        until = _next_reset(); self.dead[key] = until
        self.db.execute("insert or replace into dead values(?,?)", (self._kh(key), until)); self.db.commit()
        print(f"[gateway] key #{self.keys.index(key)+1} hit its daily quota -> failing over")

    def _acquire(self):
        """Fixed priority order: always the first key that is neither exhausted nor at its RPM cap."""
        t0, told = time.time(), False
        while True:
            now = time.time()
            with self.lock:
                for idx, k in enumerate(self.keys):
                    if self.dead.get(k, 0) > now: continue
                    h = [t for t in self.hist[k] if now - t < 60]; self.hist[k] = h
                    if len(h) < C.RPM_PER_KEY:
                        h.append(now)
                        if idx != self.active: print(f"[gateway] now using key #{idx+1}", flush=True); self.active = idx
                        return k, idx
                if all(self.dead.get(k, 0) > now + 120 for k in self.keys):   # every key is out for the day
                    reset = min(self.dead.values())
                    if not getattr(C, "WAIT_FOR_RESET", False):
                        raise QuotaExhausted("All keys exhausted their daily quota; quota resets at midnight Pacific time (~12:30 PM IST). "
                                             "Add keys from other projects, or rerun later (progress is saved), or set WAIT_FOR_RESET=1.")
                    if not told:
                        print(f"[gateway] all keys exhausted; sleeping until reset in {int((reset-now)/60)} min (WAIT_FOR_RESET=1)", flush=True); told = True
                    t0 = now   # do not trip the 10-minute wait guard while sleeping on purpose
            if now - t0 > 600: raise QuotaExhausted("Waited >10 min for a free key")
            time.sleep(5 if told else 1)

    @staticmethod
    def _err_info(r):
        """-> (is_daily_quota, retry_after_seconds_or_None, is_bad_key) from a Gemini/OpenAI error response."""
        txt = r.text; low = txt.lower(); retry = None; daily = False
        try:
            for d in r.json().get("error", {}).get("details", []):
                for v in d.get("violations", []) or []:
                    if "perday" in (v.get("quotaId") or "").lower().replace("_", ""): daily = True
                m = re.match(r"([\d.]+)s", str(d.get("retryDelay", "")))
                if m: retry = float(m.group(1))
        except Exception: pass
        daily = daily or any(x in txt for x in ("PerDay", "insufficient_quota")) or "per day" in low
        bad = r.status_code in (400, 401, 403) and any(x in txt for x in ("API_KEY_INVALID", "API key not valid", "PERMISSION_DENIED", "invalid_api_key", "has been suspended"))
        return daily, retry, bad

    def _request(self, model, prompt, system, json_mode, max_out):
        if C.PROVIDER == "openai":
            url = "https://api.openai.com/v1/chat/completions"
            body = {"model": model, "temperature": 0, "max_tokens": max_out,
                    "messages": ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt}]}
            if json_mode: body["response_format"] = {"type": "json_object"}
            hdr = lambda k: {"Authorization": "Bearer " + k}
        else:
            gc = {"temperature": 0, "maxOutputTokens": max_out}
            if json_mode: gc["responseMimeType"] = "application/json"
            if "2.5" in model: gc["thinkingConfig"] = {"thinkingBudget": 0}
            elif C.GEMINI_THINKING != "off" and model not in self._no_think:
                gc["thinkingConfig"] = {"thinkingLevel": C.GEMINI_THINKING}; gc["maxOutputTokens"] = max_out + 1024   # headroom: thinking tokens count against the cap
            body = {"contents": [{"role": "user", "parts": [{"text": prompt}]}], "generationConfig": gc}
            if system: body["systemInstruction"] = {"parts": [{"text": system}]}
            url = URL.format(model=model); hdr = lambda k: {"x-goog-api-key": k}
        last, other, throttled = None, 0, 0
        while other < 8 and throttled < 400:   # rate-limit waits never use up the normal retry budget
            key, idx = self._acquire(); t = time.time()
            try: r = requests.post(url, headers=hdr(key), json=body, timeout=120)
            except requests.RequestException as e: last = str(e); other += 1; time.sleep(2 * other); continue
            if r.status_code == 200:
                j = r.json()
                if C.PROVIDER == "openai":
                    u = j.get("usage", {}); text = j["choices"][0]["message"]["content"] or ""
                    tin, tout = u.get("prompt_tokens", 0), u.get("completion_tokens", 0)
                else:
                    u = j.get("usageMetadata", {}); cands = j.get("candidates") or [{}]
                    text = "".join(p.get("text", "") for p in cands[0].get("content", {}).get("parts", []))
                    tin, tout = u.get("promptTokenCount", 0), u.get("candidatesTokenCount", 0) + u.get("thoughtsTokenCount", 0)
                return {"text": text, "in": tin, "out": tout, "orig_latency_s": round(time.time() - t, 3), "key_idx": idx + 1}
            last = f"{r.status_code}: {r.text[:300]}"
            if r.status_code == 400 and "thinking" in r.text.lower() and "thinkingConfig" in body.get("generationConfig", {}):
                self._no_think.add(model); body["generationConfig"].pop("thinkingConfig"); body["generationConfig"]["maxOutputTokens"] = max_out
                print(f"[gateway] {model} rejected thinkingConfig; retrying without it", flush=True); continue
            daily, retry, bad = self._err_info(r)
            if bad:   # revoked / wrong key: skip it for this process and move on to the next key
                with self.lock: self.dead[key] = time.time() + 10 ** 9
                print(f"[gateway] key #{idx+1} was rejected ({r.status_code}); skipping it. Check that key.", flush=True); continue
            if r.status_code == 429:
                throttled += 1
                if daily:
                    with self.lock: self._mark_daily_dead(key)
                else:   # per-minute limit: park this key for the time Google asks (default 30 s); next key takes over
                    with self.lock: self.dead[key] = time.time() + min(max(retry or 30, 5), 90)
            elif r.status_code in (500, 502, 503, 504): other += 1; time.sleep(2 * other)
            else: raise RuntimeError(last)
        raise RuntimeError(f"LLM call failed: {last}")

    def call(self, role, prompt, system="", json_mode=False, model=None, max_out=1024):
        model = model or C.GEN_MODEL
        if C.ENFORCE_FREE_TIER and model not in C.FREE_TIER_MODELS:
            raise ValueError(f"{model} is not in FREE_TIER_MODELS; refusing to call.")
        ck = hashlib.sha256(json.dumps([model, system, prompt, json_mode, max_out]).encode()).hexdigest()
        row = self.db.execute("select v from c where k=?", (ck,)).fetchone(); t = time.time()
        if row: d, cached = json.loads(row[0]), True
        else:
            d, cached = self._request(model, prompt, system, json_mode, max_out), False
            self.db.execute("insert or replace into c values(?,?)", (ck, json.dumps(d))); self.db.commit()
        rec = {"role": role, "model": model, "in": d["in"], "out": d["out"], "cached": cached, "key_idx": d.get("key_idx"),
               "latency_s": round(time.time() - t, 3), "orig_latency_s": d["orig_latency_s"]}
        m = current_meter()
        if m: m.calls.append(rec)
        return d["text"], rec

_gw = None
def llm(role, prompt, system="", json_mode=False, model=None, max_out=1024):
    global _gw
    _gw = _gw or Gateway()
    return _gw.call(role, prompt, system, json_mode, model, max_out)

def jparse(text, default=None):
    t = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.M).strip()
    try: return json.loads(t)
    except Exception:
        m = re.search(r"\{.*\}", t, re.S)
        if m:
            try: return json.loads(m.group(0))
            except Exception: pass
    return default
