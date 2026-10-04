"""python -m gateway.keycheck   -> tests every key in API_KEYS with one tiny call. Never prints the keys."""
import time, requests
import config as C
from gateway.llm import URL

def main():
    if not C.API_KEYS: print("No keys in API_KEYS (.env)"); return
    print(f"model={C.GEN_MODEL}  keys={len(C.API_KEYS)}  RPM_PER_KEY={C.RPM_PER_KEY}")
    for i, k in enumerate(C.API_KEYS, 1):
        tag = f"key #{i} (…{k[-4:]})"
        try:
            r = requests.post(URL.format(model=C.GEN_MODEL), headers={"x-goog-api-key": k}, timeout=60,
                              json={"contents": [{"role": "user", "parts": [{"text": "Reply with the single word: ok"}]}], "generationConfig": {"maxOutputTokens": 8}})
        except requests.RequestException as e: print(f"{tag}: network error {e}"); continue
        if r.status_code == 200: print(f"{tag}: OK")
        elif r.status_code == 429:
            daily = "PerDay" in r.text or "per day" in r.text.lower()
            print(f"{tag}: LIMIT HIT ({'daily quota used up - resets ~12:30 PM IST' if daily else 'per-minute limit, retry shortly'})")
        elif r.status_code in (400, 401, 403): print(f"{tag}: REJECTED ({r.status_code}) - wrong, revoked or restricted key: {r.text[:120]!r}")
        elif r.status_code == 404: print(f"{tag}: model not available to this key ({C.GEN_MODEL})")
        else: print(f"{tag}: HTTP {r.status_code} {r.text[:120]!r}")
        time.sleep(1)
    print("\nKeys from the SAME Google project share one quota. Use one key per project (or per teammate).")

if __name__ == "__main__": main()
