# One command per deliverable. Activate your venv first.
PUB=data/eval_public.jsonl
HID=data/eval_hidden.jsonl
setup:     ; python -m tgdb.setup schema && python -m tgdb.setup queries
ingest:    ; python -m ingest.run --corpus data/corpus.jsonl
probe:     ; python -m tgdb.mcp_probe
smoke:     ; python -m evaluation.run_eval --file $(PUB) --split public --limit 5
public:    ; python -m evaluation.run_eval --file $(PUB) --split public && python -m evaluation.report --split public
# Hidden: run ONCE after freezing code/prompts. Use a fresh cache so tokens/latency are genuine:  CACHE_PATH=runs/hidden_cache.sqlite make hidden
hidden:    ; python -m evaluation.run_eval --file $(HID) --split hidden && python -m evaluation.report --split hidden && python -m evaluation.export_submission
report:    ; python -m evaluation.report --split public
dashboard: ; streamlit run dashboard/app.py
test:      ; python -m pytest -q tests
