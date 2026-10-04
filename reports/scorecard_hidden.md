# Scorecard: hidden questions

Questions: **50** | LLM used by ALL pipelines: **gemini-3.1-flash-lite** (PASS: single model)

## Per-pipeline scorecard

| Metric | rag | graphrag | agentic |
|---|---|---|---|
| Accuracy (PASS rate) | n/a (hidden) | n/a (hidden) | n/a (hidden) |
| Completeness (gold items found) | n/a | n/a | n/a |
| Semantic similarity (embedding) | n/a | n/a | n/a |
| Accuracy per 1k tokens | n/a | n/a | n/a |
| Avg tokens / question (total) | 3379.1 | 4838.7 | 3145.8 |
| &nbsp;&nbsp;context tokens | 2139.8 | 3199.2 | 738.5 |
| &nbsp;&nbsp;LLM input tokens | 3300.2 | 4755.3 | 2754.9 |
| &nbsp;&nbsp;LLM output tokens | 78.9 | 83.4 | 390.9 |
| Avg latency (s) | 7.58 | 6.1 | 12.37 |
| Median / p95 latency (s) | 6.11 / 15.14 | 5.26 / 10.13 | 11.49 / 20.3 |
| Avg retrieval steps | 1.0 | 4.0 | 2.22 |
| Avg reasoning steps | 1.0 | 1.0 | 2.88 |
| Avg LLM calls | 1.0 | 1.0 | 2.22 |
| Avg chunks / citations | 10.0 / 2.46 | 9.92 / 1.4 | 0.0 / 1.28 |
| Citation validity | 1.0 | 0.98 | 1.0 |
| Answer supported by cited evidence | 1.0 | 0.957 | 1.0 |
| Lift over blind-guess baseline | n/a | n/a | n/a |
| Avg doc recall vs gold docs | n/a | n/a | n/a |

## Retrieval methods used (share of questions using each)

- **rag**: vector_search 100%
- **graphrag**: graph_traversal:expand_chunks 100%, vector_search 100%, entity_linking 100%, graph_traversal:entity_events 100%
- **agentic**: graph_traversal:entity_events 100%, entity_linking 86%, graph_traversal:adjacent_games 16%, graph_traversal:graph_traversal 4%

## Agentic GraphRAG

- Avg tokens / question: **3145.8**
- Avg time / question: **12.37 s**
- Avg steps: 5.1 (2.22 retrieval + 2.88 reasoning); avg orchestrator rounds: 1.22
- Strategy-change rate: 0.0
- Stop reasons: {'rule_sufficient': 49, 'orchestrator_answer': 1}
- Tools used: {'entity_events': 50, 'entity_linking': 43, 'count': 22, 'max': 10, 'adjacent_games': 8, 'graph_traversal': 2}
- Agents invoked: {'entity_linking': 43, 'graph_traversal': 50, 'aggregation': 32}
