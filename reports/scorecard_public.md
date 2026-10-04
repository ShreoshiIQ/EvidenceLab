# Scorecard: public questions

Questions: **100** | LLM used by ALL pipelines: **gemini-3.1-flash-lite** (PASS: single model)

## Per-pipeline scorecard

| Metric | rag | graphrag | agentic |
|---|---|---|---|
| Accuracy (PASS rate) | 0.53 | 0.62 | 0.98 |
| Completeness (gold items found) | 0.583 | 0.703 | 1.0 |
| Semantic similarity (embedding) | 0.817 | 0.869 | 0.997 |
| Accuracy per 1k tokens | 0.156 | 0.126 | 0.313 |
| Avg tokens / question (total) | 3402.7 | 4930.5 | 3131.9 |
| &nbsp;&nbsp;context tokens | 2111.6 | 3242.5 | 614.1 |
| &nbsp;&nbsp;LLM input tokens | 3326.2 | 4848.4 | 2726.6 |
| &nbsp;&nbsp;LLM output tokens | 76.5 | 82.1 | 405.3 |
| Avg latency (s) | 8.74 | 10.86 | 12.1 |
| Median / p95 latency (s) | 8.51 / 11.85 | 10.52 / 15.06 | 11.07 / 22.66 |
| Avg retrieval steps | 1.0 | 4.0 | 2.33 |
| Avg reasoning steps | 1.0 | 1.0 | 2.83 |
| Avg LLM calls | 1.0 | 1.0 | 2.3 |
| Avg chunks / citations | 10.0 / 2.31 | 10.0 / 1.7 | 0.0 / 1.32 |
| Citation validity | 0.99 | 0.99 | 1.0 |
| Answer supported by cited evidence | 1.0 | 0.989 | 1.0 |
| Lift over blind-guess baseline | 0.53 | 0.62 | 0.98 |
| Avg doc recall vs gold docs | 0.664 | 0.874 | 0.89 |

## Retrieval methods used (share of questions using each)

- **rag**: vector_search 100%
- **graphrag**: graph_traversal:entity_events 100%, graph_traversal:expand_chunks 100%, vector_search 100%, entity_linking 100%
- **agentic**: graph_traversal:entity_events 100%, entity_linking 82%, graph_traversal:adjacent_games 22%, graph_traversal:graph_traversal 2%

## Agentic GraphRAG

- Avg tokens / question: **3131.9**
- Avg time / question: **12.1 s**
- Avg steps: 5.16 (2.33 retrieval + 2.83 reasoning); avg orchestrator rounds: 1.3
- Strategy-change rate: 0.0
- Stop reasons: {'rule_sufficient': 99, 'orchestrator_answer': 1}
- Tools used: {'count': 37, 'entity_events': 100, 'entity_linking': 82, 'adjacent_games': 22, 'max': 10, 'graph_traversal': 2}
- Agents invoked: {'aggregation': 47, 'entity_linking': 82, 'graph_traversal': 100}

## By question type

| qtype | rag acc | rag tokens | rag s | graphrag acc | graphrag tokens | graphrag s | agentic acc | agentic tokens | agentic s |
|---|---|---|---|---|---|---|---|---|---|
| aggregation | 0.0 | 3335.9 | 8.79 | 0.19 | 4665.1 | 10.82 | 1.0 | 3765.8 | 14.83 |
| lookup | 1.0 | 3403.2 | 8.46 | 1.0 | 5001.2 | 10.57 | 1.0 | 4196.2 | 13.95 |
| multi_hop | 0.393 | 3374.8 | 8.79 | 0.571 | 5040.8 | 11.04 | 0.929 | 2620.4 | 11.06 |
| superlative | 0.4 | 3345.9 | 8.35 | 0.4 | 5059.1 | 10.68 | 1.0 | 3977.8 | 11.97 |
| temporal | 0.864 | 3527.3 | 9.07 | 0.864 | 4923.8 | 11.01 | 1.0 | 1874.0 | 9.27 |
