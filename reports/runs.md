# Evaluation runs

Updated 2026-11-11.

| run_id | label | model | mode | top_k | prompt_name | fault | n | exact_match | f1 | contains_answer | retrieval_hit | mrr | latency_ms |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2 | baseline | qwen3:1.7b | rag | 3 | concise | none | 30 | 0.633 | 0.707 | 0.633 | 1.000 | 0.900 | 8802.742 |
| 3 | broken | qwen3:1.7b | rag | 3 | concise | random-retrieval | 30 | 0.033 | 0.049 | 0.033 | 0.000 | 0.000 | 7932.685 |
