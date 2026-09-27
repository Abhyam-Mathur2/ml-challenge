# Candidate Recall Benchmark

Sampled S1 rows: 5000
Random state: 42

| Candidate budget | Avg candidates | Overall recall | S2 recall | S3 recall |
|---:|---:|---:|---:|---:|
| 50 | 100.00 | 0.4295 | 0.4486 | 0.4114 |
| 80 | 160.00 | 0.5106 | 0.5425 | 0.4806 |
| 120 | 240.00 | 0.6257 | 0.6564 | 0.5969 |
| 160 | 319.65 | 0.6362 | 0.6670 | 0.6071 |

Zero-match entities are excluded from recall and reported under `singleton` in the JSON.

Detailed metrics are in `candidate_recall_benchmark.json`.