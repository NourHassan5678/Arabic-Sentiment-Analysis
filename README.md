# Arabic Sentiment Analysis

A lightweight MLOps API for classifying Arabic e-commerce reviews into Positive, Negative, or Neutral sentiments.

## Quickstart

Run these 3 commands to build, serve, and test the API locally:

1. **Build the container:**
   ```bash
   docker compose build
## Quality gate: AraBERT must beat the baseline

`src/arabic_sentiment/evaluate.py` is the last stage of the DVC pipeline (`dvc repro`).
It scores both models on the same held-out test split (6,003 rows) and exits with code 1
if AraBERT's test macro-F1 is below the baseline's, which fails `dvc repro` and any CI job
that runs it.

| Model    | Test macro-F1 |
|----------|---------------|
| Baseline | 0.6153        |
| AraBERT  | 0.6641        |

**Why this gate**
- A transformer is only worth its serving cost if it beats the simple baseline. The gate
  blocks regressions caused by retraining, data changes or code changes.
- It uses macro-F1, not accuracy. The classes are imbalanced (neutral is about 5% of the
  test set). The baseline reaches 0.78 accuracy but only 0.21 F1 on neutral. Macro-F1
  weights every class equally, so a model that ignores the minority class cannot pass.
- The comparison uses the baseline recomputed on the current split. The documented
  reference value (0.6153) is a sanity check: if the recomputed baseline differs by more
  than 0.005, `evaluate.py` prints a warning, because the data split or baseline changed.

## Serving benchmark (final, no optimization baseline)

This is the **final serving benchmark** of the Production AraBERT model behind BentoML.
The project has no optimization phase, so there is no "before" number to compare against.

- **Service:** `bentoml serve src.arabic_sentiment.bento_service:AraBERTService --port 8000`.
  `/predict` takes `{"text": "..."}` and returns `{"label", "confidence", "model_version"}`,
  the same schema as the FastAPI version. Concurrent requests are merged by a
  `batchable=True` worker (max batch 32, max latency 1000 ms).
- **Hardware:** GPU | Intel64 Family 6 Model 154 Stepping 3, GenuineIntel (the service and Locust ran on the same machine).
- **Load:** `locust -f locustfile.py --headless -u 50 -r 5 -t 60s --host http://localhost:8000 --html reports/locust_report.html`
- **Result:** 4,731 requests, **0 failures**. Median 220 ms, **p95 650 ms**, p99 about 2 s.
- **Report:** `reports/locust_report.html`
