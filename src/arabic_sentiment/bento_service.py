"""
BentoML service for AraBERT sentiment analysis: the FINAL serving benchmark target.
(There is no optimization phase in this project, so there is nothing to compare against.)

Public API (same contract as the FastAPI version):
  POST /predict  {"text": "..."} -> {"label": ..., "confidence": ..., "model_version": ...}

Inside, a separate batchable worker (AraBERTModel.classify, batchable=True) merges concurrent
single-text requests into one forward pass.

Drift metrics (Prometheus text):
  GET /metrics        -> BentoML's built-in metrics + drift metrics appended via __metrics__
  GET /drift/metrics  -> drift metrics only
"""

import bentoml
import torch
import torch.nn.functional as F
from fastapi import FastAPI, Response

from src.arabic_sentiment.monitor_drift import get_prometheus_metrics

MODEL_TAG = "arabert_production:latest"

drift_app = FastAPI()


@drift_app.get("/metrics")
def drift_metrics_endpoint():
    """Evidently PSI drift scores in Prometheus text format."""
    data, content_type = get_prometheus_metrics()
    return Response(content=data, media_type=content_type)


@bentoml.service(name="arabert_model")
class AraBERTModel:
    """Batchable inference worker. Holds the model; the public service never loads it."""

    def __init__(self):
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        bento_model = bentoml.models.get(MODEL_TAG)
        self.model_version = str(bento_model.tag)
        self.tokenizer = AutoTokenizer.from_pretrained(bento_model.path)
        self.model = AutoModelForSequenceClassification.from_pretrained(bento_model.path)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        self.model.eval()
        self.id2label = self.model.config.id2label

        self._infer(["تجربة"])  # warm-up so the first real request is not a slow outlier

    def _infer(self, texts: list[str]) -> list[dict]:
        inputs = self.tokenizer(
            texts, padding=True, truncation=True, max_length=64, return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            probs = F.softmax(self.model(**inputs).logits, dim=-1)
            confidences, predictions = torch.max(probs, dim=-1)

        return [
            {
                "label": self.id2label.get(pred.item(), str(pred.item())),
                "confidence": round(float(conf.item()), 4),
                "model_version": self.model_version,
            }
            for pred, conf in zip(predictions, confidences)
        ]

    @bentoml.api(batchable=True, max_batch_size=32, max_latency_ms=1000)
    def classify(self, texts: list[str]) -> list[dict]:
        return self._infer(texts)


# Order matters: mount_asgi_app closest to the class, @bentoml.service on top.
@bentoml.service(name="arabert_final_benchmark_service")
@bentoml.mount_asgi_app(drift_app, path="/drift")
class AraBERTService:
    model = bentoml.depends(AraBERTModel)

    def __metrics__(self, text: str) -> str:
        """Append drift metrics to BentoML's built-in /metrics output."""
        try:
            drift, _ = get_prometheus_metrics()
            return text.rstrip("\n") + "\n" + drift.decode("utf-8")
        except Exception as e:  # noqa: BLE001
            print(f"[WARNING] Could not append drift metrics: {e}")
            return text

    @bentoml.api
    def predict(self, text: str) -> dict:
        if not text.strip():
            raise bentoml.exceptions.InvalidArgument("text must not be empty")
        return self.model.classify([text])[0]