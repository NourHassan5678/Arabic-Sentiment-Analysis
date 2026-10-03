"""
BentoML service for AraBERT sentiment analysis with Prometheus drift metrics.

Drift PSI is exposed two ways:
  1. GET /metrics        -> BentoML's built-in metrics + drift metrics appended via __metrics__
  2. GET /drift/metrics  -> drift metrics only (mounted FastAPI app; handy for debugging)
"""

import bentoml
import torch
import torch.nn.functional as F
from fastapi import FastAPI, Response

from src.arabic_sentiment.monitor_drift import get_prometheus_metrics

drift_app = FastAPI()


@drift_app.get("/metrics")
def drift_metrics_endpoint():
    """Evidently PSI drift scores in Prometheus text format."""
    data, content_type = get_prometheus_metrics()
    return Response(content=data, media_type=content_type)


# Order matters: mount_asgi_app goes closest to the class, @bentoml.service on top.
# Mount under a prefix (not "/"), otherwise it shadows / collides with BentoML's own routes.
@bentoml.service(name="arabert_final_benchmark_service")
@bentoml.mount_asgi_app(drift_app, path="/drift")
class AraBERTService:
    def __init__(self):
        bento_model = bentoml.models.get("arabert_production:latest")

        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(bento_model.path)
        self.model = AutoModelForSequenceClassification.from_pretrained(bento_model.path)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        self.model.eval()
        self.id2label = self.model.config.id2label

    def __metrics__(self, text: str) -> str:
        """Append drift metrics to BentoML's built-in /metrics output."""
        try:
            drift, _ = get_prometheus_metrics()
            return text.rstrip("\n") + "\n" + drift.decode("utf-8")
        except Exception as e:  # noqa: BLE001
            print(f"[WARNING] Could not append drift metrics: {e}")
            return text

    @bentoml.api(batchable=True, max_batch_size=32, max_latency_ms=200)
    def predict(self, text: list[str]) -> list[dict]:
        inputs = self.tokenizer(
            text, padding=True, truncation=True, max_length=64, return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            probs = F.softmax(self.model(**inputs).logits, dim=-1)
            confidences, predictions = torch.max(probs, dim=-1)

        return [
            {
                "label": self.id2label.get(pred.item(), str(pred.item())),
                "confidence": round(float(conf.item()), 4),
                "model_version": "final-serving-benchmark",
            }
            for pred, conf in zip(predictions, confidences)
        ]