"""
BentoML service for AraBERT sentiment analysis with Prometheus metrics exposure.
"""

import bentoml
from fastapi import FastAPI, Response
import torch
import torch.nn.functional as F

# Import the Prometheus metrics exporter from monitor_drift
from src.arabic_sentiment.monitor_drift import get_prometheus_metrics

# 1. Create a native FastAPI app for HTTP GET requests
metrics_app = FastAPI()


@metrics_app.get("")
@metrics_app.get("/")
def drift_metrics_endpoint():
    """Exposes Evidently PSI drift scores in Prometheus format."""
    data, content_type = get_prometheus_metrics()
    return Response(content=data, media_type=content_type)


# 2. Mount the FastAPI app specifically at path="/drift_metrics"
@bentoml.mount_asgi_app(metrics_app, path="/drift_metrics")
@bentoml.service(name="arabert_final_benchmark_service")
class AraBERTService:
    def __init__(self):
        # Move model lookup inside __init__ to resolve import-time warnings
        bento_model = bentoml.models.get("arabert_production:latest")

        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(bento_model.path)
        self.model = AutoModelForSequenceClassification.from_pretrained(bento_model.path)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        self.model.eval()
        self.id2label = getattr(
            self.model.config,
            "id2label",
            {0: "negative", 1: "neutral", 2: "positive"},
        )

    @bentoml.api(batchable=True, max_batch_size=32, max_latency_ms=200)
    def predict(self, text: list[str]) -> list[dict]:
        inputs = self.tokenizer(
            text, padding=True, truncation=True, max_length=64, return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = F.softmax(outputs.logits, dim=-1)
            confidences, predictions = torch.max(probs, dim=-1)

        results = []
        for pred, conf in zip(predictions, confidences):
            label_str = self.id2label.get(pred.item(), str(pred.item()))
            results.append(
                {
                    "label": label_str,
                    "confidence": round(float(conf.item()), 4),
                    "model_version": "final-serving-benchmark",
                }
            )
        return results