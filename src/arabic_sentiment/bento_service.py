import bentoml
import torch
import torch.nn.functional as F


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