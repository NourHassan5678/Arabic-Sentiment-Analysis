import bentoml
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_DIR = "models/production_model"
BENTO_MODEL_NAME = "arabert_production"

def save_arabert_to_bento():
    print(f"Loading fine-tuned model from {MODEL_DIR}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)

    # Use bentoml.models.create to avoid deprecation warnings and assertion errors
    with bentoml.models.create(
        name=BENTO_MODEL_NAME,
        labels={
            "stage": "final_serving_benchmark",
            "framework": "pytorch",
            "architecture": "arabert"
        },
        metadata={
            "model_version": "v0.2.0-arabert",
            "optimization_phase": "false"
        }
    ) as model_ref:
        model.save_pretrained(model_ref.path)
        tokenizer.save_pretrained(model_ref.path)
        print(f"Model saved successfully to BentoStore: {model_ref.tag}")

if __name__ == "__main__":
    save_arabert_to_bento()