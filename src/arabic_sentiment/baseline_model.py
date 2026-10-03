from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import joblib
import mlflow
import pandas as pd
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.pipeline import Pipeline
from transformers import AutoModelForSequenceClassification, AutoTokenizer


class SentimentModel:
    def __init__(self, version: str = "v0.1.0-baseline"):
        self.version = version
        self.pipeline: Pipeline | None = None
        
    def train(self, df_train: pd.DataFrame, text_col: str = "review_description", label_col: str = "rating") -> None:
        X_train = df_train[text_col].astype(str)
        y_train = df_train[label_col].astype(str)
        
        self.pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(max_features=10000, ngram_range=(1, 2))),
            ('clf', LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42))
        ])
        
        print("Training baseline model...")
        self.pipeline.fit(X_train, y_train)
        print("Training complete.")

    def evaluate(self, df_eval: pd.DataFrame, dataset_name: str = "Validation", text_col: str = "review_description", label_col: str = "rating") -> dict[str, Any]:
        if not self.pipeline:
            raise ValueError("Model must be trained or loaded before evaluation.")
            
        X_eval = df_eval[text_col].astype(str)
        y_eval = df_eval[label_col].astype(str)
        
        y_pred = self.pipeline.predict(X_eval)
        
        # Calculate metrics as standard Python floats
        acc = float(accuracy_score(y_eval, y_pred))
        macro_f1 = float(f1_score(y_eval, y_pred, average='macro'))
        
        # Print for visual inspection
        print(f"\n=== {dataset_name} Evaluation Metrics ===")
        print(f"Accuracy: {acc:.4f}")
        print(f"Macro-F1: {macro_f1:.4f}")
        print("\nPer-Class Precision / Recall / F1:")
        print(classification_report(y_eval, y_pred))
        
        # Return as a structured dictionary for tracking/automation
        return {
            "accuracy": acc,
            "macro_f1": macro_f1,
            "report": classification_report(y_eval, y_pred, output_dict=True)
        }

    def predict(self, text: str) -> dict[str, Any]:
        if not self.pipeline:
            raise ValueError("Model must be trained or loaded before prediction.")
            
        probs = self.pipeline.predict_proba([text])[0]
        pred_idx = probs.argmax()
        
        return {
            # Explicitly cast the numpy string to a native Python string
            "label": str(self.pipeline.classes_[pred_idx]),
            "confidence": round(float(probs[pred_idx]), 4),
            "model_version": self.version
        }

    def save(self, filepath: str) -> None:
        if not self.pipeline:
            raise ValueError("No trained pipeline to save.")
        joblib.dump({"pipeline": self.pipeline, "version": self.version}, filepath)
        print(f"Model saved to {filepath}")

    def load(self, filepath: str) -> None:
        data = joblib.load(filepath)
        self.pipeline = data["pipeline"]
        self.version = data["version"]
        print(f"Model {self.version} loaded from {filepath}")


class TransformerSentimentModel:
    def __init__(self, model_alias_uri: str = "models:/ArabicSentiment@Production"):
        self.model_alias_uri = model_alias_uri
        self.tokenizer = None
        self.model = None
        self.version = "v0.2.0-arabert"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def load(self, local_path: str | None = None) -> None:
        """Loads model and tokenizer from a local directory or the MLflow Registry."""
        path_to_check = local_path if local_path else "models/production_model"
        path_obj = Path(path_to_check)

        # 1. Direct local directory loading via Hugging Face Transformers
        # Checks if it's a standard HF directory rather than an MLflow artifact
        if path_obj.exists() and path_obj.is_dir() and (path_obj / "config.json").exists():
            print(f"[INFO] Loading Transformer directly from HuggingFace directory: {path_obj}")
            self.tokenizer = AutoTokenizer.from_pretrained(str(path_obj))
            self.model = AutoModelForSequenceClassification.from_pretrained(str(path_obj)).to(self.device)
            self.model.eval()
            return

        # 2. MLflow Registry fallback
        model_uri = local_path or self.model_alias_uri
        tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
        mlflow.set_tracking_uri(tracking_uri)
        print(f"[INFO] Loading Transformer from MLflow tracking URI: {tracking_uri} with URI: {model_uri}")
        
        pipeline = mlflow.transformers.load_model(model_uri)
        self.model = pipeline.model.to(self.device)
        self.tokenizer = pipeline.tokenizer
        self.model.eval()
        print(f"Loaded Transformer model successfully (source: {model_uri})")
            
    def predict(self, text: str) -> dict[str, Any]:
        if not self.model or not self.tokenizer:
            raise ValueError("Transformer model is not loaded.")

        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
        inputs = inputs.to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)[0]

        pred_idx = torch.argmax(probs).item()
        confidence = float(probs[pred_idx].item())
        
        # Fallback to a default mapping if id2label isn't properly configured in the HF config
        if hasattr(self.model.config, "id2label") and self.model.config.id2label:
            label = self.model.config.id2label[pred_idx]
        else:
            label_map = {0: "NEGATIVE", 1: "NEUTRAL", 2: "POSITIVE"}
            label = label_map.get(pred_idx, "UNKNOWN")

        return {
            "label": str(label),
            "confidence": round(confidence, 4),
            "model_version": self.version
        }


if __name__ == "__main__":
    train_data = pd.read_csv("data/processed/train.csv")
    val_data = pd.read_csv("data/processed/val.csv")
    test_data = pd.read_csv("data/processed/test.csv")
    
    # Keeping the original script functionality intact for regression testing
    model = SentimentModel()
    model.train(train_data)
    
    # Evaluate and capture metrics (can be logged to MLflow, W&B, or standard output)
    val_metrics = model.evaluate(val_data, dataset_name="Validation")
    test_metrics = model.evaluate(test_data, dataset_name="Held-Out Test")
    
    model.save("models/baseline_model.joblib")