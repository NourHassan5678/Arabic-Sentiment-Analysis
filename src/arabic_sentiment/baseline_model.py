import joblib
import pandas as pd
import torch
import mlflow
from typing import Dict, Any, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score, f1_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification

class SentimentModel:
    def __init__(self, version: str = "v0.1.0-baseline"):
        self.version = version
        self.pipeline: Optional[Pipeline] = None
        
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

    def evaluate(self, df_eval: pd.DataFrame, dataset_name: str = "Validation", text_col: str = "review_description", label_col: str = "rating") -> Dict[str, Any]:
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

    def predict(self, text: str) -> Dict[str, Any]:
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

    def load(self, local_path: str = None) -> None:
        """Loads model and tokenizer artifacts from MLflow Registry or local directory."""
        if local_path:
            self.tokenizer = AutoTokenizer.from_pretrained(local_path)
            self.model = AutoModelForSequenceClassification.from_pretrained(local_path)
        else:
            pipeline = mlflow.transformers.load_model(self.model_alias_uri)
            self.model = pipeline.model
            self.tokenizer = pipeline.tokenizer

        self.model.eval()
        print(f"Loaded Transformer model successfully (source: {local_path or self.model_alias_uri})")
        
    def predict(self, text: str) -> Dict[str, Any]:
        if not self.model or not self.tokenizer:
            raise ValueError("Transformer model is not loaded.")

        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)[0]

        pred_idx = torch.argmax(probs).item()
        confidence = float(probs[pred_idx].item())
        label = self.model.config.id2label[pred_idx]

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