"""
Batch scoring pipeline with automatic Evidently drift monitoring trigger.
"""

from pathlib import Path
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import bentoml

from src.arabic_sentiment.monitor_drift import run_drift_monitoring


def run_batch_scoring(input_data_path: str, output_data_path: str = "data/scored_batch_latest.csv"):
    print(f"[INFO] Loading batch data from {input_data_path}...")
    df = pd.read_csv(input_data_path)
    
    if "text" not in df.columns:
        raise ValueError("Input CSV must contain a 'text' column.")

    # Load model and tokenizer for offline batch inference (or use bentoml client)
    bento_model = bentoml.models.get("arabert_production:latest")
    tokenizer = AutoTokenizer.from_pretrained(bento_model.path)
    model = AutoModelForSequenceClassification.from_pretrained(bento_model.path)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    texts = df["text"].astype(str).tolist()
    
    # 1. Compute text features
    df["text_length"] = df["text"].apply(len)
    
    print("[INFO] Running model predictions on batch...")
    confidences = []
    
    # Process in chunks to prevent OOM
    batch_size = 32
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]
        inputs = tokenizer(
            batch_texts, padding=True, truncation=True, max_length=64, return_tensors="pt"
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs)
            probs = F.softmax(outputs.logits, dim=-1)
            batch_confs, _ = torch.max(probs, dim=-1)
            confidences.extend(batch_confs.cpu().numpy().tolist())

    df["confidence_score"] = confidences

    # 2. Save scored batch
    Path(output_data_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_data_path, index=False)
    print(f"[SUCCESS] Scored batch saved to {output_data_path}")

    # 3. TRIGGER AUTOMATIC DRIFT MONITORING
    print("[INFO] Triggering automatic Evidently drift monitoring...")
    run_drift_monitoring(current_batch_df=df)


if __name__ == "__main__":
    import sys
    import os
    
    # Use command-line argument if provided, otherwise default to "data/test_input.csv"
    input_file = sys.argv[1] if len(sys.argv) > 1 else "data/test_input.csv"
    
    # Ensure the directory and a dummy file exist for quick testing if nothing is provided
    if not os.path.exists(input_file):
        os.makedirs(os.path.dirname(input_file), exist_ok=True)
        pd.DataFrame({'text': ['المنتج ممتاز جدا', 'خدمة سيئة وشحن بطيء']}).to_csv(input_file, index=False)
        print(f"[SETUP] Created a sample test file at {input_file}")

    run_batch_scoring(input_file)