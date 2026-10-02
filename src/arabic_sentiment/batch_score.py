"""
Batch scoring pipeline with automatic Evidently drift monitoring trigger.

Usage:
  python -m src.arabic_sentiment.batch_score [input.csv] [--text-col COL] [--sample N]
      score a batch -> data/scored_batch_latest.csv -> run drift monitoring

  python -m src.arabic_sentiment.batch_score data/processed/train.csv \
      --baseline --text-col review_description --sample 2000
      score training texts with the SAME model -> data/reference_baseline.csv
      (no drift run). This file becomes the PSI baseline.
"""

import argparse
from pathlib import Path

import bentoml
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.arabic_sentiment.monitor_drift import run_drift_monitoring


def run_batch_scoring(
    input_data_path: str,
    output_data_path: str = "data/scored_batch_latest.csv",
    trigger_drift: bool = True,
    text_column: str = "text",
    sample_size: int = None,
):
    print(f"[INFO] Loading batch data from {input_data_path}...")
    df = pd.read_csv(input_data_path)

    if text_column not in df.columns:
        raise ValueError(
            f"Input CSV must contain a '{text_column}' column (found: {list(df.columns)}). "
            "Use --text-col to name the right one."
        )
    if text_column != "text":
        df = df.rename(columns={text_column: "text"})

    # Drop empty rows and force str so text_length matches what the model actually sees
    df = df.dropna(subset=["text"]).reset_index(drop=True)
    df["text"] = df["text"].astype(str)
    if sample_size and sample_size < len(df):
        df = df.sample(sample_size, random_state=42).reset_index(drop=True)
        print(f"[INFO] Using a random sample of {sample_size} rows.")
    if df.empty:
        raise ValueError("Input CSV has no usable rows in the text column.")

    # Same model as the BentoML service
    bento_model = bentoml.models.get("arabert_production:latest")
    tokenizer = AutoTokenizer.from_pretrained(bento_model.path)
    model = AutoModelForSequenceClassification.from_pretrained(bento_model.path)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    texts = df["text"].tolist()
    df["text_length"] = df["text"].str.len()

    print("[INFO] Running model predictions on batch...")
    confidences = []

    batch_size = 32  # chunks to prevent OOM
    for i in range(0, len(texts), batch_size):
        inputs = tokenizer(
            texts[i : i + batch_size],
            padding=True,
            truncation=True,
            max_length=64,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            probs = F.softmax(model(**inputs).logits, dim=-1)
            batch_confs, _ = torch.max(probs, dim=-1)
            confidences.extend(batch_confs.cpu().numpy().tolist())

    df["confidence_score"] = confidences

    Path(output_data_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_data_path, index=False)
    print(f"[SUCCESS] Scored batch saved to {output_data_path}")

    if trigger_drift:
        print("[INFO] Triggering automatic Evidently drift monitoring...")
        run_drift_monitoring(current_batch_df=df)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file", nargs="?", default="data/test_input.csv")
    parser.add_argument(
        "--baseline",
        action="store_true",
        help="Score the input as the training baseline (data/reference_baseline.csv), no drift run.",
    )
    parser.add_argument("--text-col", default="text", help="Name of the text column in the CSV.")
    parser.add_argument("--sample", type=int, default=None, help="Score only N random rows.")
    args = parser.parse_args()

    input_path = Path(args.input_file)
    if not input_path.exists():
        input_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"text": ["المنتج ممتاز جدا", "خدمة سيئة وشحن بطيء"]}).to_csv(
            input_path, index=False
        )
        print(f"[SETUP] Created a sample test file at {input_path}")

    if args.baseline:
        run_batch_scoring(
            str(input_path),
            "data/reference_baseline.csv",
            trigger_drift=False,
            text_column=args.text_col,
            sample_size=args.sample,
        )
    else:
        run_batch_scoring(
            str(input_path),
            text_column=args.text_col,
            sample_size=args.sample,
        )