"""Evaluate the baseline and AraBERT on the held-out test set and enforce the quality gate.

Two modes:
  python src/arabic_sentiment/evaluate.py               full evaluation (loads both models),
                                                        writes results/metrics.json, then applies the gate
  python src/arabic_sentiment/evaluate.py --check-only  applies the gate to an existing metrics.json
                                                        (standard library only, used by CI)

Gate: fail (exit code 1) if AraBERT's test macro-F1 is below the baseline's test macro-F1
computed in the same run. All heavy imports live inside run_evaluation() so that
--check-only works in CI without installing torch, transformers, pandas or scikit-learn.
"""

import argparse
import json
import pathlib
import sys

DEFAULT_METRICS_PATH = pathlib.Path("results/metrics.json")
TEST_CSV_PATH = pathlib.Path("data/processed/test.csv")
BASELINE_MODEL_PATH = pathlib.Path("models/baseline_model.joblib")
ARABERT_MODEL_DIR = pathlib.Path("models/production_model")

TEXT_COL = "review_description"
LABEL_COL = "rating"
LABELS = ["negative", "neutral", "positive"]  # fixed order for per-class metrics
MAX_LENGTH = 64  # must match training
BATCH_SIZE = 32

# Documented reference for the baseline's test macro-F1. The gate compares against the value
# computed in the same run; this constant is only used to warn if the baseline has drifted.
REFERENCE_BASELINE_MACRO_F1 = 0.6153
BASELINE_DRIFT_TOLERANCE = 0.005


def check_gate(metrics_path: pathlib.Path = DEFAULT_METRICS_PATH) -> None:
    """Applies the quality gate to a metrics file. Exits with code 1 if the gate fails."""
    if not metrics_path.exists():
        raise FileNotFoundError(f"Missing {metrics_path}. Run the evaluate stage first.")

    with open(metrics_path, "r", encoding="utf-8") as f:
        metrics = json.load(f)

    arabert_f1 = metrics["arabert"]["macro_f1"]
    baseline_f1 = metrics["baseline"]["macro_f1"]

    print(
        f"AraBERT Test Macro-F1: {arabert_f1:.4f} | Computed Baseline: {baseline_f1:.4f} "
        f"| Reference Baseline: {REFERENCE_BASELINE_MACRO_F1:.4f}"
    )

    if abs(baseline_f1 - REFERENCE_BASELINE_MACRO_F1) > BASELINE_DRIFT_TOLERANCE:
        print(
            f"WARNING: computed baseline macro-F1 differs from the documented reference "
            f"({REFERENCE_BASELINE_MACRO_F1:.4f}) by more than {BASELINE_DRIFT_TOLERANCE}. "
            "The baseline or the data split may have changed.",
            file=sys.stderr,
        )

    if arabert_f1 < baseline_f1:
        print(
            "QUALITY GATE FAILED: AraBERT test macro-F1 is below the baseline's.",
            file=sys.stderr,
        )
        sys.exit(1)

    print("QUALITY GATE PASSED: AraBERT test macro-F1 is at or above the baseline's.")


def compute_metrics(y_true, y_pred) -> dict:
    from sklearn.metrics import accuracy_score, f1_score

    per_class = f1_score(y_true, y_pred, labels=LABELS, average=None, zero_division=0)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0)),
        "per_class_f1": {label: float(score) for label, score in zip(LABELS, per_class)},
    }


def load_baseline_predictor(path: pathlib.Path = BASELINE_MODEL_PATH):
    """Returns predict(texts) for the saved baseline.

    Handles either a fitted sklearn Pipeline, or a dict holding the fitted vectorizer and
    classifier under any key names. Fails with the dict's keys if it cannot tell them apart.
    """
    import joblib

    obj = joblib.load(path)
    if hasattr(obj, "predict"):
        return obj.predict
    if not isinstance(obj, dict):
        raise TypeError(f"Unsupported baseline artifact type: {type(obj).__name__}")

    classifier = next((v for v in obj.values() if hasattr(v, "predict")), None)
    vectorizer = next(
        (v for v in obj.values() if hasattr(v, "transform") and not hasattr(v, "predict")), None
    )
    if classifier is None:
        raise ValueError(f"No classifier with .predict found in {path}; keys: {list(obj)}")
    if vectorizer is None:
        return classifier.predict  # a Pipeline stored inside the dict

    def predict(texts):
        return classifier.predict(vectorizer.transform(texts))

    return predict


def predict_arabert(texts: list) -> list:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(str(ARABERT_MODEL_DIR))
    model = AutoModelForSequenceClassification.from_pretrained(str(ARABERT_MODEL_DIR))
    model.to(device)
    model.eval()

    pred_ids = []
    with torch.no_grad():
        for i in range(0, len(texts), BATCH_SIZE):
            batch = texts[i : i + BATCH_SIZE]
            inputs = tokenizer(
                batch, padding=True, truncation=True, max_length=MAX_LENGTH, return_tensors="pt"
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            logits = model(**inputs).logits
            pred_ids.extend(logits.argmax(dim=-1).tolist())

    id2label = model.config.id2label
    return [id2label[idx] for idx in pred_ids]


def run_evaluation() -> None:
    import pandas as pd

    # 1. Load data, failing loudly if anything is off
    df = pd.read_csv(TEST_CSV_PATH)
    missing = {TEXT_COL, LABEL_COL} - set(df.columns)
    if missing:
        raise ValueError(f"{TEST_CSV_PATH} is missing required columns: {sorted(missing)}")
    if df[TEXT_COL].isna().any():
        raise ValueError(f"{TEST_CSV_PATH} contains empty values in '{TEXT_COL}'")
    unknown_labels = set(df[LABEL_COL]) - set(LABELS)
    if unknown_labels:
        raise ValueError(f"Unexpected labels in '{LABEL_COL}': {sorted(unknown_labels)}")

    texts = df[TEXT_COL].astype(str).tolist()
    y_true = df[LABEL_COL].tolist()

    # 2. Baseline
    baseline_predict = load_baseline_predictor()
    baseline_metrics = compute_metrics(y_true, list(baseline_predict(texts)))

    # 3. AraBERT
    arabert_metrics = compute_metrics(y_true, predict_arabert(texts))

    # 4. Save, then apply the gate to what was just written
    results = {
        "n_test_samples": len(texts),
        "baseline": baseline_metrics,
        "arabert": arabert_metrics,
    }
    DEFAULT_METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(DEFAULT_METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"AraBERT per-class F1: {arabert_metrics['per_class_f1']}")
    check_gate(DEFAULT_METRICS_PATH)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Only apply the quality gate to an existing results/metrics.json (no models loaded)",
    )
    args = parser.parse_args()

    if args.check_only:
        check_gate()
    else:
        run_evaluation()


if __name__ == "__main__":
    main()