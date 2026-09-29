import shutil
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from mlflow.tracking import MlflowClient
from sklearn.metrics import accuracy_score, classification_report, f1_score
from torch import nn
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

# Environment & MLflow Setup
MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"
EXPERIMENT_NAME = "arabic-sentiment-transformers"

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(EXPERIMENT_NAME)

LABEL2ID = {"negative": 0, "neutral": 1, "positive": 2}
ID2LABEL = {0: "negative", 1: "neutral", 2: "positive"}

# Local copy of the winning model. This is the DVC output of the train_arabert
# stage, and it is what evaluate.py loads to compute the test macro-F1.
PRODUCTION_MODEL_DIR = Path("models/production_model")
REGISTRY_MODEL_NAME = "ArabicSentiment"


class WeightedLossTrainer(Trainer):
    """Custom Trainer overriding compute_loss to handle class imbalance with weighted CrossEntropy."""

    def __init__(self, class_weights: torch.FloatTensor, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")

        weight = self.class_weights.to(logits.device)
        loss_fct = nn.CrossEntropyLoss(weight=weight)
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))

        return (loss, outputs) if return_outputs else loss


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)

    acc = accuracy_score(labels, preds)
    macro_f1 = f1_score(labels, preds, average="macro")

    report = classification_report(
        labels,
        preds,
        target_names=["negative", "neutral", "positive"],
        output_dict=True,
        zero_division=0,
    )

    metrics = {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "neg_precision": report["negative"]["precision"],
        "neg_recall": report["negative"]["recall"],
        "neg_f1": report["negative"]["f1-score"],
        "neu_precision": report["neutral"]["precision"],
        "neu_recall": report["neutral"]["recall"],
        "neu_f1": report["neutral"]["f1-score"],
        "pos_precision": report["positive"]["precision"],
        "pos_recall": report["positive"]["recall"],
        "pos_f1": report["positive"]["f1-score"],
    }
    return metrics


def check_gpu_status() -> bool:
    """Detects CUDA hardware availability and prints GPU name and VRAM."""
    cuda_available = torch.cuda.is_available()
    print("=" * 60)
    print(" GPU Hardware Status Check:")
    if cuda_available:
        gpu_name = torch.cuda.get_device_name(0)
        vram_bytes = torch.cuda.get_device_properties(0).total_memory
        vram_gb = vram_bytes / (1024**3)
        print("  CUDA Available : TRUE")
        print(f"  GPU Device Name: {gpu_name}")
        print(f"  Total VRAM     : {vram_gb:.2f} GB")
        print("  FP16 Precision : ENABLED")
    else:
        print("  CUDA Available : FALSE (Warning: Running on CPU will be slow)")
    print("=" * 60)
    return cuda_available


def save_production_model(model, tokenizer, target_dir: Path = PRODUCTION_MODEL_DIR) -> None:
    """Saves model + tokenizer to target_dir, replacing any previous contents.

    Writes to a sibling temp folder first and swaps it in afterwards, so a crash
    mid-save never leaves a half-written production_model/ behind.
    """
    tmp_dir = target_dir.with_name(target_dir.name + "_tmp")
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.parent.mkdir(parents=True, exist_ok=True)

    model.save_pretrained(tmp_dir)
    tokenizer.save_pretrained(tmp_dir)

    if target_dir.exists():
        shutil.rmtree(target_dir)
    tmp_dir.rename(target_dir)
    print(f" Saved best-so-far model + tokenizer to {target_dir}")


def run_dry_run_test(model_name: str, train_df: pd.DataFrame, class_weights: torch.FloatTensor):
    """Runs a 1-epoch dry run on a 2,000-row subset to measure peak VRAM allocated."""
    print("\n--- Running GPU VRAM Dry-Run Test (2,000 rows, 1 Epoch) ---")

    subset_df = train_df.sample(n=min(2000, len(train_df)), random_state=42)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def tokenize_fn(examples):
        return tokenizer(examples["review_description"], truncation=True, max_length=64)

    subset_ds = Dataset.from_pandas(subset_df[["review_description", "rating"]])
    subset_ds = subset_ds.map(lambda x: {"label": LABEL2ID[str(x["rating"])]}).map(
        tokenize_fn, batched=True
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=3,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    use_fp16 = torch.cuda.is_available()

    training_args = TrainingArguments(
        output_dir="./results/dry_run",
        num_train_epochs=1,
        per_device_train_batch_size=8,
        gradient_accumulation_steps=4,
        fp16=use_fp16,
        logging_steps=10,
        save_strategy="no",
        eval_strategy="no",
        report_to="none",
    )

    trainer = WeightedLossTrainer(
        class_weights=class_weights,
        model=model,
        args=training_args,
        train_dataset=subset_ds,
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
    )

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    trainer.train()

    if torch.cuda.is_available():
        peak_vram_mb = torch.cuda.max_memory_allocated() / (1024**2)
        print(f" Dry-Run Complete! Peak VRAM Allocated: {peak_vram_mb:.2f} MB")
    else:
        print(" Dry-Run Complete on CPU!")
    print("------------------------------------------------------------------\n")


def run_experiment(
    model_name: str,
    lr: float,
    epochs: int = 3,
    best_val_f1_so_far: float = float("-inf"),
) -> dict[str, Any]:
    """Trains one config, logs it to MLflow, and saves it locally if it is the best so far.

    Selection is on validation macro-F1 only. Test metrics are logged for reporting
    but never used to pick the model.
    """
    print(f"\n Starting Full Run: model={model_name}, lr={lr}, micro_bs=8, accum=4 (effective_bs=32)")

    # Load Data Splits
    train_df = pd.read_csv("data/processed/train.csv")
    val_df = pd.read_csv("data/processed/val.csv")
    test_df = pd.read_csv("data/processed/test.csv")

    # Calculate Class Weights
    class_counts = train_df["rating"].map(LABEL2ID).value_counts().sort_index().values
    total_samples = len(train_df)
    class_weights = torch.FloatTensor(total_samples / (len(class_counts) * class_counts))

    # Tokenization (Capped at 64 tokens)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def tokenize_fn(examples):
        return tokenizer(examples["review_description"], truncation=True, max_length=64)

    train_ds = Dataset.from_pandas(train_df[["review_description", "rating"]])
    val_ds = Dataset.from_pandas(val_df[["review_description", "rating"]])
    test_ds = Dataset.from_pandas(test_df[["review_description", "rating"]])

    train_ds = train_ds.map(lambda x: {"label": LABEL2ID[str(x["rating"])]}).map(
        tokenize_fn, batched=True
    )
    val_ds = val_ds.map(lambda x: {"label": LABEL2ID[str(x["rating"])]}).map(
        tokenize_fn, batched=True
    )
    test_ds = test_ds.map(lambda x: {"label": LABEL2ID[str(x["rating"])]}).map(
        tokenize_fn, batched=True
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=3,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    output_dir = f"./results/{model_name.replace('/', '_')}_lr{lr}_bs32"
    use_fp16 = torch.cuda.is_available()

    training_args = TrainingArguments(
        output_dir=output_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=lr,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=16,
        gradient_accumulation_steps=4,
        fp16=use_fp16,
        num_train_epochs=epochs,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        logging_steps=50,
        save_total_limit=1,
        report_to="none",
    )

    trainer = WeightedLossTrainer(
        class_weights=class_weights,
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics,
    )

    with mlflow.start_run(run_name=f"{model_name.split('/')[-1]}-lr{lr}-bs32") as run:
        mlflow.log_params(
            {
                "model_name": model_name,
                "learning_rate": lr,
                "per_device_train_batch_size": 8,
                "gradient_accumulation_steps": 4,
                "effective_batch_size": 32,
                "fp16": use_fp16,
                "max_length": 64,
                "epochs": epochs,
            }
        )

        trainer.train()

        # Evaluate on Validation Set
        val_metrics = trainer.evaluate(eval_dataset=val_ds)
        val_logs = {
            f"val_{k.replace('eval_', '')}": v
            for k, v in val_metrics.items()
            if isinstance(v, (int, float))
        }
        mlflow.log_metrics(val_logs)

        # Evaluate on Held-out Test Set (reporting only, never used for selection)
        test_metrics = trainer.evaluate(eval_dataset=test_ds)
        test_logs = {
            f"test_{k.replace('eval_', '')}": v
            for k, v in test_metrics.items()
            if isinstance(v, (int, float))
        }
        mlflow.log_metrics(test_logs)

        # Log PyTorch Model Artifact. `name` replaces the deprecated `artifact_path`,
        # and the returned model_uri is what gets registered later.
        model_info = mlflow.transformers.log_model(
            transformers_model={"model": trainer.model, "tokenizer": tokenizer},
            name="model",
            task="text-classification",
        )

        run_id = run.info.run_id
        val_f1 = val_logs["val_macro_f1"]

        # trainer.model is the best-epoch model (load_best_model_at_end=True).
        # Keep a local copy only when this run beats every earlier run.
        saved_as_best = val_f1 > best_val_f1_so_far
        if saved_as_best:
            save_production_model(trainer.model, tokenizer)

        print(
            f"Run Finished. Run ID: {run_id} | Val Macro-F1: {val_f1:.4f} | "
            f"Test Macro-F1: {test_logs['test_macro_f1']:.4f} | "
            f"Test Neutral F1: {test_logs['test_neu_f1']:.4f}"
        )

    return {
        "run_id": run_id,
        "val_macro_f1": val_f1,
        "model_name": model_name,
        "model_uri": model_info.model_uri,
        "saved_as_best": saved_as_best,
    }


if __name__ == "__main__":
    check_gpu_status()

    train_df = pd.read_csv("data/processed/train.csv")
    class_counts = train_df["rating"].map(LABEL2ID).value_counts().sort_index().values
    total_samples = len(train_df)
    class_weights = torch.FloatTensor(total_samples / (len(class_counts) * class_counts))

    # 1. Run dry-run validation first on 2,000 rows
    run_dry_run_test("aubmindlab/bert-base-arabertv2", train_df, class_weights)

    # 2. Hyperparameter Grid Execution
    experiments = [
        {"model_name": "aubmindlab/bert-base-arabertv2", "lr": 2e-5},
        {"model_name": "aubmindlab/bert-base-arabertv2", "lr": 3e-5},
        {"model_name": "aubmindlab/bert-base-arabertv2", "lr": 5e-5},
        {"model_name": "CAMeL-Lab/bert-base-arabic-camelbert-mix", "lr": 2e-5},
        {"model_name": "CAMeL-Lab/bert-base-arabic-camelbert-mix", "lr": 3e-5},
    ]

    best_run = None
    best_val_f1 = float("-inf")
    for config in experiments:
        res = run_experiment(**config, best_val_f1_so_far=best_val_f1)
        if res["saved_as_best"]:
            best_run = res
            best_val_f1 = res["val_macro_f1"]

    if best_run is None:
        raise RuntimeError("No experiment produced a model; nothing to register.")

    print("\n" + "=" * 42)
    print(f"Best Run ID: {best_run['run_id']} ({best_run['model_name']})")
    print(f"Best Val Macro-F1: {best_run['val_macro_f1']:.4f}")
    print(f"Local model saved at: {PRODUCTION_MODEL_DIR}")
    print("=" * 42)

    # 3. Promote Best Model in MLflow Registry (same model that was saved locally)
    client = MlflowClient()
    registered_model = mlflow.register_model(
        model_uri=best_run["model_uri"], name=REGISTRY_MODEL_NAME
    )

    client.set_registered_model_alias(
        name=REGISTRY_MODEL_NAME,
        alias="Production",
        version=registered_model.version,
    )
    print(f"Model version {registered_model.version} promoted to 'Production' alias in MLflow Registry.")