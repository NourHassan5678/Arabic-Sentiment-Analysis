"""API tests against an isolated, temporary MLflow registry.

Nothing here touches the repo's mlflow.db or mlruns/, so the same tests pass locally and on a
clean CI runner. The registry holds a tiny randomly initialised BERT classifier logged with the
same MLflow flavor as the real model (mlflow.transformers), so the API loads and post-processes it
exactly like the real one. Predictions are meaningless; these tests check plumbing and schema only.
"""

import mlflow
import pytest
from fastapi.testclient import TestClient
from mlflow import MlflowClient

MODEL_NAME = "ArabicSentiment"
ALIAS = "Production"
LABELS = {"negative", "neutral", "positive"}


def _log_tiny_sentiment_model(work_dir):
    """Logs a tiny 3-class BERT (same id2label as the real model) and returns the ModelInfo."""
    import mlflow.transformers
    from transformers import (
        BertConfig,
        BertForSequenceClassification,
        BertTokenizerFast,
    )

    vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"] + list("abcdefghijklmnopqrstuvwxyz")
    vocab_file = work_dir / "vocab.txt"
    vocab_file.write_text("\n".join(vocab), encoding="utf-8")
    tokenizer = BertTokenizerFast(vocab_file=str(vocab_file))

    config = BertConfig(
        vocab_size=len(vocab),
        hidden_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=64,
        max_position_embeddings=64,
        num_labels=3,
        id2label={0: "negative", 1: "neutral", 2: "positive"},
        label2id={"negative": 0, "neutral": 1, "positive": 2},
    )
    model = BertForSequenceClassification(config)

    return mlflow.transformers.log_model(
        transformers_model={"model": model, "tokenizer": tokenizer},
        name="model",
        task="text-classification",
    )


@pytest.fixture(scope="module")
def registry(tmp_path_factory):
    """Temporary MLflow store with a registered model carrying the 'Production' alias."""
    work_dir = tmp_path_factory.mktemp("mlflow_registry")
    db_uri = f"sqlite:///{(work_dir / 'mlflow.db').as_posix()}"
    artifact_root = (work_dir / "artifacts").as_uri()  # keeps artifacts out of the repo's mlruns/

    previous_tracking_uri = mlflow.get_tracking_uri()
    previous_registry_uri = mlflow.get_registry_uri()
    try:
        with pytest.MonkeyPatch.context() as mp:
            # Env vars cover an API that reads them; set_*_uri covers one that relies on the global.
            mp.setenv("MLFLOW_TRACKING_URI", db_uri)
            mp.setenv("MLFLOW_REGISTRY_URI", db_uri)
            mp.delenv("MODEL_PATH", raising=False)  # api.py loads from this path if set
            mlflow.set_tracking_uri(db_uri)
            mlflow.set_registry_uri(db_uri)

            experiment_id = mlflow.create_experiment("api-tests", artifact_location=artifact_root)
            with mlflow.start_run(experiment_id=experiment_id):
                model_info = _log_tiny_sentiment_model(work_dir)

            version = mlflow.register_model(model_info.model_uri, MODEL_NAME).version
            MlflowClient().set_registered_model_alias(MODEL_NAME, ALIAS, version)

            yield {"version": str(version)}
    finally:
        mlflow.set_tracking_uri(previous_tracking_uri)
        mlflow.set_registry_uri(previous_registry_uri)


@pytest.fixture(scope="module")
def client(registry):
    # Imported here, not at module top, so a model load at import time also hits the temporary registry.
    from arabic_sentiment.api import app

    with TestClient(app) as test_client:
        yield test_client


def test_production_model_loads_from_registry(registry):
    version = MlflowClient().get_model_version_by_alias(MODEL_NAME, ALIAS)
    assert str(version.version) == registry["version"]

    model = mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}@{ALIAS}")
    assert model is not None


def test_health(client):
    assert client.get("/health").status_code == 200


def test_predict_returns_expected_schema(client):
    response = client.post("/predict", json={"text": "المنتج ممتاز وسريع"})
    assert response.status_code == 200, response.text

    body = response.json()
    assert set(body) == {"label", "confidence", "model_version"}
    assert body["label"] in LABELS
    assert isinstance(body["confidence"], float)
    assert 0.0 <= body["confidence"] <= 1.0
    assert body["model_version"]  # non-empty; tighten once the exact format is confirmed


def test_predict_rejects_empty_text(client):
    response = client.post("/predict", json={"text": ""})
    assert response.status_code in (400, 422)


def test_predict_rejects_missing_text(client):
    response = client.post("/predict", json={})
    assert response.status_code == 422