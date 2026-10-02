"""Tests for drift monitoring: PSI computation, auto-written state, Prometheus export.

Run:  pytest tests/test_drift_monitoring.py -q
No model or live traffic needed: everything uses synthetic data in a temp folder.
"""

import json

import numpy as np
import pandas as pd
import pytest

from src.arabic_sentiment import monitor_drift as md


@pytest.fixture
def isolated(tmp_path, monkeypatch):
    """Point the module at a temp folder so tests never touch real reports/baseline."""
    reports = tmp_path / "reports"
    monkeypatch.setattr(md, "REPORTS_DIR", reports)
    monkeypatch.setattr(md, "LATEST_JSON", reports / "drift_latest.json")
    monkeypatch.setattr(md, "BASELINE_CSV", tmp_path / "reference_baseline.csv")
    return tmp_path


def _frame(seed, n, len_mu, len_sd, conf_a, conf_b):
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "text_length": np.clip(rng.normal(len_mu, len_sd, n), 1, None).round(),
            "confidence_score": rng.beta(conf_a, conf_b, n),
        }
    )


def _write_baseline(tmp_path):
    _frame(1, 2000, 60, 20, 18, 2).to_csv(tmp_path / "reference_baseline.csv", index=False)


def test_prometheus_output_format(isolated):
    md.LATEST_JSON.parent.mkdir(parents=True)
    md.LATEST_JSON.write_text(
        json.dumps(
            {
                "simulated": False,
                "baseline_placeholder": False,
                "psi": {"text_length": 0.05, "confidence_score": 0.31},
                "dataset_drift": False,
            }
        )
    )

    payload, content_type = md.get_prometheus_metrics()
    text = payload.decode("utf-8")

    assert content_type.startswith("text/plain")
    assert 'model_drift_psi{feature_name="text_length"} 0.05' in text
    assert 'model_drift_psi{feature_name="confidence_score"} 0.31' in text
    assert "model_dataset_drift_detected 0.0" in text
    assert "model_drift_batch_simulated 0.0" in text
    # exactly one family definition -> Prometheus will accept the scrape
    assert text.count("# TYPE model_drift_psi gauge") == 1


def test_metrics_empty_before_first_run(isolated):
    payload, _ = md.get_prometheus_metrics()
    assert b"model_drift_psi" not in payload  # and no crash


def test_no_drift_on_same_distribution(isolated):
    pytest.importorskip("evidently")
    _write_baseline(isolated)

    scores = md.run_drift_monitoring(_frame(2, 300, 60, 20, 18, 2))

    assert set(scores) == {"text_length", "confidence_score"}
    assert all(s < md.PSI_THRESHOLD for s in scores.values())
    state = json.loads(md.LATEST_JSON.read_text())
    assert state["dataset_drift"] is False
    assert state["simulated"] is False
    assert state["baseline_placeholder"] is False
    assert (md.REPORTS_DIR / "drift_report.html").exists()


def test_drift_detected_on_shifted_batch(isolated):
    pytest.importorskip("evidently")
    _write_baseline(isolated)

    # much shorter texts and much lower confidence than the baseline
    scores = md.run_drift_monitoring(_frame(3, 300, 8, 2, 5, 5))

    assert scores["text_length"] > md.PSI_THRESHOLD
    assert scores["confidence_score"] > md.PSI_THRESHOLD
    assert json.loads(md.LATEST_JSON.read_text())["dataset_drift"] is True
    assert b"model_dataset_drift_detected 1.0" in md.get_prometheus_metrics()[0]


def test_simulated_batch_is_flagged(isolated, capsys):
    pytest.importorskip("evidently")

    md.run_drift_monitoring()  # no batch passed -> simulated

    assert "SIMULATED" in capsys.readouterr().out
    assert json.loads(md.LATEST_JSON.read_text())["simulated"] is True
    assert b"model_drift_batch_simulated 1.0" in md.get_prometheus_metrics()[0]