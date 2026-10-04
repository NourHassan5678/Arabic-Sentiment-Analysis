"""Sanity checks for the monitoring configuration files."""

import json
from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml")

MON = Path(__file__).resolve().parents[1] / "monitoring"


def test_dashboard_has_latency_and_psi_panels():
    path = MON / "grafana" / "dashboards" / "arabic_sentiment.json"
    dash = json.loads(path.read_text(encoding="utf-8"))
    assert len(dash["panels"]) >= 2
    titles = " ".join(p["title"].lower() for p in dash["panels"])
    assert "p95" in titles
    assert "psi" in titles


def test_alert_rule_fires_above_025():
    text = (MON / "alert_rules.yml").read_text(encoding="utf-8")
    rule = yaml.safe_load(text)["groups"][0]["rules"][0]
    assert rule["alert"] == "HighFeatureDrift"
    assert "> 0.25" in rule["expr"]


def test_prometheus_config_loads_rules_and_scrapes_metrics():
    cfg = yaml.safe_load((MON / "prometheus.yml").read_text(encoding="utf-8"))
    assert cfg["rule_files"]
    assert cfg["scrape_configs"][0]["metrics_path"] == "/metrics"
