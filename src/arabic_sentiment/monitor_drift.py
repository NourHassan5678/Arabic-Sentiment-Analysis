"""Drift monitoring: PSI for text_length and confidence_score (Evidently 0.7+ + Prometheus).

Flow:
  batch_score.py  -> run_drift_monitoring(df) -> Evidently PSI -> reports/drift_latest.json
  BentoML service -> /metrics (via __metrics__) and /drift/metrics -> DriftCollector reads that JSON

The JSON hand-off matters: batch scoring runs in its own process, and the serving process
must not depend on in-memory state from it. The exporter is a custom Prometheus collector
that reads the JSON on every scrape (no Gauge objects, so BentoML's multiprocess mode
cannot duplicate the series or add a pid label).
"""

import json
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from prometheus_client import CONTENT_TYPE_LATEST, CollectorRegistry, generate_latest
from prometheus_client.core import GaugeMetricFamily

warnings.filterwarnings("ignore", category=RuntimeWarning, module="scipy.stats")

TARGET_COLUMNS = ["text_length", "confidence_score"]
PSI_THRESHOLD = 0.2  # rule of thumb: <0.1 stable, 0.1-0.2 moderate, >0.2 significant
DRIFT_SHARE = 0.5  # dataset drift = at least this share of columns above threshold
MIN_ROWS = 30  # PSI on fewer rows is mostly noise

PROJECT_ROOT = Path(__file__).resolve().parents[2]  # repo root (src/arabic_sentiment/ -> ..)
REPORTS_DIR = PROJECT_ROOT / "reports"
LATEST_JSON = REPORTS_DIR / "drift_latest.json"
BASELINE_CSV = PROJECT_ROOT / "data" / "reference_baseline.csv"


# ------------------------------------------------------------------
# 1. Prometheus exporter (custom collector, reads the latest JSON on each scrape)
# ------------------------------------------------------------------
def _read_latest():
    if not LATEST_JSON.exists():
        return None
    try:
        return json.loads(LATEST_JSON.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        print(f"[WARNING] Could not read {LATEST_JSON}: {e}")
        return None


class DriftCollector:
    def collect(self):
        data = _read_latest()
        if not data:
            return

        psi = GaugeMetricFamily(
            "model_drift_psi",
            "Population Stability Index (PSI) between training baseline and recent batch",
            labels=["feature_name"],
        )
        for col, score in data["psi"].items():
            psi.add_metric([col], float(score))
        yield psi

        yield GaugeMetricFamily(
            "model_dataset_drift_detected",
            "1 if overall dataset drift is detected based on threshold, else 0",
            value=1.0 if data["dataset_drift"] else 0.0,
        )
        yield GaugeMetricFamily(
            "model_drift_batch_simulated",
            "1 if the batch used for the last drift run was SIMULATED, 0 if real scored data",
            value=1.0 if data["simulated"] else 0.0,
        )
        yield GaugeMetricFamily(
            "model_drift_baseline_placeholder",
            "1 if the baseline is placeholder data (no data/reference_baseline.csv), 0 if real",
            value=1.0 if data.get("baseline_placeholder", False) else 0.0,
        )


DRIFT_REGISTRY = CollectorRegistry()
DRIFT_REGISTRY.register(DriftCollector())


def get_prometheus_metrics():
    """Payload + content-type for /metrics (drift metrics only)."""
    return generate_latest(DRIFT_REGISTRY), CONTENT_TYPE_LATEST


# ------------------------------------------------------------------
# 2. Baseline & simulated batch
# ------------------------------------------------------------------
def _placeholder_reference() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    train_texts = [
        "المنتج ممتاز جدا وسريع التوصيل والخدمة رائعة",
        "التجربة كانت سيئة للغاية والمنتج غير مطابق للمواصفات اطلاقا",
        "التطبيق جيد نوعا ما ولكن يتطلب بعض التحسينات في الواجهة",
        "جودة عالية وتغليف ممتاز شكرا لكم على المعاملة الراقية",
    ] * 50
    df = pd.DataFrame({"text": train_texts})
    df["text_length"] = df["text"].str.len()
    df["confidence_score"] = rng.uniform(0.85, 0.99, size=len(df))
    return df


def load_reference_data() -> tuple:
    """Returns (baseline_df, is_placeholder).

    Real baseline = training texts scored by the SAME model:
        python -m src.arabic_sentiment.batch_score data/processed/train.csv \\
            --baseline --text-col review_description --sample 2000
    """
    if BASELINE_CSV.exists():
        df = pd.read_csv(BASELINE_CSV)
        if "text_length" not in df.columns:
            df["text_length"] = df["text"].astype(str).str.len()
        if "confidence_score" not in df.columns:
            raise ValueError(f"{BASELINE_CSV} needs a 'confidence_score' column.")
        return df, False

    print(
        f"[WARNING] {BASELINE_CSV} not found -> using PLACEHOLDER baseline. "
        "PSI values are NOT meaningful until you build a real baseline (see --baseline)."
    )
    return _placeholder_reference(), True


def generate_simulated_scored_batch() -> pd.DataFrame:
    """SIMULATED batch, used only when no real scored batch is passed in."""
    print("\n[NOTICE] No live traffic found. Using an EXPLICITLY SIMULATED scored batch.")
    rng = np.random.default_rng(123)
    texts = ["ممتاز", "سيء جدا", "مش شغال", "عايز استرجاع"] * 50
    df = pd.DataFrame({"text": texts})
    df["text_length"] = df["text"].str.len()
    df["confidence_score"] = rng.uniform(0.50, 0.78, size=len(df))
    return df


# ------------------------------------------------------------------
# 3. Helpers
# ------------------------------------------------------------------
def _snapshot_to_dict(snapshot) -> dict:
    """Evidently 0.7+: Snapshot has .dict() / .json(); .as_dict() no longer exists."""
    if hasattr(snapshot, "dict"):
        return snapshot.dict()
    return json.loads(snapshot.json())


def _extract_psi(summary: dict) -> dict:
    """Pull each column's PSI out of the ValueDrift metrics."""
    scores = {}
    for metric in summary.get("metrics", []):
        metric_id = str(metric.get("metric_id", ""))
        config = metric.get("config", {}) or {}
        if "ValueDrift" not in metric_id and "ValueDrift" not in str(config.get("type", "")):
            continue
        value = metric.get("value")
        if isinstance(value, dict):
            value = value.get("value", value.get("drift_score"))
        for col in TARGET_COLUMNS:
            if config.get("column") == col or f"column={col}" in metric_id:
                try:
                    value = float(value)
                except (TypeError, ValueError):
                    continue
                if np.isfinite(value):
                    scores[col] = round(value, 4)
    return scores


def _psi_numpy(reference: pd.Series, current: pd.Series, bins: int = 10) -> float:
    """Fallback PSI (reference-quantile bins) if Evidently's output can't be parsed."""
    edges = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    ref_pct = np.histogram(reference, edges)[0] / len(reference)
    cur_pct = np.histogram(current, edges)[0] / len(current)
    ref_pct, cur_pct = np.clip(ref_pct, 1e-4, None), np.clip(cur_pct, 1e-4, None)
    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


# ------------------------------------------------------------------
# 4. Core drift run
# ------------------------------------------------------------------
def run_drift_monitoring(current_batch_df: pd.DataFrame = None) -> dict:
    """Compute PSI for text_length and confidence_score; write HTML report + latest JSON."""
    # Lazy import: keeps the serving process (which only needs get_prometheus_metrics) light.
    from evidently import Report
    from evidently.metrics import ValueDrift

    reference_df, placeholder = load_reference_data()

    simulated = current_batch_df is None
    current_df = generate_simulated_scored_batch() if simulated else current_batch_df.copy()

    if "text_length" not in current_df.columns and "text" in current_df.columns:
        current_df["text_length"] = current_df["text"].astype(str).str.len()

    if "confidence_score" not in current_df.columns:
        if "confidence" in current_df.columns:
            current_df["confidence_score"] = current_df["confidence"]
        else:
            raise ValueError(
                "Scored batch has no 'confidence_score' (or 'confidence') column. "
                "Save the model's max softmax probability in batch_score.py."
            )

    if len(current_df) < MIN_ROWS:
        print(f"[WARNING] Only {len(current_df)} rows in the batch (<{MIN_ROWS}); PSI will be noisy.")

    ref = reference_df[TARGET_COLUMNS].astype(float)
    cur = current_df[TARGET_COLUMNS].astype(float)

    report = Report(
        [ValueDrift(column=c, method="psi", threshold=PSI_THRESHOLD) for c in TARGET_COLUMNS]
    )
    snapshot = report.run(current_data=cur, reference_data=ref)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    snapshot.save_html(str(REPORTS_DIR / "drift_report.html"))

    psi_scores = _extract_psi(_snapshot_to_dict(snapshot))
    for col in TARGET_COLUMNS:
        if col not in psi_scores:
            print(f"[WARNING] No PSI parsed from Evidently for '{col}', using numpy fallback.")
            psi_scores[col] = round(_psi_numpy(ref[col], cur[col]), 4)

    drifted = sum(score >= PSI_THRESHOLD for score in psi_scores.values())
    dataset_drift = drifted / len(psi_scores) >= DRIFT_SHARE

    LATEST_JSON.write_text(
        json.dumps(
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "simulated": simulated,
                "baseline_placeholder": placeholder,
                "psi": psi_scores,
                "dataset_drift": dataset_drift,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    label = "SIMULATED batch" if simulated else "real scored batch"
    base = "PLACEHOLDER baseline" if placeholder else "real baseline"
    print(
        f"[SUCCESS] Drift monitoring done ({label}, {base}). "
        f"PSI: {psi_scores}, dataset_drift={dataset_drift}"
    )
    return psi_scores


if __name__ == "__main__":
    run_drift_monitoring()