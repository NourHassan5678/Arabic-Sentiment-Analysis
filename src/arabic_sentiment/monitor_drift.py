"""
Drift Monitoring Module using Evidently AI and Prometheus Client.
Computes Population Stability Index (PSI) for text_length and confidence_score using DataDriftPreset.
"""

import os
from pathlib import Path
import warnings
import numpy as np
import pandas as pd
from prometheus_client import Gauge, generate_latest, CONTENT_TYPE_LATEST
from evidently import Report
from evidently.presets import DataDriftPreset

# Suppress division-by-zero warnings from scipy stats when handling synthetic variance
warnings.filterwarnings("ignore", category=RuntimeWarning, module="scipy.stats")

# ------------------------------------------------------------------
# 1. Prometheus Metrics Definition
# ------------------------------------------------------------------
PSI_GAUGE = Gauge(
    "model_drift_psi",
    "Population Stability Index (PSI) score between baseline and recent batch",
    ["feature_name"]
)

DATASET_DRIFT_GAUGE = Gauge(
    "model_dataset_drift_detected",
    "1 if overall dataset drift is detected based on threshold, else 0"
)


# ------------------------------------------------------------------
# 2. Reference & Simulated Batch Data Generators
# ------------------------------------------------------------------
def load_reference_data() -> pd.DataFrame:
    """Loads/simulates training baseline data with text_length and confidence_score."""
    np.random.seed(42)
    train_texts = [
        "المنتج ممتاز جدا وسريع التوصيل والخدمة رائعة",
        "التجربة كانت سيئة للغاية والمنتج غير مطابق للمواصفات اطلاقا",
        "التطبيق جيد نوعا ما ولكن يتطلب بعض التحسينات في الواجهة",
        "جودة عالية وتغليف ممتاز شكرا لكم على المعاملة الراقية",
    ] * 50  # 200 samples

    df = pd.DataFrame({"text": train_texts})
    df["text_length"] = df["text"].apply(len)
    df["confidence_score"] = np.random.uniform(0.85, 0.99, size=len(df))
    return df


def generate_simulated_scored_batch() -> pd.DataFrame:
    """
    [EXPLICIT NOTICE: SIMULATED BATCH]
    Simulates a recent scored batch from the deployed model when live traffic is unavailable.
    """
    print("\n[NOTICE] No live traffic stream found. Using EXPLICITLY SIMULATED scored batch for drift monitoring.")
    np.random.seed(123)
    simulated_texts = [
        "ممتاز",
        "سيء جدا",
        "مش شغال",
        "عايز استرجاع",
    ] * 50  # 200 samples

    df = pd.DataFrame({"text": simulated_texts})
    df["text_length"] = df["text"].apply(len)
    df["confidence_score"] = np.random.uniform(0.50, 0.78, size=len(df))
    return df


# ------------------------------------------------------------------
# 3. Core Drift Monitoring Execution
# ------------------------------------------------------------------
def run_drift_monitoring(current_batch_df: pd.DataFrame = None) -> dict:
    """
    Computes PSI for text_length and confidence_score using DataDriftPreset.
    Automatically triggered post-batch scoring run.
    Updates Prometheus Gauges.
    """
    reference_df = load_reference_data()

    if current_batch_df is None:
        current_batch_df = generate_simulated_scored_batch()

    # Isolate target numerical columns so Evidently ignores raw text strings
    target_columns = ["text_length", "confidence_score"]
    valid_cols = [col for col in target_columns if col in current_batch_df.columns]
    
    reference_df_subset = reference_df[valid_cols]
    current_batch_df_subset = current_batch_df[valid_cols]

    # Evidently Report configured with DataDriftPreset
    report = Report(metrics=[DataDriftPreset()])
    
    # FIX: In Evidently 0.7+, run() returns a Snapshot object. 
    # Methods to save and export moved to this returned object.
    snapshot = report.run(reference_data=reference_df_subset, current_data=current_batch_df_subset)

    # Save HTML report using the snapshot object
    reports_dir = Path("reports")
    reports_dir.mkdir(parents=True, exist_ok=True)
    snapshot.save_html(str(reports_dir / "drift_report.html"))

    # Extract metrics from preset dictionary and update Prometheus Registry
    # Use .as_dict() or .dict() depending on exact sub-version
    summary = snapshot.as_dict() if hasattr(snapshot, "as_dict") else snapshot.dict()
    
    psi_scores = {}

    try:
        metrics_list = summary.get("metrics", [])
        for metric in metrics_list:
            result = metric.get("result", {})
            
            if "dataset_drift" in result:
                drift_flag = 1.0 if result.get("dataset_drift", False) else 0.0
                DATASET_DRIFT_GAUGE.set(drift_flag)

            drift_by_cols = result.get("drift_by_columns", {}) or result.get("by_columns", {})
            for col_name, col_data in drift_by_cols.items():
                if col_name in target_columns:
                    score = col_data.get("drift_score", 0.0)
                    psi_scores[col_name] = score
                    PSI_GAUGE.labels(feature_name=col_name).set(score)
    except Exception as e:
        print(f"[WARNING] Could not fully parse metrics summary dict: {e}")

    print(f"[SUCCESS] Drift Monitoring Completed. Computed PSI: {psi_scores}")
    return psi_scores


def get_prometheus_metrics():
    """Returns latest Prometheus metrics payload and content-type."""
    return generate_latest(), CONTENT_TYPE_LATEST


if __name__ == "__main__":
    # Test script standalone
    run_drift_monitoring()