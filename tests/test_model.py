import json
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

TRAIN_PATH = PROJECT_ROOT / "data" / "prepared" / "train.csv"
TEST_PATH = PROJECT_ROOT / "data" / "prepared" / "test.csv"

MODEL_PATH = PROJECT_ROOT / "models" / "model.pkl"
METRICS_PATH = PROJECT_ROOT / "artifacts" / "metrics.json"
CM_PATH = PROJECT_ROOT / "artifacts" / "confusion_matrix.png"
REPORT_PATH = PROJECT_ROOT / "artifacts" / "classification_report.txt"

TARGET_COL = "stroke"

EXPECTED_COLUMNS = {
    "age",
    "hypertension",
    "heart_disease",
    "avg_glucose_level",
    "bmi",
    TARGET_COL,
}

ROC_AUC_THRESHOLD = 0.75


def test_train_file_exists():
    assert TRAIN_PATH.exists(), f"Train file not found: {TRAIN_PATH}"


def test_test_file_exists():
    assert TEST_PATH.exists(), f"Test file not found: {TEST_PATH}"


def test_train_data_has_target_and_basic_columns():
    df = pd.read_csv(TRAIN_PATH)

    missing = EXPECTED_COLUMNS - set(df.columns)
    assert not missing, f"Missing expected columns: {missing}"


def test_target_column_not_empty():
    df = pd.read_csv(TRAIN_PATH)

    assert TARGET_COL in df.columns, f"Target column '{TARGET_COL}' not found"
    assert df[TARGET_COL].notna().all(), "Target column contains missing values"


def test_target_is_binary():
    df = pd.read_csv(TRAIN_PATH)

    unique_values = set(df[TARGET_COL].unique())
    assert unique_values.issubset({0, 1}), f"Target has invalid values: {unique_values}"


def test_numeric_columns_have_valid_ranges():
    df = pd.read_csv(TRAIN_PATH)

    assert (df["age"] >= 0).all(), "Column 'age' contains negative values"
    assert (df["avg_glucose_level"] > 0).all(), "Column 'avg_glucose_level' must be > 0"
    assert (df["bmi"] > 0).all(), "Column 'bmi' must be > 0"


def test_model_file_exists():
    assert MODEL_PATH.exists(), f"Model file not found: {MODEL_PATH}"


def test_metrics_file_exists():
    assert METRICS_PATH.exists(), f"Metrics file not found: {METRICS_PATH}"


def test_confusion_matrix_exists():
    assert CM_PATH.exists(), f"Confusion matrix file not found: {CM_PATH}"


def test_classification_report_exists():
    assert REPORT_PATH.exists(), f"Classification report file not found: {REPORT_PATH}"


def test_metrics_have_required_keys():
    metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))

    required_keys = {
        "train_accuracy",
        "train_f1",
        "test_accuracy",
        "test_f1",
        "test_precision",
        "test_recall",
        "test_roc_auc",
        "test_pr_auc",
        "threshold",
    }

    missing = required_keys - set(metrics.keys())
    assert not missing, f"Missing metric keys: {missing}"


def test_quality_gate_roc_auc():
    metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    roc_auc = metrics["test_roc_auc"]

    assert roc_auc >= ROC_AUC_THRESHOLD, (
        f"Quality gate failed: test_roc_auc={roc_auc:.4f} "
        f"< {ROC_AUC_THRESHOLD:.2f}"
    )