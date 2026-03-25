from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.empty import EmptyOperator
from airflow.operators.python import BranchPythonOperator, PythonOperator


PROJECT_ROOT = Path("/opt/airflow/project")
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "healthcare-dataset-stroke-data.csv"
METRICS_PATH = PROJECT_ROOT / "artifacts" / "metrics.json"


def check_data_exists() -> None:
    if not RAW_DATA_PATH.exists():
        raise FileNotFoundError(f"Data file not found: {RAW_DATA_PATH}")


def read_metrics(**context):
    if not METRICS_PATH.exists():
        raise FileNotFoundError(f"Metrics file not found: {METRICS_PATH}")

    with open(METRICS_PATH, "r", encoding="utf-8") as f:
        metrics = json.load(f)

    return metrics


def choose_next_step(**context):
    ti = context["ti"]
    metrics = ti.xcom_pull(task_ids="evaluate_model")

    test_accuracy = float(metrics.get("test_accuracy", 0.0))

    if test_accuracy >= 0.85:
        return "register_model"
    return "stop_pipeline"


with DAG(
    dag_id="ml_training_pipeline",
    start_date=datetime(2026, 3, 25),
    schedule="@daily",
    catchup=False,
    tags=["mlops", "lab5"],
) as dag:

    check_data = PythonOperator(
        task_id="check_data",
        python_callable=check_data_exists,
    )

    prepare_data = BashOperator(
        task_id="prepare_data",
        bash_command="""
        cd /opt/airflow/project && dvc repro prepare
        """,
    )

    train_model = BashOperator(
        task_id="train_model",
        bash_command="""
        cd /opt/airflow/project && python src/train.py
        """,
    )

    evaluate_model = PythonOperator(
        task_id="evaluate_model",
        python_callable=read_metrics,
    )

    branch_on_quality = BranchPythonOperator(
        task_id="branch_on_quality",
        python_callable=choose_next_step,
    )

    register_model = BashOperator(
        task_id="register_model",
        bash_command="""
        cd /opt/airflow/project && python src/register_model.py
        """,
    )

    stop_pipeline = EmptyOperator(
        task_id="stop_pipeline",
    )

    check_data >> prepare_data >> train_model >> evaluate_model >> branch_on_quality
    branch_on_quality >> register_model
    branch_on_quality >> stop_pipeline
