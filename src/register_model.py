from __future__ import annotations

import json
from pathlib import Path

import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
import joblib


PROJECT_ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
MODELS_DIR = PROJECT_ROOT / "models"

METRICS_PATH = ARTIFACTS_DIR / "metrics.json"
MODEL_PATH = MODELS_DIR / "model.pkl"

MODEL_NAME = "StrokeRiskModel"
EXPERIMENT_NAME = "MLOps_Lab_5"


def main() -> None:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

    if not METRICS_PATH.exists():
        raise FileNotFoundError(f"Metrics file not found: {METRICS_PATH}")

    metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    model = joblib.load(MODEL_PATH)

    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(run_name="register_model_run") as run:
        mlflow.log_metrics(metrics)
        mlflow.log_artifact(str(METRICS_PATH))
        mlflow.log_artifact(str(MODEL_PATH))

        model_info = mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model"
        )

        model_uri = model_info.model_uri

    client = MlflowClient()

    registered_model = mlflow.register_model(
        model_uri=model_uri,
        name=MODEL_NAME,
    )

    client.transition_model_version_stage(
        name=MODEL_NAME,
        version=registered_model.version,
        stage="Staging",
        archive_existing_versions=False,
    )

    print("Model registered successfully.")
    print(f"Model name: {MODEL_NAME}")
    print(f"Version: {registered_model.version}")
    print("Stage: Staging")


if __name__ == "__main__":
    main()
