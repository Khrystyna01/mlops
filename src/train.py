import argparse
import json
import os
from pathlib import Path

import joblib
import matplotlib
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_confusion_matrix_png(y_true, y_pred, out_path: Path) -> None:
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(5, 4))
    plt.imshow(cm, interpolation="nearest")
    plt.title("Confusion Matrix")
    plt.colorbar()
    ticks = range(2)
    plt.xticks(ticks, ["0", "1"])
    plt.yticks(ticks, ["0", "1"])
    plt.xlabel("Predicted")
    plt.ylabel("True")

    thresh = cm.max() / 2.0 if cm.max() > 0 else 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j,
                i,
                str(cm[i, j]),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )

    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def save_feature_importance_top25(
    model, feature_names, out_png: Path, out_csv: Path
) -> None:
    fi = (
        pd.DataFrame(
            {"feature": feature_names, "importance": model.feature_importances_}
        )
        .sort_values("importance", ascending=False)
        .head(25)
    )

    fi.to_csv(out_csv, index=False)

    plt.figure(figsize=(8, 6))
    plt.barh(fi["feature"][::-1], fi["importance"][::-1])
    plt.title("Top-25 Feature Importances (RandomForest)")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train RandomForest on prepared Stroke dataset with MLflow logging"
    )

    parser.add_argument("--train_path", type=str, default="data/prepared/train.csv")
    parser.add_argument("--test_path", type=str, default="data/prepared/test.csv")
    parser.add_argument("--target_col", type=str, default="stroke")

    parser.add_argument("--experiment_name", type=str, default="MLOps_Lab_4")
    parser.add_argument("--run_name", type=str, default=None)

    parser.add_argument("--author", type=str, default=None)
    parser.add_argument("--dataset_version", type=str, default=None)

    parser.add_argument("--n_estimators", type=int, default=200)
    parser.add_argument(
        "--max_depth", type=int, default=6, help="<=0 means None (unlimited)"
    )
    parser.add_argument("--min_samples_split", type=int, default=2)
    parser.add_argument("--random_state", type=int, default=42)

    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--use_class_weight", action="store_true")

    parser.add_argument("--models_dir", type=str, default="models")
    parser.add_argument("--artifacts_dir", type=str, default="artifacts")
    parser.add_argument("--skip_local_model_save", action="store_true")

    args = parser.parse_args()

    ci_mode = os.getenv("CI", "false").lower() == "true"

    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent

    train_path = Path(args.train_path)
    if not train_path.is_absolute():
        train_path = (project_root / train_path).resolve()

    test_path = Path(args.test_path)
    if not test_path.is_absolute():
        test_path = (project_root / test_path).resolve()

    models_dir = Path(args.models_dir)
    if not models_dir.is_absolute():
        models_dir = (project_root / models_dir).resolve()

    artifacts_dir = Path(args.artifacts_dir)
    if not artifacts_dir.is_absolute():
        artifacts_dir = (project_root / artifacts_dir).resolve()

    ensure_dir(models_dir)
    ensure_dir(artifacts_dir)

    if not train_path.exists():
        raise FileNotFoundError(f"Train file not found: {train_path}")
    if not test_path.exists():
        raise FileNotFoundError(f"Test file not found: {test_path}")

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    if ci_mode and len(train_df) > 1000:
        train_df = train_df.sample(n=1000, random_state=args.random_state)

    if args.target_col not in train_df.columns:
        raise ValueError(f"Target column '{args.target_col}' not found in train data")
    if args.target_col not in test_df.columns:
        raise ValueError(f"Target column '{args.target_col}' not found in test data")

    y_train = train_df[args.target_col].astype(int)
    X_train = train_df.drop(columns=[args.target_col], errors="ignore")

    y_test = test_df[args.target_col].astype(int)
    X_test = test_df.drop(columns=[args.target_col], errors="ignore")

    class_weight = "balanced" if args.use_class_weight else None

    model = RandomForestClassifier(
        n_estimators=args.n_estimators,
        max_depth=None if args.max_depth <= 0 else args.max_depth,
        min_samples_split=args.min_samples_split,
        random_state=args.random_state,
        n_jobs=-1,
        class_weight=class_weight,
    )

    mlflow.set_experiment(args.experiment_name)

    if not args.run_name:
        args.run_name = f"rf_depth{args.max_depth}_est{args.n_estimators}"

    tags = {"model_type": "RandomForest", "lab": "lab_4"}
    if args.author:
        tags["author"] = args.author
    if args.dataset_version:
        tags["dataset_version"] = args.dataset_version

    with mlflow.start_run(run_name=args.run_name) as run:
        mlflow.set_tags(tags)

        mlflow.log_param("train_path", str(train_path))
        mlflow.log_param("test_path", str(test_path))
        mlflow.log_param("random_state", args.random_state)
        mlflow.log_param("threshold", args.threshold)
        mlflow.log_param("ci_mode", ci_mode)

        mlflow.log_param("n_estimators", args.n_estimators)
        mlflow.log_param("max_depth", args.max_depth)
        mlflow.log_param("min_samples_split", args.min_samples_split)
        mlflow.log_param(
            "class_weight", "balanced" if args.use_class_weight else "None"
        )

        model.fit(X_train, y_train)

        y_test_proba = model.predict_proba(X_test)[:, 1]
        y_test_pred = (y_test_proba >= args.threshold).astype(int)

        y_train_proba = model.predict_proba(X_train)[:, 1]
        y_train_pred = (y_train_proba >= args.threshold).astype(int)

        train_acc = accuracy_score(y_train, y_train_pred)
        train_f1 = f1_score(y_train, y_train_pred, zero_division=0)

        test_acc = accuracy_score(y_test, y_test_pred)
        test_f1 = f1_score(y_test, y_test_pred, zero_division=0)
        test_precision = precision_score(y_test, y_test_pred, zero_division=0)
        test_recall = recall_score(y_test, y_test_pred, zero_division=0)
        test_roc_auc = roc_auc_score(y_test, y_test_proba)
        test_pr_auc = average_precision_score(y_test, y_test_proba)

        mlflow.log_metric("train_accuracy", train_acc)
        mlflow.log_metric("train_f1", train_f1)
        mlflow.log_metric("test_accuracy", test_acc)
        mlflow.log_metric("test_f1", test_f1)
        mlflow.log_metric("test_precision", test_precision)
        mlflow.log_metric("test_recall", test_recall)
        mlflow.log_metric("test_roc_auc", test_roc_auc)
        mlflow.log_metric("test_pr_auc", test_pr_auc)

        report_text = classification_report(
            y_test, y_test_pred, digits=4, zero_division=0
        )

        report_path = artifacts_dir / f"classification_report_{run.info.run_id}.txt"
        report_path.write_text(report_text, encoding="utf-8")
        mlflow.log_artifact(str(report_path))

        stable_report_path = artifacts_dir / "classification_report.txt"
        stable_report_path.write_text(report_text, encoding="utf-8")

        cm_path = artifacts_dir / f"confusion_matrix_{run.info.run_id}.png"
        save_confusion_matrix_png(y_test, y_test_pred, cm_path)
        mlflow.log_artifact(str(cm_path))

        stable_cm_path = artifacts_dir / "confusion_matrix.png"
        save_confusion_matrix_png(y_test, y_test_pred, stable_cm_path)

        fi_png = artifacts_dir / f"feature_importance_top25_{run.info.run_id}.png"
        fi_csv = artifacts_dir / f"feature_importance_top25_{run.info.run_id}.csv"
        save_feature_importance_top25(model, X_train.columns.tolist(), fi_png, fi_csv)
        mlflow.log_artifact(str(fi_png))
        mlflow.log_artifact(str(fi_csv))

        mlflow.sklearn.log_model(model, artifact_path="model")

        if not args.skip_local_model_save:
            local_model_path = models_dir / f"model_rf_{run.info.run_id}.joblib"
            joblib.dump(model, local_model_path)

            stable_model_path = models_dir / "model.pkl"
            joblib.dump(model, stable_model_path)

        metrics = {
            "train_accuracy": train_acc,
            "train_f1": train_f1,
            "test_accuracy": test_acc,
            "test_f1": test_f1,
            "test_precision": test_precision,
            "test_recall": test_recall,
            "test_roc_auc": test_roc_auc,
            "test_pr_auc": test_pr_auc,
            "threshold": args.threshold,
        }

        metrics_path = artifacts_dir / "metrics.json"
        metrics_path.write_text(
            json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8"
        )

        summary = {
            "run_id": run.info.run_id,
            "run_name": args.run_name,
            "experiment_name": args.experiment_name,
            "params": {
                "n_estimators": args.n_estimators,
                "max_depth": args.max_depth,
                "min_samples_split": args.min_samples_split,
                "threshold": args.threshold,
                "class_weight": "balanced" if args.use_class_weight else "None",
            },
            "metrics": {
                "train_accuracy": train_acc,
                "train_f1": train_f1,
                "test_accuracy": test_acc,
                "test_f1": test_f1,
                "test_precision": test_precision,
                "test_recall": test_recall,
                "test_roc_auc": test_roc_auc,
                "test_pr_auc": test_pr_auc,
            },
            "tags": tags,
        }

        summary_path = artifacts_dir / f"run_summary_{run.info.run_id}.json"
        summary_path.write_text(
            json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        mlflow.log_artifact(str(summary_path))

        print("\nFinished")
        print(f"Run name: {args.run_name}")
        print(f"Run ID:   {run.info.run_id}")
        print(f"CI mode:  {ci_mode}")
        print(f"TRAIN: acc={train_acc:.4f} f1={train_f1:.4f}")
        print(
            f"TEST : acc={test_acc:.4f} "
            f"f1={test_f1:.4f} "
            f"precision={test_precision:.4f} "
            f"recall={test_recall:.4f} "
            f"roc_auc={test_roc_auc:.4f} "
            f"pr_auc={test_pr_auc:.4f}"
        )
        print(f"Artifacts dir: {artifacts_dir}")
        print(f"Models dir:    {models_dir}")


if __name__ == "__main__":
    main()
