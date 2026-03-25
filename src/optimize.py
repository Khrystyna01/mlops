import json
import random
from pathlib import Path
from typing import Any, Dict, Optional

import hydra
import joblib
import mlflow
import numpy as np
import optuna
import pandas as pd
from omegaconf import DictConfig, OmegaConf
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def load_split_data(cfg: DictConfig):
    train_df = pd.read_csv(cfg.data.train_path)
    val_df = pd.read_csv(cfg.data.val_path)
    test_df = pd.read_csv(cfg.data.test_path)

    target_col = cfg.data.target_col

    X_train = train_df.drop(columns=[target_col]).values
    y_train = train_df[target_col].values

    X_val = val_df.drop(columns=[target_col]).values
    y_val = val_df[target_col].values

    X_test = test_df.drop(columns=[target_col]).values
    y_test = test_df[target_col].values

    return X_train, X_val, X_test, y_train, y_val, y_test


def build_model(model_type: str, params: Dict[str, Any], seed: int):
    if model_type == "random_forest":
        return RandomForestClassifier(
            n_estimators=params["n_estimators"],
            max_depth=params["max_depth"],
            min_samples_split=params["min_samples_split"],
            min_samples_leaf=params["min_samples_leaf"],
            class_weight=params["class_weight"],
            random_state=seed,
            n_jobs=-1,
        )

    if model_type == "logistic_regression":
        return LogisticRegression(
            C=params["C"],
            solver=params["solver"],
            penalty=params["penalty"],
            class_weight=params["class_weight"],
            max_iter=2000,
            random_state=seed,
        )

    raise ValueError(f"Unknown model.type='{model_type}'")


def evaluate(model, X_train, y_train, X_eval, y_eval, metric: str) -> float:
    model.fit(X_train, y_train)

    if metric == "f1":
        y_pred = model.predict(X_eval)
        return float(f1_score(y_eval, y_pred, zero_division=0))

    if metric == "roc_auc":
        y_proba = model.predict_proba(X_eval)[:, 1]
        return float(roc_auc_score(y_eval, y_proba))

    raise ValueError("metric should be 'f1' or 'roc_auc'")


def evaluate_final_metrics(model, X_train, y_train, X_test, y_test) -> Dict[str, float]:
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    return {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, y_pred)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, y_proba)),
    }


def evaluate_cv(model, X, y, metric: str, seed: int, n_splits: int) -> float:
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    if metric == "f1":
        scores = cross_val_score(model, X, y, cv=cv, scoring="f1", n_jobs=-1)
        return float(np.mean(scores))

    if metric == "roc_auc":
        scores = cross_val_score(model, X, y, cv=cv, scoring="roc_auc", n_jobs=-1)
        return float(np.mean(scores))

    raise ValueError("metric should be 'f1' or 'roc_auc'")


def make_sampler(
    sampler_name: str,
    seed: int,
    grid_space: Optional[Dict[str, Any]] = None,
):
    if sampler_name == "tpe":
        return optuna.samplers.TPESampler(seed=seed)

    if sampler_name == "random":
        return optuna.samplers.RandomSampler(seed=seed)

    if sampler_name == "grid":
        if grid_space is None:
            raise ValueError("For sampler='grid' need to set grid_space.")
        return optuna.samplers.GridSampler(search_space=grid_space)

    raise ValueError("sampler should be: tpe, random, grid")


def suggest_params(trial: optuna.Trial, model_type: str, cfg: DictConfig) -> Dict[str, Any]:
    if model_type == "random_forest":
        space = cfg.random_forest
        return {
            "n_estimators": trial.suggest_int(
                "n_estimators",
                space.n_estimators.low,
                space.n_estimators.high,
            ),
            "max_depth": trial.suggest_int(
                "max_depth",
                space.max_depth.low,
                space.max_depth.high,
            ),
            "min_samples_split": trial.suggest_int(
                "min_samples_split",
                space.min_samples_split.low,
                space.min_samples_split.high,
            ),
            "min_samples_leaf": trial.suggest_int(
                "min_samples_leaf",
                space.min_samples_leaf.low,
                space.min_samples_leaf.high,
            ),
            "class_weight": trial.suggest_categorical(
                "class_weight",
                list(space.class_weight),
            ),
        }

    if model_type == "logistic_regression":
        space = cfg.logistic_regression
        return {
            "C": trial.suggest_float(
                "C",
                space.C.low,
                space.C.high,
                log=True,
            ),
            "solver": trial.suggest_categorical(
                "solver",
                list(space.solver),
            ),
            "penalty": trial.suggest_categorical(
                "penalty",
                list(space.penalty),
            ),
            "class_weight": trial.suggest_categorical(
                "class_weight",
                list(space.class_weight),
            ),
        }

    raise ValueError(f"Unknown model.type='{model_type}'.")


def objective_factory(cfg: DictConfig, X_train, X_val, y_train, y_val):
    def objective(trial: optuna.Trial) -> float:
        params = suggest_params(trial, cfg.model.type, cfg)

        with mlflow.start_run(nested=True, run_name=f"trial_{trial.number:03d}"):
            mlflow.set_tag("trial_number", trial.number)
            mlflow.set_tag("model_type", cfg.model.type)
            mlflow.set_tag("sampler", cfg.hpo.sampler)
            mlflow.set_tag("seed", cfg.seed)

            mlflow.log_params(params)

            model = build_model(cfg.model.type, params=params, seed=cfg.seed)

            if cfg.hpo.use_cv:
                X = np.concatenate([X_train, X_val], axis=0)
                y = np.concatenate([y_train, y_val], axis=0)
                score = evaluate_cv(
                    model=model,
                    X=X,
                    y=y,
                    metric=cfg.hpo.metric,
                    seed=cfg.seed,
                    n_splits=cfg.hpo.cv_folds,
                )
            else:
                score = evaluate(
                    model=model,
                    X_train=X_train,
                    y_train=y_train,
                    X_eval=X_val,
                    y_eval=y_val,
                    metric=cfg.hpo.metric,
                )

            mlflow.log_metric(cfg.hpo.metric, score)
            return score

    return objective


def save_json(data: Dict[str, Any], path: str) -> None:
    path_obj = Path(path)
    ensure_dir(path_obj.parent)
    with open(path_obj, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


def main(cfg: DictConfig) -> None:
    set_global_seed(cfg.seed)

    X_train, X_val, X_test, y_train, y_val, y_test = load_split_data(cfg)

    mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
    mlflow.set_experiment(cfg.mlflow.experiment_name)

    grid_space = None
    if cfg.hpo.sampler == "grid":
        if cfg.model.type == "random_forest":
            grid_space = {
                "n_estimators": list(cfg.grid.random_forest.n_estimators),
                "max_depth": list(cfg.grid.random_forest.max_depth),
                "min_samples_split": list(cfg.grid.random_forest.min_samples_split),
                "min_samples_leaf": list(cfg.grid.random_forest.min_samples_leaf),
                "class_weight": list(cfg.grid.random_forest.class_weight),
            }
        elif cfg.model.type == "logistic_regression":
            grid_space = {
                "C": list(cfg.grid.logistic_regression.C),
                "solver": list(cfg.grid.logistic_regression.solver),
                "penalty": list(cfg.grid.logistic_regression.penalty),
                "class_weight": list(cfg.grid.logistic_regression.class_weight),
            }

    sampler = make_sampler(cfg.hpo.sampler, seed=cfg.seed, grid_space=grid_space)

    study = optuna.create_study(
        direction=cfg.hpo.direction,
        sampler=sampler,
    )

    with mlflow.start_run(run_name=f"{cfg.model.type}_{cfg.hpo.sampler}_study"):
        mlflow.log_param("model_type", cfg.model.type)
        mlflow.log_param("sampler", cfg.hpo.sampler)
        mlflow.log_param("n_trials", cfg.hpo.n_trials)
        mlflow.log_param("metric", cfg.hpo.metric)
        mlflow.log_param("use_cv", cfg.hpo.use_cv)
        mlflow.log_param("cv_folds", cfg.hpo.cv_folds)
        mlflow.log_param("seed", cfg.seed)

        config_path = Path(cfg.artifacts.output_dir) / "used_config.yaml"
        ensure_dir(config_path.parent)
        with open(config_path, "w", encoding="utf-8") as f:
            f.write(OmegaConf.to_yaml(cfg))
        mlflow.log_artifact(str(config_path))

        objective = objective_factory(cfg, X_train, X_val, y_train, y_val)
        study.optimize(objective, n_trials=cfg.hpo.n_trials)

        best_trial = study.best_trial
        mlflow.log_metric(f"best_{cfg.hpo.metric}", float(best_trial.value))
        mlflow.log_dict(best_trial.params, "best_params.json")

        best_model = build_model(
            cfg.model.type,
            params=best_trial.params,
            seed=cfg.seed,
        )

        X_train_final = np.concatenate([X_train, X_val], axis=0)
        y_train_final = np.concatenate([y_train, y_val], axis=0)

        final_metrics = evaluate_final_metrics(
            model=best_model,
            X_train=X_train_final,
            y_train=y_train_final,
            X_test=X_test,
            y_test=y_test,
        )
        mlflow.log_metrics({f"final_{k}": v for k, v in final_metrics.items()})

        ensure_dir(Path("models"))
        trained_best_model = best_model.fit(X_train_final, y_train_final)
        joblib.dump(trained_best_model, cfg.artifacts.best_model_path)
        mlflow.log_artifact(cfg.artifacts.best_model_path)

        save_json(best_trial.params, cfg.artifacts.best_params_path)
        save_json(final_metrics, cfg.artifacts.best_metrics_path)
        mlflow.log_artifact(cfg.artifacts.best_params_path)
        mlflow.log_artifact(cfg.artifacts.best_metrics_path)

    print("Optimization completed successfully.")
    print(f"Best params: {best_trial.params}")
    print(f"Best {cfg.hpo.metric}: {best_trial.value:.4f}")
    print(f"Final metrics: {final_metrics}")


@hydra.main(version_base=None, config_path="../config", config_name="config")
def hydra_entry(cfg: DictConfig) -> None:
    main(cfg)


if __name__ == "__main__":
    hydra_entry()
