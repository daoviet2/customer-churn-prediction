"""Train the churn models from the experiment notebook and persist the best one.

Example:
    python model_trainer.py --tracking-uri http://127.0.0.1:5000
"""

from __future__ import annotations

import argparse
import json
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from prepare_data import main as prepare_dataset


ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_PATH = ROOT / "data" / "processed" / "Customer-Churn-processed.csv"
DEFAULT_MODEL_DIR = ROOT / "artifacts"
TARGET_COLUMN = "Churn"


def build_models(random_state: int) -> dict[str, Any]:
    """Return the candidate models used in the original experiment."""
    models: dict[str, Any] = {
        "Logistic Regression": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        max_iter=2000,
                        class_weight="balanced",
                        random_state=random_state,
                    ),
                ),
            ]
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            min_samples_leaf=2,
            random_state=random_state,
            class_weight="balanced",
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=3,
            random_state=random_state,
        ),
    }

    try:
        from xgboost import XGBClassifier
    except ImportError:
        print("XGBoost is not installed; skipping the XGBoost candidate.")
    else:
        models["XGBoost"] = XGBClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=5,
            subsample=0.9,
            colsample_bytree=0.8,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=random_state,
        )
    return models


def load_data(data_path: Path) -> pd.DataFrame:
    if not data_path.exists() and data_path == DEFAULT_DATA_PATH:
        prepare_dataset()
    if not data_path.exists():
        raise FileNotFoundError(f"Processed dataset not found: {data_path}")

    data = pd.read_csv(data_path)
    if TARGET_COLUMN not in data.columns:
        raise ValueError(f"Dataset must contain target column '{TARGET_COLUMN}'.")
    if data[TARGET_COLUMN].nunique() != 2:
        raise ValueError(f"'{TARGET_COLUMN}' must be a binary target.")
    return data


def configure_mlflow(enabled: bool, tracking_uri: str | None, experiment_name: str):
    if not enabled:
        return None
    try:
        import mlflow
    except ImportError as exc:
        raise RuntimeError(
            "MLflow is enabled but not installed. Install it with: pip install mlflow "
            "or run with --no-mlflow."
        ) from exc

    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
    return mlflow


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and select a customer churn model.")
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=1,
        help="Parallel CV workers; use -1 to use all cores when supported.",
    )
    parser.add_argument("--experiment-name", default="customer-churn")
    parser.add_argument("--tracking-uri", help="For example: http://127.0.0.1:5000")
    parser.add_argument("--no-mlflow", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0 < args.test_size < 1:
        raise ValueError("--test-size must be between 0 and 1.")
    if args.cv_folds < 2:
        raise ValueError("--cv-folds must be at least 2.")

    data_path = args.data_path.resolve()
    model_dir = args.model_dir.resolve()
    data = load_data(data_path)
    features = data.drop(columns=TARGET_COLUMN)
    target = data[TARGET_COLUMN]
    x_train, _, y_train, _ = train_test_split(
        features,
        target,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=target,
    )

    mlflow = configure_mlflow(
        enabled=not args.no_mlflow,
        tracking_uri=args.tracking_uri,
        experiment_name=args.experiment_name,
    )
    run_context = mlflow.start_run(run_name="model-training") if mlflow else nullcontext()

    with run_context:
        if mlflow:
            mlflow.log_params(
                {
                    "data_path": str(data_path),
                    "rows": len(data),
                    "features": features.shape[1],
                    "test_size": args.test_size,
                    "random_state": args.random_state,
                    "cv_folds": args.cv_folds,
                    "n_jobs": args.n_jobs,
                    "selection_metric": "f1_mean",
                }
            )

        cv = StratifiedKFold(
            n_splits=args.cv_folds, shuffle=True, random_state=args.random_state
        )
        results: list[dict[str, float | str]] = []
        models = build_models(args.random_state)
        for name, model in models.items():
            print(f"Training and cross-validating: {name}")
            scores = cross_validate(
                model,
                x_train,
                y_train,
                cv=cv,
                scoring={
                    "f1": "f1",
                    "precision": "precision",
                    "recall": "recall",
                    "roc_auc": "roc_auc",
                },
                n_jobs=args.n_jobs,
            )
            results.append(
                {
                    "Model": name,
                    "F1_mean": scores["test_f1"].mean(),
                    "F1_std": scores["test_f1"].std(),
                    "Precision_mean": scores["test_precision"].mean(),
                    "Recall_mean": scores["test_recall"].mean(),
                    "ROC_AUC_mean": scores["test_roc_auc"].mean(),
                }
            )

        cv_results = pd.DataFrame(results).sort_values("F1_mean", ascending=False)
        best_name = str(cv_results.iloc[0]["Model"])
        best_model = models[best_name]
        best_model.fit(x_train, y_train)

        model_dir.mkdir(parents=True, exist_ok=True)
        model_path = model_dir / "best_model.joblib"
        metrics_path = model_dir / "cv_results.csv"
        metadata_path = model_dir / "model_metadata.json"
        joblib.dump(best_model, model_path)
        cv_results.to_csv(metrics_path, index=False)
        metadata = {
            "best_model_name": best_name,
            "target_column": TARGET_COLUMN,
            "feature_names": features.columns.tolist(),
            "data_path": str(data_path),
            "test_size": args.test_size,
            "random_state": args.random_state,
            "cv_folds": args.cv_folds,
        }
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

        print("\n=== Cross-validation results ===")
        print(cv_results.round(4).to_string(index=False))
        print(f"\nSelected model: {best_name}")
        print(f"Model saved to: {model_path}")

        if mlflow:
            best_scores = cv_results.iloc[0]
            mlflow.log_param("selected_model", best_name)
            mlflow.log_metrics(
                {
                    "cv_f1_mean": float(best_scores["F1_mean"]),
                    "cv_f1_std": float(best_scores["F1_std"]),
                    "cv_precision_mean": float(best_scores["Precision_mean"]),
                    "cv_recall_mean": float(best_scores["Recall_mean"]),
                    "cv_roc_auc_mean": float(best_scores["ROC_AUC_mean"]),
                }
            )
            mlflow.log_artifacts(str(model_dir), artifact_path="training")


if __name__ == "__main__":
    main()
