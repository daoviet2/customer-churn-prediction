"""Evaluate the model selected by model_trainer.py on the held-out test split.

Example:
    python model_evaluation.py --tracking-uri http://127.0.0.1:5000
"""

from __future__ import annotations

import argparse
import json
from contextlib import nullcontext
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split

from model_trainer import DEFAULT_DATA_PATH, DEFAULT_MODEL_DIR, TARGET_COLUMN, configure_mlflow, load_data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the selected customer churn model.")
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--experiment-name", default="customer-churn")
    parser.add_argument("--tracking-uri", help="For example: http://127.0.0.1:5000")
    parser.add_argument("--no-mlflow", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_dir = args.model_dir.resolve()
    model_path = model_dir / "best_model.joblib"
    metadata_path = model_dir / "model_metadata.json"
    if not model_path.exists() or not metadata_path.exists():
        raise FileNotFoundError("Model artifacts not found. Run model_trainer.py first.")

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    data = load_data(args.data_path.resolve())
    features = data.drop(columns=metadata["target_column"])
    target = data[metadata["target_column"]]
    expected_features = metadata["feature_names"]
    if list(features.columns) != expected_features:
        raise ValueError("Dataset features differ from the model training features.")

    _, x_test, _, y_test = train_test_split(
        features,
        target,
        test_size=metadata["test_size"],
        random_state=metadata["random_state"],
        stratify=target,
    )
    model = joblib.load(model_path)
    predictions = model.predict(x_test)
    probabilities = model.predict_proba(x_test)[:, 1]
    metrics = {
        "test_precision": precision_score(y_test, predictions, zero_division=0),
        "test_recall": recall_score(y_test, predictions, zero_division=0),
        "test_f1": f1_score(y_test, predictions, zero_division=0),
        "test_roc_auc": roc_auc_score(y_test, probabilities),
    }
    matrix = confusion_matrix(y_test, predictions)

    report_path = model_dir / "evaluation_metrics.json"
    matrix_path = model_dir / "confusion_matrix.csv"
    report_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    pd.DataFrame(matrix, index=["actual_0", "actual_1"], columns=["predicted_0", "predicted_1"]).to_csv(matrix_path)

    print("=== Final test evaluation ===")
    print(f"Model: {metadata['best_model_name']}")
    print("Confusion matrix:")
    print(matrix)
    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")

    mlflow = configure_mlflow(not args.no_mlflow, args.tracking_uri, args.experiment_name)
    run_context = mlflow.start_run(run_name="model-evaluation") if mlflow else nullcontext()
    with run_context:
        if mlflow:
            mlflow.log_params(
                {
                    "evaluated_model": metadata["best_model_name"],
                    "model_path": str(model_path),
                    "test_size": metadata["test_size"],
                    "random_state": metadata["random_state"],
                }
            )
            mlflow.log_metrics({name: float(value) for name, value in metrics.items()})
            mlflow.log_artifact(str(report_path), artifact_path="evaluation")
            mlflow.log_artifact(str(matrix_path), artifact_path="evaluation")


if __name__ == "__main__":
    main()
