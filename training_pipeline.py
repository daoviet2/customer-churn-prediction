"""Train and save the preprocessing and churn-model artifacts.

The saved artifacts are intentionally separate so the prediction pipeline can
apply ``preprocessor.transform`` before it calls ``model.predict``.

Example:
    python training_pipeline.py --data-path data/raw/Customer-Churn.csv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_PATH = ROOT / "data" / "raw" / "Customer-Churn.csv"
DEFAULT_ARTIFACT_DIR = ROOT / "artifacts"
TARGET_COLUMN = "Churn"
ID_COLUMN = "customerID"


def prepare_training_data(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Return raw form features and a binary churn target."""
    if TARGET_COLUMN not in data:
        raise ValueError(f"Dataset must contain the '{TARGET_COLUMN}' column.")

    features = data.drop(columns=[TARGET_COLUMN, ID_COLUMN], errors="ignore").copy()
    if "TotalCharges" in features:
        features["TotalCharges"] = pd.to_numeric(features["TotalCharges"], errors="coerce")

    target = data[TARGET_COLUMN].replace({"Yes": 1, "No": 0})
    target = pd.to_numeric(target, errors="raise").astype(int)
    if set(target.unique()) - {0, 1}:
        raise ValueError(f"'{TARGET_COLUMN}' must contain only Yes/No or 0/1 values.")
    return features, target


def build_preprocessor(features: pd.DataFrame) -> ColumnTransformer:
    """Build preprocessing for the raw fields collected by the form."""
    numeric_columns = features.select_dtypes(include="number").columns.tolist()
    categorical_columns = [column for column in features.columns if column not in numeric_columns]

    return ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline([("imputer", SimpleImputer(strategy="median"))]),
                numeric_columns,
            ),
            (
                "categorical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("one_hot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_columns,
            ),
        ],
        remainder="drop",
    )


def train(data_path: Path = DEFAULT_DATA_PATH, artifact_dir: Path = DEFAULT_ARTIFACT_DIR) -> None:
    """Fit artifacts from raw Telco churn data and write them to *artifact_dir*."""
    if not data_path.exists():
        raise FileNotFoundError(
            f"Raw data was not found at {data_path}. Retrieve it first (for example: dvc pull)."
        )

    features, target = prepare_training_data(pd.read_csv(data_path))
    preprocessor = build_preprocessor(features)
    transformed_features = preprocessor.fit_transform(features)
    model = RandomForestClassifier(
        n_estimators=300,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
    )
    model.fit(transformed_features, target)

    artifact_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(preprocessor, artifact_dir / "preprocessor.joblib")
    joblib.dump(model, artifact_dir / "churn_model.joblib")
    (artifact_dir / "prediction_metadata.json").write_text(
        json.dumps(
            {
                "feature_names": features.columns.tolist(),
                "numeric_features": features.select_dtypes(include="number").columns.tolist(),
                "target_column": TARGET_COLUMN,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Saved preprocessor: {artifact_dir / 'preprocessor.joblib'}")
    print(f"Saved model: {artifact_dir / 'churn_model.joblib'}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the customer churn prediction pipeline.")
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACT_DIR)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args.data_path, args.artifact_dir)
