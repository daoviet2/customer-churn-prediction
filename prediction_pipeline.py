"""Predict customer churn from a form payload.

``predict_from_form`` accepts a dictionary whose keys are the raw fields used
during training, such as ``tenure``, ``Contract`` and ``MonthlyCharges``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import joblib
import pandas as pd


ROOT = Path(__file__).resolve().parent
DEFAULT_ARTIFACT_DIR = ROOT / "artifacts"


def load_artifacts(artifact_dir: Path = DEFAULT_ARTIFACT_DIR):
    """Load the fitted preprocessor, model, and required form fields."""
    preprocessor_path = artifact_dir / "preprocessor.joblib"
    model_path = artifact_dir / "churn_model.joblib"
    metadata_path = artifact_dir / "prediction_metadata.json"
    missing = [str(path) for path in (preprocessor_path, model_path, metadata_path) if not path.exists()]
    if missing:
        raise FileNotFoundError("Prediction artifacts are missing. Run training_pipeline.py first: " + ", ".join(missing))
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    return joblib.load(preprocessor_path), joblib.load(model_path), metadata


def form_to_dataframe(
    form_data: Mapping[str, Any], metadata: Mapping[str, Any]
) -> pd.DataFrame:
    """Validate one form submission and preserve the training-column order."""
    feature_names = metadata["feature_names"]
    missing = [name for name in feature_names if name not in form_data]
    if missing:
        raise ValueError("Missing form fields: " + ", ".join(missing))
    row = {name: form_data[name] for name in feature_names}
    numeric_features = metadata.get("numeric_features", ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"])
    for name in numeric_features:
        if name in row:
            row[name] = pd.to_numeric(row[name], errors="coerce")
            if pd.isna(row[name]):
                raise ValueError(f"'{name}' must be a valid number.")
    return pd.DataFrame([row], columns=feature_names)


def predict_from_form(form_data: Mapping[str, Any], artifact_dir: Path = DEFAULT_ARTIFACT_DIR) -> dict[str, Any]:
    """Return a churn prediction for one form submission.

    Flow: form input -> preprocessor.transform -> model.predict -> result.
    """
    preprocessor, model, metadata = load_artifacts(artifact_dir)
    form_frame = form_to_dataframe(form_data, metadata)
    transformed_features = preprocessor.transform(form_frame)
    prediction = int(model.predict(transformed_features)[0])
    result: dict[str, Any] = {
        "prediction": prediction,
        "label": "Churn" if prediction == 1 else "No Churn",
    }
    if hasattr(model, "predict_proba"):
        result["churn_probability"] = float(model.predict_proba(transformed_features)[0, 1])
    return result


def collect_form_input(feature_names: list[str]) -> dict[str, str]:
    """Ask a terminal user for every feature required by the trained model."""
    print("\n=== CUSTOMER CHURN PREDICTION ===")
    print("Enter the customer's information. Example categorical values: Yes, No, Male, Female.\n")
    return {name: input(f"{name}: ").strip() for name in feature_names}


def main() -> None:
    """Run an interactive prediction from one user's terminal input."""
    _, _, metadata = load_artifacts()
    form_data = collect_form_input(metadata["feature_names"])
    result = predict_from_form(form_data)
    print("\n=== PREDICTION RESULT ===")
    print(f"Result: {result['label']}")
    print(f"Churn probability: {result.get('churn_probability', 0.0):.2%}")


if __name__ == "__main__":
    main()
