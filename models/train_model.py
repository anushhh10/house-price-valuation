"""
Model Training and SHAP Explainer Pipeline (INR Valuation).

Trains Random Forest, XGBoost, and CatBoost regressors on real estate data in Indian Rupees,
evaluates performance metrics (RMSE, MAE, R²), selects the best-performing model,
and persists model artifacts and SHAP explainer for downstream inference.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
import shap


FEATURE_COLUMNS = [
    "sqft",
    "bhk",
    "bathrooms",
    "property_age",
    "floor_number",
    "total_floors",
    "parking_spaces",
    "school_score",
    "hospital_distance_km",
    "transit_score",
    "crime_rate",
    "police_proximity_km",
    "aqi",
    "green_cover_pct",
    "noise_level_db",
    "growth_index",
    "commercial_hub_distance_km",
]

TARGET_COLUMN = "price"


def format_inr(val: float) -> str:
    """Format an INR number to Lakhs or Crores."""
    abs_val = abs(val)
    sign = "-" if val < 0 else ""
    if abs_val >= 10000000:
        return f"{sign}INR {abs_val / 10000000:.2f} Cr"
    elif abs_val >= 100000:
        return f"{sign}INR {abs_val / 100000:.2f} L"
    return f"{sign}INR {abs_val:,.0f}"


def load_and_split_data(
    csv_path: str = "data/house_prices.csv",
    test_size: float = 0.20,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Load dataset and split into train and test sets."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset not found at {csv_path}. Run data/generate_dataset.py first.")

    df = pd.read_csv(csv_path)
    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    return X_train, X_test, y_train, y_test


def train_and_evaluate_models(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> Tuple[Dict[str, Any], Dict[str, Dict[str, Any]], str]:
    """
    Train Random Forest, XGBoost, and CatBoost regressors,
    compute performance metrics, and select the optimal model.
    """
    candidate_models = {
        "RandomForest": RandomForestRegressor(
            n_estimators=120,
            max_depth=14,
            min_samples_split=4,
            random_state=42,
            n_jobs=-1,
        ),
        "XGBoost": XGBRegressor(
            n_estimators=200,
            learning_rate=0.06,
            max_depth=6,
            subsample=0.85,
            colsample_bytree=0.85,
            random_state=42,
            n_jobs=-1,
        ),
        "CatBoost": CatBoostRegressor(
            iterations=250,
            learning_rate=0.07,
            depth=6,
            random_seed=42,
            verbose=0,
        ),
    }

    metrics_report: Dict[str, Dict[str, Any]] = {}
    best_model_name = ""
    best_r2 = -float("inf")

    print("\n" + "=" * 70)
    print(f"{'Model':<16} | {'RMSE (INR)':<16} | {'MAE (INR)':<16} | {'R2 Score':<10}")
    print("=" * 70)

    for name, model in candidate_models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
        mae = float(mean_absolute_error(y_test, preds))
        r2 = float(r2_score(y_test, preds))

        metrics_report[name] = {
            "RMSE": round(rmse, 2),
            "MAE": round(mae, 2),
            "R2": round(r2, 4),
            "RMSE_display": format_inr(rmse).replace("INR", "₹"),
            "MAE_display": format_inr(mae).replace("INR", "₹"),
        }

        print(f"{name:<16} | {format_inr(rmse):<16} | {format_inr(mae):<16} | {r2:<10.4f}")

        if r2 > best_r2:
            best_r2 = r2
            best_model_name = name

    print("=" * 70)
    print(f"Optimal Model Selected: {best_model_name} (R2 = {best_r2:.4f})\n")

    return candidate_models, metrics_report, best_model_name


def build_shap_explainer(best_model: Any, X_train: pd.DataFrame) -> shap.Explainer:
    """Initialize a TreeExplainer for the selected gradient boosted / tree regressor."""
    print("Initializing SHAP TreeExplainer on training background distribution...")
    background_sample = X_train.sample(n=min(200, len(X_train)), random_state=42)
    try:
        explainer = shap.TreeExplainer(best_model, data=background_sample)
    except Exception as e:
        print(f"Standard TreeExplainer init note ({e}), falling back to direct TreeExplainer...")
        explainer = shap.TreeExplainer(best_model)
    return explainer


def save_artifacts(
    best_model_name: str,
    best_model: Any,
    explainer: Any,
    metrics_report: Dict[str, Any],
    X_train: pd.DataFrame,
    output_dir: str = "models",
):
    """Save serialized models, feature metadata, metrics, and explainer."""
    os.makedirs(output_dir, exist_ok=True)

    model_path = os.path.join(output_dir, "best_model.joblib")
    explainer_path = os.path.join(output_dir, "shap_explainer.joblib")
    metrics_path = os.path.join(output_dir, "metrics.json")
    features_path = os.path.join(output_dir, "feature_names.json")
    background_sample_path = os.path.join(output_dir, "background_sample.csv")

    joblib.dump(best_model, model_path)
    joblib.dump(explainer, explainer_path)

    background_df = X_train.sample(n=min(250, len(X_train)), random_state=42)
    background_df.to_csv(background_sample_path, index=False)

    metadata = {
        "best_model": best_model_name,
        "metrics": metrics_report,
        "features": FEATURE_COLUMNS,
        "currency": "INR",
        "currency_symbol": "₹",
    }

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)

    with open(features_path, "w", encoding="utf-8") as f:
        json.dump(FEATURE_COLUMNS, f, indent=4)

    print(f"Artifacts successfully saved to '{output_dir}/':")
    print(f" - Best Model: {model_path} ({best_model_name})")
    print(f" - SHAP Explainer: {explainer_path}")
    print(f" - Metrics & Config: {metrics_path}")
    print(f" - Feature Columns: {features_path}")
    print(f" - Background Samples: {background_sample_path}")


def main():
    print("Step 1: Loading real estate dataset (INR)...")
    X_train, X_test, y_train, y_test = load_and_split_data("data/house_prices.csv")
    print(f"Dataset split: {len(X_train)} training records, {len(X_test)} testing records.")

    print("\nStep 2: Training candidate models (RandomForest, XGBoost, CatBoost)...")
    models, metrics, best_name = train_and_evaluate_models(X_train, X_test, y_train, y_test)
    best_model = models[best_name]

    print("\nStep 3: Building SHAP Explainer...")
    explainer = build_shap_explainer(best_model, X_train)

    print("\nStep 4: Persisting model artifacts...")
    save_artifacts(best_name, best_model, explainer, metrics, X_train)
    print("\nTraining and Explainer Pipeline successfully completed!")


if __name__ == "__main__":
    main()
