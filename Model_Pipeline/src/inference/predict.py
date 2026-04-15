"""
EcoPulse Inference Pipeline — predict.py
=========================================
This is the CORE of the inference pipeline.

WHAT IT DOES:
    Loads a trained XGBoost model (.joblib file) and uses it to predict
    future carbon intensity given a set of features.

WHY WE NEED IT:
    During training, we called model.predict(X_test) inside train_xgboost.py
    and threw away the predictions after computing MAE. This file makes
    prediction a REUSABLE function that can be called by:
    - The FastAPI endpoint (real-time predictions)
    - The Airflow DAG (batch forecasting)
    - generate_predictions.py (evaluation)
    - Anyone who wants to use our model

HOW IT WORKS:
    1. Load the .joblib model file (once, at startup — not per request)
    2. Accept a pandas DataFrame of features
    3. Return predicted carbon intensity in gCO2/kWh

USAGE:
    from inference.predict import CarbonPredictor
    
    predictor = CarbonPredictor()
    predictions = predictor.predict(features_df, horizon=6)
    # → array of predicted carbon intensity values
"""

import os
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Optional, List, Union
import logging

logger = logging.getLogger(__name__)

# ============================================================
# PATH CONFIGURATION
# ============================================================
# Figure out where we are relative to Model_Pipeline/
# This works whether you run from src/, src/inference/, or repo root
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_MODEL_PIPELINE_DIR = os.path.abspath(os.path.join(_THIS_DIR, "..", ".."))
_MODELS_DIR = os.path.join(_MODEL_PIPELINE_DIR, "models")

# Available forecast horizons (each has its own trained model)
HORIZONS = [1, 12, 24]

# Columns that should NOT be used as features
# These must match exactly what train_xgboost.py drops during training
DROP_COLUMNS = [
    "datetime", "zone",
    "aws_region", "gcp_region", "azure_region",  # string columns
    "carbon_intensity_gco2_per_kwh",              # the target itself
    "carbon_intensity_target_1h",                  # forecast targets
    "carbon_intensity_target_12h",
    "carbon_intensity_target_24h",
]


class CarbonPredictor:
    """
    Loads trained models and predicts carbon intensity.
    
    WHY A CLASS?
        Loading a .joblib model takes ~0.5 seconds. If we loaded it
        every time someone asks for a prediction, the API would be slow.
        By using a class, we load ONCE at startup and reuse forever.
    
    USAGE:
        # Create predictor (loads all 4 models into memory)
        predictor = CarbonPredictor()
        
        # Predict 6 hours ahead for one row of data
        result = predictor.predict(features_df, horizon=6)
        
        # Predict all 4 horizons at once
        all_results = predictor.predict_all_horizons(features_df)
    """

    def __init__(self, models_dir: str = _MODELS_DIR):
        """
        Initialize the predictor by loading all trained models.
        
        WHAT HAPPENS HERE:
            - Looks for xgboost_tuned_*.joblib first (best models)
            - Falls back to xgboost_*.joblib if tuned not found
            - Stores models in a dictionary: {1: model_1h, 12: model_12h, 24: model_24h}
            - Also stores the feature column names each model expects
        
        Args:
            models_dir: Path to folder containing .joblib model files
        """
        self.models_dir = models_dir
        self.models: Dict[int, object] = {}           # horizon → model
        self.feature_names: Dict[int, List[str]] = {}  # horizon → expected columns
        self.model_types: Dict[int, str] = {}          # horizon → "xgboost_tuned" etc.
        
        self._load_all_models()

    def _load_all_models(self):
        """
        Load models for all 4 horizons.
        
        PRIORITY ORDER for each horizon:
            1. xgboost_tuned_{h}h.joblib  (Optuna-optimized — BEST)
            2. xgboost_{h}h.joblib        (default hyperparameters)
            3. lightgbm_{h}h.joblib       (backup model)
        
        WHY THIS ORDER:
            Tuned XGBoost beat everything — it's the production model.
            But if someone accidentally deletes the tuned file, we
            gracefully fall back instead of crashing.
        """
        for horizon in HORIZONS:
            model = None
            model_type = None

            # Try each model type in priority order
            for prefix in ["xgboost_tuned", "xgboost", "lightgbm"]:
                path = os.path.join(self.models_dir, f"{prefix}_{horizon}h.joblib")
                if os.path.exists(path):
                    model = joblib.load(path)
                    model_type = prefix
                    logger.info(f"Loaded {prefix}_{horizon}h.joblib")
                    break

            if model is None:
                logger.warning(
                    f"No model found for {horizon}h horizon in {self.models_dir}"
                )
                continue

            self.models[horizon] = model
            self.model_types[horizon] = model_type

            # Store feature names if the model remembers them
            # XGBoost stores this after fitting — it tells us exactly
            # which columns the model expects, in which order
            if hasattr(model, "feature_names_in_"):
                self.feature_names[horizon] = list(model.feature_names_in_)
            else:
                self.feature_names[horizon] = None

        logger.info(
            f"Loaded {len(self.models)} models: "
            f"{list(self.models.keys())} horizons"
        )

    def _prepare_features(self, df: pd.DataFrame, horizon: int) -> pd.DataFrame:
        """
        Transform raw input DataFrame into the exact features the model expects.
        
        WHAT THIS DOES:
            1. One-hot encode the 'zone' column
               - "US-MIDA-PJM" → zone_US-MIDA-PJM = 1, zone_US-NW-PACW = 0
               - This is how train_xgboost.py encoded it during training
            
            2. Drop non-feature columns (datetime, zone, target, strings)
            
            3. Align columns with what the model expects
               - If a column is missing → add it as 0 (safe default)
               - If an extra column exists → drop it
               - Reorder to match training order
        
        WHY THIS MATTERS:
            XGBoost expects features in the EXACT same order and format
            as during training. If column 5 was "temperature_2m_c" during
            training but "wind_speed_100m_ms" during inference, the model
            would use wind speed values with temperature's learned weights.
            Predictions would be nonsense.
        
        Args:
            df: Input DataFrame with raw features
            horizon: Forecast horizon (1, 12, or 24)
            
        Returns:
            DataFrame with features aligned to model's expectations
        """
        X = df.copy()

        # Step 1: One-hot encode zone
        if "zone" in X.columns:
            zone_dummies = pd.get_dummies(X["zone"], prefix="zone")
            X = pd.concat([X, zone_dummies], axis=1)

        # Step 2: Drop non-feature columns
        cols_to_drop = [c for c in DROP_COLUMNS if c in X.columns]
        X = X.drop(columns=cols_to_drop, errors="ignore")

        # Also drop any remaining string/object columns
        string_cols = X.select_dtypes(include=["object", "category"]).columns
        if len(string_cols) > 0:
            logger.debug(f"Dropping string columns: {list(string_cols)}")
            X = X.drop(columns=string_cols, errors="ignore")

        # Step 3: Align with model's expected features
        expected = self.feature_names.get(horizon)
        if expected is not None:
            # Add missing columns as 0
            for col in expected:
                if col not in X.columns:
                    X[col] = 0
                    logger.debug(f"Added missing column: {col}")

            # Select only expected columns, in the right order
            X = X[expected]

        return X

    def predict(
        self,
        features: pd.DataFrame,
        horizon: int,
    ) -> np.ndarray:
        """
        Predict carbon intensity for a given horizon.
        
        THIS IS THE MAIN FUNCTION everyone calls.
        
        Args:
            features: DataFrame with input features (can be 1 row or many)
                      Must contain the same columns as the training data
            horizon: How far ahead to predict (1, 12, or 24 hours)
        
        Returns:
            numpy array of predicted carbon intensity values (gCO2/kWh)
        
        Example:
            predictor = CarbonPredictor()
            
            # Single prediction
            preds = predictor.predict(current_features_df, horizon=12)
            print(f"Carbon intensity in 12h: {preds[0]:.1f} gCO2/kWh")
            
            # Batch prediction (24 rows = 24 hours)
            preds = predictor.predict(next_24h_features, horizon=1)
            # → array of 24 predictions
        """
        if horizon not in self.models:
            available = list(self.models.keys())
            raise ValueError(
                f"No model loaded for {horizon}h horizon. "
                f"Available: {available}"
            )

        # Prepare features (one-hot encode, align columns)
        X = self._prepare_features(features, horizon)

        # Predict
        model = self.models[horizon]
        predictions = model.predict(X)

        # Safety: carbon intensity can't be negative
        predictions = np.maximum(predictions, 0)

        return predictions

    def predict_all_horizons(
        self, features: pd.DataFrame
    ) -> Dict[int, np.ndarray]:
        """
        Predict all 4 horizons at once.
        
        Useful for generating a complete forecast:
        "In 1h it'll be X, in 12h it'll be Z, in 24h it'll be W"
        
        Args:
            features: DataFrame with input features
            
        Returns:
            Dict mapping horizon → prediction array
            {1: array([287.3]), 6: array([312.5]), 12: array([245.8]), 24: array([198.2])}
        """
        results = {}
        for horizon in sorted(self.models.keys()):
            results[horizon] = self.predict(features, horizon)
        return results

    def get_model_info(self) -> Dict:
        """
        Return metadata about loaded models.
        Useful for the /model-info API endpoint.
        """
        info = {}
        for horizon in sorted(self.models.keys()):
            model = self.models[horizon]
            info[f"{horizon}h"] = {
                "model_type": self.model_types[horizon],
                "n_features": len(self.feature_names.get(horizon, [])),
                "file": f"{self.model_types[horizon]}_{horizon}h.joblib",
            }
            # Add XGBoost-specific info if available
            if hasattr(model, "n_estimators"):
                info[f"{horizon}h"]["n_estimators"] = model.n_estimators
            if hasattr(model, "best_iteration"):
                info[f"{horizon}h"]["best_iteration"] = model.best_iteration
        return info


# ============================================================
# STANDALONE USAGE
# Run this file directly to test that models load and predict
# ============================================================
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("=" * 70)
    print("EcoPulse Inference — Model Loading Test")
    print("=" * 70)

    # 1. Load all models
    predictor = CarbonPredictor()

    # 2. Print model info
    info = predictor.get_model_info()
    for horizon, details in info.items():
        print(f"\n  {horizon}:")
        for k, v in details.items():
            print(f"    {k}: {v}")

    # 3. Test with actual test data (if available)
    test_paths = [
        os.path.join(_MODEL_PIPELINE_DIR, "..", "Data_Pipeline", "data",
                     "processed", "test_split.parquet"),
        os.path.join(_MODEL_PIPELINE_DIR, "..", "Data_Pipeline", "data",
                     "splits", "test.parquet"),
    ]

    test_df = None
    for path in test_paths:
        if os.path.exists(path):
            test_df = pd.read_parquet(path)
            print(f"\n  Loaded test data: {test_df.shape} from {path}")
            break

    if test_df is not None:
        # Predict on first 5 rows as a sanity check
        sample = test_df.head(5)

        print(f"\n  Sample predictions (first 5 test rows):")
        print(f"  {'Horizon':<10} {'Predictions':>50}")
        print(f"  {'-'*60}")

        all_preds = predictor.predict_all_horizons(sample)
        for horizon, preds in all_preds.items():
            preds_str = ", ".join(f"{p:.1f}" for p in preds)
            print(f"  {horizon}h{'':<8} [{preds_str}]")

        print(f"\n  ✅ Inference pipeline working correctly!")
    else:
        print("\n  ⚠️  No test data found — model loading verified, "
              "but couldn't run test predictions")