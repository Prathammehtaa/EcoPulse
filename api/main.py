"""
EcoPulse — FastAPI Backend
===========================
Run:
    cd EcoPulse
    uvicorn api.main:app --reload --port 8000

Uses the real inference pipeline:
    Model_Pipeline/src/inference/predict.py      → CarbonPredictor
    Model_Pipeline/src/inference/feature_builder.py → FeatureBuilder
    Model_Pipeline/src/inference/green_window.py  → WorkloadScheduler, GreenWindowDetector
"""

import os
import sys
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional, List
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ── Path setup ────────────────────────────────────────────────────────────────
API_DIR        = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT      = os.path.dirname(API_DIR)
MODEL_PIPE_DIR = os.path.join(REPO_ROOT, "Model_Pipeline")
SRC_DIR        = os.path.join(MODEL_PIPE_DIR, "src")
DATA_DIR       = os.path.join(REPO_ROOT, "Data_Pipeline", "data", "processed")

sys.path.insert(0, SRC_DIR)

from inference.predict import CarbonPredictor
from inference.feature_builder import FeatureBuilder
from inference.green_window import GreenWindowDetector, WorkloadScheduler
from utils import load_split, ZONE_COL, HORIZONS

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("ecopulse-api")

# ── Zone metadata ─────────────────────────────────────────────────────────────
ZONES = {
    "US-MIDA-PJM": {"carbon_free_pct": 18.4, "renewable_pct": 12.1},
    "US-NW-PACW":  {"carbon_free_pct": 61.2, "renewable_pct": 58.3},
    "US-NE-ISNE":  {"carbon_free_pct": 44.7, "renewable_pct": 39.5},
}

def intensity_bucket(v: float) -> str:
    if v < 100:  return "Very Low (<100)"
    if v < 200:  return "Low (100-200)"
    if v < 350:  return "Medium (200-350)"
    if v < 500:  return "High (350-500)"
    return "Very High (500+)"

# ── Global inference objects ──────────────────────────────────────────────────
predictor: CarbonPredictor   = None
scheduler: WorkloadScheduler = None
detector:  GreenWindowDetector = None
test_df:   pd.DataFrame      = None   # used for generating forecasts

# ── Startup ───────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    global predictor, scheduler, detector, test_df

    logger.info("Loading inference pipeline...")

    # Load models via CarbonPredictor
    predictor = CarbonPredictor(
        models_dir=os.path.join(MODEL_PIPE_DIR, "models")
    )

    # WorkloadScheduler and GreenWindowDetector (stateless — just methods)
    scheduler = WorkloadScheduler()
    detector  = GreenWindowDetector(method="percentile", percentile=25.0)

    # Load test data once — used to build feature vectors for forecasting
    test_path = os.path.join(DATA_DIR, "test_split.parquet")
    if os.path.exists(test_path):
        test_df = pd.read_parquet(test_path)
        test_df["datetime"] = pd.to_datetime(test_df["datetime"])
        logger.info(f"Test data loaded: {test_df.shape}")
    else:
        logger.warning(f"Test data not found at {test_path} — using fallback forecasts")

    logger.info("Inference pipeline ready.")
    yield

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="EcoPulse API",
    description="Carbon-aware workload scheduling backend",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Pydantic schemas ──────────────────────────────────────────────────────────
class PredictRequest(BaseModel):
    zone: str
    energy_kwh: float = 120.0
    runtime_hours: int = 4
    horizon: int = 6
    priority_hours: int = 6

class PredictResponse(BaseModel):
    recommended_start: str
    hours_to_wait: int
    expected_intensity_gco2_kwh: float
    immediate_intensity_gco2_kwh: float
    runtime_hours: int
    energy_kwh: float
    immediate_co2_kg: float
    optimal_co2_kg: float
    co2_saved_kg: float
    co2_savings_pct: float
    recommendation: str
    confidence: float
    zone: str
    horizon: int

# ── Forecast helper ───────────────────────────────────────────────────────────
def get_forecast_for_zone(zone: str, horizon: int, n_hours: int = 24) -> list:
    """
    Generate n_hours of carbon intensity predictions using the real model.
    Uses the last n_hours rows of test data for the given zone as feature input.
    Falls back to sinusoidal mock if model or data is unavailable.
    """
    fallback_bases = {"US-MIDA-PJM": 287, "US-NW-PACW": 134, "US-NE-ISNE": 176}

    if predictor is None or test_df is None or horizon not in predictor.models:
        base = fallback_bases.get(zone, 200)
        np.random.seed(hash(zone) % 1000)
        return [round(max(50, base + np.random.randint(-40, 40) + 20 * np.sin(i / 4)), 1)
                for i in range(n_hours)]

    try:
        zone_df = test_df[test_df[ZONE_COL] == zone].copy()
        if len(zone_df) < n_hours:
            raise ValueError(f"Only {len(zone_df)} rows for zone {zone}, need {n_hours}")

        # Use the last n_hours rows as our "next 24 hours" feature set
        zone_df = zone_df.tail(n_hours).reset_index(drop=True)

        # Predict using CarbonPredictor — handles feature alignment internally
        preds = predictor.predict(zone_df, horizon=horizon)
        return [round(float(max(50, p)), 1) for p in preds[:n_hours]]

    except Exception as e:
        logger.warning(f"Real forecast failed for {zone} {horizon}h: {e} — using fallback")
        base = fallback_bases.get(zone, 200)
        np.random.seed(hash(zone) % 1000)
        return [round(max(50, base + np.random.randint(-40, 40) + 20 * np.sin(i / 4)), 1)
                for i in range(n_hours)]


def build_forecast_df(zone: str, values: list) -> pd.DataFrame:
    """
    Build a forecast DataFrame for WorkloadScheduler.find_optimal_schedule().
    WorkloadScheduler expects: [{"datetime": ..., "predicted_carbon_intensity": ...}]
    """
    now = datetime.now().replace(minute=0, second=0, microsecond=0)
    return [
        {"datetime": str(now + timedelta(hours=i)), "predicted_carbon_intensity": v}
        for i, v in enumerate(values)
    ]

# ══════════════════════════════════════════════════════════════════════════════
#  ROUTES
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "models_loaded": list(predictor.models.keys()) if predictor else [],
        "model_types": predictor.model_types if predictor else {},
        "test_data_loaded": test_df is not None,
        "timestamp": datetime.now().isoformat(),
    }


@app.get("/regions")
def get_regions():
    """
    Current carbon intensity for all zones.
    Matches React mockData.js `regions` shape exactly.
    """
    result = []
    for zone, meta in ZONES.items():
        values = get_forecast_for_zone(zone, horizon=1, n_hours=1)
        intensity = values[0]
        result.append({
            "zone": zone,
            "intensity": intensity,
            "bucket": intensity_bucket(intensity),
            "carbonFreePct": meta["carbon_free_pct"],
            "renewablePct":  meta["renewable_pct"],
        })
    return result


@app.get("/forecast/{zone}")
def get_forecast(zone: str, horizon: int = 6):
    """
    24-hour forecast as a flat list of floats.
    Matches React mockData.js `forecast24h[zone]` shape exactly.
    """
    if zone not in ZONES:
        raise HTTPException(status_code=404, detail=f"Zone '{zone}' not found")
    values = get_forecast_for_zone(zone, horizon=horizon, n_hours=24)
    return {"zone": zone, "horizon": horizon, "values": values}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    """
    Core scheduling endpoint using real WorkloadScheduler.
    Response shape matches SchedulerPage.jsx buildRecommendation() exactly.
    """
    if req.zone not in ZONES:
        raise HTTPException(status_code=404, detail=f"Zone '{req.zone}' not found")
    if req.horizon not in HORIZONS:
        raise HTTPException(status_code=400, detail=f"Horizon must be one of {HORIZONS}")

    # Step 1 — get real forecast values
    forecast_values = get_forecast_for_zone(req.zone, horizon=req.horizon, n_hours=24)

    # Step 2 — build forecast dicts for WorkloadScheduler
    forecast_dicts = build_forecast_df(req.zone, forecast_values)

    # Step 3 — find optimal schedule using real WorkloadScheduler
    result = scheduler.find_optimal_schedule(
        forecast=forecast_dicts,
        runtime_hours=req.runtime_hours,
        flexibility_hours=req.priority_hours,
        energy_kwh=req.energy_kwh,
    )

    return PredictResponse(
        recommended_start=result["recommended_start"],
        hours_to_wait=result["hours_to_wait"],
        expected_intensity_gco2_kwh=result["expected_intensity_gco2_kwh"],
        immediate_intensity_gco2_kwh=result["immediate_intensity_gco2_kwh"],
        runtime_hours=req.runtime_hours,
        energy_kwh=req.energy_kwh,
        immediate_co2_kg=result["immediate_co2_kg"],
        optimal_co2_kg=result["optimal_co2_kg"],
        co2_saved_kg=result["co2_saved_kg"],
        co2_savings_pct=result["co2_savings_pct"],
        recommendation=result["recommendation"],
        confidence=0.87,
        zone=req.zone,
        horizon=req.horizon,
    )


@app.get("/green-windows/{zone}")
def get_green_windows(zone: str, horizon: int = 6):
    """
    Find green carbon windows using GreenWindowDetector.
    """
    if zone not in ZONES:
        raise HTTPException(status_code=404, detail=f"Zone '{zone}' not found")

    values   = get_forecast_for_zone(zone, horizon=horizon, n_hours=24)
    forecast = build_forecast_df(zone, values)
    forecast_df = pd.DataFrame(forecast).rename(
        columns={"predicted_carbon_intensity": "predicted_carbon_intensity"}
    )
    result = detector.find_green_windows(forecast_df)
    return result


@app.get("/metrics")
def get_metrics():
    """Model performance — matches React metricsRows shape."""
    if predictor:
        info = predictor.get_model_info()
        logger.info(f"Model info: {info}")
    return [
        {"horizon": "1h",  "xgboostMae": 25.14, "lightgbmMae": 26.8,  "baselineMae": 57.48, "xgboostR2": 0.94},
        {"horizon": "6h",  "xgboostMae": 34.34, "lightgbmMae": 32.1,  "baselineMae": 71.33, "xgboostR2": 0.89},
        {"horizon": "12h", "xgboostMae": 39.97, "lightgbmMae": 38.4,  "baselineMae": 76.36, "xgboostR2": 0.84},
        {"horizon": "24h", "xgboostMae": 43.01, "lightgbmMae": 41.2,  "baselineMae": 68.79, "xgboostR2": 0.81},
    ]


@app.get("/drift")
def get_drift():
    """Drift monitoring — matches React driftRows shape."""
    return [
        {"feature": "wind_speed_100m_ms", "psi": 0.28, "status": "Warning", "action": "Retrain recommended"},
        {"feature": "hour_of_day",         "psi": 0.08, "status": "OK",      "action": "-"},
        {"feature": "solar_potential",      "psi": 0.11, "status": "OK",      "action": "-"},
        {"feature": "demand_lag_1h",        "psi": 0.07, "status": "OK",      "action": "-"},
        {"feature": "temperature_2m_c",     "psi": 0.09, "status": "OK",      "action": "-"},
    ]


@app.get("/shap")
def get_shap():
    """SHAP feature importance — matches React shapRows shape."""
    return [
        {"feature": "hour_of_day",             "value": 0.42, "direction": "Positive"},
        {"feature": "wind_speed_100m_ms",       "value": 0.33, "direction": "Positive"},
        {"feature": "carbon_intensity_lag_1h",  "value": 0.29, "direction": "Positive"},
        {"feature": "total_load_mw",            "value": 0.25, "direction": "Positive"},
        {"feature": "solar_potential",          "value": 0.21, "direction": "Negative"},
        {"feature": "temperature_2m_c",         "value": 0.17, "direction": "Positive"},
        {"feature": "carbon_intensity_lag_24h", "value": 0.15, "direction": "Positive"},
        {"feature": "cloud_cover_pct",          "value": 0.13, "direction": "Negative"},
    ]


@app.get("/alerts")
def get_alerts():
    """Active alerts — matches React alerts shape."""
    return [
        {"type": "error",   "title": "Drift detected — 6h LightGBM model",
         "detail": "wind_speed PSI 0.28 > threshold 0.2. Retraining recommended.",
         "time": "34 min ago", "active": True},
        {"type": "success", "title": "Carbon drop — US-MIDA-PJM resolved",
         "detail": "Grid dropped below threshold. Workloads were scheduled.",
         "time": "2 hrs ago", "active": False},
        {"type": "warning", "title": "High carbon — US-NW-PACW elevated",
         "detail": "Intensity at 312 gCO2/kWh. Consider deferring non-urgent workloads.",
         "time": "3 hrs ago", "active": False},
    ]


@app.get("/logs")
def get_logs():
    """System logs — matches React logs shape."""
    return [
        ["10:49", "INFO",  "MLflow run — XGBoost 1h — MAE 25.14 RMSE 11.2 R² 0.94 — artifacts saved"],
        ["10:45", "INFO",  "Workload approved — ML retraining — US-MIDA-PJM — CO2 saved 24.8 kg"],
        ["10:30", "WARN",  "Drift alert — wind_speed_100m_ms PSI 0.28 > threshold 0.2"],
        ["09:58", "INFO",  "GCS fetch — ecopulse-shared-data — 1440 rows — parquet loaded"],
        ["09:45", "INFO",  "Model retrain — LightGBM 6h — R² 0.89 — pushed to registry"],
        ["09:30", "WARN",  "Override logged — Database backup — recommended 13:00 → ran 09:30"],
        ["09:00", "WARN",  "Bias audit — flagged Very Low (0-100) subgroup — R² 0.71"],
    ]


@app.get("/users")
def get_users():
    """Registered users — matches React usersSeed shape."""
    return [
        {"initials": "HU", "email": "hitarth@example.com",  "status": "Active",   "role": "Operator"},
        {"initials": "KG", "email": "kapish@example.com",   "status": "Active",   "role": "Operator"},
        {"initials": "PM", "email": "pratham@example.com",  "status": "Inactive", "role": "Operator"},
        {"initials": "AA", "email": "aaditya@example.com",  "status": "Active",   "role": "Operator"},
        {"initials": "CL", "email": "clyde@example.com",    "status": "Active",   "role": "Operator"},
    ]


@app.post("/retrain")
def retrain(model: str = "lightgbm", horizon: int = 6):
    """Trigger model retraining — stub for now."""
    return {
        "status": "triggered",
        "model": f"{model}_{horizon}h",
        "message": f"Retraining {model}_{horizon}h triggered. Check MLflow for run status.",
        "timestamp": datetime.now().isoformat(),
    }


@app.get("/model-info")
def model_info():
    """Return metadata about loaded models."""
    if predictor is None:
        raise HTTPException(status_code=503, detail="Models not loaded yet")
    return predictor.get_model_info()
