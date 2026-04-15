export const regions = [
  {
    zone: "US-MIDA-PJM",
    name: "Northern Virginia",
    intensity: 287,
    bucket: "Medium (200-350)",
    carbonFreePct: 18.4,
    renewablePct: 12.1
  },
  {
    zone: "US-NW-PACW",
    name: "Portland Oregon",
    intensity: 134,
    bucket: "Low (100-200)",
    carbonFreePct: 61.2,
    renewablePct: 58.3
  }
];

export const zoneDisplayNames = {
  "US-MIDA-PJM": "Northern Virginia",
  "US-NW-PACW": "Portland Oregon"
};

export function getZoneDisplayName(zone) {
  return zoneDisplayNames[zone] ?? zone;
}

export const forecast24h = {
  "US-MIDA-PJM": [279, 268, 254, 243, 232, 221, 208, 198, 205, 219, 237, 251, 243, 226, 210, 196, 188, 176, 162, 149, 142, 136, 131, 138],
  "US-NW-PACW": [141, 138, 134, 136, 142, 148, 157, 169, 177, 184, 191, 197, 189, 180, 170, 161, 153, 146, 138, 130, 122, 118, 121, 126]
};

export const alerts = [
  {
    type: "error",
    title: "Drift detected - 6h LightGBM model",
    detail: "wind_speed PSI 0.28 > threshold 0.2. Retraining recommended.",
    time: "34 min ago",
    active: true
  },
  {
    type: "success",
    title: "Carbon drop - Northern Virginia resolved",
    detail: "Grid dropped below threshold. Workloads were scheduled.",
    time: "2 hrs ago",
    active: false
  },
  {
    type: "warning",
    title: "High carbon - Portland Oregon elevated",
    detail: "Intensity at 312 gCO2/kWh. Consider deferring non-urgent workloads.",
    time: "3 hrs ago",
    active: false
  }
];

export const metricsRows = [
  { horizon: "1h", xgboostMae: 25.14, lightgbmMae: 26.8, baselineMae: 57.48, xgboostR2: 0.94 },
  { horizon: "6h", xgboostMae: 34.34, lightgbmMae: 32.1, baselineMae: 71.33, xgboostR2: 0.89 },
  { horizon: "12h", xgboostMae: 39.97, lightgbmMae: 38.4, baselineMae: 76.36, xgboostR2: 0.84 },
  { horizon: "24h", xgboostMae: 43.01, lightgbmMae: 41.2, baselineMae: 68.79, xgboostR2: 0.81 }
];

export const driftRows = [
  { feature: "wind_speed_100m_ms", psi: 0.28, status: "Warning", action: "Retrain recommended" },
  { feature: "hour_of_day", psi: 0.08, status: "OK", action: "-" },
  { feature: "solar_potential", psi: 0.11, status: "OK", action: "-" },
  { feature: "demand_lag_1h", psi: 0.07, status: "OK", action: "-" },
  { feature: "temperature_2m_c", psi: 0.09, status: "OK", action: "-" }
];

export const shapRows = [
  { feature: "hour_of_day", value: 0.42, direction: "Positive" },
  { feature: "wind_speed_100m_ms", value: 0.33, direction: "Positive" },
  { feature: "carbon_intensity_lag_1h", value: 0.29, direction: "Positive" },
  { feature: "total_load_mw", value: 0.25, direction: "Positive" },
  { feature: "solar_potential", value: 0.21, direction: "Negative" }
];

export const usersSeed = [
  { initials: "HU", email: "hitarth@example.com", status: "Active", role: "Operator" },
  { initials: "KG", email: "kapish@example.com", status: "Active", role: "Operator" },
  { initials: "PM", email: "pratham@example.com", status: "Inactive", role: "Operator" },
  { initials: "AA", email: "aaditya@example.com", status: "Active", role: "Operator" }
];

export const logs = [
  ["10:49", "INFO", "MLflow run - XGBoost 1h - MAE 25.14 RMSE 11.2 R2 0.94 - artifacts saved"],
  ["10:45", "INFO", "Workload approved - ML retraining - Northern Virginia - CO2 saved 24.8 kg"],
  ["10:30", "WARN", "Drift alert - wind_speed_100m_ms PSI 0.28 > threshold 0.2"],
  ["09:58", "INFO", "GCS fetch - ecopulse-shared-data - 1440 rows - parquet loaded"],
  ["09:45", "INFO", "Model retrain - LightGBM 6h - R2 0.89 - pushed to registry"]
];
