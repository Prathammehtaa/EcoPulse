/**
 * EcoPulse API Client
 * All calls to FastAPI backend (http://localhost:8000)
 * Replace mockData.js imports with these functions.
 */

const BASE_URL = "http://localhost:8000";

async function get(path) {
  const res = await fetch(`${BASE_URL}${path}`);
  if (!res.ok) throw new Error(`GET ${path} failed: ${res.status}`);
  return res.json();
}

async function post(path, body) {
  const res = await fetch(`${BASE_URL}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) throw new Error(`POST ${path} failed: ${res.status}`);
  return res.json();
}

// GET /regions → [{zone, intensity, bucket, carbonFreePct, renewablePct}]
export const getRegions = () => get("/regions");

// GET /forecast/{zone}?horizon=6 → {zone, horizon, values: [24 floats]}
export const getForecast = (zone, horizon = 6) =>
  get(`/forecast/${encodeURIComponent(zone)}?horizon=${horizon}`);

// POST /predict → full recommendation object
export const predict = (zone, energyKwh, runtimeHours, horizon, priorityHours) =>
  post("/predict", {
    zone,
    energy_kwh: energyKwh,
    runtime_hours: runtimeHours,
    horizon,
    priority_hours: priorityHours,
  });

// GET /metrics
export const getMetrics = () => get("/metrics");

// GET /drift
export const getDrift = () => get("/drift");

// GET /shap
export const getShap = () => get("/shap");

// GET /alerts
export const getAlerts = () => get("/alerts");

// GET /logs
export const getLogs = () => get("/logs");

// GET /users
export const getUsers = () => get("/users");

// POST /retrain
export const retrain = (model = "lightgbm", horizon = 6) =>
  post(`/retrain?model=${model}&horizon=${horizon}`, {});

// GET /health
export const getHealth = () => get("/health");
