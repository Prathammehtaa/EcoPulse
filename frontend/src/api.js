/**
 * EcoPulse API Client
 * In development: calls http://localhost:8000 directly
 * In production (K8s): calls /api/ which nginx proxies to backend-service
 */

const BASE_URL = import.meta.env.PROD
  ? "/api"
  : "http://localhost:8000";

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

export const getRegions = () => get("/regions");
export const getForecast = (zone, horizon = 6) =>
  get(`/forecast/${encodeURIComponent(zone)}?horizon=${horizon}`);
export const predict = (zone, energyKwh, runtimeHours, horizon, priorityHours) =>
  post("/predict", {
    zone,
    energy_kwh: energyKwh,
    runtime_hours: runtimeHours,
    horizon,
    priority_hours: priorityHours,
  });
export const getMetrics = () => get("/metrics");
export const getDrift = () => get("/drift");
export const getShap = () => get("/shap");
export const getAlerts = () => get("/alerts");
export const getLogs = () => get("/logs");
export const getUsers = () => get("/users");
export const retrain = (model = "lightgbm", horizon = 6) =>
  post(`/retrain?model=${model}&horizon=${horizon}`, {});
export const getHealth = () => get("/health");