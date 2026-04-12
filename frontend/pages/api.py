import streamlit as st
import json
import numpy as np
from datetime import datetime, timedelta


# ── Mock helpers ──────────────────────────────────────────────────────────────
ZONES    = ["US-MIDA-PJM", "US-NW-PACW"]
HORIZONS = [1, 6, 12, 24]

def mock_predict(energy_kwh, runtime_hours, zone, horizon, priority):
    imm      = 287 if zone == "US-MIDA-PJM" else 134
    best_int = imm - 40 if priority != "urgent" else imm
    wait     = 3 if priority == "flexible" else (1 if priority == "moderate" else 0)
    imm_co2  = round((imm * energy_kwh) / 1000, 3)
    opt_co2  = round((best_int * energy_kwh) / 1000, 3)
    saved    = round(max(0, imm_co2 - opt_co2), 3)
    pct      = round((saved / imm_co2 * 100) if imm_co2 > 0 else 0, 1)
    start_dt = datetime.now() + timedelta(hours=wait)

    return {
        "recommended_start":            start_dt.strftime("%Y-%m-%d %H:%M"),
        "hours_to_wait":                wait,
        "expected_intensity_gco2_kwh":  best_int,
        "immediate_intensity_gco2_kwh": imm,
        "immediate_co2_kg":             imm_co2,
        "optimal_co2_kg":               opt_co2,
        "co2_saved_kg":                 saved,
        "co2_savings_pct":              pct,
        "recommendation":               f"Wait {wait}h — save {saved} kg CO2.",
        "confidence":                   0.87,
        "zone":                         zone,
        "horizon":                      horizon,
    }


# ── Endpoints list ────────────────────────────────────────────────────────────
ENDPOINTS = [
    ("POST", "/predict",          "Get recommended schedule for a workload"),
    ("GET",  "/forecast/{zone}",  "Fetch 24h carbon intensity forecast"),
    ("GET",  "/regions",          "List available grid zones"),
    ("GET",  "/health",           "API health check"),
    ("POST", "/retrain",          "Trigger model retraining"),
    ("GET",  "/metrics/{model}",  "Get latest model performance"),
]


# ── Page ──────────────────────────────────────────────────────────────────────
def page_api():
    st.markdown("### ⚡ API / Inference")

    col_f, col_e = st.columns(2)

    # ── Try /predict ──────────────────────────────────────────────────────────
    with col_f:
        st.markdown("#### Try /predict")
        api_zone    = st.selectbox("Zone", ZONES, key="api_zone")
        api_energy  = st.number_input("Energy (kWh)", 10.0, 2000.0, 120.0, key="api_e")
        api_runtime = st.number_input("Runtime (hours)", 0.5, 48.0, 4.0, key="api_rt")
        api_hz      = st.selectbox("Horizon", HORIZONS, index=1, key="api_hz",
                                    format_func=lambda x: f"{x}h")
        api_pri     = st.selectbox("Priority",
                                    ["flexible", "moderate", "urgent"], key="api_pri")

        if st.button("POST /predict", use_container_width=True):
            with st.spinner("Calling inference..."):
                result = mock_predict(api_energy, api_runtime, api_zone, api_hz, api_pri)
                st.code(json.dumps(result, indent=2), language="json")

    # ── Endpoints ─────────────────────────────────────────────────────────────
    with col_e:
        st.markdown("#### Available endpoints")
        for method, path, desc in ENDPOINTS:
            bg = "#dcfce7" if method == "GET" else "#dbeafe"
            fg = "#166534" if method == "GET" else "#1e3a8a"
            st.markdown(f"""
            <div style="display:flex;align-items:center;gap:10px;padding:8px 0;
                        border-bottom:1px solid #f1f5f9;">
              <span style="background:{bg};color:{fg};padding:2px 8px;
                           border-radius:5px;font-size:11px;font-weight:600;
                           min-width:36px;text-align:center;">{method}</span>
              <code style="font-size:12px;">{path}</code>
              <span style="font-size:12px;color:#475569;">{desc}</span>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Health check ──────────────────────────────────────────────────────────
    st.markdown("#### Health check")
    if st.button("GET /health", use_container_width=False):
        st.success("✅ API is healthy — all models loaded, data pipeline connected.")