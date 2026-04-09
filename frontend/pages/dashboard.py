import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from datetime import datetime


# ── Mock data ─────────────────────────────────────────────────────────────────
def mock_regions():
    return [
        {"zone": "US-MIDA-PJM",  "intensity_gco2_kwh": 287, "bucket": "Medium (200-350)",
         "carbon_free_pct": 18.4, "renewable_pct": 12.1},
        {"zone": "US-NW-PACW",   "intensity_gco2_kwh": 134, "bucket": "Low (100-200)",
         "carbon_free_pct": 61.2, "renewable_pct": 58.3},
    ]

def mock_forecast(zone, n_hours=24):
    np.random.seed(42 if zone == "US-MIDA-PJM" else 7)
    base = 287 if zone == "US-MIDA-PJM" else 134
    preds = []
    for i in range(n_hours):
        val = base + np.random.randint(-40, 40) + 20 * np.sin(i / 4)
        preds.append({"hour_offset": i + 1, "intensity": round(max(50, val), 1)})
    return preds

def intensity_color(v):
    if v < 200:  return "#22874f"
    if v < 350:  return "#d97706"
    return "#dc2626"

def trees_saved(co2_kg):
    return round(co2_kg / 13.3, 2)


# ── Page ──────────────────────────────────────────────────────────────────────
def page_dashboard():
    st.markdown(f"### Good afternoon, {st.session_state.get('username', 'User')} 👋")
    st.caption("Grid carbon is being monitored across all zones.")

    regions = mock_regions()
    pjm  = regions[0]
    pacw = regions[1]
    pjm_val  = pjm["intensity_gco2_kwh"]
    pacw_val = pacw["intensity_gco2_kwh"]

    # ── KPI Row ───────────────────────────────────────────────────────────────
    k1, k2, k3, k4 = st.columns(4)

    with k1:
        st.markdown(f"""<div class="kpi-card">
          <div class="kpi-label">US-MIDA-PJM now</div>
          <div class="kpi-value" style="color:{intensity_color(pjm_val)};">{pjm_val}</div>
          <div class="kpi-sub">gCO2/kWh · {pjm['bucket']}</div>
        </div>""", unsafe_allow_html=True)

    with k2:
        st.markdown(f"""<div class="kpi-card amber">
          <div class="kpi-label">US-NW-PACW now</div>
          <div class="kpi-value" style="color:{intensity_color(pacw_val)};">{pacw_val}</div>
          <div class="kpi-sub">gCO2/kWh · {pacw['bucket']}</div>
        </div>""", unsafe_allow_html=True)

    with k3:
        saved = sum(w.get("co2_saved_kg", 0) for w in st.session_state.workload_history)
        st.markdown(f"""<div class="kpi-card">
          <div class="kpi-label">CO2 saved today</div>
          <div class="kpi-value" style="color:#1a6b43;">{saved:.1f} <span style="font-size:14px;font-weight:400;">kg</span></div>
          <div class="kpi-sub">= {trees_saved(saved)} trees / year</div>
        </div>""", unsafe_allow_html=True)

    with k4:
        n_wl = len(st.session_state.workload_history)
        st.markdown(f"""<div class="kpi-card blue">
          <div class="kpi-label">Workloads optimized</div>
          <div class="kpi-value">{n_wl}</div>
          <div class="kpi-sub">0 SLA violations</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Forecast chart ────────────────────────────────────────────────────────
    st.markdown("#### 24-hour carbon intensity forecast")
    zone_sel = st.selectbox("Zone", ["US-MIDA-PJM", "US-NW-PACW"], key="dash_zone")

    fc     = mock_forecast(zone_sel)
    hours  = [f["hour_offset"] for f in fc]
    preds  = [f["intensity"]   for f in fc]
    colors = [intensity_color(v) for v in preds]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=hours, y=preds,
        mode="lines+markers",
        line=dict(color="#22874f", width=2),
        marker=dict(color=colors, size=8, line=dict(color="white", width=1)),
        name="Forecast",
        hovertemplate="Hour +%{x}: %{y:.0f} gCO2/kWh<extra></extra>",
    ))
    for h, v in zip(hours, preds):
        if v < 200:
            fig.add_vrect(x0=h - 0.5, x1=h + 0.5,
                          fillcolor="rgba(34,135,79,0.1)", layer="below", line_width=0)

    fig.update_layout(
        paper_bgcolor="rgba(15,61,40,1)",
        plot_bgcolor="rgba(15,61,40,1)",
        font=dict(color="#bfedcf", family="DM Sans"),
        margin=dict(l=40, r=20, t=20, b=40),
        height=260,
        xaxis=dict(title="Hours from now", gridcolor="rgba(255,255,255,0.05)"),
        yaxis=dict(title="gCO2/kWh",       gridcolor="rgba(255,255,255,0.05)"),
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── Recommendation panel ──────────────────────────────────────────────────
    st.markdown("#### Recommendation")
    best   = min(fc, key=lambda x: x["intensity"])
    wait   = best["hour_offset"]
    now_v  = pjm_val if zone_sel == "US-MIDA-PJM" else pacw_val
    saved_co2 = round(max(0, (now_v - best["intensity"]) * 200 / 1000), 2)
    start_t = datetime.now().strftime("%Y-%m-%d") + f" +{wait}h"

    st.markdown(f"""
    <div class="rec-card">
      <div style="font-size:12px;color:#94a3b8;">XGBoost forecast · {zone_sel}</div>
      <div class="rec-win">{start_t}</div>
      <div class="why-box">
        Wait {wait}h — intensity drops to {best['intensity']:.0f} gCO2/kWh.
        Save ~{saved_co2} kg CO2 vs running now ({now_v} gCO2/kWh).
      </div>
      <div style="font-size:13px;color:#475569;">
        CO2 saved: <b style="color:#1a6b43;">{saved_co2} kg</b> &nbsp;·&nbsp;
        Delay: <b>{wait}h</b> &nbsp;·&nbsp;
        SLA risk: <span style="color:#1a6b43;font-weight:600;">None</span>
      </div>
    </div>""", unsafe_allow_html=True)