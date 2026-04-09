import streamlit as st
import pandas as pd


# ── Mock data ─────────────────────────────────────────────────────────────────
MOCK_ALERTS = [
    {
        "type":    "error",
        "icon":    "⚠",
        "title":   "Drift detected — 6h LightGBM model",
        "detail":  "wind_speed PSI 0.28 > threshold 0.2. Retraining recommended.",
        "time":    "34 min ago",
        "active":  True,
    },
    {
        "type":    "success",
        "icon":    "✅",
        "title":   "Carbon drop — US-MIDA-PJM resolved",
        "detail":  "Grid dropped below threshold. Workloads were scheduled.",
        "time":    "2 hrs ago",
        "active":  False,
    },
    {
        "type":    "warning",
        "icon":    "⚠",
        "title":   "High carbon — US-NW-PACW elevated",
        "detail":  "Intensity at 312 gCO2/kWh. Consider deferring non-urgent workloads.",
        "time":    "3 hrs ago",
        "active":  False,
    },
]

MOCK_ZONES = [
    {"zone": "US-MIDA-PJM", "intensity": 287, "bucket": "Medium (200-350)",
     "carbon_free_pct": 18.4, "renewable_pct": 12.1, "status": "🟡 Elevated"},
    {"zone": "US-NW-PACW",  "intensity": 134, "bucket": "Low (100-200)",
     "carbon_free_pct": 61.2, "renewable_pct": 58.3, "status": "🟢 OK"},
]


# ── Page ──────────────────────────────────────────────────────────────────────
def page_alerts():
    st.markdown("### 🔔 Alerts & Reliability")

    # ── Active alerts ─────────────────────────────────────────────────────────
    st.markdown("#### Active alerts")
    for alert in MOCK_ALERTS:
        opacity  = "1" if alert["active"] else "0.5"
        bg_color = "#fee2e2" if alert["type"] == "error" else "#dcfce7"
        fg_color = "#7f1d1d" if alert["type"] == "error" else "#166534"

        st.markdown(f"""
        <div style="background:white;border:1px solid #e2e8f0;border-radius:10px;
                    padding:12px 16px;margin-bottom:10px;
                    display:flex;align-items:flex-start;gap:12px;opacity:{opacity};">
          <div style="width:28px;height:28px;border-radius:50%;
                      background:{bg_color};color:{fg_color};
                      display:flex;align-items:center;justify-content:center;
                      flex-shrink:0;font-size:14px;">{alert['icon']}</div>
          <div style="flex:1;">
            <div style="font-size:13px;font-weight:600;color:#0f172a;">{alert['title']}</div>
            <div style="font-size:12px;color:#475569;margin-top:2px;">{alert['detail']}</div>
          </div>
          <span style="font-size:11px;color:#94a3b8;white-space:nowrap;">{alert['time']}</span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Fail-open mode ────────────────────────────────────────────────────────
    st.markdown("#### Fail-open mode")
    fail_open = st.toggle("Enable fail-open mode (use last known forecast if data is stale)")
    if fail_open:
        st.warning("Fail-open mode is ON — using last known forecast. Data may be stale.")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Zone status ───────────────────────────────────────────────────────────
    st.markdown("#### Real-time zone status")
    rows = [{
        "Zone":        z["zone"],
        "Intensity":   f"{z['intensity']} gCO2/kWh",
        "Bucket":      z["bucket"],
        "Carbon-free": f"{z['carbon_free_pct']}%",
        "Renewable":   f"{z['renewable_pct']}%",
        "Status":      z["status"],
    } for z in MOCK_ZONES]
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Alert settings ────────────────────────────────────────────────────────
    st.markdown("#### Alert settings")
    c1, c2 = st.columns(2)
    with c1:
        st.number_input("Carbon threshold (gCO2/kWh)", 50, 500, 300, 10)
        st.number_input("Drift PSI threshold", 0.05, 1.0, 0.2, 0.05)
    with c2:
        st.selectbox("Notify via", ["Email", "Slack", "Both"])
        st.text_input("Notification email", placeholder="ops@datacenter.com")

    if st.button("Save alert settings", use_container_width=True):
        st.success("Alert settings saved!")
        