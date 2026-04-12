import streamlit as st


# ── Mock data ─────────────────────────────────────────────────────────────────
LOGS = [
    ("10:49", "INFO",  "MLflow run — XGBoost 1h — MAE 25.14 RMSE 11.2 R² 0.94 — artifacts saved"),
    ("10:45", "INFO",  "Workload approved — 'ML retraining' — US-MIDA-PJM — CO2 saved 24.8 kg"),
    ("10:30", "WARN",  "Drift alert — wind_speed_100m_ms PSI 0.28 > threshold 0.2"),
    ("09:58", "INFO",  "GCS fetch — ecopulse-shared-data — 1440 rows — parquet loaded"),
    ("09:45", "INFO",  "Model retrain — LightGBM 6h — R² 0.89 — pushed to registry"),
    ("09:30", "WARN",  "Override logged — 'Database backup' — recommended 13:00 → ran 09:30"),
    ("09:00", "WARN",  "Bias audit — flagged Very Low (0-100) subgroup in 6h model — R² 0.71"),
    ("08:45", "INFO",  "Pipeline run complete — 65/65 tests passing"),
    ("08:30", "INFO",  "User login — hitarth@example.com — Operator"),
    ("08:00", "INFO",  "System startup — all services healthy"),
]

LEVEL_COLORS = {
    "INFO": ("#dcfce7", "#166534"),
    "WARN": ("#fef3c7", "#92400e"),
    "ERROR": ("#fee2e2", "#7f1d1d"),
}


# ── Page ──────────────────────────────────────────────────────────────────────
def page_logs():
    st.markdown("### 📋 System Logs")

    # ── Filters ───────────────────────────────────────────────────────────────
    c1, c2 = st.columns([2, 1])
    with c1:
        search = st.text_input("Search logs", placeholder="e.g. MLflow, drift, retrain...")
    with c2:
        level_filter = st.selectbox("Level", ["All", "INFO", "WARN", "ERROR"])

    # ── Filter logs ───────────────────────────────────────────────────────────
    filtered = LOGS
    if search:
        filtered = [(t, l, m) for t, l, m in filtered if search.lower() in m.lower()]
    if level_filter != "All":
        filtered = [(t, l, m) for t, l, m in filtered if l == level_filter]

    st.markdown(f"<div style='font-size:12px;color:#94a3b8;margin-bottom:8px;'>"
                f"Showing {len(filtered)} of {len(LOGS)} entries</div>",
                unsafe_allow_html=True)

    # ── Log entries ───────────────────────────────────────────────────────────
    for time, level, msg in filtered:
        bg, fg = LEVEL_COLORS.get(level, ("#f1f5f9", "#475569"))
        st.markdown(f"""
        <div style="display:flex;align-items:center;gap:12px;padding:8px 0;
                    border-bottom:1px solid #f1f5f9;">
          <span style="color:#94a3b8;width:44px;flex-shrink:0;
                       font-size:12px;font-weight:500;">{time}</span>
          <span style="background:{bg};color:{fg};padding:1px 7px;
                       border-radius:4px;font-size:10px;font-weight:600;
                       min-width:36px;text-align:center;">{level}</span>
          <span style="font-size:12px;color:#475569;">{msg}</span>
        </div>""", unsafe_allow_html=True)

    if not filtered:
        st.info("No log entries match your filter.")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Clear / export ────────────────────────────────────────────────────────
    c3, c4 = st.columns(2)
    with c3:
        if st.button("🗑 Clear logs", use_container_width=True):
            st.warning("Log clearing is disabled in demo mode.")
    with c4:
        if st.button("⬇️ Export logs", use_container_width=True):
            log_text = "\n".join([f"{t} [{l}] {m}" for t, l, m in LOGS])
            st.download_button("Download logs.txt", log_text,
                               file_name="ecopulse_logs.txt",
                               mime="text/plain",
                               use_container_width=True)