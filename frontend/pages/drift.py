import streamlit as st
import pandas as pd
import plotly.graph_objects as go


# ── Mock data ─────────────────────────────────────────────────────────────────
DRIFT_DATA = {
    "Feature":    ["wind_speed_100m_ms", "hour_of_day", "solar_potential",
                   "demand_lag_1h", "temperature_2m_c"],
    "PSI Score":  [0.28, 0.08, 0.11, 0.07, 0.09],
    "Threshold":  [0.2,  0.2,  0.2,  0.2,  0.2 ],
    "Status":     ["⚠️ Warning", "✅ OK", "✅ OK", "✅ OK", "✅ OK"],
    "Action":     ["Retrain recommended", "—", "—", "—", "—"],
}

CONCEPT_DATA = {
    "Model":          ["XGBoost 1h", "XGBoost 6h", "LightGBM 1h", "LightGBM 6h"],
    "Training MAE":   [24.8,  31.2,  25.1,  28.1],
    "Recent MAE":     [25.14, 34.34, 26.8,  32.1],
    "Status":         ["✅ Stable", "⚠️ Degraded", "✅ Stable", "⚠️ Degraded"],
}


# ── Page ──────────────────────────────────────────────────────────────────────
def page_drift():
    st.markdown("### 🌊 Drift Monitoring")

    df_drift   = pd.DataFrame(DRIFT_DATA)
    df_concept = pd.DataFrame(CONCEPT_DATA)

    # ── PSI chart ─────────────────────────────────────────────────────────────
    st.markdown("#### Feature drift — PSI scores")
    colors = ["#dc2626" if p > 0.2 else "#22874f" for p in df_drift["PSI Score"]]
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=df_drift["Feature"], y=df_drift["PSI Score"],
        marker_color=colors,
        hovertemplate="%{x}: PSI %{y:.2f}<extra></extra>",
        name="PSI Score",
    ))
    fig.add_hline(
        y=0.2, line_dash="dash", line_color="#d97706",
        annotation_text="Threshold (0.2)",
        annotation_position="top right",
    )
    fig.update_layout(
        height=300,
        margin=dict(l=40, r=20, t=20, b=80),
        plot_bgcolor="white",
        paper_bgcolor="white",
        yaxis=dict(title="PSI Score"),
        xaxis=dict(tickangle=-20),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.dataframe(df_drift, use_container_width=True, hide_index=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Concept drift ─────────────────────────────────────────────────────────
    st.markdown("#### Concept drift — training vs recent MAE")
    st.dataframe(df_concept, use_container_width=True, hide_index=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Retrain button ────────────────────────────────────────────────────────
    st.markdown("#### Actions")
    c1, c2 = st.columns(2)
    with c1:
        if st.button("🔁 Trigger retraining — LightGBM 6h", use_container_width=True):
            st.success("Retraining triggered! Check MLflow for run status.")
    with c2:
        if st.button("🔁 Trigger retraining — XGBoost 6h", use_container_width=True):
            st.success("Retraining triggered! Check MLflow for run status.")