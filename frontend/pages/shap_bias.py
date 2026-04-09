import streamlit as st
import pandas as pd
import plotly.graph_objects as go


# ── Mock data ─────────────────────────────────────────────────────────────────
SHAP_DATA = {
    "Feature": [
        "hour_of_day", "wind_speed_100m_ms",
        "carbon_intensity_gco2_per_kwh_lag_1h", "total_load_mw",
        "solar_potential", "temperature_2m_c",
        "carbon_intensity_gco2_per_kwh_lag_24h", "cloud_cover_pct",
        "hour_sin", "carbon_free_energy_pct",
    ],
    "Mean |SHAP|": [0.42, 0.33, 0.29, 0.25, 0.21, 0.17, 0.15, 0.13, 0.11, 0.09],
    "Direction":   ["+",  "+",  "+",  "+",  "-",  "+",  "+",  "-",  "+",  "-" ],
}

BIAS_DATA = {
    "Bucket":  ["Very Low (<100)", "Low (100-200)", "Medium (200-350)", "High (350-500)"],
    "MAE":     [18.4, 10.2, 11.9, 13.7],
    "R²":      [0.71, 0.88, 0.90, 0.85],
    "Samples": [312,  4821, 8103, 3201],
    "Status":  ["🔴 Flagged", "🟢 OK", "🟢 OK", "🟡 Watch"],
}


# ── Page ──────────────────────────────────────────────────────────────────────
def page_shap():
    st.markdown("### 🧠 SHAP Feature Importance & Bias Audit")

    df_shap = pd.DataFrame(SHAP_DATA)
    df_bias = pd.DataFrame(BIAS_DATA)

    col_s, col_b = st.columns(2)

    # ── SHAP bar chart ────────────────────────────────────────────────────────
    with col_s:
        st.markdown("#### XGBoost 1h — top features")
        colors = ["#22874f" if d == "+" else "#f97316"
                  for d in df_shap["Direction"]]
        fig = go.Figure(go.Bar(
            x=df_shap["Mean |SHAP|"],
            y=df_shap["Feature"],
            orientation="h",
            marker_color=colors,
            hovertemplate="%{y}: %{x:.2f}<extra></extra>",
        ))
        fig.update_layout(
            height=400,
            margin=dict(l=10, r=20, t=10, b=40),
            plot_bgcolor="white",
            paper_bgcolor="white",
            yaxis=dict(autorange="reversed"),
            xaxis=dict(title="Mean |SHAP value|"),
        )
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("""
        <div style="display:flex;gap:16px;font-size:12px;color:#475569;margin-top:4px;">
          <span><span style="color:#22874f;font-weight:600;">■</span> Positive impact</span>
          <span><span style="color:#f97316;font-weight:600;">■</span> Negative impact</span>
        </div>
        """, unsafe_allow_html=True)

    # ── Bias audit table ──────────────────────────────────────────────────────
    with col_b:
        st.markdown("#### Bias audit — 6h model subgroups")
        st.dataframe(df_bias, use_container_width=True, hide_index=True)
        st.warning("**Very Low (0-100)** bucket flagged — data imbalance detected. "
                   "R² drops to 0.71. Consider oversampling this subgroup.")

        # Bias MAE bar
        st.markdown("#### MAE by carbon bucket")
        fig2 = go.Figure(go.Bar(
            x=df_bias["Bucket"],
            y=df_bias["MAE"],
            marker_color=["#dc2626", "#22874f", "#22874f", "#d97706"],
            hovertemplate="%{x}: MAE %{y}<extra></extra>",
        ))
        fig2.update_layout(
            height=240,
            margin=dict(l=10, r=10, t=10, b=80),
            plot_bgcolor="white",
            paper_bgcolor="white",
            xaxis=dict(tickangle=-20),
            yaxis=dict(title="MAE"),
        )
        st.plotly_chart(fig2, use_container_width=True)

    # ── LIME note ─────────────────────────────────────────────────────────────
    st.markdown("---")
    st.info("💡 LIME HTML reports are saved in `Model_Pipeline/reports/lime/`. "
            "Open them in a browser for per-prediction explanations.")