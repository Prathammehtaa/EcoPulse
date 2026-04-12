import streamlit as st
import pandas as pd
import plotly.graph_objects as go


# ── Mock data ─────────────────────────────────────────────────────────────────
METRICS = {
    "Horizon": ["1h", "6h", "12h", "24h"],
    "XGBoost MAE":  [25.14, 34.34, 39.97, 43.01],
    "LightGBM MAE": [26.8,  32.1,  38.4,  41.2 ],
    "Baseline MAE": [57.48, 71.33, 76.36, 68.79],
    "XGBoost R²":   [0.94,  0.89,  0.84,  0.81 ],
    "XGBoost RMSE": [11.2,  14.7,  19.4,  24.1 ],
    "XGBoost MAPE": [4.1,   5.8,   7.3,   9.1  ],
}


# ── Page ──────────────────────────────────────────────────────────────────────
def page_metrics():
    st.markdown("### 📈 Model Performance Overview")

    df = pd.DataFrame(METRICS)

    # ── KPI Row ───────────────────────────────────────────────────────────────
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Best MAE (1h)",    "25.14", delta="-32.34 vs baseline")
    k2.metric("Best R² (1h)",     "0.94")
    k3.metric("Best RMSE (1h)",   "11.2")
    k4.metric("vs baseline avg",  "43% better")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Bar chart ─────────────────────────────────────────────────────────────
    st.markdown("#### MAE — XGBoost vs LightGBM vs Baseline")
    fig = go.Figure()
    for model, color in [
        ("XGBoost MAE",  "#22874f"),
        ("LightGBM MAE", "#2563eb"),
        ("Baseline MAE", "#94a3b8"),
    ]:
        fig.add_trace(go.Bar(
            name=model, x=df["Horizon"], y=df[model],
            marker_color=color,
        ))
    fig.update_layout(
        barmode="group",
        height=320,
        margin=dict(l=40, r=20, t=20, b=40),
        plot_bgcolor="white",
        paper_bgcolor="white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── R² line chart ─────────────────────────────────────────────────────────
    st.markdown("#### R² across horizons — XGBoost")
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(
        x=df["Horizon"], y=df["XGBoost R²"],
        mode="lines+markers",
        line=dict(color="#22874f", width=2),
        marker=dict(size=8, color="#22874f"),
        name="R²",
    ))
    fig2.update_layout(
        height=250,
        margin=dict(l=40, r=20, t=20, b=40),
        plot_bgcolor="white",
        paper_bgcolor="white",
        yaxis=dict(range=[0.75, 1.0], title="R²"),
        xaxis=dict(title="Horizon"),
    )
    st.plotly_chart(fig2, use_container_width=True)

    # ── Full metrics table ────────────────────────────────────────────────────
    st.markdown("#### Full metrics table")
    st.dataframe(df, use_container_width=True, hide_index=True)