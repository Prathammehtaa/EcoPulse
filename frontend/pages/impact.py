import streamlit as st
import pandas as pd
import plotly.express as px


# ── Helpers ───────────────────────────────────────────────────────────────────
def trees_saved(co2_kg):
    return round(co2_kg / 13.3, 2)


# ── Page ──────────────────────────────────────────────────────────────────────
def page_impact():
    st.markdown("### 📊 Impact & ESG Reporting")

    history     = st.session_state.workload_history
    total_saved = sum(w.get("co2_saved_kg", 0) for w in history)
    n_wl        = len(history)

    # ── KPI Row ───────────────────────────────────────────────────────────────
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total CO2 avoided",   f"{total_saved:.1f} kg")
    k2.metric("Trees equivalent",    f"{trees_saved(total_saved)}")
    k3.metric("Workloads optimized", n_wl)
    k4.metric("SLA violations",      "0")

    st.markdown("<br>", unsafe_allow_html=True)

    if history:
        # ── Bar chart ─────────────────────────────────────────────────────────
        st.markdown("#### CO2 savings per workload")
        df_h = pd.DataFrame(history)

        if "scheduled_at" in df_h.columns:
            df_h["scheduled_at"] = pd.to_datetime(df_h["scheduled_at"])
            df_h = df_h.sort_values("scheduled_at")

            fig = px.bar(
                df_h,
                x="name" if "name" in df_h.columns else df_h.index,
                y="co2_saved_kg",
                color_discrete_sequence=["#22874f"],
                labels={"co2_saved_kg": "CO2 saved (kg)", "name": "Workload"},
                title="CO2 saved per workload",
            )
            fig.update_layout(
                height=300,
                margin=dict(l=40, r=20, t=40, b=40),
                plot_bgcolor="white",
                paper_bgcolor="white",
            )
            st.plotly_chart(fig, use_container_width=True)

        # ── Audit log ─────────────────────────────────────────────────────────
        st.markdown("#### Audit log — all decisions")
        rows = [{
            "Time":        w.get("scheduled_at", "—"),
            "Workload":    w.get("name", "—"),
            "Zone":        w.get("zone", "—"),
            "Recommended": w.get("recommended_start", "—"),
            "CO2 saved":   f"{w.get('co2_saved_kg', 0)} kg",
            "Savings %":   f"{w.get('co2_savings_pct', 0):.1f}%",
            "Decision":    "Followed",
        } for w in reversed(history)]
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    else:
        st.info("No workloads scheduled yet. Go to **Workload Scheduler** to get started.")