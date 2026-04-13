import streamlit as st


# ── Page ──────────────────────────────────────────────────────────────────────
def page_settings():
    st.markdown("### ⚙️ Model & System Settings")

    # ── Forecast settings ─────────────────────────────────────────────────────
    st.markdown("#### Forecast settings")
    c1, c2 = st.columns(2)
    with c1:
        st.slider("Carbon threshold (gCO2/kWh)", 50, 400, 200, 10,
                  help="Workloads will be deferred if intensity is above this.")
        st.slider("Default forecast horizon (h)", 1, 24, 6, 1,
                  help="Default horizon used for scheduling recommendations.")
    with c2:
        st.slider("Min CO2 savings to defer (%)", 1, 50, 10, 1,
                  help="Only defer workloads if savings exceed this threshold.")
        st.slider("Fail-open threshold (data age h)", 1, 12, 2, 1,
                  help="Use last known forecast if data is older than this.")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Model settings ────────────────────────────────────────────────────────
    st.markdown("#### Model settings")
    c3, c4 = st.columns(2)
    with c3:
        st.selectbox("Default model", ["XGBoost", "LightGBM"],
                     help="Model used for scheduling recommendations.")
        st.selectbox("Default zone", ["US-MIDA-PJM", "US-NW-PACW"],
                     help="Default grid zone for new workloads.")
    with c4:
        st.number_input("Retraining interval (days)", 1, 30, 7, 1,
                        help="How often to retrain models automatically.")
        st.number_input("Drift PSI threshold", 0.05, 1.0, 0.2, 0.05,
                        help="Trigger retraining if PSI exceeds this value.")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Notification settings ─────────────────────────────────────────────────
    st.markdown("#### Notification settings")
    c5, c6 = st.columns(2)
    with c5:
        st.selectbox("Notify via", ["Email", "Slack", "Both"])
        st.text_input("Notification email", placeholder="ops@datacenter.com")
    with c6:
        st.text_input("Slack webhook URL", placeholder="https://hooks.slack.com/...")
        st.toggle("Enable daily ESG report email")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Save ──────────────────────────────────────────────────────────────────
    if st.button("Save settings", use_container_width=True):
        st.success("✅ Settings saved successfully!")