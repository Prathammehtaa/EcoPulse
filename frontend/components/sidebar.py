import streamlit as st


def render_sidebar():
    with st.sidebar:
        # ── LOGO / TITLE ──
        st.markdown("""
        <div style="display:flex;align-items:center;gap:10px;padding:8px 0 16px 0;">
            <span style="font-size:24px;">⚡</span>
            <span style="font-size:18px;font-weight:700;color:#9FE1CB;
                         letter-spacing:0.04em;">EcoPulse</span>
        </div>
        """, unsafe_allow_html=True)

        # ── DIVIDER ──
        st.markdown("""
        <hr style="border:none;border-top:1px solid rgba(255,255,255,0.1);margin:0 0 12px 0;"/>
        """, unsafe_allow_html=True)

        # ── NAVIGATION ──
        if st.session_state.get("role") == "admin":
            pages = {
                "📈  Model Metrics":   "metrics",
                "🧠  SHAP & Bias":     "shap",
                "🌊  Drift Monitor":   "drift",
                "⚡  API / Inference": "api",
                "👥  Users":           "users",
                "⚙️  Settings":        "settings",
                "📋  System Logs":     "logs",
            }
        else:
            pages = {
                "🏠  Dashboard":         "dashboard",
                "⚙️  Workload Scheduler": "scheduler",
                "📊  Impact & ESG":       "impact",
                "🔔  Alerts":             "alerts",
            }

        if "page" not in st.session_state:
            st.session_state["page"] = list(pages.values())[0]

        for label, key in pages.items():
            if st.button(
                label,
                key=f"nav_{key}",
                use_container_width=True,
            ):
                st.session_state["page"] = key
                st.rerun()

        st.markdown("<br>", unsafe_allow_html=True)

        # ── ALERT BADGE ──
        st.markdown("""
        <div style="background:#3d1010;border-radius:8px;
                    padding:8px 12px;margin-top:8px;
                    display:flex;align-items:center;gap:8px;">
            <span style="width:7px;height:7px;border-radius:50%;
                         background:#e24b4a;display:inline-block;
                         animation:pulse 2s infinite;"></span>
            <span style="font-size:12px;color:#f09595;font-weight:500;">
                2 active alerts
            </span>
        </div>
        <style>
        @keyframes pulse{0%,100%{opacity:1}50%{opacity:.3}}
        </style>
        """, unsafe_allow_html=True)

        # ── SPACER ──
        st.markdown("<div style='flex:1;'></div>", unsafe_allow_html=True)
        st.markdown("<br><br><br><br><br><br><br>", unsafe_allow_html=True)

        # ── DIVIDER ──
        st.markdown("""
        <hr style="border:none;border-top:1px solid rgba(255,255,255,0.1);margin:8px 0;"/>
        """, unsafe_allow_html=True)

        # ── USER INFO ──
        username = st.session_state.get("username", "User")
        initials = username[:2].upper()

        st.markdown(f"""
        <div style="display:flex;align-items:center;gap:9px;padding:6px 0;">
            <div style="width:30px;height:30px;border-radius:50%;
                        background:#22874f;color:white;
                        display:flex;align-items:center;justify-content:center;
                        font-size:11px;font-weight:600;flex-shrink:0;">
                {initials}
            </div>
            <div style="flex:1;min-width:0;">
                <div style="font-size:12px;color:#d6f5e8;font-weight:600;
                            white-space:nowrap;overflow:hidden;text-overflow:ellipsis;">
                    {username}
                </div>
                <div style="font-size:10px;color:#9FE1CB;">Operator</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # ── SIGN OUT BUTTON ──
        if st.button("→  Sign out", use_container_width=True, key="signout_btn"):
            for key in ["logged_in", "username", "email", "page",
                        "workload_history", "session_saved"]:
                st.session_state.pop(key, None)
            st.rerun()