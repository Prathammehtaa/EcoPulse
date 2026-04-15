"""
EcoPulse — Main Entry Point
============================
Run:  cd ~/EcoPulse && streamlit run streamlit_app.py
"""

import os
import sys
import streamlit as st

# ── Path setup ────────────────────────────────────────────────────────────────
REPO_ROOT    = os.path.dirname(os.path.abspath(__file__))
FRONTEND_DIR = os.path.join(REPO_ROOT, "frontend")
sys.path.insert(0, FRONTEND_DIR)

# ── Page config (must be FIRST streamlit call) ────────────────────────────────
st.set_page_config(
    page_title="EcoPulse",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600&display=swap');
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
.main  { background: linear-gradient(to right, #0f3d28 44%, #f3fcf6 44%); }
.stApp { background: linear-gradient(to right, #0f3d28 44%, #f3fcf6 44%); }
.block-container { padding: 1rem 1rem 0 1rem !important; }
header { display: none !important; }
[data-testid="stSidebarHeader"] { display: none !important; }
[data-testid="stSidebarNav"] { display: none !important; }
iframe { border: none !important; display: block; }

.kpi-card {
    background: white; border-radius: 12px;
    padding: 1rem 1.25rem; border: 1px solid #e2e8f0;
    border-top: 3px solid #2da866; margin-bottom: 0.5rem;
}
.kpi-card.amber { border-top-color: #d97706; }
.kpi-card.red   { border-top-color: #dc2626; }
.kpi-card.blue  { border-top-color: #2563eb; }
.kpi-label { font-size: 11px; color: #94a3b8; text-transform: uppercase;
             letter-spacing: .05em; font-weight: 500; margin-bottom: 4px; }
.kpi-value { font-size: 26px; font-weight: 600; color: #0f172a; line-height: 1.1; }
.kpi-sub   { font-size: 12px; color: #475569; margin-top: 4px; }

.rec-card {
    background: white; border-radius: 12px; padding: 1.25rem;
    border: 1px solid #e2e8f0; border-top: 3px solid #22874f;
    margin-bottom: 1rem;
}
.rec-card.warn { border-top-color: #d97706; }
.rec-win { font-size: 20px; font-weight: 600; color: #145233; margin: 6px 0; }
.why-box { background: #f3fcf6; border: 1px solid #bfedcf; border-radius: 8px;
           padding: 9px 12px; font-size: 13px; color: #145233; margin: 8px 0; }

section[data-testid="stSidebar"] { background: #0f3d28 !important; }
section[data-testid="stSidebar"] .stMarkdown,
section[data-testid="stSidebar"] label { color: #bfedcf !important; }

.stButton > button {
    background: #1a6b43; color: white; border: none;
    border-radius: 8px; font-weight: 600;
}
.stButton > button:hover { background: #145233; }
</style>
""", unsafe_allow_html=True)

# ── Session state defaults ────────────────────────────────────────────────────
defaults = {
    "logged_in":        False,
    "role":             "user",
    "username":         "",
    "workload_history": [],
    "page":             "dashboard",
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ── Imports from frontend ─────────────────────────────────────────────────────
from components.sidebar import render_sidebar
from pages.login        import show_login
from pages.dashboard    import page_dashboard
from pages.scheduler    import page_scheduler
from pages.impact       import page_impact
from pages.alerts       import page_alerts
from pages.metrics      import page_metrics
from pages.shap_bias    import page_shap
from pages.drift        import page_drift
from pages.api          import page_api
from pages.users        import page_users
from pages.settings     import page_settings
from pages.logs         import page_logs

# ── Router ────────────────────────────────────────────────────────────────────
def main():
    if not st.session_state.logged_in:
        show_login()
        return

    render_sidebar()
    page = st.session_state.page

    if st.session_state.role == "user":
        if   page == "dashboard": page_dashboard()
        elif page == "scheduler": page_scheduler()
        elif page == "impact":    page_impact()
        elif page == "alerts":    page_alerts()
    else:
        if   page == "metrics":   page_metrics()
        elif page == "shap":      page_shap()
        elif page == "drift":     page_drift()
        elif page == "api":       page_api()
        elif page == "users":     page_users()
        elif page == "settings":  page_settings()
        elif page == "logs":      page_logs()


if __name__ == "__main__":
    main()