import streamlit as st


def show_login():
    col_left, col_right = st.columns([1, 1])

    # ── Left panel — branding ──────────────────────────────────────────────
    with col_left:
        st.markdown("""
        <div style="background:#0f3d28;border-radius:16px;padding:3rem 2rem;
                    text-align:center;min-height:520px;display:flex;
                    flex-direction:column;align-items:center;
                    justify-content:center;gap:1.5rem;">

          <div style="width:100px;height:100px;border-radius:50%;
                      background:#062e1f;border:2px solid rgba(255,255,255,0.15);
                      display:flex;align-items:center;justify-content:center;margin:0 auto;">
            <span style="font-size:52px;">🌿</span>
          </div>

          <div>
            <div style="font-size:30px;font-weight:600;color:white;
                        letter-spacing:-.01em;">EcoPulse</div>
            <div style="font-size:13px;color:#86dba9;margin-top:6px;
                        max-width:220px;line-height:1.6;">
              Carbon-aware workload scheduling for greener data centers
            </div>
          </div>



        </div>
        """, unsafe_allow_html=True)

    # ── Right panel — login form ───────────────────────────────────────────
    with col_right:
        st.markdown("<br><br>", unsafe_allow_html=True)
        st.markdown("### Sign in")
        st.markdown("Access your carbon decision support dashboard")

        role = st.radio(
            "Role", ["Operator / Engineer", "Admin"],
            horizontal=True, label_visibility="collapsed"
        )
        email    = st.text_input("Email address", placeholder="you@datacenter.com")
        password = st.text_input("Password", type="password", placeholder="••••••••")

        if st.button("Sign in to EcoPulse", use_container_width=True):
            if email and password == "ecopulse":
                st.session_state.logged_in = True
                st.session_state.role      = "admin" if role == "Admin" else "user"
                st.session_state.username  = email.split("@")[0].capitalize()
                st.session_state.page      = "dashboard" if st.session_state.role == "user" else "metrics"
                st.rerun()
            else:
                st.error("Incorrect credentials. Password is **ecopulse**.")

        st.caption("Demo — any email · password **ecopulse**")