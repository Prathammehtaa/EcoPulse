import streamlit as st
import pandas as pd


# ── Mock data ─────────────────────────────────────────────────────────────────
if "users" not in st.session_state:
    st.session_state.users = [
        {"Initials": "HU", "Email": "hitarth@example.com",  "Status": "Active",   "Role": "Operator"},
        {"Initials": "KG", "Email": "kapish@example.com",   "Status": "Active",   "Role": "Operator"},
        {"Initials": "PM", "Email": "pratham@example.com",  "Status": "Inactive", "Role": "Operator"},
        {"Initials": "AA", "Email": "aaditya@example.com",  "Status": "Active",   "Role": "Operator"},
    ]


# ── Page ──────────────────────────────────────────────────────────────────────
def page_users():
    st.markdown("### 👥 Registered Users")

    # ── Users table ───────────────────────────────────────────────────────────
    df = pd.DataFrame(st.session_state.users)
    st.dataframe(df, use_container_width=True, hide_index=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Add user ──────────────────────────────────────────────────────────────
    st.markdown("#### Add new user")
    c1, c2, c3 = st.columns([3, 2, 1])
    with c1:
        new_email = st.text_input("Email", placeholder="new@example.com",
                                   label_visibility="collapsed")
    with c2:
        new_role  = st.selectbox("Role", ["Operator", "Admin"],
                                  label_visibility="collapsed")
    with c3:
        if st.button("Add", use_container_width=True):
            if new_email:
                initials = new_email[:2].upper()
                st.session_state.users.append({
                    "Initials": initials,
                    "Email":    new_email,
                    "Status":   "Active",
                    "Role":     new_role,
                })
                st.success(f"Added {new_email}!")
                st.rerun()
            else:
                st.error("Please enter an email.")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Remove user ───────────────────────────────────────────────────────────
    st.markdown("#### Remove user")
    emails = [u["Email"] for u in st.session_state.users]
    del_email = st.selectbox("Select user to remove", emails,
                              label_visibility="collapsed")
    if st.button("Remove user", use_container_width=False):
        st.session_state.users = [
            u for u in st.session_state.users if u["Email"] != del_email
        ]
        st.success(f"Removed {del_email}!")
        st.rerun()