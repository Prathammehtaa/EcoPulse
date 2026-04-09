import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


# ── Mock helpers ──────────────────────────────────────────────────────────────
ZONES    = ["US-MIDA-PJM", "US-NW-PACW"]
HORIZONS = [1, 6, 12, 24]

def mock_current_intensity(zone):
    return 287 if zone == "US-MIDA-PJM" else 134

def mock_forecast(zone, n_hours=24):
    np.random.seed(42 if zone == "US-MIDA-PJM" else 7)
    base = mock_current_intensity(zone)
    return [
        {"hour_offset": i + 1,
         "intensity": round(max(50, base + np.random.randint(-40, 40) + 20 * np.sin(i / 4)), 1)}
        for i in range(n_hours)
    ]

def mock_schedule(energy_kwh, runtime_hours, zone, horizon, priority):
    fc       = mock_forecast(zone, n_hours=horizon)
    imm      = mock_current_intensity(zone)
    best     = min(fc, key=lambda x: x["intensity"]) if priority != "urgent" else {"hour_offset": 0, "intensity": imm}
    wait     = best["hour_offset"]
    best_int = best["intensity"]
    imm_co2  = round((imm * energy_kwh) / 1000, 3)
    opt_co2  = round((best_int * energy_kwh) / 1000, 3)
    saved    = round(max(0, imm_co2 - opt_co2), 3)
    pct      = round((saved / imm_co2 * 100) if imm_co2 > 0 else 0, 1)
    start_dt = datetime.now() + timedelta(hours=wait)

    return {
        "recommended_start":            start_dt.strftime("%Y-%m-%d %H:%M"),
        "hours_to_wait":                wait,
        "expected_intensity_gco2_kwh":  round(best_int, 2),
        "immediate_intensity_gco2_kwh": imm,
        "immediate_co2_kg":             imm_co2,
        "optimal_co2_kg":               opt_co2,
        "co2_saved_kg":                 saved,
        "co2_savings_pct":              pct,
        "recommendation": (
            f"Run now — already optimal window. Intensity: {imm} gCO2/kWh."
            if wait == 0 else
            f"Wait {wait}h — start at {start_dt.strftime('%H:%M')}. "
            f"Save {saved} kg CO2 ({pct}% reduction)."
        ),
        "confidence": 0.87,
        "zone":       zone,
        "horizon":    horizon,
    }

def trees_saved(co2_kg):
    return round(co2_kg / 13.3, 2)


# ── Page ──────────────────────────────────────────────────────────────────────
def page_scheduler():
    st.markdown("### ⚙️ Workload Scheduler")

    # ── What-if simulator ─────────────────────────────────────────────────────
    with st.expander("🔮 What-if simulator", expanded=True):
        c1, c2 = st.columns(2)
        with c1:
            delay_h = st.slider("Delay by (hours)", 0, 24, 0)
        with c2:
            energy  = st.slider("Energy (kWh)", 10, 500, 120, 10)

        zone_wi = st.selectbox("Zone", ZONES, key="wi_zone")
        imm     = mock_current_intensity(zone_wi)
        fc      = mock_forecast(zone_wi)
        opt_int = fc[delay_h - 1]["intensity"] if delay_h > 0 else imm
        co2_now = round((imm * energy) / 1000, 2)
        co2_opt = round((opt_int * energy) / 1000, 2)
        saved   = round(max(0, co2_now - co2_opt), 2)

        w1, w2, w3 = st.columns(3)
        w1.metric("Expected intensity", f"{opt_int:.0f} gCO2/kWh")
        w2.metric("CO2 this run",       f"{co2_opt} kg")
        w3.metric("CO2 saved vs now",   f"{saved} kg",
                  delta=f"-{saved} kg" if saved > 0 else "0 kg")

    st.markdown("---")
    st.markdown("#### Schedule a new workload")

    # ── Schedule form ─────────────────────────────────────────────────────────
    fc1, fc2 = st.columns(2)
    with fc1:
        wl_name    = st.text_input("Workload name", placeholder="e.g. ML training job")
        wl_region  = st.selectbox("Region", ZONES)
        wl_runtime = st.number_input("Runtime (hours)", 0.5, 48.0, 4.0, 0.5)
        wl_energy  = st.number_input("Energy (kWh)", 1.0, 2000.0, 120.0, 10.0)
    with fc2:
        wl_hz      = st.selectbox("Forecast horizon", HORIZONS,
                                   index=1, format_func=lambda x: f"{x}h")
        wl_pri     = st.selectbox("Priority", ["flexible", "moderate", "urgent"])
        wl_type    = st.selectbox("Workload type",
                                   ["ML training", "Batch processing", "Database backup",
                                    "Video rendering", "Data pipeline", "Other"])
        wl_deadline= st.number_input("Deadline (hours from now)", 1, 72, 24, 1)

    if st.button("Find green window 🌿", use_container_width=True):
        if not wl_name:
            st.error("Please enter a workload name.")
        else:
            with st.spinner("Finding optimal green window..."):
                result = mock_schedule(wl_energy, wl_runtime, wl_region, wl_hz, wl_pri)
                result["name"]         = wl_name
                result["type"]         = wl_type
                result["scheduled_at"] = datetime.now().strftime("%Y-%m-%d %H:%M")
                st.session_state.workload_history.append(result)

                st.success("Green window found!")
                st.markdown(f"""
                <div class="rec-card">
                  <div style="font-size:12px;color:#94a3b8;">
                    {wl_name} · {wl_region} · {wl_hz}h horizon
                  </div>
                  <div class="rec-win">{result['recommended_start']}</div>
                  <div class="why-box">{result['recommendation']}</div>
                </div>""", unsafe_allow_html=True)

                r1, r2, r3, r4, r5, r6 = st.columns(6)
                r1.metric("Start at",    result["recommended_start"][-5:])
                r2.metric("Wait",        f"{result['hours_to_wait']}h")
                r3.metric("Intensity",   f"{result['expected_intensity_gco2_kwh']:.0f} gCO2")
                r4.metric("CO2 if now",  f"{result['immediate_co2_kg']} kg")
                r5.metric("CO2 optimal", f"{result['optimal_co2_kg']} kg")
                r6.metric("CO2 saved",   f"{result['co2_saved_kg']} kg",
                          delta=f"{result['co2_savings_pct']}% less")

                trees = trees_saved(result["co2_saved_kg"])
                if trees > 0:
                    st.info(f"🌳 Equivalent to **{trees} trees** absorbing CO₂ for a year.")

    # ── Workload queue ────────────────────────────────────────────────────────
    if st.session_state.workload_history:
        st.markdown("#### Scheduled workloads")
        rows = [{
            "Name":          w.get("name", "—"),
            "Type":          w.get("type", "—"),
            "Zone":          w.get("zone", "—"),
            "Start at":      w.get("recommended_start", "—")[-5:],
            "Wait (h)":      w.get("hours_to_wait", 0),
            "CO2 saved (kg)":w.get("co2_saved_kg", 0),
            "Savings %":     f"{w.get('co2_savings_pct', 0):.1f}%",
        } for w in reversed(st.session_state.workload_history)]
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)