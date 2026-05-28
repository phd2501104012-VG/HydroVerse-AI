import streamlit as st
from typing import List, Optional, Callable
from datetime import datetime, timedelta

from config import CFG, DataSource
from dashboard.config import *


def render_sidebar(
    districts: List[str],
    data_sources: List[str],
    on_source_change: Optional[Callable] = None,
    on_district_change: Optional[Callable] = None,
    on_mode_change: Optional[Callable] = None,
):
    with st.sidebar:
        st.markdown(f'''
        <div class="sidebar-logo">
            <span class="logo-icon">{APP_ICON}</span>
            <div class="logo-title">{APP_NAME}</div>
            <p class="logo-subtitle">{APP_SUBTITLE}</p>
        </div>
        ''', unsafe_allow_html=True)

        st.markdown("---")

        # Navigation
        nav_items = [
            ("📊", "Overview", "tab-0"),
            ("🛰️", "Real-Time Monitor", "tab-1"),
            ("📜", "Historical Analysis", "tab-2"),
            ("🗺️", "Hazard Maps", "tab-3"),
            ("🔮", "Forecasting", "tab-4"),
            ("✅", "Validation", "tab-5"),
            ("🔔", "Alerts", "tab-6"),
        ]

        active_tab = st.session_state.get("nav_tab", 0)
        for i, (icon, label, _) in enumerate(nav_items):
            cls = "sidebar-nav-item active" if i == active_tab else "sidebar-nav-item"
            if st.button(f"{icon} {label}", key=f"nav_{i}", help=f"Go to {label}",
                         use_container_width=True, type="secondary" if i != active_tab else "primary"):
                st.session_state.nav_tab = i
                st.rerun()

        st.markdown("---")

        # Data Source
        st.markdown("#### 🎯 Data Source")
        source = st.selectbox(
            "Select Source",
            options=data_sources,
            index=0,
            label_visibility="collapsed",
            key="data_source",
        )
        if source == DataSource.ERA5.value:
            CFG.active_data_source = DataSource.ERA5
            st.markdown('<span style="font-size:0.65rem;color:var(--primary-dark);">🔵 ERA5-Land (Reanalysis)</span>', unsafe_allow_html=True)
        elif source == DataSource.IMD.value:
            CFG.active_data_source = DataSource.IMD
            st.markdown('<span style="font-size:0.65rem;color:#C2410C;">🟠 IMD Gridded (Observed)</span>', unsafe_allow_html=True)
        else:
            CFG.active_data_source = DataSource.AUTO
            st.markdown('<span style="font-size:0.65rem;color:#7C3AED;">🟣 Auto-detect best source</span>', unsafe_allow_html=True)

        if on_source_change:
            on_source_change(source)

        # District
        st.markdown("#### 📍 District")
        if districts:
            district = st.selectbox(
                "Select District",
                options=districts,
                index=0,
                label_visibility="collapsed",
                key="district",
            )
        else:
            district = None
            st.warning("No districts loaded")

        st.markdown("---")

        # Time Period
        st.markdown("#### ⏱️ Time Period")
        today = datetime.now()
        hist_start = datetime.strptime(CFG.hist_start, "%Y-%m-%d") if hasattr(CFG, 'hist_start') else today - timedelta(days=365*25)
        start_date = st.date_input("Start", value=hist_start, min_value=hist_start, max_value=today, label_visibility="collapsed")
        end_date = st.date_input("End", value=today, min_value=hist_start, max_value=today, label_visibility="collapsed")

        # Forecast Horizon
        st.markdown("#### 🔮 Forecast Horizon")
        horizon = st.select_slider(
            "Horizon",
            options=[30, 60, 90, 180, 365, 730, 1825, 3650, 5475],
            value=90,
            format_func=lambda d: f"{d} days ({d//365}y {d%365}d)" if d >= 365 else f"{d} days",
            label_visibility="collapsed",
        )

        st.markdown("---")

        # System Status
        with st.expander("⚙️ System Status", expanded=False):
            c1, c2 = st.columns(2)
            mode = c2.selectbox("Mode", ["Operational", "Research", "Validation"], label_visibility="collapsed", key="system_mode")
            st.markdown(f"**Source:** {CFG.active_data_source.value}")
            st.markdown(f"**Districts:** {len(districts)}")
            st.markdown(f"**Scenario:** SSP2-4.5")
            st.markdown(f"**Period:** 2000–2040")
            st.markdown(f"**Forecast:** ML + CMIP6 Ensemble")

        st.markdown("---")

        st.markdown(f'''
        <div class="sidebar-footer">
            <span style="font-weight:600;color:var(--primary);font-size:0.65rem;">HydroVerse AI</span> v3.0<br>
            Water Climate & Sustainability Lab<br>
            <strong>IIT Indore</strong>
        </div>
        ''', unsafe_allow_html=True)

    return source, district, start_date, end_date, horizon, mode
