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
        # ── Brand Header ──
        try:
            with open(r'D:\cri\IITI_Logo.svg', 'r', encoding='utf-8') as _f:
                _svg = _f.read()
            import base64
            _b64 = base64.b64encode(_svg.encode('utf-8')).decode()
            _logo_html = f'<img src="data:image/svg+xml;base64,{_b64}" style="height:48px;width:auto;">'
        except Exception:
            _logo_html = ''
        st.markdown(f'''
        <div style="text-align:center;padding:6px 0 12px;">
            <div style="display:flex;align-items:center;justify-content:center;gap:12px;">
                {_logo_html}
                <div style="text-align:left;">
                    <div style="font-size:1.6rem;font-weight:800;color:var(--primary);letter-spacing:-0.03em;line-height:1.1;">HydroVerse AI</div>
                    <div style="font-size:0.72rem;color:var(--text-muted);font-weight:500;margin-top:2px;">Water, Climate &amp; Sustainability Lab<br>Indian Institute of Technology Indore</div>
                </div>
            </div>
        </div>
        ''', unsafe_allow_html=True)

        st.markdown("<hr>", unsafe_allow_html=True)

        # ── Navigation ──
        st.markdown("#### 🧭 View")
        nav = st.radio(
            "Navigation",
            ["📊 Dashboard", "📈 Historical Analysis", "🔮 Forecast", "🤖 AI Assistant"],
            index=0,
            label_visibility="collapsed",
            key="nav_view",
            horizontal=False,
        )

        st.markdown("<hr>", unsafe_allow_html=True)

        # ── District ──
        st.markdown("#### 📍 District")
        if districts:
            default_idx = districts.index("Bhopal") if "Bhopal" in districts else 0
            district = st.selectbox(
                "Select District",
                options=districts,
                index=default_idx,
                label_visibility="collapsed",
                key="district",
            )
        else:
            district = None
            st.warning("No districts loaded")

        st.markdown("<hr>", unsafe_allow_html=True)

        # ── Data Source ──
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
        elif source == DataSource.IMD.value:
            CFG.active_data_source = DataSource.IMD
        else:
            CFG.active_data_source = DataSource.AUTO

        if on_source_change:
            on_source_change(source)

        st.markdown("<hr>", unsafe_allow_html=True)

        # ── Time Period ──
        st.markdown("#### ⏱️ Time Period")
        today = datetime.now()
        try:
            hist_start = datetime.strptime(CFG.hist_start, "%Y-%m-%d") if hasattr(CFG, 'hist_start') else today - timedelta(days=365*25)
        except Exception:
            hist_start = today - timedelta(days=365*25)
        start_date = st.date_input("Start", value=hist_start, min_value=hist_start, max_value=today, label_visibility="collapsed")
        end_date = st.date_input("End", value=today, min_value=hist_start, max_value=today, label_visibility="collapsed")

        # ── Forecast Horizon ──
        st.markdown("#### 🔮 Forecast Horizon")
        horizon = st.select_slider(
            "Horizon",
            options=[30, 60, 90, 180, 365, 730, 1825, 3650, 5475],
            value=90,
            format_func=lambda d: f"{d} days ({d//365}y {d%365}d)" if d >= 365 else f"{d} days",
            label_visibility="collapsed",
        )

        st.markdown("<hr>", unsafe_allow_html=True)

        # ── Quick Actions ──
        st.markdown("#### ⚡ Quick Actions")
        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("📊 Forecast", key="sidebar_forecast", use_container_width=True):
                st.session_state.trigger_forecast = True
                st.session_state["nav_view"] = "🔮 Forecast"
                st.rerun()
        with col_b:
            if st.button("🔔 Alerts", key="sidebar_alerts", use_container_width=True):
                st.session_state.trigger_alerts = True

        with st.expander("⚙️ System Status", expanded=False):
            mode = st.selectbox("Mode", ["Operational", "Research", "Validation"], label_visibility="collapsed", key="system_mode")
            st.markdown(f"**Source:** {CFG.active_data_source.value}")
            st.markdown(f"**Districts:** {len(districts)}")
            st.markdown(f"**Scenario:** SSP2-4.5")
            st.markdown(f"**Period:** 2000–2040")
            st.markdown(f"**Forecast:** ML + CMIP6 Ensemble")

        st.markdown("<hr>", unsafe_allow_html=True)

    return source, district, start_date, end_date, horizon, mode
