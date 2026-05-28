import streamlit as st
from typing import Optional, Dict, List, Any
import pandas as pd
from datetime import datetime

from dashboard.config import *


def render_realtime_status(status: Dict[str, Any]):
    st.markdown(
        f"""
        <div style="display: inline-flex; align-items: center; 
                    background: var(--bg-card); padding: 8px 16px; 
                    border-radius: 20px; border: 1px solid var(--border-color);
                    margin-bottom: 16px;">
            <span class="realtime-dot"></span>
            <span style="font-size: 0.85rem; color: var(--text-secondary);">
                Real-time monitoring · Update #{status.get("update_count", 0)}
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    cols = st.columns(4)
    with cols[0]:
        ok = status.get("precip", False)
        st.markdown(f"""
            <div style="background: var(--bg-card); border-radius: 12px; padding: 12px; 
                        border: 1px solid {'rgba(34,197,94,0.2)' if ok else 'rgba(239,68,68,0.2)'};">
                <span style="font-size: 1.5rem;">🌧️</span>
                <br>
                <span style="color: var(--text-secondary); font-size: 0.75rem;">PRECIP</span>
                <br>
                <span style="color: {'#22c55e' if ok else '#ef4444'}; font-size: 0.8rem;">
                    {"● Online" if ok else "○ Offline"}
                </span>
            </div>
        """, unsafe_allow_html=True)

    with cols[1]:
        ok = status.get("ndvi", False)
        st.markdown(f"""
            <div style="background: var(--bg-card); border-radius: 12px; padding: 12px; 
                        border: 1px solid {'rgba(34,197,94,0.2)' if ok else 'rgba(239,68,68,0.2)'};">
                <span style="font-size: 1.5rem;">🌿</span>
                <br>
                <span style="color: var(--text-secondary); font-size: 0.75rem;">NDVI</span>
                <br>
                <span style="color: {'#22c55e' if ok else '#ef4444'}; font-size: 0.8rem;">
                    {"● Online" if ok else "○ Offline"}
                </span>
            </div>
        """, unsafe_allow_html=True)

    with cols[2]:
        ok = status.get("lst", False)
        st.markdown(f"""
            <div style="background: var(--bg-card); border-radius: 12px; padding: 12px; 
                        border: 1px solid {'rgba(34,197,94,0.2)' if ok else 'rgba(239,68,68,0.2)'};">
                <span style="font-size: 1.5rem;">🌡️</span>
                <br>
                <span style="color: var(--text-secondary); font-size: 0.75rem;">LST</span>
                <br>
                <span style="color: {'#22c55e' if ok else '#ef4444'}; font-size: 0.8rem;">
                    {"● Online" if ok else "○ Offline"}
                </span>
            </div>
        """, unsafe_allow_html=True)

    with cols[3]:
        ok = status.get("soil_moisture", False)
        st.markdown(f"""
            <div style="background: var(--bg-card); border-radius: 12px; padding: 12px; 
                        border: 1px solid {'rgba(34,197,94,0.2)' if ok else 'rgba(239,68,68,0.2)'};">
                <span style="font-size: 1.5rem;">💧</span>
                <br>
                <span style="color: var(--text-secondary); font-size: 0.75rem;">SOIL</span>
                <br>
                <span style="color: {'#22c55e' if ok else '#ef4444'}; font-size: 0.8rem;">
                    {"● Online" if ok else "○ Offline"}
                </span>
            </div>
        """, unsafe_allow_html=True)


def render_anomaly_table(anomaly_df: pd.DataFrame):
    if anomaly_df.empty:
        st.info("No anomalies detected.")
        return

    st.markdown("### 🔍 Active Anomalies")

    def color_anomaly(val):
        if val == "Extreme":
            return "background-color: rgba(127,29,29,0.3); color: #fca5a5"
        elif val == "Severe":
            return "background-color: rgba(239,68,68,0.2); color: #fca5a5"
        elif val == "Moderate":
            return "background-color: rgba(249,115,22,0.2); color: #fdba74"
        return ""

    styled = anomaly_df.style.applymap(color_anomaly, subset=["anomaly_class"])
    st.dataframe(styled, width='stretch')


def render_alert_summary(alert_summary: Dict):
    if alert_summary.get("total", 0) == 0:
        st.success("✅ No active alerts - All clear")
        return

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Alerts", alert_summary["total"])
    c2.metric("Critical (75+)", alert_summary.get("critical", 0))
    c3.metric("Hazards Active", len(alert_summary.get("by_hazard", {})))
    c4.metric("Districts Affected", len(alert_summary.get("by_district", {})))

    st.markdown("#### By Hazard Type")
    for hazard, count in alert_summary.get("by_hazard", {}).items():
        st.markdown(f"- {HAZARD_NAMES.get(hazard, hazard)}: {count}")

    st.markdown("#### By Risk Level")
    for cls, count in alert_summary.get("by_severity", {}).items():
        color = SEVERITY_COLORS.get(cls, "#888")
        st.markdown(f"- <span style='color: {color};'>●</span> {cls}: {count}", unsafe_allow_html=True)
