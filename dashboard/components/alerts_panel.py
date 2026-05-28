import streamlit as st
from typing import Optional, Dict, List, Any
import pandas as pd
from datetime import datetime

from dashboard.config import *
from hazards.categories import HazardClassifier


def render_alert_panel(alerts_df: pd.DataFrame):
    st.markdown(
        f"""
        <div class="dashboard-header fade-in">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <div>
                    <h1 style="margin: 0;">🚨 Real-Time Alerts</h1>
                    <p class="subtitle" style="margin: 4px 0 0 0;">
                        <span class="realtime-dot"></span>
                        Live monitoring active
                    </p>
                </div>
                <div style="text-align: right;">
                    <span style="font-size: 2rem; font-weight: 800; color: var(--primary);">
                        {len(alerts_df)}
                    </span>
                    <br>
                    <span style="color: var(--text-secondary); font-size: 0.8rem;">Active Alerts</span>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if alerts_df.empty:
        st.markdown(
            """
            <div style="text-align: center; padding: 48px; color: var(--text-secondary);">
                <span style="font-size: 3rem;">✅</span>
                <h3>No Active Alerts</h3>
                <p>All districts are at Normal risk levels.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    col_order = ["severity", "district", "hazard", "risk_class", "recommended_action", "confidence"]
    display_cols = [c for c in col_order if c in alerts_df.columns]

    for _, alert in alerts_df.sort_values("severity", ascending=False).iterrows():
        severity = alert.get("severity", 0)
        risk_class = alert.get("risk_class", "Normal")
        hazard = alert.get("hazard", "unknown")
        district = alert.get("district", "unknown")

        class_css = "alert-critical" if severity >= 90 else "alert-severe" if severity >= 75 else "alert-warning" if severity >= 50 else "alert-watch"
        severity_color = SEVERITY_COLORS.get(risk_class, "#22c55e")

        st.markdown(
            f"""
            <div class="alert-panel {class_css} fade-in">
                <div style="display: flex; justify-content: space-between; align-items: start;">
                    <div>
                        <span style="font-size: 1.2rem;">{HAZARD_NAMES.get(hazard, hazard)}</span>
                        <br>
                        <strong style="font-size: 1.1rem;">{district}</strong>
                    </div>
                    <div style="text-align: right;">
                        <span class="severity-badge severity-{risk_class.lower()}">
                            {risk_class}
                        </span>
                        <br>
                        <span style="font-size: 1.5rem; font-weight: 700; color: {severity_color};">
                            {severity:.0f}
                        </span>
                        <span style="color: var(--text-muted); font-size: 0.8rem;">/100</span>
                    </div>
                </div>
                <div style="margin-top: 8px; padding-top: 8px; border-top: 1px solid var(--border-color);">
                    <span style="color: var(--text-secondary); font-size: 0.85rem;">
                        {alert.get("recommended_action", "Monitor situation.")}
                    </span>
                </div>
                <div style="margin-top: 4px; font-size: 0.75rem; color: var(--text-muted);">
                    Confidence: {alert.get("confidence", 0.5)*100:.0f}%
                    {" | " + alert.get("timestamp", "")[:19] if "timestamp" in alert else ""}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown(
        f"""
        <div style="text-align: center; padding: 16px; color: var(--text-muted); font-size: 0.8rem;">
            Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} UTC
            | {len(alerts_df)} active alert{'s' if len(alerts_df) != 1 else ''}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_alert_history(history_df: pd.DataFrame):
    if history_df.empty:
        st.info("No alert history available.")
        return

    st.dataframe(
        history_df.sort_values("timestamp", ascending=False).head(100),
        width='stretch',
    )
