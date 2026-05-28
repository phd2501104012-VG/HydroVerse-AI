"""Event Detection Cards — upcoming extreme weather events."""
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


def render_event_cards(data: pd.DataFrame, hazards: pd.DataFrame, district: str):
    """Render dynamic event detection cards for upcoming extreme events."""
    st.markdown('<p class="section-title">⚠️ Event Detection</p>', unsafe_allow_html=True)

    events = []

    if hazards.empty:
        st.info("No hazard data available for event detection.")
        return

    today = pd.Timestamp(datetime.now().date())

    # Heatwave events
    if "heatwave_severity" in hazards.columns and "heatwave_event" in hazards.columns:
        hw_events = hazards[(hazards["heatwave_event"] == 1) & (hazards.index >= today)]
        if not hw_events.empty:
            peak_idx = hw_events["heatwave_severity"].idxmax()
            peak_val = float(hw_events["heatwave_severity"].max())
            duration = len(hw_events)
            severity_cls = "severe" if peak_val >= 75 else "warning" if peak_val >= 50 else "watch"
            events.append({
                "type": "🔥 Heatwave",
                "severity": peak_val,
                "severity_cls": severity_cls,
                "title": f"Heatwave Event — {duration}d duration",
                "detail": f"Peak severity {peak_val:.0f}/100 on {peak_idx.strftime('%b %d')}. {duration} consecutive days above threshold.",
                "confidence": min(90, 50 + duration * 10),
                "impact": "High" if peak_val >= 75 else "Medium" if peak_val >= 50 else "Low",
            })

    # Extreme rainfall events
    if "flood_severity" in hazards.columns and "precip" in data.columns:
        recent_precip = data[data.index >= today - timedelta(days=7)]["precip"].dropna() if not data.empty else pd.Series(dtype=float)
        heavy_days = (recent_precip >= 64.5).sum()
        very_heavy_days = (recent_precip >= 115.6).sum()
        if very_heavy_days > 0 or heavy_days > 1:
            events.append({
                "type": "🌊 Extreme Rainfall",
                "severity": min(100, heavy_days * 20 + very_heavy_days * 35),
                "severity_cls": "severe" if very_heavy_days > 0 else "warning",
                "title": f"Extreme Rainfall Warning — {heavy_days + very_heavy_days} events",
                "detail": f"{heavy_days} heavy (≥64.5mm) + {very_heavy_days} very heavy (≥115.6mm) rain days in past week.",
                "confidence": 85,
                "impact": "High" if very_heavy_days > 0 else "Medium",
            })

    # Flood events
    if "flood_severity" in hazards.columns:
        fl = hazards["flood_severity"].dropna()
        if not fl.empty:
            latest_fl = fl.iloc[-1]
            if latest_fl >= 50:
                events.append({
                    "type": "🌊 Flood Risk",
                    "severity": latest_fl,
                    "severity_cls": "severe" if latest_fl >= 75 else "warning",
                    "title": f"Active Flood Risk — {latest_fl:.0f}/100",
                    "detail": f"Current flood severity at {latest_fl:.0f}/100. Monitor low-lying areas and waterways.",
                    "confidence": 80,
                    "impact": "High" if latest_fl >= 75 else "Medium",
                })

    # Drought escalation
    if "drought_severity" in hazards.columns and "spi_3m" in hazards.columns:
        dr = hazards["drought_severity"].dropna()
        spi = hazards["spi_3m"].dropna()
        if not dr.empty and not spi.empty:
            latest_dr = dr.iloc[-1]
            latest_spi = spi.iloc[-1]
            if latest_dr >= 40 or latest_spi < -1.3:
                events.append({
                    "type": "🏜️ Drought Escalation",
                    "severity": latest_dr,
                    "severity_cls": "severe" if latest_dr >= 75 else "warning" if latest_dr >= 50 else "watch",
                    "title": f"Drought Watch — SPI {latest_spi:.2f}",
                    "detail": f"SPI-3 at {latest_spi:.2f} with drought severity {latest_dr:.0f}/100. Monitor water resources.",
                    "confidence": 75,
                    "impact": "High" if latest_dr >= 75 else "Medium",
                })

    # Agricultural stress
    if "agri_severity" in hazards.columns and "vhi" in hazards.columns:
        ag = hazards["agri_severity"].dropna()
        vhi = hazards["vhi"].dropna()
        if not ag.empty and not vhi.empty:
            latest_ag = ag.iloc[-1]
            latest_vhi = vhi.iloc[-1]
            if latest_ag >= 50 or (pd.notna(latest_vhi) and latest_vhi < 35):
                events.append({
                    "type": "🌾 Agriculture Stress",
                    "severity": latest_ag,
                    "severity_cls": "severe" if latest_ag >= 75 else "warning" if latest_ag >= 50 else "watch",
                    "title": f"Crop Stress Alert — VHI {latest_vhi:.0f}/100" if pd.notna(latest_vhi) else "Crop Stress Alert",
                    "detail": f"Agricultural stress at {latest_ag:.0f}/100. Vegetation health declining. Consider irrigation planning.",
                    "confidence": 70,
                    "impact": "Medium",
                })

    # Temperature anomaly
    if "tmax_anom" in hazards.columns:
        ta = hazards["tmax_anom"].dropna()
        if not ta.empty:
            latest_ta = ta.iloc[-1]
            if abs(latest_ta) > 4:
                events.append({
                    "type": "🌡️ Extreme Temperature Anomaly",
                    "severity": min(100, abs(latest_ta) * 12),
                    "severity_cls": "severe" if abs(latest_ta) > 6 else "warning",
                    "title": f"{'+' if latest_ta > 0 else ''}{latest_ta:.1f}°C Anomaly",
                    "detail": f"Temperature {'above' if latest_ta > 0 else 'below'} normal by {abs(latest_ta):.1f}°C. {'Heatwave risk elevated.' if latest_ta > 0 else 'Unusual cooling detected.'}",
                    "confidence": 90,
                    "impact": "High" if abs(latest_ta) > 6 else "Medium",
                })

    # Display events
    if events:
        events = sorted(events, key=lambda e: e["severity"], reverse=True)
        cols = st.columns(min(len(events), 3))
        for i, event in enumerate(events[:6]):
            with cols[i % len(cols)]:
                sev_cls = event["severity_cls"]
                sev_color = {"severe": "#EF4444", "warning": "#F97316", "watch": "#EAB308"}.get(sev_cls, "#22C55E")

                st.markdown(f'''
                <div class="event-card fade-in" style="border-left: 3px solid {sev_color};">
                    <div class="event-top">
                        <span class="event-type" style="color:{sev_color};">{event["type"]}</span>
                        <span class="event-severity-tag" style="background: {sev_color}22; color: {sev_color}; border: 1px solid {sev_color}44;">
                            {sev_cls.upper()}
                        </span>
                    </div>
                    <div style="font-size:0.82rem;font-weight:600;margin:4px 0;">{event["title"]}</div>
                    <div class="event-detail">{event["detail"]}</div>
                    <div class="event-footer">
                        <span>🎯 Confidence: {event["confidence"]}%</span>
                        <span>📊 Impact: {event["impact"]}</span>
                    </div>
                </div>
                ''', unsafe_allow_html=True)
    else:
        st.info("No extreme events detected for the current period.")
