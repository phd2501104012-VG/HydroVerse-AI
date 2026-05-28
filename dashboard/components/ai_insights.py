"""AI Climate Insight Engine — generates natural-language summaries from computed hazard data."""
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime


def generate_insights(data: pd.DataFrame, hazards: pd.DataFrame, district: str) -> list:
    """Generate natural-language climate insights from computed data."""
    insights = []

    if data.empty or hazards.empty:
        return ["Select a district with data to generate AI climate insights."]

    today = pd.Timestamp(datetime.now().date())
    latest_date = data.index[-1] if not data.empty else today
    recents = hazards.loc[hazards.index >= latest_date - pd.Timedelta(days=30)] if latest_date in hazards.index else hazards.tail(30)

    # 1. Heatwave insight
    if "heatwave_severity" in hazards.columns:
        hw = hazards["heatwave_severity"].dropna()
        if not hw.empty:
            max_hw = hw.max()
            recent_hw = recents["heatwave_severity"].dropna() if not recents.empty else hw.tail(15)
            recent_max = recent_hw.max() if not recent_hw.empty else 0
            hw_days = (recent_hw >= 50).sum() if not recent_hw.empty else 0
            hw_trend = "increasing" if len(recent_hw) > 5 and recent_hw.iloc[-3:].mean() > recent_hw.iloc[:3].mean() else "stable" if len(recent_hw) > 5 else "observed"

            if recent_max >= 75:
                insights.append(f"🔥 **Severe heatwave conditions** detected in {district}. Maximum severity {max_hw:.0f}/100 with {hw_days} high-risk days in the past month. Heatwave intensity is {hw_trend}.")
            elif recent_max >= 50:
                insights.append(f"🌡️ **Heatwave conditions** present in {district}. Peak severity {recent_max:.0f}/100 with {hw_days} elevated days. Trend is {hw_trend}.")
            elif "tmax" in data.columns and not data["tmax"].dropna().empty:
                tmax_latest = data["tmax"].dropna().iloc[-1]
                if tmax_latest > 35:
                    insights.append(f"🌡️ Temperatures in {district} reaching {tmax_latest:.1f}°C. Below heatwave threshold but monitoring recommended.")

    # 2. Flood insight
    if "flood_severity" in hazards.columns:
        fl = hazards["flood_severity"].dropna()
        if not fl.empty:
            max_fl = fl.max()
            recent_fl = recents["flood_severity"].dropna() if not recents.empty else fl.tail(30)
            recent_max_fl = recent_fl.max() if not recent_fl.empty else 0

            if recent_max_fl >= 50:
                heavy_days = 0
                if "precip" in data.columns:
                    recent_precip = data.loc[data.index.isin(recent_fl.index), "precip"].dropna() if not recent_fl.empty else pd.Series(dtype=float)
                    heavy_days = (recent_precip >= 64.5).sum()
                insights.append(f"🌊 **Flood risk elevated** in {district}. Severity {recent_max_fl:.0f}/100 with {heavy_days} heavy rainfall days observed.")
            elif max_fl > 0:
                insights.append(f"💧 Flood conditions in {district} are currently normal. Seasonal monitoring active.")

    # 3. Drought insight
    if "drought_severity" in hazards.columns:
        dr = hazards["drought_severity"].dropna()
        if not dr.empty:
            max_dr = dr.max()
            recent_dr = recents["drought_severity"].dropna() if not recents.empty else dr.tail(30)
            recent_max_dr = recent_dr.max() if not recent_dr.empty else 0

            spi_val = None
            if "spi_3m" in hazards.columns:
                spi_vals = hazards["spi_3m"].dropna()
                if not spi_vals.empty:
                    spi_val = spi_vals.iloc[-1]

            cdd_val = None
            if "cdd" in hazards.columns:
                cdd_vals = hazards["cdd"].dropna()
                if not cdd_vals.empty:
                    cdd_val = cdd_vals.iloc[-1]

            if recent_max_dr >= 50:
                spi_txt = f"SPI-3 at {spi_val:.2f}" if spi_val is not None else "elevated dryness indicators"
                cdd_txt = f"{int(cdd_val)} consecutive dry days" if cdd_val is not None and cdd_val > 5 else ""
                insights.append(f"🏜️ **Drought conditions** in {district} (severity {recent_max_dr:.0f}/100). {spi_txt}. {cdd_txt}")
            elif spi_val is not None and spi_val < -1:
                insights.append(f"📊 SPI-3 in {district} at {spi_val:.2f}, indicating moderately dry conditions. Continue monitoring.")
            elif spi_val is not None:
                insights.append(f"📊 SPI-3 in {district} at {spi_val:.2f}. Conditions within normal range.")

    # 4. Agricultural stress insight
    if "agri_severity" in hazards.columns:
        ag = hazards["agri_severity"].dropna()
        if not ag.empty:
            recent_ag = recents["agri_severity"].dropna() if not recents.empty else ag.tail(30)
            recent_max_ag = recent_ag.max() if not recent_ag.empty else 0
            max_ag = ag.max()

            vhi_val = None
            if "vhi" in hazards.columns:
                vhi_vals = hazards["vhi"].dropna()
                if not vhi_vals.empty:
                    vhi_val = vhi_vals.iloc[-1]

            if recent_max_ag >= 50:
                vhi_txt = f"VHI at {vhi_val:.0f}" if vhi_val is not None else "stress indicators elevated"
                insights.append(f"🌾 **Agricultural stress** in {district} (severity {recent_max_ag:.0f}/100). {vhi_txt}. Monitor crop health.")
            elif vhi_val is not None and vhi_val < 40:
                insights.append(f"🌾 Vegetation health (VHI) in {district} at {vhi_val:.0f}, below normal. Agricultural watch recommended.")

    # 5. Temperature anomaly insight
    if "tmax_anom" in hazards.columns:
        tanom = hazards["tmax_anom"].dropna()
        if not tanom.empty:
            latest_tanom = tanom.iloc[-1]
            recent_tanom = tanom.tail(15)
            if latest_tanom > 3:
                insights.append(f"📈 **Significant temperature anomaly** in {district}: +{latest_tanom:.1f}°C above normal. Consistent with heatwave development.")
            elif latest_tanom > 1.5:
                insights.append(f"📈 Temperature anomaly in {district} at +{latest_tanom:.1f}°C. Above normal but within seasonal range.")
            elif latest_tanom < -2:
                insights.append(f"📉 Below-normal temperatures in {district} ({latest_tanom:+.1f}°C anomaly). Cooler conditions prevailing.")

    # 6. Precipitation insight
    if "precip_anom" in hazards.columns:
        panom = hazards["precip_anom"].dropna()
        if not panom.empty:
            latest_panom = panom.iloc[-1]
            if latest_panom < -20:
                insights.append(f"☀️ Rainfall deficit of {abs(latest_panom):.0f}mm in {district}. Below-normal precipitation may affect water resources.")
            elif latest_panom > 20:
                insights.append(f"🌧️ Rainfall surplus of {latest_panom:.0f}mm in {district}. Above-normal precipitation observed.")

    # 7. Compound risk insight
    if "compound" in hazards.columns:
        comp = hazards["compound"].dropna()
        if not comp.empty:
            latest_comp = comp.iloc[-1]
            if isinstance(latest_comp, (int, float)) and latest_comp > 50:
                insights.append(f"⚠️ **Compound hazard risk** elevated in {district} (score {latest_comp:.0f}/100). Multiple hazard factors active simultaneously.")

    # 8. Summary insight
    if "flood_severity" in hazards.columns and "heatwave_severity" in hazards.columns and "drought_severity" in hazards.columns:
        active_hazards = []
        for hname, hcol in [("flood", "flood_severity"), ("heatwave", "heatwave_severity"), ("drought", "drought_severity")]:
            if hcol in hazards.columns:
                last_val = hazards[hcol].dropna()
                if not last_val.empty and last_val.iloc[-1] >= 25:
                    active_hazards.append(hname)
        if active_hazards:
            insights.append(f"📋 **Active hazards** in {district}: {', '.join(active_hazards)}. {'Multi-hazard scenario requires integrated response.' if len(active_hazards) > 1 else 'Focused monitoring recommended.'}")

    if not insights:
        insights.append(f"✅ {district} currently shows no significant hazard activity. All climate indicators within normal ranges.")

    return insights[:6]


def render_ai_insights(data: pd.DataFrame, hazards: pd.DataFrame, district: str):
    """Render the AI insight engine card."""
    insights = generate_insights(data, hazards, district)

    st.markdown(f'''
    <div class="ai-insight-card fade-in">
        <div class="ai-header">
            <div class="ai-icon">🧠</div>
            <span class="ai-title">AI Climate Intelligence</span>
            <span class="ai-badge">Auto-generated</span>
        </div>
        <div class="ai-content">
            {"".join(f"<div style='padding:4px 0;'>{insight}</div>" for insight in insights)}
        </div>
    </div>
    ''', unsafe_allow_html=True)
