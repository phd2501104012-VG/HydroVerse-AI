"""Government Policy Advisory & Farmer Advisory components."""
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


def generate_policy_advisories(data: pd.DataFrame, hazards: pd.DataFrame, district: str) -> list:
    """Generate government policy advisories from computed hazard data."""
    advisories = []
    if data.empty or hazards.empty:
        return ["Load district data to generate policy advisories."]

    today = pd.Timestamp(datetime.now().date())
    sev = {}
    for h in ["flood", "drought", "heatwave", "agri_stress"]:
        col = f"{h}_severity"
        if col in hazards.columns:
            vals = hazards[col].dropna()
            sev[h] = vals.iloc[-1] if not vals.empty else 0
        else:
            sev[h] = 0

    # Reservoir management
    if sev["flood"] >= 50:
        advisories.append({
            "icon": "💧", "title": "Reservoir Management",
            "content": f"Increased flood risk ({sev['flood']:.0f}/100) in {district}. Recommend pre-emptive reservoir release and monitoring of dam levels. Activate flood control protocols for downstream communities.",
            "tag": "High Priority", "tag_color": "#EF4444",
        })
    elif sev["drought"] >= 50:
        advisories.append({
            "icon": "💧", "title": "Reservoir Management",
            "content": f"Drought conditions intensifying ({sev['drought']:.0f}/100) in {district}. Implement water rationing, restrict non-essential use, and prioritize drinking water supply.",
            "tag": "Advisory", "tag_color": "#F97316",
        })

    # Irrigation planning
    if sev["drought"] >= 40:
        advisories.append({
            "icon": "🌾", "title": "Irrigation Planning",
            "content": f"Drought stress elevated in {district}. Recommend micro-irrigation adoption, scheduling irrigation during early morning/late evening, and mulching to reduce evaporation.",
            "tag": "Agriculture", "tag_color": "#14B8A6",
        })

    # Heatwave preparedness
    if sev["heatwave"] >= 50:
        advisories.append({
            "icon": "🔥", "title": "Heatwave Preparedness",
            "content:": f"Heatwave severity at {sev['heatwave']:.0f}/100 in {district}. Activate cooling centers, issue public health warnings, adjust school/work timings, and ensure medical readiness for heat-related illnesses.",
            "tag": "Urgent", "tag_color": "#EF4444",
        })
    elif "tmax" in data.columns:
        tmax_latest = data["tmax"].dropna().iloc[-1] if not data["tmax"].dropna().empty else 0
        if tmax_latest > 38:
            advisories.append({
                "icon": "🔥", "title": "Heatwave Preparedness",
                "content": f"Temperatures reaching {tmax_latest:.1f}°C in {district}. Issue heat advisory for vulnerable populations. Ensure water availability in public spaces.",
                "tag": "Watch", "tag_color": "#F97316",
            })

    # Disaster response
    if sev["flood"] >= 75 or sev["heatwave"] >= 75:
        advisories.append({
            "icon": "🚨", "title": "Disaster Response",
            "content": f"Severe hazard conditions detected ({max(sev['flood'], sev['heatwave']):.0f}/100) in {district}. Activate emergency operations center, deploy rapid response teams, and coordinate with district disaster management authority.",
            "tag": "Critical", "tag_color": "#7F1D1D",
        })

    # Water resource management
    if "spi_3m" in hazards.columns:
        spi = hazards["spi_3m"].dropna()
        if not spi.empty and spi.iloc[-1] < -1:
            advisories.append({
                "icon": "💧", "title": "Water Resource Management",
                "content": f"SPI-3 at {spi.iloc[-1]:.2f} indicates water deficit in {district}. Implement demand-side management, promote rainwater harvesting, and prepare for groundwater recharge programs.",
                "tag": "Long-term", "tag_color": "#0EA5E9",
            })

    # Climate adaptation
    if sev["agri_stress"] >= 50 or sev["drought"] >= 50:
        advisories.append({
            "icon": "🌱", "title": "Climate Adaptation",
            "content": f"Agricultural and drought stress detected in {district}. Recommend climate-resilient crop varieties, diversification to stress-tolerant cultivars, and strengthening of crop insurance coverage.",
            "tag": "Strategic", "tag_color": "#14B8A6",
        })

    if not advisories:
        advisories.append({
            "icon": "✅", "title": "Normal Conditions",
            "content": f"No significant hazard triggers in {district}. Continue routine monitoring and maintain preparedness protocols. Regular maintenance of water infrastructure recommended.",
            "tag": "Routine", "tag_color": "#22C55E",
        })

    return advisories


def generate_farmer_advisories(data: pd.DataFrame, hazards: pd.DataFrame, district: str) -> list:
    """Generate farmer-specific advisories from hazard data."""
    advisories = []
    if data.empty or hazards.empty:
        return ["Load district data to generate farmer advisories."]

    sev = {}
    for h in ["flood", "drought", "heatwave", "agri_stress"]:
        col = f"{h}_severity"
        if col in hazards.columns:
            vals = hazards[col].dropna()
            sev[h] = vals.iloc[-1] if not vals.empty else 0
        else:
            sev[h] = 0

    # Crop advisory
    if sev["agri_stress"] >= 50:
        advisories.append({
            "icon": "🌾", "title": "Crop Advisory",
            "content": f"Agricultural stress at {sev['agri_stress']:.0f}/100 in {district}. Monitor crops for pest/disease. Delay nitrogen fertilizer application if rainfall expected. Consider foliar spray of micronutrients.",
        })
    elif "vhi" in hazards.columns:
        vhi = hazards["vhi"].dropna()
        if not vhi.empty and vhi.iloc[-1] < 40:
            advisories.append({
                "icon": "🌾", "title": "Crop Advisory",
                "content": f"Vegetation health (VHI {vhi.iloc[-1]:.0f}/100) below normal in {district}. Scout fields for pest infestation, ensure adequate irrigation, and consider bio-stimulant application.",
            })

    # Irrigation recommendation
    if sev["drought"] >= 40:
        advisories.append({
            "icon": "💧", "title": "Irrigation Recommendation",
            "content": f"Drought conditions in {district}. Irrate during early morning (4-8 AM) to minimize evaporation. Use drip irrigation for horticulture. Apply mulch to conserve soil moisture.",
        })
    elif "precip" in data.columns and "tmax" in data.columns:
        recent_precip = data["precip"].dropna().tail(7).sum()
        tmax_avg = data["tmax"].dropna().tail(7).mean()
        if recent_precip < 10 and tmax_avg > 35:
            advisories.append({
                "icon": "💧", "title": "Irrigation Recommendation",
                "content": f"Dry and hot conditions (7d rain: {recent_precip:.0f}mm, avg tmax: {tmax_avg:.1f}°C). Irrigate immediately for standing crops. Increase irrigation frequency for vegetables.",
            })

    # Sowing guidance
    if "precip" in data.columns:
        recent_rain = data["precip"].dropna().tail(15).sum()
        if recent_rain > 60:
            advisories.append({
                "icon": "🌱", "title": "Sowing Guidance",
                "content": f"Adequate rainfall received ({recent_rain:.0f}mm in 15 days) in {district}. Proceed with Kharif sowing for soybean, paddy, and maize. Ensure good seedbed preparation.",
            })
        elif recent_rain < 20:
            advisories.append({
                "icon": "🌱", "title": "Sowing Guidance",
                "content": f"Insufficient rainfall ({recent_rain:.0f}mm in 15 days) in {district}. Delay sowing until adequate soil moisture. Consider dry sowing for drought-tolerant varieties.",
            })

    # Heat stress alert
    if sev["heatwave"] >= 40:
        advisories.append({
            "icon": "🔥", "title": "Heat Stress Alert",
            "content": f"Heatwave conditions in {district}. Provide shade for livestock, increase drinking water frequency. Avoid field work during peak heat hours (11 AM - 4 PM). Apply kaolin spray to fruit trees.",
        })

    # Fertilizer timing
    if "precip" in data.columns:
        forecast_rain = data["precip"].dropna().tail(3).sum() if len(data) >= 3 else 0
        if forecast_rain > 30:
            advisories.append({
                "icon": "🧪", "title": "Fertilizer Timing",
                "content": f"Heavy rainfall expected in {district}. Delay fertilizer application to prevent nutrient runoff. Apply after heavy rain subsides using split application method for better uptake.",
            })
        else:
            advisories.append({
                "icon": "🧪", "title": "Fertilizer Timing",
                "content": f"Conditions favorable for fertilizer application in {district}. Apply recommended dose of NPK based on soil test. Use neem-coated urea for slow nitrogen release.",
            })

    if not advisories:
        advisories.append({
            "icon": "✅", "title": "Normal Farming Conditions",
            "content": f"Conditions normal in {district}. Continue regular farm operations. Monitor weather updates and maintain pest surveillance.",
        })

    return advisories


def render_policy_advisory(data: pd.DataFrame, hazards: pd.DataFrame, district: str):
    advisories = generate_policy_advisories(data, hazards, district)
    st.markdown(f'<p class="section-title">🏛️ Government Policy Advisory</p>', unsafe_allow_html=True)
    for adv in advisories:
        if isinstance(adv, dict):
            st.markdown(f'''
            <div class="policy-card fade-in">
                <div class="policy-header">
                    <div class="policy-icon">{adv["icon"]}</div>
                    <span class="policy-title">{adv["title"]}</span>
                </div>
                <div class="policy-content">{adv["content"]}</div>
                <span class="policy-tag" style="background:{adv.get("tag_color","#0EA5E9")}22;color:{adv.get("tag_color","#0EA5E9")};border:1px solid {adv.get("tag_color","#0EA5E9")}44;">
                    {adv.get("tag","Info")}
                </span>
            </div>
            ''', unsafe_allow_html=True)
        else:
            st.info(adv)


def render_farmer_advisory(data: pd.DataFrame, hazards: pd.DataFrame, district: str):
    advisories = generate_farmer_advisories(data, hazards, district)
    st.markdown(f'<p class="section-title">👨‍🌾 Farmer Advisory</p>', unsafe_allow_html=True)
    cols = st.columns(2)
    for i, adv in enumerate(advisories):
        if isinstance(adv, dict):
            with cols[i % 2]:
                st.markdown(f'''
                <div class="policy-card fade-in" style="background:linear-gradient(135deg,#FFF7ED,#FFFBEB);border-color:rgba(249,115,22,0.15);">
                    <div class="policy-header">
                        <div class="policy-icon" style="background:linear-gradient(135deg,#F97316,#EA580C);">{adv["icon"]}</div>
                        <span class="policy-title" style="color:#C2410C;">{adv["title"]}</span>
                    </div>
                    <div class="policy-content">{adv["content"]}</div>
                </div>
                ''', unsafe_allow_html=True)
        else:
            st.info(adv)
