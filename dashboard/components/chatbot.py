"""HydroVerse AI Assistant — ChatGPT-style floating climate intelligence assistant."""
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime

SUGGESTED_PROMPTS = [
    "Which districts are under heatwave risk?",
    "Explain current drought severity",
    "Show rainfall anomaly trends",
    "What is the flood probability?",
    "Summarize climate conditions",
    "Show temperature anomaly forecast",
]


def answer_query(query: str, data: pd.DataFrame, hazards: pd.DataFrame, district: str, districts_list: list) -> str:
    """Generate a response to a natural-language query using computed data."""
    q = query.lower().strip()

    if any(w in q for w in ["heatwave", "hot", "temperature", "heat"]):
        if "heatwave_severity" in hazards.columns:
            hw = hazards["heatwave_severity"].dropna()
            if not hw.empty:
                latest = hw.iloc[-1]
                max_hw = hw.max()
                hw_days = (hw >= 50).sum()
                if "tmax" in data.columns:
                    tmax_latest = data["tmax"].dropna().iloc[-1] if not data["tmax"].dropna().empty else "N/A"
                    tmax_str = f"{tmax_latest:.1f}°C" if isinstance(tmax_latest, (int, float)) else "N/A"
                    return f"In **{district}**, current max temperature is **{tmax_str}**. Heatwave severity is **{latest:.0f}/100** (peak: {max_hw:.0f}). **{hw_days}** days with elevated heat risk observed.\n\n{'🟠 **Heatwave active** — take precautions, stay hydrated, avoid outdoor exposure during peak hours.' if latest >= 50 else '🟢 **No active heatwave warning.** Conditions are within normal range.'}"
                return f"Heatwave severity in {district}: **{latest:.0f}/100** (peak {max_hw:.0f}, {hw_days} elevated days)."
            return "No heatwave data available for this district."
        return "Heatwave detection not enabled for this dataset."

    if any(w in q for w in ["flood", "rainfall", "precip", "rain", "water"]):
        if "flood_severity" in hazards.columns:
            fl = hazards["flood_severity"].dropna()
            latest = fl.iloc[-1] if not fl.empty else 0
            max_fl = fl.max() if not fl.empty else 0
            if "precip" in data.columns:
                precip_latest = data["precip"].dropna().iloc[-1] if not data["precip"].dropna().empty else 0
                precip_str = f"{precip_latest:.1f}mm" if isinstance(precip_latest, (int, float)) else "N/A"
                return f"Current precipitation in **{district}**: **{precip_str}**. Flood severity: **{latest:.0f}/100** (historical peak: {max_fl:.0f}).\n\n{'🌊 **Elevated flood risk** — monitor waterways and low-lying areas.' if latest >= 50 else '✅ **No active flood risk.** Conditions are normal.'}"
            return f"Flood severity in {district}: **{latest:.0f}/100** (peak {max_fl:.0f})."
        return "Flood detection not enabled."

    if any(w in q for w in ["drought", "dry", "spi", "arid"]):
        if "drought_severity" in hazards.columns:
            dr = hazards["drought_severity"].dropna()
            latest_dr = dr.iloc[-1] if not dr.empty else 0
            spi_text = ""
            if "spi_3m" in hazards.columns:
                spi = hazards["spi_3m"].dropna()
                if not spi.empty:
                    spi_text = f"\n\nSPI-3 index: **{spi.iloc[-1]:.2f}** — {'drought conditions' if spi.iloc[-1] < -1 else 'wet conditions' if spi.iloc[-1] > 1 else 'normal conditions'}."
            cdd_text = ""
            if "cdd" in hazards.columns:
                cdd = hazards["cdd"].dropna()
                if not cdd.empty and cdd.iloc[-1] > 5:
                    cdd_text = f"\n**{int(cdd.iloc[-1])} consecutive dry days** recorded."
            return f"Drought severity in **{district}**: **{latest_dr:.0f}/100**.{spi_text}{cdd_text}\n\n{'🏜️ **Drought conditions warrant monitoring** — water conservation measures recommended.' if latest_dr >= 50 else '✅ **No significant drought.**'}"
        return "Drought detection not enabled."

    if any(w in q for w in ["agriculture", "agri", "crop", "vegetation", "vhi", "ndvi", "farm"]):
        if "vhi" in hazards.columns:
            vhi = hazards["vhi"].dropna()
            vhi_latest = vhi.iloc[-1] if not vhi.empty else "N/A"
            vhi_str = f"{vhi_latest:.0f}" if isinstance(vhi_latest, (int, float)) else "N/A"
            agri_text = ""
            if "agri_severity" in hazards.columns:
                ag = hazards["agri_severity"].dropna()
                ag_latest = ag.iloc[-1] if not ag.empty else "N/A"
                agri_text = f"\nAgricultural stress index: **{ag_latest:.0f}/100**." if isinstance(ag_latest, (int, float)) else ""
            ndvi_text = ""
            if "ndvi" in data.columns:
                ndvi = data["ndvi"].dropna()
                if not ndvi.empty:
                    ndvi_text = f"\nNDVI: **{ndvi.iloc[-1]:.3f}**."
            return f"Vegetation health (VHI) in **{district}**: **{vhi_str}**/100.{ndvi_text}{agri_text}\n\n{'🌾 **Crop stress detected** — monitor field conditions.' if isinstance(vhi_latest, (int, float)) and vhi_latest < 40 else '✅ **Vegetation conditions normal.**'}"
        return "Satellite vegetation data not available."

    if any(w in q for w in ["anomaly", "anomalies", "abnormal", "departure"]):
        parts = []
        if "tmax_anom" in hazards.columns:
            ta = hazards["tmax_anom"].dropna()
            if not ta.empty:
                parts.append(f"Temperature anomaly: **{ta.iloc[-1]:+.2f}°C**")
        if "precip_anom" in hazards.columns:
            pa = hazards["precip_anom"].dropna()
            if not pa.empty:
                parts.append(f"Precipitation anomaly: **{pa.iloc[-1]:+.2f}mm**")
        if parts:
            return f"Climate anomalies in **{district}**:\n\n{chr(10).join('• ' + p for p in parts)}\n\nConditions are {'**above normal**' if any('+' in p for p in parts) else '**below normal**'} in some indicators."
        return "Anomaly data not computed for this district."

    if any(w in q for w in ["summary", "overview", "conditions", "situation", "status", "current"]):
        parts = []
        for hname, hcol in [("flood", "flood_severity"), ("drought", "drought_severity"), ("heatwave", "heatwave_severity")]:
            if hcol in hazards.columns:
                val = hazards[hcol].dropna()
                if not val.empty:
                    v = val.iloc[-1]
                    status = "🔴" if v >= 75 else "🟠" if v >= 50 else "🟡" if v >= 25 else "🟢"
                    parts.append(f"{hname.title()} {status} ({v:.0f}/100)")
        if parts:
            active_count = sum(1 for hcol in ["flood_severity", "drought_severity", "heatwave_severity"]
                              if hcol in hazards.columns and not hazards[hcol].dropna().empty and hazards[hcol].dropna().iloc[-1] >= 50)
            rec = "**Multiple hazards active** — integrated monitoring recommended." if active_count > 1 else "**Single hazard active** — focused monitoring recommended." if active_count == 1 else "**All clear** — routine monitoring."
            return f"## {district} — Current Status\n\n{'  |  '.join(parts)}\n\n{rec}"
        return f"Load data for **{district}** to see current conditions."

    if any(w in q for w in ["forecast", "future", "prediction", "outlook", "next"]):
        if "tmax_anom" in hazards.columns:
            ta = hazards["tmax_anom"].dropna()
            trend = "warming" if not ta.empty and ta.tail(7).mean() > ta.head(7).mean() else "cooling" if not ta.empty else "stable"
            return f"Short-term outlook for **{district}**: **{trend}** trend observed in temperature anomalies.\n\n{'🔍 **Monitor heatwave development.** Generate a full forecast for detailed projections to 2040.' if trend == 'warming' else '📊 **No extreme signals** in near term. Generate a full forecast in the Forecasting tab.'}"
        return "Generate a forecast in the Forecasting tab for outlook information."

    if any(w in q for w in ["district", "districts"]):
        if districts_list:
            return f"**{len(districts_list)}** districts are configured in the system. Currently viewing **{district}**. Select a different district from the sidebar to compare conditions."
        return "District data not loaded."

    if any(w in q for w in ["help", "what can", "commands", "capabilities"]):
        return """I can answer questions about:

🌡️ **Heatwave conditions** — severity, temperature, hot days
🌊 **Flood risk** — precipitation, severity, warnings
🏜️ **Drought status** — SPI indices, dry days, severity
🌾 **Agricultural stress** — VHI, NDVI, crop conditions
📊 **Climate anomalies** — temperature & precipitation departures
🔮 **Forecast outlook** — trends and projections
📍 **District information** — data availability

Try one of the suggested prompts below!"""

    return f"I understand you're asking about '{query}'. I can help with information about heatwave, flood, drought, climate anomalies, agricultural stress, and forecasts for **{district}**. Try one of the suggested prompts!"


def render_chatbot(data, hazards, district, districts_list):
    """Render the floating ChatGPT-style AI assistant."""
    if "chatbot_open" not in st.session_state:
        st.session_state.chatbot_open = False
    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = [
            {"role": "bot", "content": f"👋 Hi! I'm the **HydroVerse AI Assistant**. Ask me about climate hazards, forecasts, or conditions in **{district}**!"}
        ]

    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        if st.button("🤖 AI Assistant", key="chatbot_toggle", type="secondary", use_container_width=True):
            st.session_state.chatbot_open = not st.session_state.chatbot_open

    if not st.session_state.chatbot_open:
        return

    # Container styled as chatbot window via CSS :has() selector
    chat_window = st.container()
    with chat_window:
        # Header
        st.markdown(f'''
        <div class="chatbot-header">
            <div class="chatbot-avatar">🤖</div>
            <div class="chatbot-title-group">
                <div class="chatbot-title-name">HydroVerse AI Assistant</div>
                <div class="chatbot-title-status">
                    <span class="status-dot-sm"></span> Online · {district}
                </div>
            </div>
        </div>
        ''', unsafe_allow_html=True)

        # Messages
        for msg in st.session_state.chat_messages[-12:]:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        # Suggested prompts
        sugs_cols = st.columns(3)
        for i, prompt in enumerate(SUGGESTED_PROMPTS[:6]):
            with sugs_cols[i % 3]:
                if st.button(prompt, key=f"chat_sug_{i}", use_container_width=True, type="secondary"):
                    _handle_user_input(prompt, data, hazards, district, districts_list)
                    st.rerun()

        # Chat input
        user_input = st.chat_input("Ask about climate conditions...", key="chat_input_main")
        if user_input:
            _handle_user_input(user_input, data, hazards, district, districts_list)
            st.rerun()


def _handle_user_input(user_input, data, hazards, district, districts_list):
    """Process user input and generate response with typing simulation."""
    st.session_state.chat_messages.append({"role": "user", "content": user_input})
    with st.spinner("🤔 Analyzing..."):
        response = answer_query(user_input, data, hazards, district, districts_list)
    st.session_state.chat_messages.append({"role": "bot", "content": response})
