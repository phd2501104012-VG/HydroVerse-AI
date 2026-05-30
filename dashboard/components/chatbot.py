"""HydroVerse AI Assistant — Bottom-right floating AI assistant powered by Gemini."""
import streamlit as st
import pandas as pd
import numpy as np
import os
from datetime import datetime
import google.generativeai as genai

SUGGESTED_PROMPTS = [
    "Which districts are under heatwave risk?",
    "Explain current drought severity",
    "Show rainfall anomaly trends",
    "What is the flood probability?",
    "Summarize climate conditions",
    "Show temperature anomaly forecast",
]

_SYSTEM_PROMPT = """You are HydroVerse AI Assistant, a climate hazard intelligence expert for Madhya Pradesh, India.

Current District: {district}
Total Districts: {num_districts}

Latest Hazard Severities (0-100 scale):
- Flood: {flood_sev}/100
- Heatwave: {heatwave_sev}/100  
- Drought: {drought_sev}/100
- Agriculture Stress: {agri_sev}/100

Latest Climate Indicators:
- Max Temperature: {tmax}°C
- Min Temperature: {tmin}°C
- Precipitation: {precip}mm
- NDVI: {ndvi}
- Soil Moisture: {soil_moisture}
- SPI-3: {spi}
- Consecutive Dry Days: {cdd}

Thresholds: 0-25=Normal, 25-50=Watch, 50-75=Warning, 75-100=Severe/Extreme

Answer questions about climate conditions. Be specific, cite data values. Keep to 3-4 sentences unless asked for details."""


def _try_gemini(query: str, context: dict) -> str | None:
    key = st.session_state.get("gemini_api_key") or os.environ.get("GEMINI_API_KEY", "")
    if not key:
        return None
    try:
        genai.configure(api_key=key)
        model = genai.GenerativeModel("gemini-pro")
        prompt = _SYSTEM_PROMPT.format(**context) + f"\n\nUser question: {query}"
        resp = model.generate_content(prompt)
        return resp.text.strip() if resp and resp.text else None
    except Exception:
        return None


def _build_context(data, hazards, district, districts_list) -> dict:
    ctx = {"district": district, "num_districts": len(districts_list)}
    for k in ["flood_severity","heatwave_severity","drought_severity","agri_severity"]:
        v = 0
        if k in hazards.columns and not hazards[k].dropna().empty:
            v = round(float(hazards[k].dropna().iloc[-1]))
        ctx[k.replace("_severity","_sev")] = v
    for k, col in [("tmax","tmax"),("tmin","tmin"),("precip","precip")]:
        v = "N/A"
        if k in data.columns and not data[k].dropna().empty:
            v = f"{data[k].dropna().iloc[-1]:.1f}"
        ctx[k] = v
    for k in ["ndvi","soil_moisture"]:
        v = "N/A"
        if k in data.columns and not data[k].dropna().empty:
            v = f"{data[k].dropna().iloc[-1]:.3f}"
        ctx[k] = v
    ctx["spi"] = "N/A"
    if "spi_3m" in hazards.columns and not hazards["spi_3m"].dropna().empty:
        ctx["spi"] = f"{hazards['spi_3m'].dropna().iloc[-1]:.2f}"
    ctx["cdd"] = "N/A"
    if "cdd" in hazards.columns and not hazards["cdd"].dropna().empty:
        ctx["cdd"] = str(int(hazards["cdd"].dropna().iloc[-1]))
    return ctx


def answer_query(query: str, data: pd.DataFrame, hazards: pd.DataFrame, district: str, districts_list: list) -> str:
    context = _build_context(data, hazards, district, districts_list)
    gemini_resp = _try_gemini(query, context)
    if gemini_resp:
        return gemini_resp
    q = query.lower().strip()
    if any(w in q for w in ["heatwave", "hot", "temperature", "heat"]):
        if "heatwave_severity" in hazards.columns and not hazards["heatwave_severity"].dropna().empty:
            hw = hazards["heatwave_severity"].dropna().iloc[-1]
            return f"In **{district}**, heatwave severity is **{hw:.0f}/100**.\n\n{'🟠 Active' if hw >= 50 else '🟢 Normal'}"
    if any(w in q for w in ["flood", "rainfall", "precip", "rain"]):
        if "flood_severity" in hazards.columns and not hazards["flood_severity"].dropna().empty:
            fl = hazards["flood_severity"].dropna().iloc[-1]
            return f"Flood severity in **{district}**: **{fl:.0f}/100**.\n\n{'🌊 Elevated' if fl >= 50 else '✅ Normal'}"
    if any(w in q for w in ["drought", "dry", "spi"]):
        if "drought_severity" in hazards.columns and not hazards["drought_severity"].dropna().empty:
            dr = hazards["drought_severity"].dropna().iloc[-1]
            return f"Drought severity in **{district}**: **{dr:.0f}/100**.\n\n{'🏜️ Monitor' if dr >= 50 else '✅ Normal'}"
    if any(w in q for w in ["summary", "overview", "conditions", "status", "current"]):
        parts = []
        for hname, hcol in [("Flood","flood_severity"),("Drought","drought_severity"),("Heatwave","heatwave_severity")]:
            if hcol in hazards.columns and not hazards[hcol].dropna().empty:
                parts.append(f"{hname}: {hazards[hcol].dropna().iloc[-1]:.0f}/100")
        if parts:
            return f"## {district}\n\n" + "  |  ".join(parts)
    if any(w in q for w in ["help", "what can"]):
        return "I answer about: heatwave, flood, drought, agriculture, anomalies, forecast."
    return f"I can help with climate hazard info for **{district}**. Try a suggested prompt."


def render_chatbot(data, hazards, district, districts_list):
    """Render floating button + inline chat panel at bottom of page."""
    if "chatbot_open" not in st.session_state:
        st.session_state.chatbot_open = False
    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = [
            {"role": "assistant", "content": f"👋 Hi! I'm the **HydroVerse AI Assistant**. Ask me about climate hazards in **{district}**!"}
        ]
    if "gemini_api_key" not in st.session_state:
        st.session_state.gemini_api_key = ""

    is_open = st.session_state.chatbot_open
    key_ok = bool(st.session_state.gemini_api_key or os.environ.get("GEMINI_API_KEY"))
    key_status = "✅ Gemini" if key_ok else "⚡ Rule-based"

    # ── Floating FAB button (position: fixed in CSS) ──
    st.markdown(f"""
    <style>
    #fabchat {{ position:fixed; bottom:24px; right:24px; z-index:9999; width:54px; height:54px;
        border-radius:50%; background:#0EA5E9; color:#fff; border:none; cursor:pointer;
        box-shadow:0 4px 20px rgba(0,0,0,0.3); font-size:24px; }}
    #fabchat:hover {{ transform:scale(1.1); }}
    </style>
    <button id="fabchat" onclick="document.getElementById('togchat').click()">{'✕' if is_open else '💬'}</button>
    """, unsafe_allow_html=True)

    if st.button("", key="togchat"):
        st.session_state.chatbot_open = not st.session_state.chatbot_open
        st.rerun()

    # API key input when closed
    if not is_open:
        if not key_ok:
            with st.sidebar:
                with st.expander("🤖 AI Key", expanded=False):
                    k = st.text_input("Gemini API Key", type="password", key="gkey_side")
                    if k:
                        st.session_state.gemini_api_key = k
                        st.rerun()
        return

    # ── Chat panel (rendered inline at bottom of app) ──
    with st.container():
        # Outer styled container
        st.markdown(f"""
        <div style="background:#fff;border-radius:16px;box-shadow:0 4px 24px rgba(0,0,0,0.12);
            border:1px solid #e5e7eb;overflow:hidden;margin-bottom:4px;">
        <div style="background:#0EA5E9;color:#fff;padding:10px 16px;display:flex;align-items:center;gap:10px;">
            <span style="font-size:1.3rem;">🤖</span>
            <div>
                <div style="font-weight:700;font-size:0.9rem;">HydroVerse AI Assistant</div>
                <div style="font-size:0.65rem;opacity:0.85;">{district} · {key_status}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Suggested prompts
        sugs = st.columns(6)
        for i, p in enumerate(SUGGESTED_PROMPTS[:6]):
            with sugs[i]:
                if st.button(p.split(" ")[0], key=f"sug_{i}", help=p, use_container_width=True):
                    _handle_user_input(p, data, hazards, district, districts_list)
                    st.rerun()

        # Messages
        for msg in st.session_state.chat_messages[-10:]:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        st.markdown("</div>", unsafe_allow_html=True)

    # Chat input
    prompt = st.chat_input("Ask about climate conditions...", key="chat_in")
    if prompt:
        _handle_user_input(prompt, data, hazards, district, districts_list)
        st.rerun()

    # API key
    if not key_ok:
        with st.expander("🔑 Gemini API Key", expanded=False):
            k = st.text_input("Enter key", type="password", key="gkey_pop")
            if k:
                st.session_state.gemini_api_key = k
                st.rerun()


def _handle_user_input(user_input, data, hazards, district, districts_list):
    st.session_state.chat_messages.append({"role": "user", "content": user_input})
    with st.spinner(""):
        response = answer_query(user_input, data, hazards, district, districts_list)
    st.session_state.chat_messages.append({"role": "assistant", "content": response})
