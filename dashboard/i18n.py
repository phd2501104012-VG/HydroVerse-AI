"""Internationalization (i18n) — English & Hindi for HydroVerse AI."""

import streamlit as st

LANGUAGES = {"English": "en", "हिन्दी": "hi"}

def get_lang():
    return st.session_state.get("_lang", "en")

def _(en_text: str) -> str:
    """Translate en_text to the current session language."""
    try:
        lang = get_lang()
        if lang == "en":
            return en_text
        return HI_TRANSLATIONS.get(en_text, en_text)
    except Exception:
        return en_text

# ─── Hindi translations ───
HI_TRANSLATIONS = {
    # Navigation
    "Overview": "अवलोकन",
    "Live Monitoring": "लाइव निगरानी",
    "Forecasting": "पूर्वानुमान",
    "Hazard Maps": "खतरा मानचित्र",
    "Climate Trends": "जलवायु रुझान",
    "AI Assistant": "AI सहायक",
    "Alerts": "चेतावनियाँ",
    "Reports": "रिपोर्ट",

    # Sidebar labels
    "📍 District": "📍 जिला",
    "🎯 Data Source": "🎯 डेटा स्रोत",
    "⏱️ Time Period": "⏱️ समय अवधि",
    "🔮 Forecast Horizon": "🔮 पूर्वानुमान क्षितिज",
    "⚡ Quick Actions": "⚡ त्वरित कार्रवाई",
    "⚙️ System Status": "⚙️ सिस्टम स्थिति",
    "Select District": "जिला चुनें",
    "No districts loaded": "कोई जिला लोड नहीं हुआ",
    "Select Source": "स्रोत चुनें",
    "Start": "आरंभ",
    "End": "अंत",
    "Forecast": "पूर्वानुमान",
    "Alerts": "चेतावनी",
    "Mode": "मोड",
    "Operational": "संचालनात्मक",
    "Research": "अनुसंधान",
    "Validation": "सत्यापन",

    # Header
    "Madhya Pradesh": "मध्य प्रदेश",
    "Climate Hazard Dashboard": "जलवायु खतरा डैशबोर्ड",
    "Time Period": "समय अवधि",
    "Days Forecast": "दिन का पूर्वानुमान",
    "District": "जिला",

    # Hazard names
    "Flood": "बाढ़",
    "Drought": "सूखा",
    "Heatwave": "लू",
    "Agri Stress": "कृषि तनाव",
    "Extreme Precip": "अत्यधिक वर्षा",
    "Compound": "संयुक्त",
    "Flood Risk": "बाढ़ जोखिम",
    "Drought Risk": "सूखा जोखिम",
    "Heatwave Risk": "लू जोखिम",
    "Agri Risk": "कृषि जोखिम",

    # Risk levels
    "Low": "निम्न",
    "Moderate": "मध्यम",
    "High": "उच्च",
    "Severe": "गंभीर",
    "Extreme": "अत्यधिक",
    "Very Low": "बहुत निम्न",
    "Very High": "बहुत उच्च",

    "Normal": "सामान्य",
    "Watch": "निगरानी",
    "Warning": "चेतावनी",
    "No Risk": "कोई खतरा नहीं",
    "Active": "सक्रिय",

    # Alert status
    "No Active Alerts": "कोई सक्रिय चेतावनी नहीं",
    "All hazard levels are within normal range for": "सभी खतरा स्तर सामान्य सीमा में हैं:",
    "Alert": "चेतावनी",

    # KPIs
    "Current Conditions": "वर्तमान स्थितियाँ",
    "Max Temperature": "अधिकतम तापमान",
    "Rainfall (24h)": "वर्षा (24 घंटे)",
    "Humidity": "आर्द्रता",
    "Wind Speed": "हवा की गति",
    "Soil Moisture": "मिट्टी की नमी",
    "NDVI (Avg)": "NDVI (औसत)",

    # Climate indicators
    "Climate Indicators": "जलवायु संकेतक",
    "Rainfall Anomaly": "वर्षा विसंगति",
    "Temperature Anomaly": "तापमान विसंगति",
    "NDVI Trend": "NDVI प्रवृत्ति",
    "Drought Severity Index": "सूखा गंभीरता सूचकांक",

    # Cards
    "Madhya Pradesh — District Hazard Map": "मध्य प्रदेश — जिला खतरा मानचित्र",
    "Live Alerts & Warnings": "लाइव चेतावनियाँ",
    "Forecast Confidence": "पूर्वानुमान विश्वसनीयता",
    "High Confidence": "उच्च विश्वसनीयता",
    "Model Agreement: High": "मॉडल सहमति: उच्च",
    "Updated": "अपडेटेड",
    "AI Summary": "AI सारांश",
    "View Full Report": "पूरी रिपोर्ट देखें",
    "View All Alerts": "सभी चेतावनियाँ देखें",
    "View Full Forecast": "पूरा पूर्वानुमान देखें",
    "View Guidelines": "दिशानिर्देश देखें",
    "View Recommendations": "सिफारिशें देखें",

    # District Quick Status table
    "District Quick Status": "जिला त्वरित स्थिति",
    "District": "जिला",
    "Flood Risk": "बाढ़ जोखिम",
    "Heatwave Risk": "लू जोखिम",
    "Drought Risk": "सूखा जोखिम",
    "Agri Risk": "कृषि जोखिम",
    "Rainfall (mm)": "वर्षा (मिमी)",
    "Max Temp (°C)": "अधिकतम तापमान (°C)",

    # Advisories
    "Govt Advisory": "सरकारी सलाह",
    "Farmer Advisory": "किसान सलाह",

    # Events & Forecast
    "Upcoming Extreme Events": "आगामी चरम घटनाएँ",
    "Potential Events": "संभावित घटनाएँ",
    "Next 7 Days": "अगले 7 दिन",
    "15-Day Forecast Overview": "15-दिन का पूर्वानुमान अवलोकन",

    # Live Monitoring
    "Live Monitoring — 7-Day Forecast for": "लाइव निगरानी — 7-दिन का पूर्वानुमान:",
    "Temperature Forecast (°C)": "तापमान पूर्वानुमान (°C)",
    "Rainfall Forecast (mm)": "वर्षा पूर्वानुमान (मिमी)",
    "Recent Temperature (°C)": "हाल का तापमान (°C)",
    "Recent Rainfall (mm)": "हाल की वर्षा (मिमी)",
    "No recent observations available.": "कोई हालिया अवलोकन उपलब्ध नहीं है।",
    "No data available for this district.": "इस जिले के लिए कोई डेटा उपलब्ध नहीं है।",

    # Forecasting tab
    "Climate Forecasting to 2040 —": "2040 तक जलवायु पूर्वानुमान —",
    "Temperature": "तापमान",
    "Rainfall": "वर्षा",
    "Hazard Forecast": "खतरा पूर्वानुमान",
    "No historical data available for forecasting.": "पूर्वानुमान के लिए कोई ऐतिहासिक डेटा उपलब्ध नहीं है।",
    "forecast not available.": "पूर्वानुमान उपलब्ध नहीं है।",
    "Hazard projection not available.": "खतरा प्रक्षेपण उपलब्ध नहीं है।",
    "Forecast engine unavailable:": "पूर्वानुमान इंजन अनुपलब्ध:",

    # Hazard Maps tab
    "No boundary data — install a shapefile or check config.": "कोई सीमा डेटा नहीं — शेपफाइल स्थापित करें या कॉन्फ़िग जांचें।",

    # Climate Trends tab
    "Climate Trends —": "जलवायु रुझान —",
    "Monthly Max Temperature": "मासिक अधिकतम तापमान",
    "Monthly Min Temperature": "मासिक न्यूनतम तापमान",
    "Monthly Total Rainfall": "मासिक कुल वर्षा",
    "Monthly NDVI": "मासिक NDVI",
    "Monthly Soil Moisture": "मासिक मिट्टी की नमी",
    "SPI-3m (Drought Index)": "SPI-3मी (सूखा सूचकांक)",
    "TCI (Thermal Condition)": "TCI (तापीय स्थिति)",
    "Consecutive Dry Days": "लगातार शुष्क दिन",
    "Consecutive Wet Days": "लगातार गीले दिन",
    "All Variables (Normalized 0–1)": "सभी चर (सामान्यीकृत 0–1)",
    "No historical data available.": "कोई ऐतिहासिक डेटा उपलब्ध नहीं है।",

    # AI Assistant tab
    "AI Climate Assistant": "AI जलवायु सहायक",
    "Ask about climate, hazards, or MP districts...": "जलवायु, खतरों या MP जिलों के बारे में पूछें...",
    "Ask about climate, hazards...": "जलवायु, खतरों के बारे में पूछें...",
    "Install google-generativeai to enable the AI Assistant.": "AI सहायक सक्षम करने के लिए google-generativeai स्थापित करें।",
    "AI Assistant unavailable:": "AI सहायक अनुपलब्ध:",
    "Gemini API error:": "Gemini API त्रुटि:",
    "AI unavailable — set a valid GEMINI_API_KEY in Secrets.": "AI अनुपलब्ध — Secrets में मान्य GEMINI_API_KEY सेट करें।",
    "Gemini error:": "Gemini त्रुटि:",

    # Active Alerts tab
    "Active Alerts & Warnings —": "सक्रिय चेतावनियाँ —",
    "All hazard levels are within normal range for": "सभी खतरा स्तर सामान्य सीमा में हैं:",

    # Methodology
    "How We Calculate Hazards (IMD Standards)": "हम खतरों की गणना कैसे करते हैं (IMD मानक)",
    "click to expand": "विस्तार के लिए क्लिक करें",
    "🌊 Flood": "🌊 बाढ़",
    "🌡️ Heatwave": "🌡️ लू",
    "🏜️ Drought": "🏜️ सूखा",
    "🌱 Agri Stress": "🌱 कृषि तनाव",

    # Footer
    "Data Sources": "डेटा स्रोत",
    "Models": "मॉडल",
    "Validation": "सत्यापन",
    "Support": "सहायता",
    "Disclaimer": "अस्वीकरण",
    "This platform utilizes datasets from the DICRA dataset along with resources and research support from the Water Climate & Sustainability Lab, IIT Indore. This platform and its predictive models are currently under validation and development.": "यह प्लेटफ़ॉर्म DICRA डेटासेट के डेटासेट और IIT इंदौर के जल जलवायु और स्थिरता प्रयोगशाला के संसाधनों और अनुसंधान सहायता का उपयोग करता है। यह प्लेटफ़ॉर्म और इसके पूर्वानुमान मॉडल वर्तमान में सत्यापन और विकास के अधीन हैं।",
    "Generated": "उत्पन्न",
    "HydroVerse AI · Powered by ERA5 · IMD · CHIRPS · MODIS · CMIP6 · GEE": "HydroVerse AI · ERA5 · IMD · CHIRPS · MODIS · CMIP6 · GEE द्वारा संचालित",

    # System Status
    "System Status": "सिस्टम स्थिति",
    "All systems operational": "सभी सिस्टम चालू हैं",
    "Source": "स्रोत",
    "districts": "जिले",
    "Scenario": "परिदृश्य",
    "Period": "अवधि",
    "Forecast": "पूर्वानुमान",
    "ML + CMIP6 Ensemble": "ML + CMIP6 समूह",
    "HydroVerse AI": "HydroVerse AI",
    "AI-Powered Climate Intelligence": "AI-संचालित जलवायु बुद्धिमत्ता",
    "Water, Climate & Sustainability Lab": "जल, जलवायु और स्थिरता प्रयोगशाला",
    "Indian Institute of Technology Indore": "भारतीय प्रौद्योगिकी संस्थान इंदौर",

    # AI Assistant chat
    "Which districts are under heatwave risk?": "कौन से जिले लू के जोखिम में हैं?",
    "Explain current drought severity": "वर्तमान सूखा गंभीरता समझाएं",
    "Show rainfall anomaly trends": "वर्षा विसंगति रुझान दिखाएं",
    "What is the flood probability?": "बाढ़ की संभावना क्या है?",
    "Summarize climate conditions": "जलवायु स्थितियों का सारांश दें",
    "Show temperature anomaly forecast": "तापमान विसंगति पूर्वानुमान दिखाएं",

    # Buttons
    "🤖 AI Assistant": "🤖 AI सहायक",
    "Toggle AI Assistant": "AI सहायक टॉगल करें",

    # AI Summary bullet items
    "Heatwave conditions likely to persist in western MP.": "पश्चिमी MP में लू की स्थिति बने रहने की संभावना है।",
    "Rainfall deficit observed in 60% of districts.": "60% जिलों में वर्षा की कमी देखी गई है।",
    "Agri stress moderate to high in soybean regions.": "सोयाबीन क्षेत्रों में कृषि तनाव मध्यम से उच्च है।",
    "Western MP": "पश्चिमी MP",
    "Thunderstorm Warning": "आंधी-तूफान की चेतावनी",
    "Rainfall Deficit Detected": "वर्षा की कमी का पता चला",
    "Several districts": "कई जिले",
    "Next 7 days": "अगले 7 दिन",
    "Severity": "गंभीरता",
    "Generating": "उत्पन्न किया जा रहा है",
    "Forecast to 2040": "2040 तक पूर्वानुमान",
    "Hazard Forecast to 2040": "2040 तक खतरा पूर्वानुमान",
    "Projecting hazards to 2040...": "2040 तक खतरों का प्रक्षेपण...",
    "Active Alerts &amp; Warnings —": "सक्रिय चेतावनियाँ —",

    # Advisory bullet items
    "Monitor heatwave in western districts.": "पश्चिमी जिलों में लू पर निगरानी रखें।",
    "Ensure sufficient water availability.": "पर्याप्त पानी की उपलब्धता सुनिश्चित करें।",
    "Prepare for thunderstorm events.": "आंधी-तूफान की घटनाओं के लिए तैयार रहें।",
    "Follow disaster management protocols.": "आपदा प्रबंधन प्रोटोकॉल का पालन करें।",
    "Irrigate in morning hours.": "सुबह के समय सिंचाई करें।",
    "Delay sowing of soybean.": "सोयाबीन की बुवाई में देरी करें।",
    "Use mulching to retain soil moisture.": "मिट्टी की नमी बनाए रखने के लिए मल्चिंग का उपयोग करें।",
    "Monitor weather updates regularly.": "नियमित रूप से मौसम अपडेट की निगरानी करें।",
    "Today": "आज",
    "Thunderstorm": "आंधी-तूफान",
    "Heavy Rainfall": "भारी वर्षा",
    "Potential Events": "संभावित घटनाएँ",
    "Upcoming Extreme Events": "आगामी चरम घटनाएँ",

    # Methodology detailed descriptions
    "Rainfall persistence (≥64.5mm/day for 2+ days) + 3-day cumulative totals. Pre-monsoon months zeroed. Based on IMD heavy/very heavy/extreme rainfall thresholds.": "वर्षा की स्थिरता (2+ दिनों के लिए ≥64.5 मिमी/दिन) + 3-दिन संचयी कुल। पूर्व-मानसून महीने शून्य। IMD भारी/बहुत भारी/अत्यधिक वर्षा सीमा पर आधारित।",
    "Max temp ≥40°C with departure ≥4.5°C from normal. Severe when departure ≥6.5°C. Follows IMD heatwave classification criteria.": "अधिकतम तापमान ≥40°C सामान्य से ≥4.5°C विचलन के साथ। गंभीर जब विचलन ≥6.5°C हो। IMD लू वर्गीकरण मानदंड का पालन करता है।",
    "SPI-3 (Standardized Precipitation Index) ≤ -1.0 indicates moderate drought. Consecutive dry days (>30 days) also flagged. Based on IMD drought monitoring.": "SPI-3 (मानकीकृत वर्षा सूचकांक) ≤ -1.0 मध्यम सूखा इंगित करता है। लगातार शुष्क दिन (>30 दिन) भी चिह्नित। IMD सूखा निगरानी पर आधारित।",
    "Vegetation Health Index (VHI) < 50 derived from NDVI &amp; LST. Combined with soil moisture anomalies for crop stress assessment.": "वनस्पति स्वास्थ्य सूचकांक (VHI) < 50 NDVI और LST से प्राप्त। फसल तनाव मूल्यांकन के लिए मिट्टी की नमी विसंगतियों के साथ संयुक्त।",

    # Error messages
    "Gemini API error:": "Gemini API त्रुटि:",
    "Gemini error:": "Gemini त्रुटि:",
    "No boundary data — install a shapefile or check config.": "कोई सीमा डेटा नहीं — शेपफाइल स्थापित करें या कॉन्फ़िग जांचें।",
}
