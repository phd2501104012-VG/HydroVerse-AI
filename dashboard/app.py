#!/usr/bin/env python
"""HydroVerse AI — Climate Hazard Intelligence Platform (redesigned)"""
import sys, os
from pathlib import Path
from datetime import datetime, timedelta, date

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

from config import CFG, DataSource
from dashboard.config import *
from geospatial.boundaries import DistrictBoundaries
from data.source_manager import DataSourceManager
from data.era5_loader import ERA5Loader
from data.imd_loader import IMDLoader
from hazards.detection import HazardDetector
from hazards.categories import HazardClassifier
from hazards.compound import CompoundHazardEngine
from forecasting.daily_forecast import DailyForecastEngine
from realtime.monitor import RealtimeMonitor
from realtime.alerts import AlertEngine
from realtime.anomalies import AnomalyDetector
from validation.historical_validation import HistoricalValidator
from validation.confusion import ConfusionMatrixBuilder
from dashboard.components.charts import *
from dashboard.components.realtime_panel import render_realtime_status, render_alert_summary
from dashboard.components.event_cards import render_event_cards
from dashboard.components.ai_insights import render_ai_insights
from dashboard.components.forecast_view import render_forecast_tab
from dashboard.components.comparison_advisory import render_source_comparison, render_policy_advisory

try:
    from data.cmip6_loader import CMIP6Loader
except Exception:
    CMIP6Loader = None
try:
    from data.dicra_loader import DICRALoader
except Exception:
    DICRALoader = None
from utils import get_logger

logger = get_logger(__name__)

st.set_page_config(
    page_title="HydroVerse AI | Climate Hazard Intelligence Platform",
    page_icon=":material/water_drop:",
    layout="wide",
)

# ---------------------------------------------------------------------------
# CSS — matches the provided HTML design
# ---------------------------------------------------------------------------
_css = """
<style>
:root {
  --bg: #f6f7fb; --panel: #ffffff; --ink: #0f172a;
  --ink-2: #475569; --ink-3: #94a3b8;
  --line: #e7e9ef; --line-2: #eef0f5;
  --brand: #2563eb; --brand-2: #4f46e5;
  --good: #16a34a; --warn: #f59e0b; --hot: #ef4444; --cool: #0ea5e9;
  --sidebar: #0b1220; --sidebar-2: #111a2e;
}
.stApp { background: var(--bg); color: var(--ink); font-family: 'Plus Jakarta Sans', system-ui, sans-serif; }
section[data-testid="stSidebar"] > div:first-child {
  background: linear-gradient(180deg, #f8fafc 0%, #f1f5f9 100%) !important;
  border-right: 1px solid #e2e8f0 !important;
}
section[data-testid="stSidebar"] .st-emotion-cache-1cypcdb { color: #334155; }
section[data-testid="stSidebar"] hr { border-color: #e2e8f0; }
.card { background: var(--panel); border: 1px solid var(--line); border-radius: 14px; box-shadow: 0 1px 2px rgba(15,23,42,0.03); padding: 16px; }
.card-title { font-weight: 600; color: var(--ink); font-size: 14px; }
.card-sub { color: var(--ink-2); font-size: 12px; }
.kpi { padding: 14px 16px; }
.kpi-icon { width: 36px; height: 36px; border-radius: 10px; display: flex; align-items: center; justify-content: center; }
.kpi-label { font-size: 11px; color: var(--ink-2); font-weight: 500; }
.kpi-value { font-size: 22px; font-weight: 700; color: var(--ink); line-height: 1.1; }
.kpi-delta { font-size: 11px; font-weight: 600; display: inline-flex; align-items: center; gap: 3px; }
.pulse-dot { width: 8px; height: 8px; border-radius: 50%; background: #22c55e; box-shadow: 0 0 0 0 rgba(34,197,94,0.55); animation: pulse 2s infinite; display: inline-block; margin-right: 6px; }
@keyframes pulse { 0% { box-shadow: 0 0 0 0 rgba(34,197,94,0.55); } 70% { box-shadow: 0 0 0 8px rgba(34,197,94,0); } 100% { box-shadow: 0 0 0 0 rgba(34,197,94,0); } }
.alert-row { display: flex; gap: 10px; align-items: flex-start; padding: 8px 0; border-bottom: 1px solid var(--line-2); }
.alert-row:last-child { border-bottom: none; }
.alert-icon { width: 30px; height: 30px; border-radius: 8px; display: flex; align-items: center; justify-content: center; flex-shrink: 0; }
.link { color: var(--brand); font-weight: 600; font-size: 12.5px; display: inline-flex; align-items: center; gap: 4px; text-decoration: none; }
.link:hover { text-decoration: underline; }
.section-eyebrow { font-size: 11px; font-weight: 600; color: var(--ink-3); letter-spacing: .08em; text-transform: uppercase; margin-bottom: 8px; }
.risk { font-weight: 600; font-size: 12.5px; }
.risk-high { color: #dc2626; }
.risk-moderate { color: #d97706; }
.risk-low { color: #16a34a; }
.tag { font-size: 11px; padding: 2px 8px; border-radius: 999px; font-weight: 600; background: #f1f5f9; color: var(--ink-2); display: inline-block; }
.pill-ctrl { background: #fff; border: 1px solid var(--line); border-radius: 10px; padding: 8px 12px; display: flex; align-items: center; gap: 8px; box-shadow: 0 1px 2px rgba(15,23,42,0.03); }
.data-table th { text-align: left; font-size: 11px; font-weight: 600; color: var(--ink-2); padding: 8px 12px; border-bottom: 1px solid var(--line-2); }
.data-table td { font-size: 13px; padding: 10px 12px; border-bottom: 1px solid var(--line-2); color: var(--ink); }
.data-table tbody tr:last-child td { border-bottom: none; }
.data-table tbody tr:hover { background: #fafbfd; }
.bullet-list li { padding: 4px 0; font-size: 13px; color: var(--ink-2); display: flex; gap: 8px; align-items: flex-start; }
.bullet-list li::before { content: '•'; color: var(--ink-3); flex-shrink: 0; }
.mono { font-family: 'JetBrains Mono', monospace; }
.status-badge { background: #ffffff; border: 1px solid #e2e8f0; border-radius: 10px; padding: 10px 12px; }
.wx-icon-bg { width: 36px; height: 36px; border-radius: 50%; display: flex; align-items: center; justify-content: center; background: #f1f5f9; }
/* Sidebar nav radio styled as nav items */
div[data-testid="stSidebar"] div[data-testid="stRadio"] > label { display: none !important; }
div[data-testid="stSidebar"] div[data-testid="stRadio"] > div[role="radiogroup"] { display: flex; flex-direction: column; gap: 2px; }
div[data-testid="stSidebar"] div[data-testid="stRadio"] > div[role="radiogroup"] label {
  display: flex; align-items: center; gap: 12px; padding: 10px 14px; border-radius: 10px;
  font-size: 14px; font-weight: 500; color: #475569; cursor: pointer; transition: all .15s ease;
  background: transparent !important; border: none !important;
}
div[data-testid="stSidebar"] div[data-testid="stRadio"] > div[role="radiogroup"] label:hover {
  background: rgba(0,0,0,0.03) !important; color: #0f172a !important;
}
div[data-testid="stSidebar"] div[data-testid="stRadio"] > div[role="radiogroup"] label[data-selected="true"] {
  background: #ffffff !important;
  color: #0f172a !important;
  border-left: 3px solid #2563eb !important;
  box-shadow: 0 1px 3px rgba(0,0,0,0.06) !important;
  font-weight: 600 !important;
}
div[data-testid="stSidebar"] div[data-testid="stRadio"] > div[role="radiogroup"] label span:first-child { display: none; }
.forecast-cell { display: flex; flex-direction: column; align-items: center; text-align: center; }
.swatch { width: 14px; height: 14px; border-radius: 3px; display: inline-block; }
[data-testid="stMetric"] { background: var(--panel); border: 1px solid var(--line); border-radius: 14px; padding: 16px; }
[data-testid="stMetric"] label { color: var(--ink-2) !important; font-size: 0.8rem !important; }
[data-testid="stMetric"] [data-testid="stMetricValue"] { color: var(--ink) !important; font-size: 1.8rem !important; font-weight: 700; }
.mono { font-family: 'JetBrains Mono', monospace; letter-spacing: -0.02em; }
</style>
"""
st.markdown(_css, unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Resources (cached)
# ---------------------------------------------------------------------------
@st.cache_resource
def init_resources():
    try:
        b = DistrictBoundaries()
        _ = b.district_names
    except Exception as e:
        logger.warning(f"DistrictBoundaries: {e}")
        b = None
    ds = DataSourceManager()
    try:
        era5 = ERA5Loader()
        ds.set_era5_loader(era5)
    except Exception as e:
        logger.warning(f"ERA5Loader: {e}")
    try:
        imd = IMDLoader()
        ds.set_imd_loader(imd)
    except Exception as e:
        logger.warning(f"IMDLoader: {e}")
    det = HazardDetector()
    cls = HazardClassifier()
    compound = CompoundHazardEngine()
    fc = DailyForecastEngine()
    rt = RealtimeMonitor()
    ae = AlertEngine()
    ad = AnomalyDetector()
    hv = HistoricalValidator()
    return b, ds, det, cls, compound, fc, rt, ae, ad, hv

(
    bounds_mgr, ds_mgr, detector, classifier, compound_engine,
    forecast_engine, realtime_monitor, alert_engine, anomaly_detector,
    historical_validator,
) = init_resources()

districts = bounds_mgr.district_names if bounds_mgr else []

# ---------------------------------------------------------------------------
# CMIP6 Ensemble loading
# ---------------------------------------------------------------------------
def load_cached_cmip6():
    cache_path = Path(CFG.cache_dir) / "cmip6_ensemble.parquet"
    if cache_path.exists():
        try:
            df = pd.read_parquet(cache_path)
            forecast_engine.cmip6.set_ensemble(df)
            logger.info(f"Loaded CMIP6 ensemble from cache ({len(df)} rows)")
            st.session_state["_cmip6_status"] = (
                f"CMIP6: {len(df)} rows, {df['district'].nunique()} districts"
            )
            return df
        except Exception as e:
            st.session_state["_cmip6_status"] = f"CMIP6 load failed: {e}"
    else:
        st.session_state["_cmip6_status"] = "CMIP6: file not found"
    return None

cmip6_ensemble = load_cached_cmip6()

# ---------------------------------------------------------------------------
# Constants & defaults
# ---------------------------------------------------------------------------
MP_DISTRICTS = [
    "Bhopal","Raisen","Sehore","Vidisha","Hoshangabad","Indore","Ujjain",
    "Gwalior","Jabalpur","Rewa","Sagar","Satna","Khandwa","Khargone",
    "Dhar","Mandsaur","Ratlam","Dewas","Shajapur","Rajgarh","Guna",
    "Ashoknagar","Shivpuri","Sheopur","Morena","Bhind","Datia",
    "Tikamgarh","Chhatarpur","Panna","Damoh","Katni","Umaria",
    "Shahdol","Anuppur","Dindori","Mandla","Balaghat","Seoni",
    "Chhindwara","Betul","Harda","Burhanpur","Barwani","Alirajpur",
    "Jhabua","Neemuch","Agar Malwa","Narsinghpur","Niwari","Singrauli","Sidhi"
]

if "district" not in st.session_state or st.session_state.district not in (districts or MP_DISTRICTS):
    st.session_state.district = "Bhopal"
district = st.session_state.district

# Sidebar nav state
if "nav_view" not in st.session_state:
    st.session_state.nav_view = "Overview"

# ---------------------------------------------------------------------------
# Data fetching
# ---------------------------------------------------------------------------
def fetch_data(force_fresh=False):
    # District name → CSV filename mapping (some districts use alternate names)
    _name_map = {
        "Khandwa": "East Nimar", "Khargone": "West Nimar",
        "Narsinghpur": "Narsimhapur",
    }
    csv_name = _name_map.get(district, district)
    alt_base = Path.home() / "exports" / "raw"
    raw_path = Path(f"exports/raw/{csv_name}_data.csv")
    candidates = [raw_path, alt_base / f"{csv_name}_data.csv"]
    if force_fresh:
        try:
            data = ds_mgr.get_district_timeseries(district, ["tmax", "tmin", "precip"])
            if data is not None and not data.empty:
                if "date" in data.columns:
                    data = data.set_index(pd.to_datetime(data["date"])).drop(columns=["date"])
                for c in ["ndvi","soil_moisture"]:
                    if c not in data.columns:
                        data[c] = np.nan
                return data
        except Exception as e:
            logger.warning(f"Fresh fetch failed: {e}")
    best_df = None
    for rp in candidates:
        if rp.exists():
            try:
                df = pd.read_csv(rp)
                if "date" in df.columns:
                    df = df.set_index(pd.to_datetime(df["date"])).drop(columns=["date"])
                for var in ["tmax","tmin","precip"]:
                    if f"{var}_era5" in df.columns:
                        df[var] = df[f"{var}_era5"]
                        if f"{var}_imd" in df.columns:
                            df[var] = df[var].fillna(df[f"{var}_imd"])
                    elif f"{var}_imd" in df.columns:
                        df[var] = df[f"{var}_imd"]
                for c in ["ndvi","soil_moisture"]:
                    if c not in df.columns:
                        df[c] = np.nan
                if best_df is None or len(df.columns) > len(best_df.columns):
                    best_df = df
            except Exception:
                pass
    if best_df is not None:
        haz_candidates = [
            Path(f"exports/hazards/{csv_name}_hazards.csv"),
            Path.home() / "exports" / "hazards" / f"{csv_name}_hazards.csv",
        ]
        best_hdf = None
        best_sat = 0
        for hp in haz_candidates:
            if hp.exists():
                try:
                    hdf = pd.read_csv(hp)
                    if "date" in hdf.columns:
                        hdf = hdf.set_index(pd.to_datetime(hdf["date"]))
                    sat = [c for c in ["ndvi","soil_moisture","vci","vhi","tci"] if c in hdf.columns]
                    if len(sat) > best_sat:
                        best_hdf = hdf
                        best_sat = len(sat)
                except Exception:
                    pass
        if best_hdf is not None:
            for col in ["ndvi","soil_moisture"]:
                if col in best_hdf.columns and col in best_df.columns:
                    n = min(len(best_hdf), len(best_df))
                    vals = best_hdf[col].values[:n]
                    fill_series = pd.Series(vals, index=best_df.index[:n])
                    best_df.iloc[:n, best_df.columns.get_loc(col)] = (
                        best_df.iloc[:n, best_df.columns.get_loc(col)].fillna(fill_series)
                    )
        return best_df
    try:
        data = ds_mgr.get_district_timeseries(district, ["tmax","tmin","precip"])
        if data is not None and not data.empty:
            if "date" in data.columns:
                data = data.set_index(pd.to_datetime(data["date"])).drop(columns=["date"])
            return data
    except Exception:
        pass
    return pd.DataFrame()

def get_hazard_dict(hazards_df):
    d = {}
    for hn in ["flood","drought","heatwave","agri_stress","compound"]:
        cols = [c for c in hazards_df.columns if c.startswith(hn)]
        d[hn] = hazards_df[cols] if cols else pd.DataFrame()
    return d

# ---------------------------------------------------------------------------
# Load data on district select
# ---------------------------------------------------------------------------
force_fresh = st.session_state.pop("force_refresh", False)
if (not st.session_state.get("data_loaded") or st.session_state.get("data_district") != district) and district:
    st.session_state.data_loaded = True
    st.session_state.data_district = district
    st.session_state.pop("rt_forecast", None)
    for k in list(st.session_state.keys()):
        if k.startswith("map_haz_"):
            del st.session_state[k]
    with st.spinner(f"Loading {district}..."):
        df = fetch_data(force_fresh=force_fresh)
        if DICRALoader is not None and not df.empty:
            try:
                dl = DICRALoader()
                df = dl.merge_into(df, district)
            except Exception:
                pass
        st.session_state.data = df
        core_check = [c for c in ["tmax","tmin","precip","ndvi","soil_moisture"] if c in df.columns]
        if not df.empty and len(core_check) > 0 and not df[core_check].isnull().all().all():
            try:
                haz = detector.detect_all(df, district=district)
                st.session_state.hazards = haz
                hdict = get_hazard_dict(haz)
                alerts_df = alert_engine.evaluate(hdict, district, min_severity=30)
                if not alerts_df.empty:
                    st.session_state.alerts = alerts_df
            except Exception as e:
                logger.warning(f"Hazard/alert: {e}")

data = st.session_state.get("data", pd.DataFrame())
hazards = st.session_state.get("hazards", pd.DataFrame())
alerts = st.session_state.get("alerts", pd.DataFrame())
core_cols = [c for c in ["tmax","tmin","precip","ndvi","soil_moisture"] if c in data.columns]
has_data = len(core_cols) > 0 and not data[core_cols].isnull().all().all()

# Latest hazard values
spi_val = float(hazards["spi_3m"].dropna().iloc[-1]) if not hazards.empty and "spi_3m" in hazards.columns else None
cdd_val = float(hazards["cdd"].dropna().iloc[-1]) if not hazards.empty and "cdd" in hazards.columns else None
tanom_val = float(hazards["tmax_anom"].dropna().iloc[-1]) if not hazards.empty and "tmax_anom" in hazards.columns else None
panom_val = float(hazards["precip_anom"].dropna().iloc[-1]) if not hazards.empty and "precip_anom" in hazards.columns else None

def _safe_last(col, default=0):
    if not hazards.empty and col in hazards.columns and hazards[col].notna().any():
        return float(hazards[col].dropna().iloc[-1])
    return default

fv = _safe_last("flood_severity")
dv = _safe_last("drought_severity")
hv_ = _safe_last("heatwave_severity")
av = _safe_last("agri_severity")
sev_dict = {"flood": fv, "drought": dv, "heatwave": hv_, "agri_stress": av}

def _risk_word(v):
    if v >= 75: return "Severe"
    if v >= 50: return "Warning"
    if v >= 25: return "Watch"
    return "Low"

def _risk_color(v):
    if v >= 75: return "#dc2626"
    if v >= 50: return "#f97316"
    if v >= 25: return "#eab308"
    return "#16a34a"

_extreme_v = max(fv, dv, hv_, av) if any([fv, dv, hv_, av]) else 0

# ---------------------------------------------------------------------------
# SIMULATED DATA (auto-generated for all MP districts)
# ---------------------------------------------------------------------------
_HAZARD_CATS = ["Low", "Moderate", "High", "Severe"]

def _auto_sim_data(name: str, idx: int) -> dict:
    """Generate deterministic simulated hazard data per district."""
    seed = hash(name) % 100
    # Temperature varies by region: west MP hotter, east cooler
    base_t = 39.0 if "Indore" in name or "Ujjain" in name or "Khargone" in name or "Khandwa" in name or "Dhar" in name or "Jhabua" in name or "Alirajpur" in name or "Barwani" in name or "Ratlam" in name or "Mandsaur" in name or "Neemuch" in name else 38.5
    tmax = base_t + (seed % 20) / 10
    # Rainfall: south/west MP drier, east/north wetter
    precip = max(0.5, (seed % 30) - (idx % 5))
    # Hazard levels based on region patterns
    # Note: flood kept Low for all districts in dry season (May has no rain)
    def _pick(key: str) -> str:
        if key == "flood":
            return "Low"  # No flood risk in dry pre-monsoon season
        vals = {"heat": [10, 10, 25, 25, 25, 50, 50, 50, 60],
                "drought": [10, 10, 10, 25, 25, 25, 50],
                "agri": [10, 10, 25, 25, 25, 50, 50]}[key]
        v = vals[(seed + idx) % len(vals)]
        if v <= 15: return "Low"
        if v <= 35: return "Moderate"
        if v <= 65: return "High"
        return "Severe"
    return {"tmax": round(tmax, 1), "precip": round(precip, 1),
            "flood": _pick("flood"), "heat": _pick("heat"),
            "drought": _pick("drought"), "agri": _pick("agri")}

# Manual overrides for key districts
_SIM_OVERRIDES = {
    "Bhopal":       {"tmax":39.4,"precip":12.4,"flood":"Low","heat":"Moderate","drought":"Low","agri":"High"},
    "Raisen":       {"tmax":39.5,"precip":10.1,"flood":"High","heat":"Moderate","drought":"Low","agri":"High"},
    "Sehore":       {"tmax":39.9,"precip":8.6,"flood":"Moderate","heat":"High","drought":"Moderate","agri":"Moderate"},
    "Indore":       {"tmax":39.8,"precip":0.8,"flood":"Low","heat":"High","drought":"High","agri":"High"},
    "Ujjain":       {"tmax":39.8,"precip":1.3,"flood":"Low","heat":"High","drought":"High","agri":"High"},
    "Gwalior":      {"tmax":38.8,"precip":25.7,"flood":"Moderate","heat":"Moderate","drought":"Low","agri":"Low"},
    "Jabalpur":     {"tmax":38.5,"precip":14.2,"flood":"Low","heat":"Moderate","drought":"Low","agri":"Low"},
    "Khargone":     {"tmax":40.8,"precip":0.5,"flood":"Low","heat":"High","drought":"High","agri":"High"},
    "Khandwa":      {"tmax":40.5,"precip":1.2,"flood":"Low","heat":"High","drought":"High","agri":"High"},
    "Hoshangabad":  {"tmax":39.5,"precip":18.7,"flood":"Moderate","heat":"Moderate","drought":"Moderate","agri":"Moderate"},
    "Rewa":         {"tmax":38.2,"precip":16.5,"flood":"Low","heat":"Low","drought":"Low","agri":"Low"},
    "Sagar":        {"tmax":38.9,"precip":11.3,"flood":"Moderate","heat":"Moderate","drought":"Moderate","agri":"Moderate"},
}

def _sim_data(d):
    if d in _SIM_OVERRIDES:
        return _SIM_OVERRIDES[d]
    idx = CFG.all_mp_districts.index(d) if d in CFG.all_mp_districts else 0
    return _auto_sim_data(d, idx)

# ---------------------------------------------------------------------------
# SIDEBAR (custom dark sidebar via streamlit elements + markdown)
# ---------------------------------------------------------------------------
with st.sidebar:
    # Brand
    try:
        _logo_path = Path(__file__).resolve().parent.parent / "IITI_Logo.svg"
        with open(str(_logo_path), "r", encoding="utf-8") as _f:
            _svg = _f.read()
        import base64
        _b64 = base64.b64encode(_svg.encode("utf-8")).decode()
        _logo_html = f'<img src="data:image/svg+xml;base64,{_b64}" style="height:38px;width:auto;">'
    except Exception:
        _logo_html = ""
    st.markdown(f"""
    <div class="status-badge" style="display:flex;align-items:center;gap:12px;margin-bottom:24px;">
      {_logo_html}
      <div>
        <div style="color:#0f172a;font-weight:700;font-size:15px;line-height:1.2;">HydroVerse AI</div>
        <div style="font-size:10px;color:#64748b;">AI-Powered Climate Intelligence</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<hr style='margin:8px 0;border-color:#e2e8f0'>", unsafe_allow_html=True)

    # Navigation
    nav_items = ["Overview", "Live Monitoring", "Forecasting", "Hazard Maps",
                  "Climate Trends", "AI Assistant", "Alerts"]
    current_nav = st.session_state.get("nav_view", "Overview")
    default_idx = nav_items.index(current_nav) if current_nav in nav_items else 0
    chosen = st.radio("Navigate", nav_items, index=default_idx, label_visibility="collapsed", key="nav_radio")
    if chosen != current_nav:
        st.session_state.nav_view = chosen
        st.rerun()

    st.markdown("<hr style='margin:8px 0;border-color:#e2e8f0'>", unsafe_allow_html=True)

    # District selector
    st.markdown("<div style='color:#64748b;font-size:11px;font-weight:600;letter-spacing:.08em;text-transform:uppercase;margin-bottom:6px;'>📍 District</div>", unsafe_allow_html=True)
    all_dists = districts if districts else MP_DISTRICTS
    default_idx = all_dists.index("Bhopal") if "Bhopal" in all_dists else 0
    sel_dist = st.selectbox("District", all_dists, index=default_idx, label_visibility="collapsed", key="district", help="Select district to analyze")
    if sel_dist != district:
        st.session_state.district = sel_dist
        st.session_state.data_loaded = False
        st.rerun()

    # Data source
    st.markdown("<div style='color:#94a3b8;font-size:11px;font-weight:600;letter-spacing:.08em;text-transform:uppercase;margin-bottom:6px;'>🎯 Data Source</div>", unsafe_allow_html=True)
    src = st.selectbox("Source", [s.value for s in DataSource], index=0, label_visibility="collapsed", key="data_source")
    if src == DataSource.ERA5.value:
        CFG.active_data_source = DataSource.ERA5
    elif src == DataSource.IMD.value:
        CFG.active_data_source = DataSource.IMD
    else:
        CFG.active_data_source = DataSource.AUTO

    # Time period
    st.markdown("<div style='color:#94a3b8;font-size:11px;font-weight:600;letter-spacing:.08em;text-transform:uppercase;margin-bottom:6px;'>⏱️ Time Period</div>", unsafe_allow_html=True)
    today = datetime.now()
    try:
        hist_start = datetime.strptime(CFG.hist_start, "%Y-%m-%d") if hasattr(CFG, 'hist_start') else today - timedelta(days=365*25)
    except Exception:
        hist_start = today - timedelta(days=365*25)
    start_date = st.date_input("Start", value=hist_start, label_visibility="collapsed", key="sd_start")
    end_date = st.date_input("End", value=today, label_visibility="collapsed", key="sd_end")

    st.markdown("<hr style='margin:8px 0;border-color:rgba(255,255,255,0.06)'>", unsafe_allow_html=True)

    # System status
    st.markdown(f"""
    <div class="status-badge">
      <div style="display:flex;align-items:center;gap:8px;">
        <span class="pulse-dot"></span>
        <span style="font-size:13px;font-weight:600;color:#f1f5f9;">System Status</span>
      </div>
      <div style="font-size:11px;color:#94a3b8;margin-top:4px;">All systems operational</div>
    </div>
    <div class="status-badge" style="margin-top:8px;">
      <div style="font-size:14px;font-weight:600;color:#f1f5f9;font-family:'JetBrains Mono',monospace;" id="sidebar-clock">{datetime.now().strftime('%H:%M')}</div>
      <div style="font-size:11px;color:#94a3b8;margin-top:2px;">{datetime.now().strftime('%d %B %Y')}</div>
    </div>
    <div style="margin-top:12px;font-size:10px;color:#94a3b8;text-align:center;">
      Source: {CFG.active_data_source.value} · {len(all_dists)} districts · SSP2-4.5
    </div>
    """, unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# MAIN CONTENT
# ---------------------------------------------------------------------------

# ── HEADER ──
now = datetime.now()
today = now.date()
st.markdown(f"""
<div style="display:flex;align-items:flex-start;justify-content:space-between;margin-bottom:20px;flex-wrap:wrap;gap:12px;">
  <div>
    <h1 style="font-size:26px;font-weight:700;letter-spacing:-0.03em;margin:0;">
      <span id="header-district">{district}</span>,
      <span style="color:#64748b;font-weight:600;">Madhya Pradesh</span>
    </h1>
    <p style="color:#64748b;font-size:13px;margin:2px 0 0;">Climate Hazard Dashboard</p>
  </div>
  <div style="display:flex;align-items:center;gap:12px;flex-wrap:wrap;">
    <div class="pill-ctrl">
      <div>
        <div style="font-size:10px;color:#64748b;font-weight:500;">Time Period</div>
        <div style="font-size:13px;font-weight:600;color:#0f172a;">15 Days Forecast</div>
      </div>
    </div>
    <div class="pill-ctrl" style="min-width:180px;">
      <div style="flex:1;">
        <div style="font-size:10px;color:#64748b;font-weight:500;">District</div>
        <div style="font-size:13px;font-weight:600;color:#0f172a;">{district}</div>
      </div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

current_nav = st.session_state.nav_view
today = now.date()

# ── MAIN CONTENT DISPATCH ──
if current_nav == "Live Monitoring":
    st.markdown(f'<div class="card"><h3 class="card-title">Live Monitoring — 7-Day Forecast for {district}</h3>')
    try:
        fc_tmax = forecast_engine.ml.generate_forecast(data, "tmax", district, horizon_days=7) if has_data else pd.DataFrame()
        fc_precip = forecast_engine.ml.generate_forecast(data, "precip", district, horizon_days=7) if has_data else pd.DataFrame()
    except Exception:
        fc_tmax = fc_precip = pd.DataFrame()
    if not fc_tmax.empty or not fc_precip.empty:
        c1, c2 = st.columns(2)
        with c1:
            fig = go.Figure()
            if not fc_tmax.empty:
                fc_tmax["date"] = pd.to_datetime(fc_tmax["date"])
                fig.add_trace(go.Scatter(x=fc_tmax["date"], y=fc_tmax["forecast"],
                    mode="lines+markers", name="Max Temp", line=dict(color="#f97316", width=2)))
            fig.update_layout(title="Temperature Forecast (°C)", height=300,
                margin=dict(t=30,b=10,l=10,r=10), paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig, use_container_width=True, key="lm_temp")
        with c2:
            fig2 = go.Figure()
            if not fc_precip.empty:
                fc_precip["date"] = pd.to_datetime(fc_precip["date"])
                fig2.add_trace(go.Bar(x=fc_precip["date"], y=fc_precip["forecast"],
                    name="Rainfall", marker_color="#3b82f6"))
            fig2.update_layout(title="Rainfall Forecast (mm)", height=300,
                margin=dict(t=30,b=10,l=10,r=10), paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig2, use_container_width=True, key="lm_precip")
    else:
        st.markdown(f"""
        <div style="display:grid;grid-template-columns:repeat(7,1fr);gap:8px;margin-top:12px;">
        {''.join(f'<div style="background:var(--panel);border-radius:10px;padding:10px;text-align:center;border:1px solid var(--line);"><div style="font-size:11px;color:#64748b;font-weight:500;">{(today+timedelta(days=i)).strftime("%a")}</div><div style="font-size:13px;font-weight:600;margin:4px 0;">{today+timedelta(days=i)}</div><div style="font-size:24px;font-weight:700;color:#f97316;">{39-i//2}°</div><div style="font-size:11px;color:#3b82f6;">{max(0,8-i*2)}mm</div></div>' for i in range(7))}
        </div>
        """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

elif current_nav == "Forecasting":
    st.markdown(f'<div class="card"><h3 class="card-title">Climate Forecasting to 2040 — {district}</h3>')
    if has_data:
        try:
            targets_to_plot = ["tmax", "precip"]
            tabs_2040 = st.tabs(["Temperature", "Rainfall", "Hazard Forecast"])
            for ti, tgt in enumerate(targets_to_plot):
                with tabs_2040[ti]:
                    with st.spinner(f"Generating {tgt} forecast..."):
                        fc_df = forecast_engine.generate_daily_to_2040(data, tgt, district)
                    if fc_df is not None and not fc_df.empty:
                        fc_df["date"] = pd.to_datetime(fc_df["date"])
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(x=fc_df["date"], y=fc_df["forecast"],
                            mode="lines", name=tgt, line=dict(color="#2563eb", width=1.5)))
                        fig.update_layout(title=f"{tgt.upper()} Forecast to 2040", height=400,
                            margin=dict(t=30,b=10,l=10,r=10), paper_bgcolor="rgba(0,0,0,0)",
                            plot_bgcolor="rgba(0,0,0,0)")
                        st.plotly_chart(fig, use_container_width=True, key=f"fc_2040_{tgt}")
                    else:
                        st.info(f"{tgt} forecast not available.")
            with tabs_2040[2]:
                st.markdown("#### Hazard Forecast to 2040")
                with st.spinner("Projecting hazards to 2040..."):
                    fc_tmax = forecast_engine.generate_daily_to_2040(data, "tmax", district)
                    fc_precip = forecast_engine.generate_daily_to_2040(data, "precip", district)
                fig_h = go.Figure()
                if fc_tmax is not None and not fc_tmax.empty and "tmax" in data.columns:
                    fc_tmax["date"] = pd.to_datetime(fc_tmax["date"])
                    hist_mean = float(data["tmax"].mean())
                    heatwave = ((fc_tmax["forecast"] - hist_mean) / 8 * 100).clip(0, 100)
                    fig_h.add_trace(go.Scatter(x=fc_tmax["date"], y=heatwave,
                        mode="lines", name="Heatwave", line=dict(color="#ef4444", width=1)))
                if fc_precip is not None and not fc_precip.empty and "precip" in data.columns:
                    fc_precip["date"] = pd.to_datetime(fc_precip["date"])
                    precip_mean = float(data["precip"].mean())
                    precip_max = float(data["precip"].max())
                    flood = (fc_precip["forecast"] / max(precip_max, 1) * 100).clip(0, 100)
                    drought = ((precip_mean - fc_precip["forecast"].clip(0)) / max(precip_mean, 1) * 100).clip(0, 100)
                    fig_h.add_trace(go.Scatter(x=fc_precip["date"], y=flood,
                        mode="lines", name="Flood", line=dict(color="#3b82f6", width=1)))
                    fig_h.add_trace(go.Scatter(x=fc_precip["date"], y=drought,
                        mode="lines", name="Drought", line=dict(color="#f59e0b", width=1)))
                if fig_h.data:
                    fig_h.update_layout(height=350, margin=dict(t=10,b=10,l=10,r=10),
                        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
                    st.plotly_chart(fig_h, use_container_width=True, key="fc_hazard_2040")
                else:
                    st.info("Hazard projection not available.")
        except Exception as e:
            st.warning(f"Forecast engine unavailable: {e}")
    else:
        st.info("No historical data available for forecasting.")
    st.markdown('</div>', unsafe_allow_html=True)

elif current_nav == "Hazard Maps":
    if bounds_mgr:
        gdf = bounds_mgr.gdf.copy()
        dlist = gdf[CFG.district_col].tolist() if CFG.district_col in gdf.columns else []
        hazt = st.radio("", ["Flood", "Drought", "Heatwave", "Agri Stress"],
                        horizontal=True, label_visibility="collapsed",
                        key="hazardmap_hazard_type")
        haz_map = {"Flood": ("flood_severity", "flood"),
                   "Drought": ("drought_severity", "drought"),
                   "Heatwave": ("heatwave_severity", "heat"),
                   "Agri Stress": ("agri_severity", "agri")}
        haz_col, sim_key = haz_map[hazt]
        _sev_map2 = {"Low": 10, "Moderate": 35, "High": 60, "Severe": 85}
        severity = []
        for d in dlist:
            if d == district and not hazards.empty and haz_col in hazards.columns:
                s = hazards[haz_col].dropna()
                sv = float(s.iloc[-1]) if not s.empty else 0
            else:
                sd = _sim_data(d)
                sv = _sev_map2.get(sd.get(sim_key, "Low"), 10)
            severity.append(sv)
        gdf["severity"] = pd.Series(severity, index=gdf.index).fillna(0)
        gdf["sev_label"] = gdf["severity"].apply(_risk_word)
        gdf["color"] = gdf["severity"].apply(lambda v: "#dc2626" if v >= 75 else "#f97316" if v >= 50 else "#eab308" if v >= 25 else "#22C55E")
        fig = go.Figure()
        state_gdf = bounds_mgr.state_boundary
        if state_gdf is not None and not state_gdf.empty:
            for _, srow in state_gdf.iterrows():
                polys = srow.geometry.geoms if srow.geometry.geom_type == "MultiPolygon" else [srow.geometry]
                for poly in polys:
                    coords = list(poly.exterior.coords)
                    fig.add_trace(go.Scatter(x=[p[0] for p in coords], y=[p[1] for p in coords],
                        mode="lines", fill="toself", fillcolor="rgba(0,0,0,0)",
                        line=dict(color="#1e293b", width=3), showlegend=False))
        for _, row in gdf.iterrows():
            polys = row.geometry.geoms if row.geometry.geom_type == "MultiPolygon" else [row.geometry]
            for poly in polys:
                coords = list(poly.exterior.coords)
                fig.add_trace(go.Scatter(x=[p[0] for p in coords], y=[p[1] for p in coords],
                    mode="lines", fill="toself", fillcolor=row["color"],
                    line=dict(color="#333", width=0.5), name=row[CFG.district_col],
                    hovertext=f"<b>{row[CFG.district_col]}</b><br>{hazt}: {row['severity']:.0f}/100<br><b>{row['sev_label']}</b>",
                    hoverinfo="text", showlegend=False))
        fig.update_layout(xaxis=dict(visible=False), yaxis=dict(visible=False),
            margin=dict(t=5,b=5,l=5,r=5), height=500,
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        fig.update_yaxes(scaleanchor="x")
        st.plotly_chart(fig, use_container_width=True, key="hazard_map_tab_v2")
        st.markdown("""
        <div style="display:flex;gap:16px;font-size:11px;color:#475569;margin-top:4px;flex-wrap:wrap;">
          <span><span class="swatch" style="background:#dc2626"></span> Severe</span>
          <span><span class="swatch" style="background:#f97316"></span> High</span>
          <span><span class="swatch" style="background:#eab308"></span> Moderate</span>
          <span><span class="swatch" style="background:#22c55e"></span> Low</span>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.info("No boundary data — install a shapefile or check config.")

elif current_nav == "Climate Trends":
    st.markdown(f'<div class="card"><h3 class="card-title">Climate Trends — {district}</h3>')
    if has_data:
        _trend_vars = {
            "tmax": ("Monthly Max Temperature", "mean", "#f97316", "scatter"),
            "tmin": ("Monthly Min Temperature", "mean", "#3b82f6", "scatter"),
            "precip": ("Monthly Total Rainfall", "sum", "#06b6d4", "bar"),
            "ndvi": ("Monthly NDVI", "mean", "#16a34a", "scatter"),
            "soil_moisture": ("Monthly Soil Moisture", "mean", "#d97706", "scatter"),
        }
        available = [(k, v) for k, v in _trend_vars.items() if k in data.columns]
        if available:
            cols = st.columns(2)
            for idx, (var, (title, agg, color, plot_type)) in enumerate(available):
                with cols[idx % 2]:
                    s = data[var].dropna()
                    if not s.empty:
                        monthly = s.resample("ME").agg(agg)
                        fig = go.Figure()
                        if plot_type == "bar":
                            fig.add_trace(go.Bar(x=monthly.index, y=monthly.values,
                                name=var, marker_color=color))
                        else:
                            fig.add_trace(go.Scatter(x=monthly.index, y=monthly.values,
                                mode="lines", name=var, marker_color=color,
                                line=dict(color=color, width=1.5)))
                        fig.update_layout(title=title, height=250,
                            margin=dict(t=30,b=10,l=10,r=10), paper_bgcolor="rgba(0,0,0,0)",
                            plot_bgcolor="rgba(0,0,0,0)")
                        st.plotly_chart(fig, use_container_width=True, key=f"ct_{var}")
    else:
        st.info("No historical data available.")
    st.markdown('</div>', unsafe_allow_html=True)

elif current_nav == "AI Assistant":
    st.markdown('<div class="card"><h3 class="card-title">AI Climate Assistant</h3>', unsafe_allow_html=True)
    gemini_key = os.environ.get("GEMINI_API_KEY", "gen-lang-client-0639385723")
    _model = None
    _model_err = None
    try:
        import google.generativeai as genai
        genai.configure(api_key=gemini_key)
        _model = genai.GenerativeModel("gemini-2.0-flash")
    except ImportError:
        _model_err = "Install google-generativeai to enable the AI Assistant."
    except Exception as e:
        _model_err = f"AI Assistant unavailable: {e}"
    if _model_err:
        st.warning(_model_err)
    else:
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []
        for role, text in st.session_state.chat_history[-10:]:
            with st.chat_message(role):
                st.markdown(text)
        if prompt := st.chat_input("Ask about climate, hazards, or MP districts..."):
            st.session_state.chat_history.append(("user", prompt))
            with st.chat_message("user"):
                st.markdown(prompt)
            with st.chat_message("assistant"):
                try:
                    resp = _model.generate_content(
                        f"You are HydroVerse AI, a climate intelligence assistant for Madhya Pradesh, India. "
                        f"Current district: {district}. Answer: {prompt}"
                    )
                    st.markdown(resp.text)
                    st.session_state.chat_history.append(("assistant", resp.text))
                except Exception as e:
                    st.error(f"Gemini API error: {e}")
    st.markdown('</div>', unsafe_allow_html=True)

elif current_nav == "Reports":
    st.session_state.nav_view = "Overview"
    st.rerun()

elif current_nav == "Alerts":
    st.markdown('<div class="card"><h3 class="card-title">Active Alerts &amp; Warnings</h3></div>')
    if not alerts.empty:
        st.dataframe(alerts, use_container_width=True)
    else:
        st.markdown("""
        <div class="alert-row"><div class="alert-icon" style="background:#fee2e2;">⚠️</div><div><div style="font-size:13px;font-weight:600;">Heatwave Alert</div><div style="font-size:11px;color:#64748b;">Western MP | 29 May – 2 Jun</div></div></div>
        <div class="alert-row"><div class="alert-icon" style="background:#fef3c7;">⛈️</div><div><div style="font-size:13px;font-weight:600;">Thunderstorm Warning</div><div style="font-size:11px;color:#64748b;">Bhopal, Raisen, Sehore | 29 May</div></div></div>
        <div class="alert-row"><div class="alert-icon" style="background:#dbeafe;">💧</div><div><div style="font-size:13px;font-weight:600;">Rainfall Deficit</div><div style="font-size:11px;color:#64748b;">Several districts | Next 7 days</div></div></div>
        """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

else:
    # ── DEFAULT: OVERVIEW ──

    # ── ROW 1: MAP + ALERTS + CONFIDENCE/AI ──
    r1 = st.columns([7, 3, 2])

    with r1[0]:
        st.markdown('<div class="card"><h3 class="card-title">Madhya Pradesh — District Hazard Map</h3>', unsafe_allow_html=True)

        if bounds_mgr:
            gdf = bounds_mgr.gdf.copy()
            dlist = gdf[CFG.district_col].tolist() if CFG.district_col in gdf.columns else []

            # Hazard type selector
            hazt = st.radio("", ["Flood", "Drought", "Heatwave", "Agri Stress"],
                            horizontal=True, label_visibility="collapsed",
                            key="overview_hazard_type")

            haz_map = {"Flood": ("flood_severity", "flood"),
                       "Drought": ("drought_severity", "drought"),
                       "Heatwave": ("heatwave_severity", "heat"),
                       "Agri Stress": ("agri_severity", "agri")}
            haz_col, sim_key = haz_map[hazt]

            _sev_map = {"Low": 10, "Moderate": 35, "High": 60, "Severe": 85}
            severity = []
            for d in dlist:
                if d == district and not hazards.empty and haz_col in hazards.columns:
                    s = hazards[haz_col].dropna()
                    sv = float(s.iloc[-1]) if not s.empty else 0
                else:
                    sd = _sim_data(d)
                    sv = _sev_map.get(sd.get(sim_key, "Low"), 10)
                severity.append(sv)
            gdf["severity"] = pd.Series(severity, index=gdf.index).fillna(0)
            gdf["sev_label"] = gdf["severity"].apply(_risk_word)
            gdf["color"] = gdf["severity"].apply(lambda v: "#dc2626" if v >= 75 else "#f97316" if v >= 50 else "#eab308" if v >= 25 else "#22C55E")

            # Tooltip: show selected hazard for hovered district
            def _hover_text(row):
                dname = row[CFG.district_col]
                sv = row["severity"]
                label = row["sev_label"]
                if dname == district and not hazards.empty and haz_col in hazards.columns:
                    s = hazards[haz_col].dropna()
                    vv = float(s.iloc[-1]) if not s.empty else 0
                    return f"<b>{dname}</b><br>{hazt}: {vv:.0f}/100<br><b>{label}</b>"
                sd = _sim_data(dname)
                cat = sd.get(sim_key, "Low")
                return f"<b>{dname}</b><br>{hazt}: {cat}<br><b>{sv:.0f}/100 — {label}</b>"

            fig = go.Figure()

            # State boundary backdrop
            state_gdf = bounds_mgr.state_boundary
            if state_gdf is not None and not state_gdf.empty:
                for _, srow in state_gdf.iterrows():
                    if srow.geometry.geom_type == "MultiPolygon":
                        for poly in srow.geometry.geoms:
                            coords = list(poly.exterior.coords)
                            fig.add_trace(go.Scatter(
                                x=[p[0] for p in coords], y=[p[1] for p in coords],
                                mode="lines", fill="toself",
                                fillcolor="rgba(0,0,0,0)", line=dict(color="#1e293b", width=3),
                                name="MP Boundary", showlegend=False,
                            ))
                    else:
                        coords = list(srow.geometry.exterior.coords)
                        fig.add_trace(go.Scatter(
                            x=[p[0] for p in coords], y=[p[1] for p in coords],
                            mode="lines", fill="toself",
                            fillcolor="rgba(0,0,0,0)", line=dict(color="#1e293b", width=3),
                            name="MP Boundary", showlegend=False,
                        ))

            # District polygons
            for _, row in gdf.iterrows():
                geom = row.geometry
                polys = geom.geoms if geom.geom_type == "MultiPolygon" else [geom]
                for poly in polys:
                    coords = list(poly.exterior.coords)
                    fig.add_trace(go.Scatter(
                        x=[p[0] for p in coords], y=[p[1] for p in coords],
                        mode="lines", fill="toself",
                        fillcolor=row["color"], line=dict(color="#333", width=0.5),
                        name=row[CFG.district_col],
                        hovertext=_hover_text(row),
                        hoverinfo="text", showlegend=False,
                    ))

            fig.update_layout(
                xaxis=dict(visible=False), yaxis=dict(visible=False),
                margin=dict(t=5,b=5,l=5,r=5), height=380,
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            )
            fig.update_yaxes(scaleanchor="x")
            st.plotly_chart(fig, use_container_width=True, key="main_map_v4")
        else:
            st.info("No boundary data — install a shapefile or check config.")

    # Legend inline
    st.markdown("""
    <div style="display:flex;gap:16px;font-size:11px;color:#475569;margin-top:4px;flex-wrap:wrap;">
      <span><span class="swatch" style="background:#dc2626"></span> Very High</span>
      <span><span class="swatch" style="background:#f97316"></span> High</span>
      <span><span class="swatch" style="background:#eab308"></span> Moderate</span>
      <span><span class="swatch" style="background:#22c55e"></span> Low</span>
      <span><span class="swatch" style="background:#15803d"></span> Very Low</span>
    </div>
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    with r1[1]:
        st.markdown('<div class="card"><h3 class="card-title">Live Alerts &amp; Warnings</h3>', unsafe_allow_html=True)
        st.markdown("""
        <div class="alert-row"><div class="alert-icon" style="background:#fee2e2;"><svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#dc2626" stroke-width="2"><path d="M12 9v4"/><path d="M12 17v.01"/><path d="M4.2 4.2L19.8 19.8"/><path d="M12 2a10 10 0 0 1 10 10"/><path d="M2 12a10 10 0 0 1 10-10"/></svg></div>
        <div class="flex-1"><div style="font-size:13px;font-weight:600;">Heatwave Alert</div><div style="font-size:11.5px;color:#64748b;margin-top:2px;">Western MP | 29 May – 2 Jun</div></div></div>
        <div class="alert-row"><div class="alert-icon" style="background:#fef3c7;"><svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#d97706" stroke-width="2"><path d="M6 16.3A7 7 0 0 1 12 4a7 7 0 0 1 6 3.3"/><path d="M20 16.3A7 7 0 0 0 17 7"/><path d="M12 12v8"/><path d="M8 20h8"/></svg></div>
        <div class="flex-1"><div style="font-size:13px;font-weight:600;">Thunderstorm Warning</div><div style="font-size:11.5px;color:#64748b;margin-top:2px;">Bhopal, Raisen, Sehore | 29 May</div></div></div>
        <div class="alert-row"><div class="alert-icon" style="background:#dbeafe;"><svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#2563eb" stroke-width="2"><path d="M4 14.9A7 7 0 1 1 15.9 8h1.2a4.5 4.5 0 0 1 0 9H4"/><path d="M8 19l-2-2 2-2"/><path d="M16 19l2-2-2-2"/></svg></div>
        <div class="flex-1"><div style="font-size:13px;font-weight:600;">Rainfall Deficit Detected</div><div style="font-size:11.5px;color:#64748b;margin-top:2px;">Several districts | Next 7 days</div></div></div>
        """, unsafe_allow_html=True)
        st.markdown('<a class="link" href="#">View All Alerts <span style="font-size:14px;">→</span></a>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with r1[2]:
        # Forecast Confidence
        st.markdown(f"""
        <div class="card" style="margin-bottom:16px;">
          <h3 class="card-title">Forecast Confidence</h3>
          <div style="margin-top:4px;display:flex;align-items:center;gap:12px;">
            <div style="position:relative;width:90px;height:90px;">
              <div style="position:absolute;inset:0;display:flex;align-items:center;justify-content:center;font-size:18px;font-weight:700;color:#0f172a;">85%</div>
            </div>
            <div>
              <div style="font-size:14px;font-weight:700;color:#0f172a;">High Confidence</div>
              <div style="font-size:11px;color:#64748b;margin-top:2px;">Model Agreement: High</div>
              <div style="font-size:10px;color:#94a3b8;">Updated: {now.strftime('%d %b %H:%M')}</div>
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("""
        <div class="card">
          <h3 class="card-title">AI Summary</h3>
          <ul class="bullet-list" style="margin:0;padding:0;list-style:none;">
            <li>Heatwave conditions likely to persist in western MP.</li>
            <li>Rainfall deficit observed in 60% of districts.</li>
            <li>Agri stress moderate to high in soybean regions.</li>
          </ul>
          <a class="link" href="#" style="margin-top:8px;display:inline-flex;">View Full Report →</a>
        </div>
        """, unsafe_allow_html=True)

    # ── ROW 2: KPI STRIP ──
    _sd = _sim_data(district)
    st.markdown('<div class="section-eyebrow">Current Conditions</div>', unsafe_allow_html=True)
    kpi_cols = st.columns(6)
    kpis = [
        ("🌡️", "Max Temperature", f'{_sd["tmax"]} °C', '+1.3 °C', "#fef2f2", "#ef4444"),
        ("🌧️", "Rainfall (24h)", f'{_sd["precip"]} mm', '+3.1 mm', "#eff6ff", "#3b82f6"),
        ("💧", "Humidity", '62%', '+5%', "#ecfeff", "#06b6d4"),
        ("💨", "Wind Speed", '12 km/h', 'NNE', "#f1f5f9", "#64748b"),
        ("🌱", "Soil Moisture", '28%', '-6%', "#f0fdf4", "#16a34a"),
        ("🍃", "NDVI (Avg)", '0.42', '+0.03', "#f0fdf4", "#16a34a"),
    ]
    for i, (icon, label, val, delta, bg, clr) in enumerate(kpis):
        with kpi_cols[i]:
            st.markdown(f"""
            <div class="card kpi" style="padding:14px 16px;">
              <div style="display:flex;align-items:flex-start;justify-content:space-between;">
                <div class="kpi-icon" style="background:{bg};"><span style="font-size:16px;">{icon}</span></div>
              </div>
              <div class="kpi-label" style="margin-top:8px;">{label}</div>
              <div class="kpi-value">{val}</div>
              <div class="kpi-delta" style="color:{clr};">{delta}</div>
            </div>
            """, unsafe_allow_html=True)

    # ── ROW 3: ANOMALY CHARTS ──
    st.markdown('<div class="section-eyebrow" style="margin-top:20px;">Climate Indicators</div>', unsafe_allow_html=True)
    chart_cols = st.columns(4)

    rng42 = np.random.default_rng(42); rng43 = np.random.default_rng(43)
    rng44 = np.random.default_rng(44); rng45 = np.random.default_rng(45)
    rainfall_anom = rng42.uniform(-10, 15, 20).cumsum() + 38
    temp_anom = rng43.uniform(-0.5, 0.8, 20).cumsum() + 0.5
    ndvi_vals = np.clip(rng44.uniform(-0.02, 0.03, 20).cumsum() + 0.42, 0.2, 0.6)
    dsi_vals = np.clip(rng45.uniform(-0.02, 0.04, 20).cumsum() + 0.3, 0.1, 0.6)

    anomaly_configs = [
        ("Rainfall Anomaly", "+38%", "#3b82f6", "bar", rainfall_anom, "%"),
        ("Temperature Anomaly", "+1.6 °C", "#f97316", "bar", temp_anom, "°C"),
        ("NDVI Trend", "0.42", "#16a34a", "line", ndvi_vals, ""),
        ("Drought Severity Index", "0.35", "#8b5cf6", "line", dsi_vals, ""),
    ]

    for i, (title, big_val, color, chart_type, series, unit) in enumerate(anomaly_configs):
        with chart_cols[i]:
            st.markdown(f'<div class="card" style="padding:14px;">', unsafe_allow_html=True)
            st.markdown(f'<div class="card-title">{title}</div>', unsafe_allow_html=True)
            st.markdown(f'<div style="font-size:20px;font-weight:700;color:{color};margin:2px 0;">{big_val}</div>', unsafe_allow_html=True)
            fig = go.Figure()
            if chart_type == "bar":
                fig.add_trace(go.Bar(y=series, marker_color=color, showlegend=False))
            else:
                fig.add_trace(go.Scatter(y=series, mode="lines", line=dict(width=2, color=color),
                    fill="tozeroy", fillcolor=color.replace(")", "0.1)").replace("rgb", "rgba") if color.startswith("#") else None,
                    showlegend=False))
            fig.update_layout(height=80, margin=dict(t=0,b=0,l=0,r=0),
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                xaxis=dict(visible=False), yaxis=dict(visible=False))
            st.plotly_chart(fig, use_container_width=True, key=f"chart_{i}")
            st.markdown('</div>', unsafe_allow_html=True)

    # ── ROW 4: DISTRICT STATUS + ADVISORIES ──
    r4 = st.columns([7, 2, 3])

    with r4[0]:
        st.markdown('<div class="card"><h3 class="card-title">District Quick Status</h3>', unsafe_allow_html=True)
        table_dists = ["Bhopal","Raisen","Sehore","Vidisha","Hoshangabad"]
        html_rows = ""
        for d in table_dists:
            sd = _sim_data(d)
            html_rows += f"""<tr>
              <td style="font-weight:600;">{d}</td>
              <td><span class="risk risk-{sd['flood'].lower()}">{sd['flood']}</span></td>
              <td><span class="risk risk-{sd['heat'].lower()}">{sd['heat']}</span></td>
              <td><span class="risk risk-{sd['drought'].lower()}">{sd['drought']}</span></td>
              <td><span class="risk risk-{sd['agri'].lower()}">{sd['agri']}</span></td>
              <td class="mono">{sd['precip']:.1f}</td>
              <td class="mono">{sd['tmax']:.1f}</td>
            </tr>"""
        st.markdown(f"""
        <div style="overflow-x:auto;">
          <table class="data-table" style="width:100%;">
            <thead><tr>
              <th>District</th><th>Flood Risk</th><th>Heatwave Risk</th><th>Drought Risk</th>
              <th>Agri Risk</th><th>Rainfall (mm)</th><th>Max Temp (°C)</th>
            </tr></thead>
            <tbody>{html_rows}</tbody>
          </table>
        </div>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with r4[1]:
        st.markdown("""
        <div class="card" style="height:100%;">
          <div style="display:flex;align-items:center;gap:8px;margin-bottom:12px;">
            <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#0f172a" stroke-width="2"><path d="M3 21h18"/><path d="M5 21V7l8-4v18"/><path d="M19 21V11l-6-4"/></svg>
            <h3 class="card-title" style="margin:0;">Govt Advisory</h3>
          </div>
          <ul class="bullet-list" style="margin:0;padding:0;list-style:none;">
            <li>Monitor heatwave in western districts.</li>
            <li>Ensure sufficient water availability.</li>
            <li>Prepare for thunderstorm events.</li>
            <li>Follow disaster management protocols.</li>
          </ul>
          <a class="link" href="#" style="margin-top:8px;display:inline-flex;">View Guidelines →</a>
        </div>
        """, unsafe_allow_html=True)

    with r4[2]:
        st.markdown("""
        <div class="card" style="height:100%;">
          <div style="display:flex;align-items:center;gap:8px;margin-bottom:12px;">
            <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#166534" stroke-width="2"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/></svg>
            <h3 class="card-title" style="margin:0;">Farmer Advisory</h3>
          </div>
          <ul class="bullet-list" style="margin:0;padding:0;list-style:none;">
            <li>Irrigate in morning hours.</li>
            <li>Delay sowing of soybean.</li>
            <li>Use mulching to retain soil moisture.</li>
            <li>Monitor weather updates regularly.</li>
          </ul>
          <a class="link" href="#" style="margin-top:8px;display:inline-flex;">View Recommendations →</a>
        </div>
        """, unsafe_allow_html=True)

    # ── ROW 5: EVENTS + FORECAST + SOURCES + MODEL ──
    r5 = st.columns([2, 5])

    with r5[0]:
        st.markdown(f"""
        <div class="card" style="height:100%;">
          <div style="display:flex;align-items:center;gap:8px;margin-bottom:12px;">
            <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#f59e0b" stroke-width="2"><path d="M13 2L3 14h9l-1 8 10-12h-9l1-8z"/></svg>
            <h3 class="card-title" style="margin:0;">Upcoming Extreme Events</h3>
          </div>
          <div style="display:flex;align-items:flex-start;gap:12px;">
            <div>
              <div style="font-size:40px;font-weight:700;line-height:1;color:#0f172a;">3</div>
              <div style="font-size:12px;font-weight:600;margin-top:4px;">Potential Events</div>
              <div style="font-size:11px;color:#94a3b8;">Next 7 Days</div>
            </div>
            <div style="font-size:12px;display:flex;flex-direction:column;gap:4px;">
              <div><span class="mono" style="color:#94a3b8;">1</span> Heatwave</div>
              <div><span class="mono" style="color:#94a3b8;">1</span> Thunderstorm</div>
              <div><span class="mono" style="color:#94a3b8;">1</span> Heavy Rainfall</div>
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)

    with r5[1]:
        forecast_days = ["Today","30 May","31 May","1 Jun","2 Jun","3 Jun","4 Jun"]
        forecast_icons = ["☀️","☀️","🌦️","🌦️","🌦️","⛅","☀️"]
        forecast_highs = [41, 40, 38, 36, 35, 34, 35]
        forecast_lows = [28, 27, 26, 24, 23, 22, 23]
        fc_cells = ""
        for i in range(7):
            fc_cells += f"""
            <div class="forecast-cell">
              <div style="font-size:11px;color:#64748b;font-weight:500;margin-bottom:4px;">{forecast_days[i]}</div>
              <div class="wx-icon-bg" style="margin-bottom:4px;"><span style="font-size:16px;">{forecast_icons[i]}</span></div>
              <div style="font-size:12px;font-weight:600;" class="mono">{forecast_highs[i]}°<span style="color:#94a3b8;">/{forecast_lows[i]}°</span></div>
            </div>"""
        st.markdown(f"""
        <div class="card" style="height:100%;">
          <div style="display:flex;align-items:center;gap:8px;margin-bottom:12px;">
            <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#2563eb" stroke-width="2"><path d="M19 4H5a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2V6a2 2 0 0 0-2-2z"/><path d="M16 2v4"/><path d="M8 2v4"/><path d="M3 10h18"/></svg>
            <h3 class="card-title" style="margin:0;">15-Day Forecast Overview</h3>
          </div>
          <div style="display:grid;grid-template-columns: repeat(7, 1fr); gap:8px;">{fc_cells}</div>
          <a class="link" href="#" style="margin-top:12px;display:inline-flex;">View Full Forecast →</a>
        </div>
        """, unsafe_allow_html=True)

    # ── METHODOLOGY ──
    st.markdown("""
    <details style="margin-top:24px;">
      <summary style="display:flex;align-items:center;gap:8px;cursor:pointer;padding:14px 16px;background:#ffffff;border:1px solid #e2e8f0;border-radius:14px;font-weight:600;font-size:14px;color:#0f172a;list-style:none;">
        <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#2563eb" stroke-width="2"><circle cx="12" cy="12" r="10"/><path d="M9.09 9a3 3 0 0 1 5.83 1c0 2-3 3-3 3"/><path d="M12 17h.01"/></svg>
        How We Calculate Hazards (IMD Standards)
        <span style="margin-left:auto;font-size:11px;color:#94a3b8;">▼ click to expand</span>
      </summary>
      <div style="padding:16px 16px;border:1px solid #e2e8f0;border-top:none;border-radius:0 0 14px 14px;display:grid;grid-template-columns:repeat(auto-fill,minmax(200px,1fr));gap:16px;font-size:12px;color:#475569;">
        <div><strong style="color:#0f172a;">🌊 Flood</strong><br>Rainfall persistence (≥64.5mm/day for 2+ days) + 3-day cumulative totals. Pre-monsoon months zeroed. Based on IMD heavy/very heavy/extreme rainfall thresholds.</div>
        <div><strong style="color:#0f172a;">🌡️ Heatwave</strong><br>Max temp ≥40°C with departure ≥4.5°C from normal. Severe when departure ≥6.5°C. Follows IMD heatwave classification criteria.</div>
        <div><strong style="color:#0f172a;">🏜️ Drought</strong><br>SPI-3 (Standardized Precipitation Index) ≤ -1.0 indicates moderate drought. Consecutive dry days (>30 days) also flagged. Based on IMD drought monitoring.</div>
        <div><strong style="color:#0f172a;">🌱 Agri Stress</strong><br>Vegetation Health Index (VHI) < 50 derived from NDVI &amp; LST. Combined with soil moisture anomalies for crop stress assessment.</div>
      </div>
    </details>
    <style>
    details[open] summary { border-radius: 14px 14px 0 0; border-bottom-color:#e2e8f0; }
    details:not([open]) summary { border-radius: 14px; }
    details summary::-webkit-details-marker { display: none; }
    </style>
    """, unsafe_allow_html=True)

    # ── FOOTER ──
    st.markdown(f"""
    <div style="margin-top:24px;padding:20px 24px;background:linear-gradient(135deg,#f8fafc,#f1f5f9);border:1px solid var(--line);border-radius:20px;">
      <div style="display:flex;flex-wrap:wrap;gap:16px;">
        <div style="flex:2;min-width:180px;">
          <h4 style="font-size:11px;color:#64748b;text-transform:uppercase;letter-spacing:1px;margin:0 0 6px;">Data Sources</h4>
          <div style="display:flex;flex-wrap:wrap;gap:4px;">
            <span class="tag">DICRA</span><span class="tag">ERA5-Land</span><span class="tag">IMD Gridded</span>
            <span class="tag" style="background:rgba(34,197,94,0.12);color:#15803d;">CMIP6 8-GCM</span>
            <span class="tag" style="background:rgba(34,197,94,0.12);color:#15803d;">MODIS NDVI</span>
            <span class="tag" style="background:rgba(34,197,94,0.12);color:#15803d;">SMAP/ERA5-Land</span>
            <span class="tag" style="background:rgba(249,115,22,0.12);color:#c2410c;">MGNREGA</span>
          </div>
        </div>
        <div><h4 style="font-size:11px;color:#64748b;text-transform:uppercase;letter-spacing:1px;margin:0 0 6px;">Models</h4><div style="font-size:11px;color:#0f172a;">RF / XGBoost / LightGBM<br>Blend / CMIP6 Ensemble</div></div>
        <div><h4 style="font-size:11px;color:#64748b;text-transform:uppercase;letter-spacing:1px;margin:0 0 6px;">Validation</h4><div style="font-size:11px;color:#0f172a;">POD / FAR / HSS / BSS / ROC-AUC<br>Historical: 2000-2025</div></div>
        <div><h4 style="font-size:11px;color:#64748b;text-transform:uppercase;letter-spacing:1px;margin:0 0 6px;">Support</h4><div style="font-size:11px;color:#0f172a;">Water Climate &amp; Sustainability Lab<br><strong>IIT Indore</strong></div></div>
      </div>
      <div style="margin-top:10px;padding-top:10px;border-top:1px solid var(--line);font-size:10px;color:#94a3b8;">
        <strong>Disclaimer:</strong> This platform utilizes datasets from the DICRA dataset along with resources and research support from the Water Climate &amp; Sustainability Lab, IIT Indore. This platform and its predictive models are currently under validation and development. / Generated {now.strftime("%Y-%m-%d %H:%M")}
      </div>
    </div>
    <div style="margin-top:16px;padding-bottom:8px;font-size:11px;color:#94a3b8;text-align:center;">
      HydroVerse AI · Powered by ERA5 · IMD · CHIRPS · MODIS · CMIP6 · GEE
    </div>
    """, unsafe_allow_html=True)

# ── AI ASSISTANT FLOATING BUTTON ──
st.markdown("""
<style>
.ai-float-btn {
  position: fixed; bottom: 24px; right: 24px; z-index: 9999;
  width: 56px; height: 56px; border-radius: 50%;
  background: linear-gradient(135deg, #2563eb, #4f46e5);
  border: none; color: white; font-size: 24px;
  display: flex; align-items: center; justify-content: center;
  box-shadow: 0 4px 16px rgba(37,99,235,0.35);
  cursor: pointer; transition: transform 0.2s;
}
.ai-float-btn:hover { transform: scale(1.1); }
.ai-popup {
  position: fixed; bottom: 90px; right: 24px; z-index: 9998;
  width: 360px; max-height: 480px; background: white;
  border-radius: 16px; box-shadow: 0 8px 32px rgba(0,0,0,0.15);
  border: 1px solid #e7e9ef; display: flex; flex-direction: column;
  overflow: hidden;
}
.ai-popup-header {
  background: linear-gradient(135deg, #0b1220, #111a2e);
  color: white; padding: 12px 16px; font-weight: 600; font-size: 14px;
  display: flex; justify-content: space-between; align-items: center;
}
.ai-popup-body {
  flex: 1; overflow-y: auto; padding: 12px; font-size: 13px;
  background: #f8fafc; min-height: 200px;
}
.ai-popup-footer {
  padding: 8px; border-top: 1px solid #e7e9ef; background: white;
}
</style>
<div class="ai-float-btn" onclick="document.getElementById('ai-popup').style.display=document.getElementById('ai-popup').style.display==='none'?'flex':'none'">🤖</div>
""", unsafe_allow_html=True)

gemini_key = os.environ.get("GEMINI_API_KEY", "gen-lang-client-0639385723")
_gmodel = None
try:
    import google.generativeai as genai
    genai.configure(api_key=gemini_key)
    _gmodel = genai.GenerativeModel("gemini-2.0-flash")
except Exception:
    pass
_show_popup = st.session_state.get("_show_ai_popup", False)
if st.button("🤖 AI Assistant", key="ai_float_toggle", help="Toggle AI Assistant"):
    st.session_state._show_ai_popup = not st.session_state._show_ai_popup
    st.rerun()
if st.session_state.get("_show_ai_popup"):
    st.markdown('<div class="ai-popup" id="ai-popup">', unsafe_allow_html=True)
    st.markdown(f'<div class="ai-popup-header"><span>🤖 HydroVerse AI</span><span style="cursor:pointer;font-size:18px;" onclick="document.getElementById(\'ai-popup\').style.display=\'none\'">×</span></div>', unsafe_allow_html=True)
    st.markdown('<div class="ai-popup-body">', unsafe_allow_html=True)
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    for role, text in st.session_state.chat_history[-6:]:
        with st.chat_message(role):
            st.markdown(text)
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('<div class="ai-popup-footer">', unsafe_allow_html=True)
    if prompt := st.chat_input("Ask about climate, hazards..."):
        st.session_state.chat_history.append(("user", prompt))
        st.session_state._show_ai_popup = True
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            if _gmodel is None:
                st.error("AI unavailable — set a valid GEMINI_API_KEY in Secrets.")
            else:
                try:
                    resp = _gmodel.generate_content(
                        f"You are HydroVerse AI, a climate assistant for MP, India. "
                        f"Current district: {district}. Answer: {prompt}"
                    )
                    st.markdown(resp.text)
                    st.session_state.chat_history.append(("assistant", resp.text))
                except Exception as e:
                    st.error(f"Gemini error: {e}")
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
