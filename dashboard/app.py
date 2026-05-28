#!/usr/bin/env python
"""HydroVerse AI — AI-Powered Climate & Water Intelligence Platform"""
import sys, os
from pathlib import Path
from datetime import datetime, timedelta
from functools import reduce
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
from dashboard.components.sidebar import render_sidebar
from dashboard.components.charts import *

from dashboard.components.sidebar import render_sidebar
from dashboard.components.realtime_panel import render_realtime_status, render_alert_summary
from dashboard.components.event_cards import render_event_cards
from dashboard.components.ai_insights import render_ai_insights
from dashboard.components.chatbot import render_chatbot
from dashboard.components.forecast_view import render_forecast_tab

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

st.set_page_config(page_title=APP_TITLE, page_icon=APP_ICON, layout="wide", initial_sidebar_state="expanded")

css_path = Path(__file__).resolve().parent / "assets" / "style.css"
if css_path.exists():
    with open(css_path, encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

st.markdown('<div class="app-top-bar"></div>', unsafe_allow_html=True)

# ─── Resources ───
@st.cache_resource
def init_resources():
    b = DistrictBoundaries()
    ds = DataSourceManager()
    try:
        era5 = ERA5Loader()
        ds.set_era5_loader(era5)
    except Exception as e: logger.warning(f"ERA5Loader: {e}")
    try:
        imd = IMDLoader()
        ds.set_imd_loader(imd)
    except Exception as e: logger.warning(f"IMDLoader: {e}")
    det = HazardDetector()
    cls = HazardClassifier()
    compound = CompoundHazardEngine()
    fc = DailyForecastEngine()
    rt = RealtimeMonitor()
    ae = AlertEngine()
    ad = AnomalyDetector()
    hv = HistoricalValidator()
    return b, ds, det, cls, compound, fc, rt, ae, ad, hv

bounds_mgr, ds_mgr, detector, classifier, compound_engine, forecast_engine, \
    realtime_monitor, alert_engine, anomaly_detector, historical_validator = init_resources()

districts = bounds_mgr.district_names if bounds_mgr else []

# ─── Load cached CMIP6 ensemble at startup ───
def load_cached_cmip6():
    cache_path = Path(CFG.cache_dir) / "cmip6_ensemble.parquet"
    if cache_path.exists():
        try:
            df = pd.read_parquet(cache_path)
            forecast_engine.set_ensemble(df)
            logger.info(f"Loaded CMIP6 ensemble from cache ({len(df)} rows)")
            return df
        except Exception as e:
            logger.warning(f"CMIP6 cache load failed: {e}")
    return None

cmip6_ensemble = load_cached_cmip6()

# ─── Initialize DICRA NDVI loader ───
dicra_loader = DICRALoader() if DICRALoader is not None else None

source, district, start_date, end_date, horizon, mode = render_sidebar(
    districts=districts,
    data_sources=[DataSource.ERA5.value, DataSource.IMD.value, DataSource.AUTO.value],
)

# ─── District change → clear session ───
if st.session_state.get("_prev_district") != district:
    clear_keys = ["data", "hazards", "forecasts", "alerts", "anomalies", "validation",
                   "realtime_status", "data_loaded", "quick_forecasted_hazards",
                   "forecasted_hazards", "rt_forecast", "chat_messages"]
    for k in clear_keys: st.session_state.pop(k, None)
    st.session_state._prev_district = district

# ─── Data Fetching ───
def fetch_data(force_fresh=False):
    alt_base = Path.home() / "exports" / "raw"
    raw_path = Path(f"exports/raw/{district}_data.csv")
    candidates = [raw_path, alt_base / f"{district}_data.csv"]

    # Force-fresh: skip file cache, go straight to data source
    if force_fresh:
        try:
            data = ds_mgr.get_district_timeseries(district, ["tmax","tmin","precip"])
            if data is not None and not data.empty:
                if "date" in data.columns:
                    data = data.set_index(pd.to_datetime(data["date"])).drop(columns=["date"])
                if "ndvi" not in data.columns: data["ndvi"] = np.nan
                if "soil_moisture" not in data.columns: data["soil_moisture"] = np.nan
                logger.info(f"Fetched fresh data ({len(data)} rows)")
                return data
        except Exception as e:
            logger.warning(f"Fresh fetch failed: {e} — falling back to cache")

    best_df = None
    for raw_path in candidates:
        if raw_path.exists():
            try:
                df = pd.read_csv(raw_path)
                if "date" in df.columns:
                    df = df.set_index(pd.to_datetime(df["date"])).drop(columns=["date"])
                for var in ["tmax","tmin","precip"]:
                    if f"{var}_era5" in df.columns:
                        df[var] = df[f"{var}_era5"]
                        if f"{var}_imd" in df.columns:
                            df[var] = df[var].fillna(df[f"{var}_imd"])
                    elif f"{var}_imd" in df.columns:
                        df[var] = df[f"{var}_imd"]
                if "ndvi" not in df.columns:
                    df["ndvi"] = np.nan
                if "soil_moisture" not in df.columns:
                    df["soil_moisture"] = np.nan
                n_cols = len(df.columns)
                if best_df is None or n_cols > len(best_df.columns):
                    best_df = df
            except Exception as e:
                logger.warning(f"Cached raw load failed ({raw_path}): {e}")

    if best_df is not None:
        haz_candidates = [
            Path(f"exports/hazards/{district}_hazards.csv"),
            Path.home() / "exports" / "hazards" / f"{district}_hazards.csv",
        ]
        best_hdf = None
        best_sat_count = 0
        for hp in haz_candidates:
            if hp.exists():
                try:
                    hdf = pd.read_csv(hp)
                    if "date" in hdf.columns:
                        hdf = hdf.set_index(pd.to_datetime(hdf["date"]))
                    sat_cols = [c for c in ["ndvi","soil_moisture","vci","vhi","tci"] if c in hdf.columns]
                    if len(sat_cols) > best_sat_count:
                        best_hdf = hdf
                        best_sat_count = len(sat_cols)
                except Exception as e:
                    logger.warning(f"Hazards load failed ({hp}): {e}")
        if best_hdf is not None:
            sat_cols = [c for c in ["ndvi","soil_moisture"] if c in best_hdf.columns]
            for col in sat_cols:
                if col not in best_df.columns:
                    best_df[col] = np.nan
                n = min(len(best_hdf), len(best_df))
                vals = best_hdf[col].values[:n]
                best_df.iloc[:n, best_df.columns.get_loc(col)] = best_df.iloc[:n, best_df.columns.get_loc(col)].fillna(
                    pd.Series(vals, index=best_df.index[:n]))
            logger.info(f"Merged {sat_cols} from hazards CSV")
        logger.info(f"Loaded cached raw data ({len(best_df)} rows, {list(best_df.columns)})")
        return best_df
    try:
        data = ds_mgr.get_district_timeseries(district, ["tmax","tmin","precip"])
        if data is not None and not data.empty:
            if "date" in data.columns:
                data = data.set_index(pd.to_datetime(data["date"])).drop(columns=["date"])
            return data
    except Exception as e: logger.warning(f"Data fetch failed: {e}")
    logger.warning(f"No data available for {district}")
    return pd.DataFrame()

def get_hazard_dict(hazards_df):
    d = {}
    for hn in ["flood","drought","heatwave","agri_stress","compound"]:
        cols = [c for c in hazards_df.columns if c.startswith(hn)]
        d[hn] = hazards_df[cols] if cols else pd.DataFrame()
    return d

# ─── Load data on district select ───
force_fresh = st.session_state.pop("force_refresh", False)
if not st.session_state.get("data_loaded") and district:
    with st.spinner(f"Loading {district}..."):
        df = fetch_data(force_fresh=force_fresh)
        # Merge DICRA NDVI if available
        if dicra_loader is not None and not df.empty:
            df = dicra_loader.merge_into(df, district)
        st.session_state.data = df
        st.session_state.data_loaded = True
        core_cols_check = [c for c in ["tmax","tmin","precip","ndvi","soil_moisture"] if c in df.columns]
        if not df.empty and len(core_cols_check) > 0 and not df[core_cols_check].isnull().all().all():
            try:
                haz = detector.detect_all(df, district=district)
                st.session_state.hazards = haz
                hdict = get_hazard_dict(haz)
                alerts_df = alert_engine.evaluate(hdict, district, min_severity=30)
                if not alerts_df.empty:
                    st.session_state.alerts = alerts_df
            except Exception as e: logger.warning(f"Hazard/alert: {e}")

data = st.session_state.get("data", pd.DataFrame())
hazards = st.session_state.get("hazards", pd.DataFrame())
alerts = st.session_state.get("alerts", pd.DataFrame())
core_cols = [c for c in ["tmax","tmin","precip","ndvi","soil_moisture"] if c in data.columns]
has_data = len(core_cols) > 0 and not data[core_cols].isnull().all().all()

# ─── Diagnostics sidebar ───
if district:
    with st.sidebar:
        with st.expander("🔍 Data Diagnostics", expanded=False):
            if data.empty:
                st.error(f"No data loaded for {district}")
                st.code(f"data_loaded={st.session_state.get('data_loaded', False)}")
            else:
                ndvi_nn = data['ndvi'].notna().sum() if 'ndvi' in data.columns else -1
                sm_nn = data['soil_moisture'].notna().sum() if 'soil_moisture' in data.columns else -1
                st.code(f"Data: {len(data)} rows × {len(data.columns)} cols")
                st.code(f"Columns: {list(data.columns)}")
                st.code(f"ndvi: {ndvi_nn} non-null\nsoil_moisture: {sm_nn} non-null")
                if not hazards.empty:
                    for c in ['vci','vhi','tci']:
                        nn = hazards[c].notna().sum() if c in hazards.columns else -1
                        st.code(f"hazards[{c}]: {nn} non-null")
                else:
                    st.code("hazards: EMPTY (detection skipped/failed)")
                st.code(f"has_data check: {has_data}")

# ─── Compute latest hazard values ───
spi_val = float(hazards["spi_3m"].dropna().iloc[-1]) if not hazards.empty and "spi_3m" in hazards.columns else None
cdd_val = float(hazards["cdd"].dropna().iloc[-1]) if not hazards.empty and "cdd" in hazards.columns else None
tanom_val = float(hazards["tmax_anom"].dropna().iloc[-1]) if not hazards.empty and "tmax_anom" in hazards.columns else None
panom_val = float(hazards["precip_anom"].dropna().iloc[-1]) if not hazards.empty and "precip_anom" in hazards.columns else None
fv = float(hazards["flood_severity"].dropna().iloc[-1]) if not hazards.empty and "flood_severity" in hazards.columns and hazards["flood_severity"].notna().any() else 0
dv = float(hazards["drought_severity"].dropna().iloc[-1]) if not hazards.empty and "drought_severity" in hazards.columns and hazards["drought_severity"].notna().any() else 0
hv_ = float(hazards["heatwave_severity"].dropna().iloc[-1]) if not hazards.empty and "heatwave_severity" in hazards.columns and hazards["heatwave_severity"].notna().any() else 0
av = float(hazards["agri_severity"].dropna().iloc[-1]) if not hazards.empty and "agri_severity" in hazards.columns and hazards["agri_severity"].notna().any() else 0

sev_dict = {"flood": fv, "drought": dv, "heatwave": hv_, "agri_stress": av}

# ─── Build ticker items from computed data ───
ticker_items = []
today_str = datetime.now().strftime("%d %b")
if has_data and not hazards.empty:
    if hv_ >= 50:
        ticker_items.append(('🔴', 'red', f"Heatwave Alert in {district} | {today_str}"))
    if hv_ >= 25 and hv_ < 50:
        ticker_items.append(('🟡', 'yellow', f"Elevated Temperatures in {district} | {today_str}"))
    if fv >= 50:
        ticker_items.append(('🔴', 'red', f"Flood Watch for {district} | {today_str}"))
    if dv >= 50:
        ticker_items.append(('🟠', 'orange', f"Drought Severity Increasing in {district}"))
    if av >= 50:
        ticker_items.append(('🟢', 'green', f"Agricultural Stress Detected in {district}"))
    if tanom_val is not None and abs(tanom_val) > 3:
        ticker_items.append(('🔵', 'blue', f"Temperature Anomaly {tanom_val:+.1f}°C in {district}"))
    if spi_val is not None and spi_val < -1:
        ticker_items.append(('🟠', 'orange', f"SPI-3 Drought Signal in {district} ({spi_val:.2f})"))
if not ticker_items:
    ticker_items.append(('🟢', 'green', f"Normal Conditions in {district} | {today_str}"))
    ticker_items.append(('🔵', 'blue', "Long-term forecast to 2040 available"))

# Duplicate for seamless scrolling
ticker_items_dup = ticker_items * 2
ticker_html = '<div class="event-ticker slide-up"><div class="event-ticker-track">'
for icon, dot_cls, text in ticker_items_dup:
    ticker_html += f'<span class="ticker-item"><span class="ticker-dot {dot_cls}"></span>{icon} {text}</span>'
ticker_html += '</div></div>'

st.markdown(ticker_html, unsafe_allow_html=True)

# ======================================================================
#   HERO SECTION
# ======================================================================
hero_col1, hero_col2 = st.columns([2.5, 1])
with hero_col1:
    st.markdown(f'''
    <div class="hero-section slide-up">
        <div class="hero-top">
            <div class="hero-title-group">
                <h1>{APP_ICON} <span class="gradient-text">{APP_NAME}</span></h1>
                <p class="subtitle">{APP_SUBTITLE} · <strong>{district or "Select a district"}</strong> · Madhya Pradesh, India</p>
            </div>
            <div class="hero-badges">
                <span class="hero-badge primary">📡 DICRA</span>
                <span class="hero-badge secondary">🌤️ IMD</span>
                <span class="hero-badge accent">🌍 ERA5</span>
                <span class="hero-badge success">🔭 CMIP6</span>
                <span class="hero-badge warning">🛰️ MODIS</span>
                <span class="hero-badge info">💧 SMAP</span>
            </div>
        </div>
        <div class="hero-stats">
            <div class="hero-stat">
                <div class="hero-stat-icon" style="background:#E0F2FE;">🌊</div>
                <div class="hero-stat-text">
                    <div class="hero-stat-value" style="color:{'#EF4444' if fv>=50 else '#22C55E'};">{fv:.0f}</div>
                    <div class="hero-stat-label">Flood Risk</div>
                </div>
            </div>
            <div class="hero-stat">
                <div class="hero-stat-icon" style="background:#FEFCE8;">🏜️</div>
                <div class="hero-stat-text">
                    <div class="hero-stat-value" style="color:{'#EF4444' if dv>=50 else '#22C55E'};">{dv:.0f}</div>
                    <div class="hero-stat-label">Drought Risk</div>
                </div>
            </div>
            <div class="hero-stat">
                <div class="hero-stat-icon" style="background:#FFF7ED;">🔥</div>
                <div class="hero-stat-text">
                    <div class="hero-stat-value" style="color:{'#EF4444' if hv_>=50 else '#22C55E'};">{hv_:.0f}</div>
                    <div class="hero-stat-label">Heatwave Risk</div>
                </div>
            </div>
            <div class="hero-stat">
                <div class="hero-stat-icon" style="background:#F0FDF4;">🌾</div>
                <div class="hero-stat-text">
                    <div class="hero-stat-value" style="color:{'#EF4444' if av>=50 else '#22C55E'};">{av:.0f}</div>
                    <div class="hero-stat-label">Agri Risk</div>
                </div>
            </div>
            <div class="hero-stat">
                <div class="hero-stat-icon" style="background:#E0F2FE;">📊</div>
                <div class="hero-stat-text">
                    <div class="hero-stat-value">{f"{spi_val:.2f}" if spi_val is not None else "N/A"}</div>
                    <div class="hero-stat-label">SPI-3</div>
                </div>
            </div>
            <div class="hero-stat">
                <div class="hero-stat-icon" style="background:#ECFEFF;">🌡️</div>
                <div class="hero-stat-text">
                    <div class="hero-stat-value">{f"{tanom_val:+.1f}°" if tanom_val is not None else "N/A"}</div>
                    <div class="hero-stat-label">Tmax Anom</div>
                </div>
            </div>
        </div>
        <div class="hero-confidence" style="margin-top:12px;">
            <span>🎯 Forecast Confidence: <strong>{'High' if not hazards.empty else 'N/A'}</strong></span>
            <span style="margin-left:12px;">🔄 Updated: {datetime.now().strftime('%H:%M')}</span>
        </div>
    </div>
    ''', unsafe_allow_html=True)

with hero_col2:
    if not hazards.empty:
        render_hazard_radar(sev_dict, height=220, key_suffix="hero")
    elif district:
        st.info("No hazard data yet. Select a district or run detection.")

# ======================================================================
#   PAGE NAVIGATION (replaces st.tabs)
# ======================================================================
pages = [
    ("📊", "Overview"),
    ("🛰️", "Real-Time Monitor"),
    ("📜", "Historical Analysis"),
    ("🗺️", "Hazard Maps"),
    ("🔮", "Forecasting"),
    ("✅", "Validation"),
    ("🔔", "Alerts"),
]
nav_tab = st.session_state.get("nav_tab", 0)
page_labels = [f"{icon} {label}" for icon, label in pages]
selected_tab = st.segmented_control(
    "Navigate", page_labels, default=page_labels[nav_tab], key="page_nav",
    label_visibility="collapsed", selection_mode="single",
)
if selected_tab:
    for i, lbl in enumerate(page_labels):
        if lbl == selected_tab:
            st.session_state.nav_tab = i
            break
current_page = st.session_state.get("nav_tab", 0)

# ======================================================================
#   PAGE 0: OVERVIEW
# ======================================================================
if current_page == 0:
    # ── AI Insights ──
    if has_data:
        render_ai_insights(data, hazards, district)
        st.markdown('<div class="divider-gradient"></div>', unsafe_allow_html=True)

    # ── Hazard Risk Gauges ──
    st.markdown('<p class="section-title">⚠️ Hazard Risk Assessment</p>', unsafe_allow_html=True)
    gcol1, gcol2 = st.columns([3, 2])
    with gcol1:
        c1, c2 = st.columns(2)
        with c1: render_risk_gauge(severity=fv, title="Flood Risk")
        with c2: render_risk_gauge(severity=dv, title="Drought Risk")
        c3, c4 = st.columns(2)
        with c3: render_risk_gauge(severity=hv_, title="Heatwave Risk")
        with c4: render_risk_gauge(severity=av, title="Agri Stress")
    with gcol2:
        with st.container():
            render_hazard_radar(sev_dict, height=360)
            render_compound_hazard_matrix(sev_dict, height=320)

    # ── Climate Indicators ──
    st.markdown('<div class="divider-soft"></div>', unsafe_allow_html=True)
    st.markdown('<p class="section-title">📊 Climate Indicators</p>', unsafe_allow_html=True)
    c_i1, c_i2, c_i3, c_i4 = st.columns(4)
    with c_i1:
        if spi_val is not None: st.metric("SPI-3 (Drought Index)", f"{spi_val:.2f}")
        else: st.metric("SPI-3", "N/A")
    with c_i2:
        if cdd_val is not None: st.metric("CDD (Dry Days)", f"{int(cdd_val)}d")
        else: st.metric("CDD", "N/A")
    with c_i3:
        if tanom_val is not None: st.metric("Tmax Anomaly", f"{tanom_val:+.2f}°C")
        else: st.metric("Tmax Anom", "N/A")
    with c_i4:
        if panom_val is not None: st.metric("Precip Anomaly", f"{panom_val:+.2f}mm")
        else: st.metric("Precip Anom", "N/A")

    # ── Event Detection Cards ──
    st.markdown('<div class="divider-soft"></div>', unsafe_allow_html=True)
    if has_data and not hazards.empty:
        render_event_cards(data, hazards, district)

    # ── Overview Time Series ──
    st.markdown('<div class="divider-gradient"></div>', unsafe_allow_html=True)
    st.markdown('<p class="section-title">📈 Climate Time Series</p>', unsafe_allow_html=True)
    c_left, c_right = st.columns([3, 1])
    with c_left:
        if has_data:
            ov_df = data.copy()
            if not hazards.empty:
                for c in ["spi_3m","spi_1m","spi_6m","cdd","cwd","tmax_anom","tmin_anom","precip_anom","tci","vci","vhi"]:
                    if c in hazards.columns:
                        if c in ov_df.columns:
                            ov_df[c] = ov_df[c].fillna(hazards.loc[ov_df.index, c])
                        else:
                            ov_df[c] = hazards.loc[ov_df.index, c]
                for c in hazards.columns:
                    if c.endswith("_anom") and c not in ov_df.columns:
                        ov_df[c] = hazards[c]
            ov_mask = (ov_df.index >= pd.Timestamp(start_date)) & (ov_df.index <= pd.Timestamp(end_date))
            ov_df = ov_df[ov_mask]
            if ov_df.empty:
                ov_df = data.tail(365).copy()
                if not hazards.empty:
                    for c in ["vci","vhi","tci","spi_1m","spi_3m","spi_6m"]:
                        if c in hazards.columns:
                            ov_df[c] = hazards.loc[ov_df.index.intersection(hazards.index), c]

            base_avail = [c for c in ["tmax","tmin","precip","ndvi","soil_moisture"] if c in ov_df.columns]
            extra_avail = [c for c in ["spi_3m","cdd","spi_1m","tmax_anom","precip_anom","spi_6m","cwd","tmin_anom","tci","vci","vhi","ndvi_anom","soil_moisture_anom"] if c in ov_df.columns]
            all_avail = base_avail + extra_avail

            ov_cols = st.multiselect("Select parameters to display", all_avail, default=all_avail[:4], key="overview_ts_cols")

            LABELS = {
                "tmax": ("Max Temperature", "°C", "#0EA5E9"),
                "tmin": ("Min Temperature", "°C", "#7c3aed"),
                "precip": ("Precipitation", "mm", "#14B8A6"),
                "spi_3m": ("SPI-3 (Drought Index)", "", "#22c55e"),
                "spi_1m": ("SPI-1 (Short-term)", "", "#10b981"),
                "spi_6m": ("SPI-6 (Long-term)", "", "#059669"),
                "cdd": ("Consecutive Dry Days", "days", "#ef4444"),
                "cwd": ("Consecutive Wet Days", "days", "#3b82f6"),
                "tmax_anom": ("Tmax Anomaly", "°C", "#f97316"),
                "tmin_anom": ("Tmin Anomaly", "°C", "#a855f7"),
                "precip_anom": ("Precip Anomaly", "mm", "#eab308"),
                "tci": ("TCI (Thermal Condition)", "", "#d946ef"),
                "vci": ("VCI (Vegetation Condition)", "", "#06b6d4"),
                "vhi": ("VHI (Vegetation Health)", "", "#8b5cf6"),
                "ndvi": ("NDVI (Vegetation Index)", "", "#22c55e"),
                "soil_moisture": ("Soil Moisture", "m³/m³", "#0ea5e9"),
                "ndvi_anom": ("NDVI Anomaly", "", "#65a30d"),
                "soil_moisture_anom": ("Soil Moisture Anomaly", "", "#0284c7"),
            }
            marker_vars = {"ndvi","vci","vhi","tci","soil_moisture","ndvi_anom","soil_moisture_anom"}
            if ov_cols:
                for i in range(0, len(ov_cols), 2):
                    pair = ov_cols[i:i+2]
                    c1, c2 = st.columns(2)
                    for ci, col in zip([c1, c2], pair):
                        with ci:
                            label, unit, clr = LABELS.get(col, (col.title(), "", "#888"))
                            render_individual_chart(ov_df, col, label, unit, clr, marker_vars, district, "ov")
            else:
                st.info("Select at least one parameter")

            # ── 15-Day Extreme Event Forecast ──
            st.markdown('<div class="divider-gradient"></div>', unsafe_allow_html=True)
            st.markdown('<p class="section-title">🔮 15-Day Extreme Event Forecast</p>', unsafe_allow_html=True)
            if st.button("Generate 15-Day Hazard Forecast", key="quick_fc_15day", type="secondary"):
                with st.spinner("Forecasting next 15 days..."):
                    qf = {}
                    for tgt in ["tmax","tmin","precip"]:
                        if tgt in data.columns:
                            try:
                                fc = forecast_engine.generate_daily_to_2040(data, tgt, district)
                                if not fc.empty:
                                    qf[tgt] = fc[["date","forecast"]].copy()
                                    qf[tgt] = qf[tgt].rename(columns={"forecast": tgt})
                                    qf[tgt]["date"] = pd.to_datetime(qf[tgt]["date"])
                            except Exception:
                                pass
                    if len(qf) >= 2:
                        fc_panel_15 = reduce(lambda a, b: a.merge(b, on="date", how="outer"), qf.values())
                        fc_panel_15 = fc_panel_15.set_index("date").sort_index()
                        fc_panel_15 = fc_panel_15[fc_panel_15.index >= pd.Timestamp(datetime.now().date())].head(15)
                        ext_15 = data.copy().combine_first(fc_panel_15)
                        try:
                            haz_15 = detector.detect_all(ext_15, district=district)
                            st.session_state.quick_forecasted_hazards = haz_15.loc[fc_panel_15.index.intersection(haz_15.index)]
                        except Exception:
                            st.session_state.quick_forecasted_hazards = pd.DataFrame()

            qh = st.session_state.get("quick_forecasted_hazards", pd.DataFrame())
            if not qh.empty:
                qsev_cols = [c for c in qh.columns if c.endswith("_severity")]
                if qsev_cols:
                    qcols15 = st.columns(len(qsev_cols))
                    qlabels = {"flood_severity":("Flood","#2563eb"),"drought_severity":("Drought","#ca8a04"),
                               "heatwave_severity":("Heatwave","#dc2626"),"agri_stress_severity":("Agri Stress","#16a34a")}
                    for qi, col in enumerate(qsev_cols):
                        with qcols15[qi % len(qsev_cols)]:
                            lbl, clr = qlabels.get(col, (col.replace("_severity",""), "#888"))
                            fig15 = go.Figure()
                            fig15.add_trace(go.Scatter(x=qh.index, y=qh[col], mode="lines+markers", name=lbl,
                                line=dict(width=2, color=clr), marker=dict(size=6)))
                            fig15.update_layout(**{**sci_layout(f"{lbl} — 15d", 200), "showlegend": False})
                            st.plotly_chart(fig15, use_container_width=True, key=f"overview_15d_{col}")

                    # Risk breakdown table
                    ov_cat_cols = {"Flood": "flood_risk", "Drought (SPI)": "drought_spi_category",
                                   "Heatwave": "heatwave_category", "Agri Risk": "agri_risk"}
                    ov_avail = {k: v for k, v in ov_cat_cols.items() if v in qh.columns}
                    if ov_avail:
                        ov_risk = qh[list(ov_avail.values())].copy()
                        ov_risk.index = ov_risk.index.strftime("%b %d")
                        ov_risk = ov_risk.rename(columns={v: k for k, v in ov_avail.items()})
                        def _ov_color_bright(val):
                            if pd.isna(val): return ""
                            if val in ("Normal","No Heatwave","Low"): return "background-color: #166534; color: #bbf7d0; font-weight: 600"
                            if val in ("Watch","Hot Day","Mild Drought","Elevated Temperature"): return "background-color: #854d0e; color: #fef08a; font-weight: 600"
                            if val in ("Moderate","Heatwave","Moderate Drought"): return "background-color: #9a3412; color: #fed7aa; font-weight: 600"
                            if val in ("High","Severe Heatwave","Severe Drought","Severe","Severe Flood"): return "background-color: #991b1b; color: #fca5a5; font-weight: 700"
                            if val in ("Extreme Drought","Extreme"): return "background-color: #7f1d1d; color: #f87171; font-weight: 700"
                            return ""
                        st.dataframe(ov_risk.style.map(_ov_color_bright), use_container_width=True)

        elif district:
            raw_path = Path(f"exports/raw/{district}_data.csv")
            if not raw_path.exists():
                st.warning(f"No cached data for **{district}**. Run: `python run_all.py --districts {district}`")
            else:
                st.info("Data loaded but all values NaN")
        else:
            st.info("Select a district from the sidebar")

    with c_right:
        if not alerts.empty: render_alert_panel(alerts.head(3))
        elif has_data: st.success("No active alerts")
        else: st.info("Select a district")

    # All Districts Quick Status
    if bounds_mgr:
        st.markdown('<div class="divider-soft"></div>', unsafe_allow_html=True)
        with st.expander("📋 All Districts — Quick Status", expanded=False):
            all_d = bounds_mgr.district_names
            st.markdown(f"**{len(all_d)} districts** in Madhya Pradesh")
            ad_rows = []
            for d in all_d:
                has_csv = Path(f"exports/raw/{d}_data.csv").exists()
                current = "✓ Active" if d == district else ("📁 Cached" if has_csv else "—")
                ad_rows.append({"District": d, "Status": current})
            adf = pd.DataFrame(ad_rows)
            st.dataframe(adf, use_container_width=True)
            with_any = (adf["Status"] != "—").sum()
            if with_any < len(all_d):
                st.caption(f"{with_any}/{len(all_d)} districts have cached data. Run pipeline for all: `python run_all.py --all-districts`")
            else:
                st.caption("All districts have cached data. Select one from the sidebar.")

# ======================================================================
#   PAGE 1: REAL-TIME MONITOR
# ======================================================================
elif current_page == 1:
    st.markdown(f'<div class="glass-card" style="padding:20px 28px;"><h1 style="font-size:1.5rem;font-weight:800;margin:0;">🛰️ Real-Time Climate Monitor</h1><p class="subtitle" style="margin:2px 0 0;">Live GEE-based monitoring · {district or "Select a district"}</p></div>', unsafe_allow_html=True)

    gee_auth = st.session_state.get("gee_authenticated", False)
    if not gee_auth:
        if st.button("🔐 Connect Google Earth Engine", type="primary"):
            with st.spinner("Initializing GEE..."):
                try:
                    import ee
                    try:
                        ee.Initialize(project=CFG.gee.project)
                    except Exception:
                        st.info("Opening browser for GEE auth...")
                        ee.Authenticate()
                        ee.Initialize(project=CFG.gee.project)
                    st.session_state.gee_authenticated = True
                    st.success("GEE connected! Fetching live data...")
                    st.rerun()
                except Exception as e:
                    st.error(f"GEE connection failed: {e}")

    c1, c2 = st.columns([2, 1])
    with c1:
        st.markdown("### Satellite Feeds")
        status = st.session_state.get("realtime_status") or {}
        if not status and district:
            if gee_auth or st.session_state.get("gee_authenticated", False):
                try:
                    status = realtime_monitor.fetch_all(district)
                    st.session_state.realtime_status = status
                except Exception as e:
                    logger.warning(f"GEE monitoring: {e}")
        st.markdown(f'''
        <div style="display:flex;gap:12px;align-items:center;margin-bottom:12px;flex-wrap:wrap;">
            <span class="hero-badge {'success' if gee_auth else 'warning'}">
                <span class="status-dot {'online' if gee_auth else 'offline'}"></span>
                {'GEE Connected' if gee_auth else 'GEE Disconnected'}
            </span>
            <span class="hero-badge info">🌐 ERA5-Land</span>
            <span class="hero-badge info">📡 MODIS</span>
            <span class="hero-badge info">💧 SMAP</span>
        </div>
        ''', unsafe_allow_html=True)
        if status and any(status.get(k, False) for k in ["precip","ndvi","lst","soil_moisture"]):
            render_realtime_status(status)
        elif gee_auth:
            st.info("GEE connected but no satellite data returned for this district.")
        else:
            st.info("Click 'Connect Google Earth Engine' to enable live satellite feeds")

        # ── Generate forecast for live today's data ──
        rt_fc = st.session_state.get("rt_forecast", None)
        if rt_fc is None and has_data:
            with st.spinner("Generating forecast..."):
                rt_fc = {}
                for tgt in ["tmax","tmin","precip"]:
                    if tgt in data.columns:
                        try:
                            fc = forecast_engine.generate_daily_to_2040(data, tgt, district)
                            if not fc.empty:
                                rt_fc[tgt] = fc[["date","forecast"]].copy()
                                rt_fc[tgt]["date"] = pd.to_datetime(rt_fc[tgt]["date"])
                        except Exception:
                            pass
                st.session_state.rt_forecast = rt_fc

        today = pd.Timestamp(datetime.now().date())

        st.markdown("### 🌡️ Today's Forecast")
        if rt_fc and len(rt_fc) >= 2:
            today_fc = {}
            for tgt in ["tmax","tmin","precip"]:
                if tgt in rt_fc:
                    match = rt_fc[tgt][pd.to_datetime(rt_fc[tgt]["date"]).dt.date == today.date()]
                    if not match.empty:
                        today_fc[tgt] = match.iloc[0]["forecast"]
            # Fallback: if no exact today match, use first forecast row after today
            if not today_fc:
                for tgt in ["tmax","tmin","precip"]:
                    if tgt in rt_fc:
                        future = rt_fc[tgt][pd.to_datetime(rt_fc[tgt]["date"]) >= today]
                        if not future.empty:
                            today_fc[tgt] = future.iloc[0]["forecast"]
            if today_fc:
                cols = st.columns(3)
                with cols[0]:
                    v = today_fc.get("tmax", "N/A")
                    st.metric("Max Temperature", f"{v:.1f}°C" if isinstance(v,(int,float)) else "N/A")
                with cols[1]:
                    v = today_fc.get("tmin", "N/A")
                    st.metric("Min Temperature", f"{v:.1f}°C" if isinstance(v,(int,float)) else "N/A")
                with cols[2]:
                    v = today_fc.get("precip", "N/A")
                    st.metric("Precipitation", f"{v:.1f}mm" if isinstance(v,(int,float)) else "N/A")
                st.markdown(f'<span class="hero-badge info">📅 Forecast for {today.strftime("%d %b %Y")} · AI-generated from ERA5/IMD + CMIP6</span>', unsafe_allow_html=True)
            else:
                st.info("Forecast data not yet available for today — generating extended outlook...")
        elif has_data:
            st.info("Generating forecast... (click Refresh to retry)")
        else:
            st.info("Load a district to see today's forecast")

        info_cols = st.columns([3, 1])
        with info_cols[0]:
            if has_data and not data.empty:
                latest_date = data.index[-1]
                st.markdown(f'<span class="hero-badge" style="background:var(--warning-light);color:var(--text-muted);font-size:0.7rem;">📜 Latest observed: {latest_date.strftime("%d %b %Y")} (historical baseline)</span>', unsafe_allow_html=True)
        with info_cols[1]:
            if st.button("🔄 Refresh", key="refresh_data", use_container_width=True):
                st.session_state.force_refresh = True
                st.session_state.data_loaded = False
                st.session_state.pop("data", None)
                st.session_state.pop("hazards", None)
                st.session_state.pop("alerts", None)
                st.session_state.pop("realtime_status", None)
                st.session_state.pop("rt_forecast", None)
                st.rerun()

        # Forecasted outlook
        if rt_fc and len(rt_fc) >= 2:
            st.markdown("### 🔮 7-Day Outlook")
            fc_tbl = None
            for tgt in ["tmax","tmin","precip"]:
                if tgt in rt_fc:
                    p = rt_fc[tgt].copy()
                    p = p[pd.to_datetime(p["date"]) >= today].head(7).copy()
                    p = p.rename(columns={"forecast": tgt})
                    fc_tbl = p if fc_tbl is None else fc_tbl.merge(p, on="date", how="outer")
            if fc_tbl is not None and not fc_tbl.empty:
                fc_tbl["date"] = fc_tbl["date"].dt.strftime("%b %d, %Y")
                fc_tbl = fc_tbl.set_index("date")
                for col, unit in [("tmax","°C"),("tmin","°C"),("precip","mm")]:
                    if col in fc_tbl.columns:
                        fc_tbl[col] = fc_tbl[col].round(1).astype(str) + unit
                st.dataframe(fc_tbl, use_container_width=True)
            else:
                st.caption("Forecast data starts after today — extended forecast being generated...")
        elif has_data:
            st.caption("Generating forecast outlook...")

            # Climate indices
            st.markdown("### Climate Indices (ERA5/IMD)")
            if not hazards.empty:
                h_latest = hazards.tail(1).iloc[0]
                pcols = st.columns(4)
                with pcols[0]:
                    v = h_latest.get("spi_3m", None)
                    st.metric("SPI-3", f"{v:.2f}" if pd.notna(v) else "N/A")
                with pcols[1]:
                    v = h_latest.get("cdd", None)
                    st.metric("CDD", f"{int(v)}d" if pd.notna(v) else "N/A")
                with pcols[2]:
                    v = h_latest.get("tmax_anom", None)
                    st.metric("Tmax Anomaly", f"{v:+.2f}°C" if pd.notna(v) else "N/A")
                with pcols[3]:
                    v = h_latest.get("precip_anom", None)
                    st.metric("Precip Anomaly", f"{v:+.2f}mm" if pd.notna(v) else "N/A")
                has_sat = any(h_latest.get(c) is not None and pd.notna(h_latest.get(c)) for c in ["vhi", "vci", "tci"])
                if has_sat:
                    st.caption("Satellite indices (VHI/VCI/TCI) available via MODIS/GEE")
            else:
                st.info("Run hazard detection to see climate indices")
        else:
            st.info("Select a district to see observations")

    with c2:
        st.markdown("### Anomaly Detection")
        if st.button("Scan for Anomalies", type="primary"):
            with st.spinner("Analyzing..."):
                try:
                    baseline = {c: anomaly_detector.compute_baseline(data[c]) for c in data.columns if c in data}
                    anomalies = anomaly_detector.detect_anomalies(data, baseline, value_col="tmax") if has_data else pd.DataFrame()
                    st.session_state.anomalies = anomalies
                except Exception as e: st.error(f"Anomaly detection: {e}")
        anomalies = st.session_state.get("anomalies", pd.DataFrame())
        if not anomalies.empty: render_anomaly_table(anomalies)
        else: st.info("No anomalies detected")

# ======================================================================
#   PAGE 2: HISTORICAL ANALYSIS
# ======================================================================
elif current_page == 2:
    st.markdown(f'<div class="glass-card" style="padding:20px 28px;"><h1 style="font-size:1.5rem;font-weight:800;margin:0;">📜 Historical Analysis</h1><p class="subtitle" style="margin:2px 0 0;">{district or "Select a district"} · {start_date} to {end_date}</p></div>', unsafe_allow_html=True)

    if has_data:
        mask = (data.index >= pd.Timestamp(start_date)) & (data.index <= pd.Timestamp(end_date))
        subset = data[mask]
        display_df = subset if len(subset) > 0 else data

        if not hazards.empty:
            merged = display_df.copy()
            for c in ["vhi","vci","tci","spi_1m","spi_3m","spi_6m","cdd","cwd"]:
                if c in hazards.columns:
                    if c in merged.columns:
                        merged[c] = merged[c].fillna(hazards.loc[display_df.index, c])
                    else:
                        merged[c] = hazards.loc[display_df.index, c]
            for c in hazards.columns:
                if c.endswith("_anom") and c not in merged.columns:
                    merged[c] = hazards[c]
        else:
            merged = display_df

        base_avail = [c for c in ["tmax","tmin","precip","ndvi","soil_moisture"] if c in merged.columns]
        extra_avail = [c for c in ["spi_1m","spi_3m","spi_6m","cdd","cwd","tmax_anom","tmin_anom","precip_anom","tci","vci","vhi","ndvi_anom","soil_moisture_anom"] if c in merged.columns]
        hist_all_avail = base_avail + extra_avail
        ha_cols = st.multiselect("Select parameters", hist_all_avail, default=hist_all_avail[:4], key="hist_ts_cols")

        LABELS = {
            "tmax": ("Max Temperature", "°C", "#0EA5E9"),
            "tmin": ("Min Temperature", "°C", "#7c3aed"),
            "precip": ("Precipitation", "mm", "#14B8A6"),
            "spi_3m": ("SPI-3 (Drought Index)", "", "#22c55e"),
            "spi_1m": ("SPI-1 (Short-term)", "", "#10b981"),
            "spi_6m": ("SPI-6 (Long-term)", "", "#059669"),
            "cdd": ("Consecutive Dry Days", "days", "#ef4444"),
            "cwd": ("Consecutive Wet Days", "days", "#3b82f6"),
            "tmax_anom": ("Tmax Anomaly", "°C", "#f97316"),
            "tmin_anom": ("Tmin Anomaly", "°C", "#a855f7"),
            "precip_anom": ("Precip Anomaly", "mm", "#eab308"),
            "vhi": ("VHI (Vegetation Health)", "", "#8b5cf6"),
            "vci": ("VCI (Vegetation Condition)", "", "#06b6d4"),
            "tci": ("TCI (Thermal Condition)", "", "#d946ef"),
            "ndvi": ("NDVI (Vegetation Index)", "", "#22c55e"),
            "soil_moisture": ("Soil Moisture", "m³/m³", "#0ea5e9"),
            "ndvi_anom": ("NDVI Anomaly", "", "#65a30d"),
            "soil_moisture_anom": ("Soil Moisture Anomaly", "", "#0284c7"),
        }
        hist_marker_vars = {"ndvi","vci","vhi","tci","soil_moisture","ndvi_anom","soil_moisture_anom"}
        if ha_cols:
            for i in range(0, len(ha_cols), 2):
                pair = ha_cols[i:i+2]
                c1, c2 = st.columns(2)
                for ci, col in zip([c1, c2], pair):
                    with ci:
                        label, unit, clr = LABELS.get(col, (col.title(), "", "#888"))
                        render_individual_chart(merged, col, label, unit, clr, hist_marker_vars, district, "hist")
        else:
            st.info("Select at least one parameter")

        # Hazard severity timeline
        if not hazards.empty:
            hcols = [c for c in hazards.columns if c.endswith("_severity")]
            display = hazards[hcols].copy()
            display["date"] = hazards.index
            for c in hcols: display[c] = display[c].fillna(0)
            render_hazard_severity_panel(display, [c.replace("_severity","") for c in hcols])

        # IMD drought classification
        if "precip" in data.columns:
            with st.expander("🌾 IMD Drought Classification (Rainfall Deficiency)", expanded=False):
                st.markdown("""
                **IMD Drought Definition**: A region is drought-affected if rainfall deficiency ≥ 26%.
                - **Moderate drought**: deficiency 26–50%
                - **Severe drought**: deficiency > 50%
                """)
                p_daily = data["precip"].dropna()
                if len(p_daily) > 365:
                    doy_clim = p_daily.groupby(p_daily.index.dayofyear).mean()
                    monthly = p_daily.resample("ME").sum()
                    monthly_norm = monthly.index.map(lambda d: doy_clim.loc[d.dayofyear] if d.dayofyear in doy_clim.index else doy_clim.mean())
                    monthly_norm = pd.Series(monthly_norm.values, index=monthly.index)
                    deficiency = ((1 - monthly / monthly_norm.replace(0, monthly_norm.median())) * 100).fillna(0)
                    imd_class = pd.cut(deficiency, bins=[-1e9, 0, 25, 50, 101], labels=["Normal", "Watch", "Moderate Drought", "Severe Drought"])
                    imd_df = pd.DataFrame({"Monthly Precip (mm)": monthly.round(1), "Normal (mm)": monthly_norm.round(1),
                                           "Deficiency (%)": deficiency.round(1), "IMD Class": imd_class})
                    imd_df.index = imd_df.index.strftime("%Y-%m")
                    st.dataframe(imd_df.tail(60), use_container_width=True)
                else:
                    st.info("Need >1 year of precipitation data for IMD classification")

        # Methodology reference
        with st.expander("📖 Hazard Methodology Reference", expanded=False):
            st.markdown("""
            ### 🌊 Flood Methodology
            | Category | IMD Threshold | Score |
            |----------|--------------|-------|
            | Light Rain | ≥ 15.6 mm/day | — |
            | Moderate Rain | ≥ 15.6 mm/day | 5 pts |
            | Heavy Rain | ≥ 64.5 mm/day | 15 pts |
            | Very Heavy Rain | ≥ 115.6 mm/day | 25 pts |
            | Extreme Rain | ≥ 204.5 mm/day | 35 pts |

            **Severity:** Persistence (consecutive heavy rain) + 3-day cumulative + Antecedent saturation + Intensity.

            ### 🏜️ Drought Methodology
            **IMD:** Deficiency ≥ 26% = drought. SPI-1/3/6 + VHI + soil moisture anomaly.

            ### 🔥 Heatwave Methodology
            **IMD:** Tmax ≥ 40°C + departure 4.5-6.4°C = Heatwave, ≥ 6.5°C = Severe.
            **Severity:** Temperature component (30pts) + Departure (45pts) + Sustained bonus (15pts) + Severe bonus (10pts).

            ### 🌾 Agricultural Stress
            **Components:** VHI (50%) + Tmax heat stress (20%) + Soil moisture anomaly (20%) + CDD (10%)
            """)
    else:
        st.info("No historical data available")

# ======================================================================
#   PAGE 3: HAZARD MAPS
# ======================================================================
elif current_page == 3:
    st.markdown(f'<div class="glass-card" style="padding:20px 28px;"><h1 style="font-size:1.5rem;font-weight:800;margin:0;">🗺️ Hazard Maps</h1><p class="subtitle" style="margin:2px 0 0;">Current hazards across Madhya Pradesh · {district or "Select a district"}</p></div>', unsafe_allow_html=True)

    if not hazards.empty and bounds_mgr:
        severity_cols = [c for c in hazards.columns if c.endswith("_severity")]
        gdf = bounds_mgr.gdf.copy()

        dates = hazards.index.unique()
        dates = sorted(dates)
        if len(dates) > 365:
            dates = dates[-365:]
        sel_idx = len(dates) - 1
        if "hazard_map_date" in st.session_state:
            saved = st.session_state.hazard_map_date
            match = [i for i, d in enumerate(dates) if d == saved]
            if match:
                sel_idx = match[0]
        selected_date = st.selectbox("Select Date", dates, index=sel_idx,
            format_func=lambda d: d.strftime("%Y-%m-%d"), key="hazard_map_date_sel")
        st.session_state.hazard_map_date = selected_date

        row = hazards.loc[selected_date] if selected_date in hazards.index else hazards.iloc[-1]
        sev_data = {}
        haz_base = Path.home() / "exports" / "hazards"
        for sc in severity_cols:
            hname = sc.replace("_severity", "")
            vals = {}
            for _, r in gdf.iterrows():
                d = r[CFG.district_col]
                if d == district:
                    val = float(row[sc]) if not pd.isna(row[sc]) else 0
                else:
                    haz_path = haz_base / f"{d}_hazards.csv"
                    val = 0.0
                    if haz_path.exists():
                        try:
                            hdf = pd.read_csv(haz_path)
                            if "date" in hdf.columns:
                                hdf = hdf.set_index(pd.to_datetime(hdf["date"]))
                            if sc in hdf.columns and selected_date in hdf.index:
                                v = hdf.loc[selected_date, sc]
                                val = float(v) if pd.notna(v) else 0.0
                        except Exception:
                            pass
                vals[d] = val
            sev_data[hname] = pd.Series(vals)

        col_map, col_info = st.columns([3, 1])
        with col_map:
            render_hazard_map_selector(gdf, sev_data)
        with col_info:
            if district:
                st.markdown(f"### {district}")
                for sc in severity_cols:
                    hname = sc.replace("_severity", "")
                    val = float(row[sc]) if not pd.isna(row[sc]) else 0
                    severity_cls = classifier.classify(val)
                    color = SEVERITY_COLORS.get(severity_cls, "#888")
                    st.markdown(f"- {HAZARD_NAMES.get(hname, hname)}: **{val:.1f}** <span style='color:{color}'>({severity_cls})</span>", unsafe_allow_html=True)
                st.markdown("---")
                st.caption("Severity loaded from pipeline exports (51/51 districts)")
            else:
                st.info("Select a district from the sidebar")

            # Historical hazard events
            st.markdown("---")
            st.markdown("### Historical Events")
            if district and not hazards.empty:
                event_thresh = 50
                has_events = False
                for sc in severity_cols:
                    hname = sc.replace("_severity", "")
                    if sc not in hazards.columns:
                        continue
                    above = (hazards[sc].fillna(0) >= event_thresh).astype(int)
                    groups = (above.diff().fillna(0) != 0).cumsum()
                    periods = above[above == 1].groupby(groups)
                    if len(periods) == 0:
                        continue
                    has_events = True
                    with st.expander(f"{HAZARD_NAMES.get(hname, hname.title())} Events", expanded=False):
                        evt_rows = []
                        for _, idx in periods:
                            block = hazards.loc[idx.index, sc]
                            evt_rows.append({
                                "Start": idx.index[0].strftime("%Y-%m-%d"),
                                "End": idx.index[-1].strftime("%Y-%m-%d"),
                                "Days": len(idx),
                                "Peak": round(float(block.max()), 1),
                                "Peak Date": block.idxmax().strftime("%Y-%m-%d"),
                            })
                        if evt_rows:
                            evt_df = pd.DataFrame(evt_rows).sort_values("Start", ascending=False).head(50)
                            st.dataframe(evt_df, use_container_width=True)
                if not has_events:
                    st.info(f"No historical events ≥ {event_thresh} severity")
            else:
                st.info("Load hazard data to see historical events")

        # Forecast events
        st.markdown("### Forecast Events (Daily)")
        forecasts = st.session_state.get("forecasts", {})
        if not forecasts:
            st.info("Generate a forecast in the Forecasting tab to see daily projections")
        else:
            matching = [k for k in forecasts if k[0] == district]
            if not matching:
                st.info(f"No forecasts for {district}")
            else:
                targets = [k[1] for k in matching]
                tgt_labels = {"tmax": "Max Temperature (°C)", "tmin": "Min Temperature (°C)", "precip": "Precipitation (mm)"}
                sel_tgt = st.selectbox("Forecast Variable", targets, format_func=lambda t: tgt_labels.get(t, t.title()), key="hazard_fc_tgt")
                fc = forecasts.get((district, sel_tgt), pd.DataFrame())
                if fc.empty:
                    st.warning("Forecast data is empty")
                else:
                    fc = fc.copy()
                    if "date" not in fc.columns:
                        fc["date"] = fc.index
                    fc["date"] = pd.to_datetime(fc["date"])
                    display_cols = ["date", "forecast"]
                    if "source" in fc.columns: display_cols.append("source")
                    if "lower" in fc.columns: display_cols.append("lower")
                    if "upper" in fc.columns: display_cols.append("upper")
                    fc_display = fc[display_cols].copy()
                    fc_display["date"] = fc_display["date"].dt.strftime("%Y-%m-%d")
                    fc_display["forecast"] = pd.to_numeric(fc_display["forecast"], errors="coerce").round(2)
                    thresh_map = {"tmax": CFG.hazard.heatwave_tmax_threshold, "precip": CFG.hazard.imd_precip_very_heavy}
                    if sel_tgt in thresh_map:
                        thresh_val = thresh_map[sel_tgt]
                        fc_display["Threshold Exceeded"] = fc_display["forecast"] >= thresh_val
                    max_rows = st.slider("Rows to show", 50, len(fc_display), min(500, len(fc_display)), 50, key="hazard_fc_rows")
                    st.dataframe(fc_display.head(max_rows), use_container_width=True)
    else:
        st.info("No hazard data to map")

# ======================================================================
#   PAGE 4: FORECASTING
# ======================================================================
elif current_page == 4:
    st.markdown(f'<div class="glass-card" style="padding:20px 28px;"><h1 style="font-size:1.5rem;font-weight:800;margin:0;">🔮 Forecasting</h1><p class="subtitle" style="margin:2px 0 0;">Multi-Model AI + CMIP6 Hybrid to 2040 · {district or "Select a district"}</p></div>', unsafe_allow_html=True)

    # ── CMIP6 ensemble status ──
    has_cmip6 = forecast_engine.cmip6._ensemble is not None and not forecast_engine.cmip6._ensemble.empty
    cmip6_col1, cmip6_col2 = st.columns([3, 1])
    with cmip6_col1:
        if has_cmip6:
            ens = forecast_engine.cmip6._ensemble
            ens_dates = pd.to_datetime(ens["date"])
            st.markdown(f'<span class="hero-badge success">✅ CMIP6 ensemble loaded ({len(ens)} rows, {ens_dates.min().strftime("%Y")}–{ens_dates.max().strftime("%Y")})</span>', unsafe_allow_html=True)
        else:
            st.markdown(f'<span class="hero-badge warning">⚠️ No CMIP6 data — forecasts will use climatology (repeating pattern)</span>', unsafe_allow_html=True)
    with cmip6_col2:
        if st.button("📡 Fetch CMIP6 (GEE)", key="fetch_cmip6", use_container_width=True):
            if not st.session_state.get("gee_authenticated", False):
                st.warning("Connect Google Earth Engine first (Real-Time Monitor page)")
            elif CMIP6Loader is None:
                st.error("CMIP6Loader not available")
            else:
                with st.spinner("Fetching CMIP6 projections from GEE... this takes 5–15 minutes..."):
                    try:
                        # Validate CMIP6 variable config before fetching
                        from config.constants import CMIP6_VARIABLES
                        for vkey, vcfg in CMIP6_VARIABLES.items():
                            if "band" not in vcfg:
                                vcfg["band"] = vkey
                        ee_fc = bounds_mgr.get_ee_fc(districts)
                        loader = CMIP6Loader(ee_fc)
                        raw = loader.fetch_all_models()
                        if not raw.empty:
                            ensemble = loader.compute_ensemble(raw)
                            cache_path = Path(CFG.cache_dir) / "cmip6_ensemble.parquet"
                            cache_path.parent.mkdir(parents=True, exist_ok=True)
                            ensemble.to_parquet(cache_path, index=False)
                            forecast_engine.set_ensemble(ensemble)
                            st.session_state.cmip6_fetched = True
                            st.success(f"CMIP6 ensemble cached ({len(ensemble)} rows)")
                            st.rerun()
                        else:
                            st.error("No CMIP6 data returned from GEE")
                    except Exception as e:
                        st.error(f"CMIP6 fetch failed: {e}")

    if st.button("Generate Forecast", type="primary"):
        with st.spinner("Generating forecast... ML training in progress..."):
            forecasts = {}
            for target in ["tmax","tmin","precip"]:
                if has_data and target in data.columns:
                    try:
                        fc = forecast_engine.generate_daily_to_2040(data, target, district)
                        if not fc.empty:
                            forecasts[(district, target)] = fc
                            logger.info(f"Forecast {target}: {len(fc)} days")
                    except Exception as e: logger.warning(f"Forecast {target}: {e}")
            st.session_state.forecasts = forecasts

            forecasted_hazards = pd.DataFrame()
            fc_parts = []
            for target in ["tmax","tmin","precip"]:
                fc = forecasts.get((district, target))
                if fc is not None and not fc.empty:
                    p = fc[["date", "forecast"]].copy()
                    p = p.rename(columns={"forecast": target})
                    p["date"] = pd.to_datetime(p["date"])
                    fc_parts.append(p)
            if len(fc_parts) >= 2:
                fc_panel = reduce(lambda a, b: a.merge(b, on="date", how="outer"), fc_parts)
                fc_panel = fc_panel.set_index("date").sort_index()
                extended = data.copy().combine_first(fc_panel)
                try:
                    haz_all = detector.detect_all(extended, district=district)
                    haz_fc = haz_all.loc[fc_panel.index.intersection(haz_all.index)]
                    forecasted_hazards = haz_fc
                    logger.info(f"Forecasted hazards: {len(haz_fc)} days")
                except Exception as e:
                    logger.warning(f"Hazard forecast failed: {e}")
            st.session_state.forecasted_hazards = forecasted_hazards

            if forecasts:
                src_txt = "climatology"
                sources = set()
                for fc in forecasts.values():
                    if "source" in fc.columns:
                        sources.update(fc["source"].unique())
                if sources:
                    src_txt = ", ".join(sorted(sources))
                n_haz = len(forecasted_hazards) if not forecasted_hazards.empty else 0
                st.success(f"Generated {len(forecasts)} variables ({src_txt}) + {n_haz}d hazard forecast")
            else:
                st.error("ML models need training — run: python run_all.py --all-districts")

    forecasts = st.session_state.get("forecasts", {})
    forecasted_hazards = st.session_state.get("forecasted_hazards", pd.DataFrame())

    if forecasts:
        if not forecasted_hazards.empty:
            sev_cols = [c for c in forecasted_hazards.columns if c.endswith("_severity")]
            event_cols = [c for c in forecasted_hazards.columns if c.endswith("_event")]
            if sev_cols or event_cols:
                with st.expander("🔮 Forecasted Hazard Events", expanded=True):
                    if sev_cols:
                        fc_haz_labels = {
                            "flood_severity": ("Flood Severity", "", "#2563eb"),
                            "drought_severity": ("Drought Severity", "", "#ca8a04"),
                            "heatwave_severity": ("Heatwave Severity", "", "#dc2626"),
                            "agri_stress_severity": ("Agri Stress Severity", "", "#16a34a"),
                        }
                        sev_icons = {
                            "flood_severity": "🌊", "drought_severity": "🏜️",
                            "heatwave_severity": "🔥", "agri_stress_severity": "🌾",
                        }
                        for col in sev_cols:
                            label, unit, clr = fc_haz_labels.get(col, (col.replace("_severity","").title(), "", "#888"))
                            icon = sev_icons.get(col, "⚠️")
                            st.markdown(f'<p style="font-size:0.9rem;font-weight:700;margin:12px 0 2px;color:{clr};">{icon} {label} — Forecasted Severity</p><p style="font-size:0.72rem;color:var(--text-muted);margin:0 0 6px;">Predicted {label.lower()} over the forecast period for <strong>{district}</strong>. Values range 0 (normal) to 100 (extreme).</p>', unsafe_allow_html=True)
                            fig = go.Figure()
                            hex_clr = clr.lstrip("#")
                            if len(hex_clr) == 6:
                                rgba = tuple(int(hex_clr[i:i+2], 16) for i in (0, 2, 4)) + (0.1,)
                            else:
                                rgba = (136, 136, 136, 0.1)
                            fig.add_trace(go.Scatter(
                                x=forecasted_hazards.index, y=forecasted_hazards[col],
                                mode="lines", name=label, line=dict(width=2, color=clr),
                                fill="tozeroy", fillcolor=f"rgba{rgba}",
                            ))
                            fig.update_layout(
                                **sci_layout(f"{label} — {district}", 200, 25, yaxis_title="Severity", xaxis_title="Date"),
                            )
                            fig.update_yaxes(range=[0, 100], tickvals=[0, 25, 50, 75, 100],
                                             ticktext=["0\nNormal", "25\nWatch", "50\nModerate", "75\nSevere", "100\nExtreme"])
                            st.plotly_chart(fig, use_container_width=True, key=f"fc_haz_{col}")

                    if event_cols:
                        st.markdown("#### Forecasted Event Days")
                        evt_summary = {}
                        for c in event_cols:
                            hname = c.replace("_event", "").title()
                            cnt = int(forecasted_hazards[c].fillna(0).sum())
                            evt_summary[hname] = cnt
                        if evt_summary:
                            st.dataframe(pd.DataFrame([evt_summary]).T.rename(columns={0:"Event Days"}), use_container_width=True)

                    cat_cols = {
                        "Flood": "flood_risk", "Drought (SPI)": "drought_spi_category",
                        "Drought (IMD)": "drought_imd_class", "Heatwave": "heatwave_category",
                        "Agri Risk": "agri_risk", "Compound": "compound_class",
                    }
                    avail_cats = {k: v for k, v in cat_cols.items() if v in forecasted_hazards.columns}
                    if avail_cats:
                        st.markdown("#### Hazard Risk Calendar")
                        risk_table = forecasted_hazards[list(avail_cats.values())].copy()
                        risk_table.index = risk_table.index.strftime("%b %d, %Y")
                        risk_table = risk_table.rename(columns={v: k for k, v in avail_cats.items()})
                        def _color_risk(val):
                            if pd.isna(val) or val in ("Normal", "No Heatwave", "Low"): return "background-color: #1a3a1a; color: #86efac"
                            if val in ("Watch", "Hot Day", "Mild Drought", "Elevated Temperature"): return "background-color: #3a3a1a; color: #fef08a"
                            if val in ("Moderate", "Heatwave", "Moderate Drought", "Moderate Flood"): return "background-color: #3a2a1a; color: #fed7aa"
                            if val in ("High", "Severe Heatwave", "Severe Drought", "Severe Flood", "Severe"): return "background-color: #3a1a1a; color: #fca5a5"
                            if val in ("Extreme Drought", "Extreme"): return "background-color: #4a0a0a; color: #ef4444"
                            return ""
                        with st.expander("📅 Daily Risk Breakdown", expanded=True):
                            max_days = st.slider("Days", 7, len(risk_table), min(90, len(risk_table)), 7, key="fc_risk_cal_days")
                            st.dataframe(risk_table.head(max_days).style.map(_color_risk), use_container_width=True)

        render_forecast_tab(data, forecasts, district)
    else:
        st.info("Click Generate Forecast to produce AI + CMIP6 predictions")
        st.markdown("""
        | Model | Horizon | Source |
        |-------|---------|--------|
        | Random Forest | 90 days | ERA5/IMD historical |
        | XGBoost | 90 days | ERA5/IMD historical |
        | LightGBM | 90 days | ERA5/IMD historical |
        | CMIP6 Ensemble | to 2040 | 8 GCMs (SSP2-4.5) |
        | Blended | Full Period | Clamp + smooth handoff |
        """)

# ======================================================================
#   PAGE 5: VALIDATION
# ======================================================================
elif current_page == 5:
    st.markdown(f'<div class="glass-card" style="padding:20px 28px;"><h1 style="font-size:1.5rem;font-weight:800;margin:0;">✅ Scientific Validation</h1><p class="subtitle" style="margin:2px 0 0;">Model skill scores & historical accuracy · {district or "Select a district"}</p></div>', unsafe_allow_html=True)

    if st.button("Run Validation", type="primary"):
        with st.spinner(f"Validating {district}..."):
            try:
                from validation.run import run_district_validation
                result = run_district_validation(district)
                st.session_state.validation = result
                if "error" in result: st.error(result["error"])
                else: st.success("Validation complete")
            except Exception as e: st.error(f"Validation failed: {e}")

    r = st.session_state.get("validation", {})
    if r and "error" not in r:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("### Skill Scores")
            skills = r.get("skill_scores", {})
            for k, v in skills.items():
                score_color = "#22c55e" if v >= 0.7 else "#eab308" if v >= 0.4 else "#ef4444"
                st.markdown(f"<span style='color:{score_color};font-weight:700;'>{k}: {v:.3f}</span>", unsafe_allow_html=True)
        with col2:
            st.markdown("### Confusion Matrix")
            conf = r.get("confusion", {})
            if conf: st.dataframe(pd.DataFrame([conf]).T.rename(columns={0:"Score"}))
        with col3:
            st.metric("Data Points", r.get("data_points", 0))
            st.metric("Hazard Events", r.get("hazard_events", 0))
        st.divider()
        st.json(r)
    else:
        st.info("Click Run Validation to evaluate model performance")
        st.markdown("""
        | Metric | Description | Target |
        |--------|-------------|--------|
        | POD | Probability of Detection | >0.7 |
        | FAR | False Alarm Ratio | <0.3 |
        | HSS | Heidke Skill Score | >0.4 |
        | BSS | Brier Skill Score | >0.3 |
        | ROC-AUC | ROC Area Under Curve | >0.8 |
        """)

# ======================================================================
#   PAGE 6: ALERTS
# ======================================================================
elif current_page == 6:
    st.markdown(f'<div class="glass-card" style="padding:20px 28px;"><h1 style="font-size:1.5rem;font-weight:800;margin:0;">🔔 Alert Center</h1><p class="subtitle" style="margin:2px 0 0;">Active hazard alerts & event history · {district or "Select a district"}</p></div>', unsafe_allow_html=True)

    if st.button("Refresh Alerts", type="primary"):
        with st.spinner("Evaluating..."):
            try:
                if has_data:
                    haz = detector.detect_all(data, district=district)
                    st.session_state.hazards = haz
                    hdict = get_hazard_dict(haz)
                    alerts_df = alert_engine.evaluate(hdict, district, min_severity=30)
                    if not alerts_df.empty: st.session_state.alerts = alerts_df
                st.success("Alerts refreshed")
            except Exception as e: st.error(f"Alert refresh: {e}")

    tab_a, tab_b = st.tabs(["Active Alerts", "Alert History"])
    with tab_a:
        if not alerts.empty: render_alert_panel(alerts)
        else: st.success("No active alerts — all districts at Normal risk")
    with tab_b:
        st.info("Alert history is saved to exports/ directory")

# ======================================================================
#   CHATBOT (floating, always available)
# ======================================================================
if district:
    render_chatbot(data, hazards, district, districts if bounds_mgr else [])

# ======================================================================
#   FOOTER
# ======================================================================
st.markdown('<div class="divider-gradient" style="margin-top:32px;"></div>', unsafe_allow_html=True)
st.markdown(f'''
<div class="sci-footer fade-in">
    <div class="footer-grid">
        <div class="footer-section" style="flex:2;min-width:200px;">
            <h4>Data Sources</h4>
            <div class="footer-badges">
                <span class="footer-badge">DICRA</span>
                <span class="footer-badge">ERA5-Land</span>
                <span class="footer-badge">IMD Gridded</span>
                <span class="footer-badge green">CMIP6 8-GCM</span>
                <span class="footer-badge green">MODIS NDVI</span>
                <span class="footer-badge green">SMAP/ERA5-Land</span>
                <span class="footer-badge orange">MGNREGA</span>
            </div>
        </div>
        <div class="footer-section">
            <h4>Forecast Models</h4>
            <div class="footer-text">
                RF · XGBoost · LightGBM<br>
                Blend · CMIP6 Ensemble (SSP2-4.5)
            </div>
        </div>
        <div class="footer-section">
            <h4>Validation</h4>
            <div class="footer-text">
                POD · FAR · HSS · BSS · ROC-AUC<br>
                Historical: 2000–2025
            </div>
        </div>
        <div class="footer-section">
            <h4>Support</h4>
            <div class="footer-text">
                Water Climate &amp; Sustainability Lab<br>
                <strong>Indian Institute of Technology Indore</strong>
            </div>
        </div>
    </div>
    <div class="footer-divider">
        <strong>Disclaimer:</strong> This platform utilizes datasets from the DICRA dataset along with resources and research support from the Water Climate & Sustainability Lab, IIT Indore. This platform and its predictive models are currently under validation and development. · Generated {datetime.now().strftime("%Y-%m-%d %H:%M")}
    </div>
</div>
''', unsafe_allow_html=True)
