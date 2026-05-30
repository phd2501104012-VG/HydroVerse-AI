"""Multi-source forecast comparison & IMD-aligned policy advisory."""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from pathlib import Path
from datetime import datetime
from config.constants import ACTION_PLAYBOOK, RISK_THRESHOLDS, RISK_CLASSES
from dashboard.components.charts import sci_layout


def _get_severity_class(val):
    if pd.isna(val) or val <= RISK_THRESHOLDS[0]:
        return RISK_CLASSES[0]
    for i in range(len(RISK_THRESHOLDS) - 1, -1, -1):
        if val > RISK_THRESHOLDS[i]:
            return RISK_CLASSES[i]
    return RISK_CLASSES[0]


def render_source_comparison(district: str, forecast_engine=None):
    st.markdown('<p class="section-title">📊 Multi-Source Forecast Comparison</p>', unsafe_allow_html=True)
    st.markdown('<p style="font-size:0.85rem;color:var(--text-muted);margin:-8px 0 14px;">ML Forecast vs CMIP6 Ensemble vs Observed (ERA5/IMD) — all from local data sources</p>', unsafe_allow_html=True)

    obs_df = st.session_state.get("data", pd.DataFrame())
    if not obs_df.empty:
        if "date" not in obs_df.columns:
            obs_df = obs_df.reset_index()
            if "date" not in obs_df.columns:
                obs_df = obs_df.rename(columns={"index": "date"})
        obs_df["date"] = pd.to_datetime(obs_df["date"])

    if not forecast_engine and district and not obs_df.empty:
        from forecasting.daily_forecast import DailyForecastEngine
        forecast_engine = DailyForecastEngine()
        try:
            ens = pd.read_parquet("data/cache/cmip6_ensemble.parquet")
            forecast_engine.set_ensemble(ens)
        except Exception:
            pass

    has_fc = forecast_engine is not None and district and not obs_df.empty

    if not has_fc and obs_df.empty:
        st.info(f"No data available for {district}")
        return

    variables = ["tmax", "tmin", "precip"]
    sel_var = st.selectbox("Variable", variables, key="cmp_var")

    fig = go.Figure()

    # Observed (historic)
    if not obs_df.empty and sel_var in obs_df.columns:
        fig.add_trace(go.Scatter(
            x=obs_df["date"], y=obs_df[sel_var],
            mode="lines", name="Observed (ERA5/IMD)",
            line=dict(width=1.5, color="#22C55E", dash="dash"),
        ))

    # On-the-fly forecast (ML + CMIP6 blend) through 2040 — cached per (district, var)
    _fc_key = f"cmp_fc_{district}_{sel_var}"
    _fc_data = st.session_state.get(_fc_key)
    if has_fc and sel_var in obs_df.columns and _fc_data is None:
        try:
            _panel = obs_df[["date", sel_var]].copy().set_index("date")
            _panel.index = pd.to_datetime(_panel.index)
            _panel["ndvi"] = 0.0
            _panel["soil_moisture"] = 0.0
            _fc = forecast_engine.generate_daily_to_2040(_panel, sel_var, district)
        except Exception as _e:
            _fc = None
            st.caption(f"Forecast gen skipped: {_e}")
        if _fc is not None and not _fc.empty:
            _fc_data = _fc.to_dict("records")
            st.session_state[_fc_key] = _fc_data
    if _fc_data is not None:
        _fc = pd.DataFrame(_fc_data)
        _fc["date"] = pd.to_datetime(_fc["date"])
        fig.add_trace(go.Scatter(
            x=_fc["date"], y=_fc["forecast"],
            mode="lines", name="Blended Forecast (ML+CMIP6)",
            line=dict(width=2, color="#0EA5E9"),
        ))
        if "lower" in _fc.columns and "upper" in _fc.columns:
            fig.add_trace(go.Scatter(
                x=_fc["date"], y=_fc["upper"],
                mode="lines", line=dict(width=0, color="#0EA5E9"),
                showlegend=False, name="upper",
            ))
            fig.add_trace(go.Scatter(
                x=_fc["date"], y=_fc["lower"],
                mode="lines", line=dict(width=0, color="#0EA5E9"),
                fillcolor="rgba(14,165,233,0.12)",
                fill="tonexty", showlegend=False, name="lower",
            ))

    # Raw CMIP6 ensemble through 2040 — cache the full dataframe
    _cmip6 = st.session_state.get("_cmip6_df")
    if _cmip6 is None:
        cmip6_path = Path(r"D:\cri\data\cache") / "cmip6_ensemble.parquet"
        if cmip6_path.exists():
            try:
                _cmip6 = pd.read_parquet(cmip6_path)
                st.session_state["_cmip6_df"] = _cmip6
            except Exception:
                _cmip6 = None
        else:
            _cmip6 = None
    if _cmip6 is not None:
        if "date" in _cmip6.columns:
            _cmip6["date"] = pd.to_datetime(_cmip6["date"])
            if "district" in _cmip6.columns:
                _cmip6 = _cmip6[_cmip6["district"].str.strip().str.lower() == district.strip().lower()]
            _cmip6_col = {"tmax":"tmax_proj_mean","tmin":"tmin_proj_mean","precip":"precip_proj_mean"}.get(sel_var)
            if _cmip6_col and not _cmip6.empty and _cmip6_col in _cmip6.columns:
                fig.add_trace(go.Scatter(
                    x=_cmip6["date"], y=_cmip6[_cmip6_col],
                    mode="lines", name="CMIP6 Ensemble",
                    line=dict(width=2, color="#F97316", dash="dot"),
                ))
            elif not _cmip6.empty:
                st.caption(f"CMIP6: column '{_cmip6_col}' not found in {list(_cmip6.columns)[:10]}...")
            elif _cmip6.empty:
                st.caption(f"CMIP6: no data for district '{district}'")

    if len(fig.data) > 0:
        labels = {"tmax": ("Max Temperature", "°C"), "tmin": ("Min Temperature", "°C"), "precip": ("Precipitation", "mm")}
        label, unit = labels.get(sel_var, (sel_var, ""))
        fig.update_layout(**sci_layout(f"{label} — {district} (1981–2040)", 350, yaxis_title=unit))
        st.plotly_chart(fig, use_container_width=True, key="cmp_chart")
    else:
        st.info(f"No {sel_var} data available for comparison")

    with st.expander("ℹ️ About These Sources", expanded=False):
        st.markdown("""
        | Source | Description |
        |--------|-------------|
        | **Blended Forecast** | ML (RF/XGB/LGBM) short-term + CMIP6 (SSP2-4.5, 8-GCM) long-term through 2040 |
        | **CMIP6 Ensemble** | 8-GCM ensemble mean (SSP2-4.5) from CMIP6 projections |
        | **Observed** | ERA5-Land reanalysis + IMD gridded observations |
        """)


def render_policy_advisory(hazards: pd.DataFrame, district: str):
    st.markdown('<p class="section-title">🏛️ IMD-Aligned Policy Advisory</p>', unsafe_allow_html=True)

    if hazards.empty:
        st.info(f"No hazard data for {district}. Run detection first.")
        return

    severities = {}
    for hazard_key in ["flood", "drought", "heatwave", "agri_stress", "compound"]:
        col = f"{hazard_key}_severity"
        if col in hazards.columns:
            vals = hazards[col].dropna()
            severities[hazard_key] = float(vals.iloc[-1]) if not vals.empty else 0
        else:
            severities[hazard_key] = 0

    has_any_severity = any(v > 0 for v in severities.values())
    if not has_any_severity:
        st.success(f"All clear — no significant hazard signals in {district}")
        return

    severity_labels = {"flood": "🌊 Flood", "drought": "🏜️ Drought", "heatwave": "🔥 Heatwave", "agri_stress": "🌾 Agri Stress", "compound": "⚠️ Compound"}
    severity_colors = {"Normal": "#22c55e", "Watch": "#eab308", "Warning": "#f97316", "Severe": "#ef4444", "Extreme": "#7f1d1d"}

    for hazard_key, sev_val in severities.items():
        if sev_val <= 0:
            continue
        cls = _get_severity_class(sev_val)
        color = severity_colors.get(cls, "#888")
        label = severity_labels.get(hazard_key, hazard_key)

        playbook_entry = ACTION_PLAYBOOK.get(hazard_key, {})
        action = playbook_entry.get(cls, "Monitor situation and follow standard protocols.")

        st.markdown(f"""
        <div class="advisory-card fade-in" style="border-left-color: {color};">
            <div class="advisory-header">
                <span class="advisory-icon">{label}</span>
                <span class="advisory-severity" style="background:{color}22;color:{color};border:1px solid {color}44;">
                    {cls} ({sev_val:.0f}/100)
                </span>
            </div>
            <div class="advisory-body">{action}</div>
        </div>
        """, unsafe_allow_html=True)

    with st.expander("📖 About IMD-Aligned Advisories", expanded=False):
        st.markdown("""
        Advisories follow India Meteorological Department (IMD) classification:

        | Severity | Threshold | Action Required |
        |----------|-----------|-----------------|
        | **Normal** | 0–25 | No action required |
        | **Watch** | 25–50 | Prepare monitoring |
        | **Warning** | 50–75 | Initiate response |
        | **Severe** | 75–90 | Mobilize resources |
        | **Extreme** | 90+ | Emergency response |

        Source: `ACTION_PLAYBOOK` in config.constants (IMD-aligned)
        """)
