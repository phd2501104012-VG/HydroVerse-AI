"""Enhanced forecast view with scientific analytics."""
import streamlit as st
from typing import Optional, Dict, List, Any
import pandas as pd
import plotly.graph_objects as go
from dashboard.config import *
from dashboard.components.charts import render_forecast_chart


def render_forecast_tab(
    panel: pd.DataFrame,
    forecasts: Dict[tuple, pd.DataFrame],
    district: str,
):
    forecast_keys = [k for k in forecasts.keys() if k[0] == district]
    if not forecast_keys:
        return

    targets = [k[1] for k in forecast_keys]
    target_labels = {
        "tmax": "Max Temperature (°C)", "tmin": "Min Temperature (°C)",
        "precip": "Precipitation (mm)", "flood_event": "Flood Probability",
        "drought_event": "Drought Probability", "heatwave_event": "Heatwave Probability",
    }

    # Analytics overview
    st.markdown('<p class="section-title">📊 Forecast Analytics</p>', unsafe_allow_html=True)

    # Summary metrics
    all_fcs = {t: forecasts.get((district, t), pd.DataFrame()) for t in targets}
    metrics_cols = st.columns(4)
    with metrics_cols[0]:
        st.metric("Variables", len([v for v in all_fcs.values() if not v.empty]))
    with metrics_cols[1]:
        if all_fcs.get("tmax") is not None and not all_fcs["tmax"].empty:
            fc = all_fcs["tmax"]
            st.metric("Start", pd.to_datetime(fc["date"]).min().strftime("%Y-%m-%d"))
    with metrics_cols[2]:
        if all_fcs.get("tmax") is not None and not all_fcs["tmax"].empty:
            fc = all_fcs["tmax"]
            st.metric("End", pd.to_datetime(fc["date"]).max().strftime("%Y-%m-%d"))
    with metrics_cols[3]:
        if all_fcs.get("tmax") is not None and not all_fcs["tmax"].empty:
            st.metric("Total Days", f"{len(fc):,}")

    # Variable selection and chart
    selected_target = st.selectbox(
        "Forecast Variable",
        targets,
        format_func=lambda t: target_labels.get(t, t.title()),
        key="fc_target_sel",
    )

    fc = forecasts.get((district, selected_target), pd.DataFrame())
    if fc.empty:
        st.warning("Forecast data is empty.")
        return

    fc["date"] = pd.to_datetime(fc["date"])

    # Historical context
    hist = None
    if selected_target in panel.columns:
        hist = panel[selected_target].dropna().tail(730)

    render_forecast_chart(
        hist, fc, selected_target,
        title=f"{district} — {target_labels.get(selected_target, selected_target)}",
    )

    st.caption(
        "ML = AI forecast (0-90 days) | Blend = Smooth transition | "
        "CMIP6 = 8-GCM ensemble (SSP2-4.5) | Climatology = Historical mean + trend"
    )

    # Source breakdown
    src_counts = fc["source"].value_counts().to_dict() if "source" in fc.columns else {"N/A": len(fc)}
    with st.expander("📊 Source Breakdown", expanded=False):
        src_df = pd.DataFrame({
            "Source": list(src_counts.keys()),
            "Days": list(src_counts.values()),
            "Percentage": [f"{v/len(fc)*100:.1f}%" for v in src_counts.values()],
        })
        st.dataframe(src_df, use_container_width=True)

    # Trend analysis
    with st.expander("📈 Trend Analysis", expanded=False):
        col_t1, col_t2 = st.columns(2)
        with col_t1:
            if len(fc) > 30:
                monthly = fc.set_index("date").resample("ME")["forecast"].mean()
                fig_trend = go.Figure()
                fig_trend.add_trace(go.Scatter(
                    x=monthly.index, y=monthly.values, mode="lines+markers",
                    line=dict(color="#0EA5E9", width=2), marker=dict(size=4, color="#0EA5E9"),
                    name="Monthly Mean",
                ))
                fig_trend.update_layout(
                    title="Monthly Trend", height=280,
                    template='plotly_white', margin=dict(t=30, b=10, l=10, r=10),
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(color=THEME_TEXT),
                )
                st.plotly_chart(fig_trend, use_container_width=True, key="fc_trend_monthly")

        with col_t2:
            seasonal = fc.set_index("date")
            seasonal["month"] = seasonal.index.month
            seas_mean = seasonal.groupby("month")["forecast"].mean()
            fig_seas = go.Figure()
            fig_seas.add_trace(go.Bar(
                x=["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"],
                y=seas_mean.values, marker_color="#14B8A6",
                marker_line=dict(color="#0D9488", width=1),
                hovertemplate="%{x}: %{y:.2f}<extra></extra>",
            ))
            fig_seas.update_layout(
                title="Seasonal Profile", height=280,
                template='plotly_white', margin=dict(t=30, b=10, l=10, r=10),
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color=THEME_TEXT),
            )
            st.plotly_chart(fig_seas, use_container_width=True, key="fc_seasonal")

    # Data table
    with st.expander("📋 Raw Forecast Data", expanded=False):
        st.dataframe(fc.head(1000), use_container_width=True)

    # Download
    with st.expander("📥 Download", expanded=False):
        csv = fc.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download CSV",
            csv,
            f"forecast_{district}_{selected_target}.csv",
            "text/csv",
        )
