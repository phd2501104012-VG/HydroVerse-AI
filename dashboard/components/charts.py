"""Scientific chart components with premium styling — no cropping, proper margins."""
import streamlit as st
from typing import Optional, Dict, List, Any
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from dashboard.config import *


def sci_layout(title="", height=320, margin_t=40, margin_b=20, margin_l=50, margin_r=20,
               xaxis_title=None, yaxis_title=None):
    """Base scientific layout — generous margins prevent cropping."""
    return dict(
        template='plotly_white',
        height=height,
        margin=dict(t=margin_t, b=margin_b, l=margin_l, r=margin_r),
        hovermode="x unified",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=THEME_TEXT, family='Inter', size=12),
        xaxis=dict(gridcolor=THEME_BORDER, gridwidth=0.5, title=xaxis_title,
                   showspikes=True, spikemode="across", spikedash="dot",
                   automargin=True),
        yaxis=dict(gridcolor=THEME_BORDER, gridwidth=0.5, title=yaxis_title,
                   showspikes=True, spikemode="across", spikedash="dot",
                   automargin=True),
        hoverlabel=dict(bgcolor="white", font_size=12, font_family="Inter"),
    )


def render_time_series(
    df: pd.DataFrame,
    columns: List[str],
    title: str = "Time Series",
    height: int = 380,
    date_col: str = "date",
    colors: Optional[List[str]] = None,
    show_legend: bool = True,
    show_range_selector: bool = False,
):
    colors = colors or ["#0EA5E9", "#14B8A6", "#F97316", "#8B5CF6", "#EF4444"]
    fig = go.Figure()
    x_vals = df[date_col] if date_col in df.columns else df.index
    for i, col in enumerate(columns):
        if col not in df.columns:
            continue
        fig.add_trace(go.Scatter(
            x=x_vals, y=df[col], mode="lines", name=col,
            line=dict(width=2, color=colors[i % len(colors)]),
            hovertemplate="%{x|%b %d, %Y}<br>%{y:.2f}<extra>" + col + "</extra>",
        ))
    layout = sci_layout(title, height)
    layout.update(
        hovermode="x unified",
        showlegend=show_legend,
        legend=dict(orientation="h", y=1.12, font=dict(size=11)),
    )
    if show_range_selector:
        layout["xaxis"].update(
            rangeselector=dict(
                buttons=list([
                    dict(count=1, label="1m", step="month", stepmode="backward"),
                    dict(count=6, label="6m", step="month", stepmode="backward"),
                    dict(count=1, label="1y", step="year", stepmode="backward"),
                    dict(step="all"),
                ]),
                bgcolor="white", activecolor=THEME_PRIMARY,
                font=dict(size=10),
            ),
            rangeslider=dict(visible=True, thickness=0.06),
        )
        layout["margin"].update(t=70)
    fig.update_layout(**layout)
    st.plotly_chart(fig, use_container_width=True, key=f"ts_{'_'.join(columns[:2])}_{title[:20]}")


def render_risk_gauge(severity: float, title: str = "Risk Level", height: int = 220):
    color = "#22C55E"
    label = "Normal"
    if severity >= 90:
        color = "#7F1D1D"; label = "Extreme"
    elif severity >= 75:
        color = "#EF4444"; label = "Severe"
    elif severity >= 50:
        color = "#F97316"; label = "Warning"
    elif severity >= 25:
        color = "#EAB308"; label = "Watch"

    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=severity,
        domain=dict(x=[0.05, 0.95], y=[0.05, 0.95]),
        title=dict(text=title, font=dict(color=THEME_TEXT, size=14, family='Inter')),
        number=dict(font=dict(color=color, size=30, family='Plus Jakarta Sans')),
        delta=dict(reference=50, increasing=dict(color="#EF4444"), decreasing=dict(color="#22C55E")),
        gauge=dict(
            axis=dict(range=[0, 100], tickwidth=1, tickcolor="#94A3B8",
                      tickvals=[0, 25, 50, 75, 100],
                      ticktext=["0", "25", "50", "75", "100"]),
            bar=dict(color=color, thickness=0.5),
            bgcolor="rgba(0,0,0,0)",
            borderwidth=0,
            steps=[
                dict(range=[0, 25], color="#F0FDF4"),
                dict(range=[25, 50], color="#FEFCE8"),
                dict(range=[50, 75], color="#FFF7ED"),
                dict(range=[75, 90], color="#FEF2F2"),
                dict(range=[90, 100], color="#FEF2F2"),
            ],
            threshold=dict(line=dict(color=color, width=5), thickness=0.85, value=severity),
        ),
    ))
    fig.update_layout(
        height=240,
        margin=dict(t=50, b=25, l=25, r=25),
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color=THEME_TEXT, size=12),
    )
    fig.add_annotation(x=0.5, y=0.08,
        text=f'<span style="font-size:0.85rem;font-weight:700;color:{color}">{label}</span>',
        showarrow=False, xref="paper", yref="paper", xanchor="center")
    st.plotly_chart(fig, use_container_width=True, key=f"gauge_{title[:10]}_{severity:.0f}")


def render_hazard_radar(hazard_severities: Dict[str, float], height: int = 380, key_suffix: str = ""):
    categories = list(hazard_severities.keys())
    labels = {"flood": "Flood", "drought": "Drought", "heatwave": "Heatwave",
              "agri_stress": "Agri Stress", "compound": "Compound"}
    cats = [labels.get(k, k.title()) for k in categories]
    vals = [hazard_severities[k] for k in categories]
    vals += vals[:1]
    cats += cats[:1]

    fig = go.Figure(go.Scatterpolar(
        r=vals, theta=cats, fill='toself',
        line=dict(color="#0EA5E9", width=2.5),
        fillcolor="rgba(14,165,233,0.12)",
        hovertemplate="%{theta}: %{r:.1f}<extra></extra>",
    ))
    fig.update_layout(
        template='plotly_white', height=height,
        margin=dict(t=30, b=30, l=60, r=60),
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color=THEME_TEXT, family='Inter', size=12),
        polar=dict(
            radialaxis=dict(range=[0, 100], tickvals=[25, 50, 75],
                           tickcolor="#94A3B8", gridcolor=THEME_BORDER,
                           linecolor=THEME_BORDER,
                           tickfont=dict(size=10)),
            angularaxis=dict(gridcolor=THEME_BORDER, linecolor=THEME_BORDER,
                            tickfont=dict(size=11)),
            bgcolor="rgba(0,0,0,0)",
        ),
    )
    st.plotly_chart(fig, use_container_width=True,
                    key=f"hazard_radar_{key_suffix}" if key_suffix else "hazard_radar")


def render_compound_hazard_matrix(hazard_severities: Dict[str, float], height: int = 420):
    hazards = list(hazard_severities.keys())
    n = len(hazards)
    matrix = np.zeros((n, n))
    for i, h1 in enumerate(hazards):
        for j, h2 in enumerate(hazards):
            if i == j:
                matrix[i, j] = hazard_severities[h1]
            else:
                matrix[i, j] = (hazard_severities[h1] + hazard_severities[h2]) / 2

    labels = {"flood": "Flood", "drought": "Drought", "heatwave": "Heatwave",
              "agri_stress": "Agri Stress", "compound": "Compound"}
    xlabels = [labels.get(h, h) for h in hazards]
    fig = go.Figure(data=go.Heatmap(
        z=matrix, x=xlabels, y=xlabels,
        colorscale=[[0, "#F0FDF4"], [0.25, "#FEFCE8"], [0.5, "#FFF7ED"],
                    [0.75, "#FEF2F2"], [1, "#7F1D1D"]],
        zmin=0, zmax=100, text=matrix.round(1), texttemplate="%{text}",
        hovertemplate="%{x} × %{y}<br>Composite: %{z:.1f}<extra></extra>",
    ))
    fig.update_layout(
        title=dict(text="Compound Hazard Interaction", font=dict(size=13, family='Inter')),
        template='plotly_white', height=height,
        margin=dict(t=50, b=20, l=20, r=20),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=THEME_TEXT, family='Inter', size=11),
        xaxis=dict(gridcolor=THEME_BORDER, automargin=True),
        yaxis=dict(gridcolor=THEME_BORDER, automargin=True),
    )
    st.plotly_chart(fig, use_container_width=True, key="compound_matrix")


def render_hazard_severity_panel(df: pd.DataFrame, hazards: List[str], height: int = 520):
    fig = make_subplots(rows=2, cols=2,
        subplot_titles=[HAZARD_NAMES.get(h, h.title()) for h in hazards[:4]],
        vertical_spacing=0.15,
        horizontal_spacing=0.08)
    positions = [(1, 1), (1, 2), (2, 1), (2, 2)]
    colors = {"flood": "#0EA5E9", "drought": "#EAB308", "heatwave": "#EF4444", "agri_stress": "#14B8A6"}
    for h, (r, c) in zip(hazards[:4], positions):
        col = f"{h}_severity"
        if col not in df.columns:
            continue
        fig.add_trace(go.Scatter(
            x=df.index, y=df[col], mode="lines", fill="tozeroy",
            line=dict(color=colors.get(h, "#888"), width=1.5),
            showlegend=False,
        ), row=r, col=c)
        fig.update_yaxes(range=[0, 100], row=r, col=c,
                        tickvals=[0, 25, 50, 75, 100],
                        tickfont=dict(size=9))
        fig.update_xaxes(tickfont=dict(size=8), row=r, col=c)
        for thr in [50, 75]:
            fig.add_hline(y=thr, line_dash="dash", line_color=THEME_BORDER, line_width=0.5, row=r, col=c)
    fig.update_layout(
        height=height, template='plotly_white',
        title=dict(text="Hazard Severity Timeline", font=dict(size=14)),
        margin=dict(t=50, b=20, l=20, r=20),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=THEME_TEXT, size=11),
    )
    st.plotly_chart(fig, use_container_width=True, key="sev_panel")


def render_forecast_chart(historical: Optional[pd.Series], forecast: pd.DataFrame, target: str, title: str = "Forecast"):
    fig = go.Figure()
    if historical is not None:
        fig.add_trace(go.Scatter(
            x=historical.index, y=historical.values, mode="lines",
            name="Historical (2y)", line=dict(color="#94A3B8", width=1.5),
            hovertemplate="%{x|%b %d, %Y}<br>%{y:.2f}</extra>Historical</extra>"))
    color_map = {"ML": "#0EA5E9", "Blend": "#F97316", "CMIP6": "#14B8A6", "Climatology": "#94A3B8"}
    for src in forecast["source"].unique():
        sub = forecast[forecast["source"] == src]
        fig.add_trace(go.Scatter(
            x=pd.to_datetime(sub["date"]), y=sub["forecast"], mode="lines",
            name=src, line=dict(color=color_map.get(src, "#888"), width=2.5 if src == "ML" else 1.5),
            hovertemplate="%{x|%b %d, %Y}<br>%{y:.2f}<extra>" + src + "</extra>"))
    if "lower" in forecast.columns and "upper" in forecast.columns:
        fig.add_trace(go.Scatter(
            x=list(pd.to_datetime(forecast["date"])) + list(pd.to_datetime(forecast["date"]))[::-1],
            y=list(forecast["upper"]) + list(forecast["lower"])[::-1],
            fill="toself", fillcolor="rgba(14,165,233,0.08)", line=dict(color="rgba(0,0,0,0)"),
            name="95% CI", hoverinfo="skip"))
    ml_rows = forecast[forecast["source"] == "ML"]
    if len(ml_rows) > 0:
        seam = pd.to_datetime(ml_rows["date"].max())
        fig.add_vline(x=seam, line_dash="dash", line_color="#0EA5E9", line_width=1)
        fig.add_annotation(x=seam, y=1, text="ML → CMIP6", showarrow=False, xref="x", yref="paper",
                          font=dict(size=10, color="#0EA5E9"))
    layout = sci_layout(title, 440, margin_t=50, margin_b=30)
    layout.update(
        legend=dict(orientation="h", y=1.1, font=dict(size=11)),
        xaxis=dict(gridcolor=THEME_BORDER, title="Date", automargin=True),
        yaxis=dict(gridcolor=THEME_BORDER, title=target.replace("_", " ").title(), automargin=True),
    )
    fig.update_layout(**layout)
    st.plotly_chart(fig, use_container_width=True, key=f"fc_chart_{target}")


def render_individual_chart(ov_df, col, label, unit, clr, marker_vars, district, prefix="chart"):
    """Render a single styled time series chart — ample margins, no cropping."""
    series = ov_df[col]
    if not series.notna().any():
        st.info(f"No {label} data available")
        return
    use_markers = col in marker_vars or series.notna().sum() < len(series) * 0.1
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=ov_df.index, y=series, mode="lines+markers" if use_markers else "lines",
        name=col, line=dict(width=2, color=clr),
        marker=dict(size=5, color=clr, symbol="circle") if use_markers else None,
        hovertemplate="%{x|%b %d, %Y}<br>%{y:.2f} " + unit + "<extra>" + label + "</extra>",
    ))
    ytitle = f"{label} ({unit})" if unit else label
    fig.update_layout(**sci_layout(f"{label} — {district}", 300, margin_t=45, margin_l=60))
    fig.update_layout(yaxis_title=ytitle)
    st.plotly_chart(fig, use_container_width=True, key=f"{prefix}_{col}")
