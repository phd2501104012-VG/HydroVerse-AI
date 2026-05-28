"""Interactive hazard maps with folium and Plotly."""
import streamlit as st
from typing import Optional, Dict, List, Any
import pandas as pd
import geopandas as gpd
import plotly.graph_objects as go
from dashboard.config import *
from geospatial.visualization import SpatialVisualizer


def render_hazard_map(
    gdf: gpd.GeoDataFrame,
    severity_series: pd.Series,
    hazard_name: str,
    title: str = "",
    height: int = 600,
):
    viz = SpatialVisualizer()
    try:
        m = viz.choropleth_map(
            gdf, severity_series, hazard_name,
            title=title or f"{hazard_name.title()} Hazard Map — Madhya Pradesh",
            center=(23.5, 78.5),
        )
    except Exception:
        m = viz.choropleth_map(
            gdf, severity_series, hazard_name,
            title=title or f"{hazard_name.title()} Hazard Map",
            center=(23.5, 78.5),
        )

    html = m.get_root().render()
    st.components.v1.html(html, height=height, scrolling=False)


def render_hazard_map_selector(
    gdf: gpd.GeoDataFrame,
    severity_data: Dict[str, pd.Series],
):
    hazard_names = list(severity_data.keys())
    if not hazard_names:
        st.warning("No hazard data available")
        return

    col1, col2 = st.columns([1, 3])
    with col1:
        selected = st.selectbox(
            "Hazard Layer",
            hazard_names,
            format_func=lambda h: HAZARD_NAMES.get(h, h.title()),
        )

        st.markdown("##### Layer Controls")
        show_boundaries = st.checkbox("District Boundaries", value=True)
        show_labels = st.checkbox("District Labels", value=True)
        opacity = st.slider("Layer Opacity", 0.3, 1.0, 0.8, 0.1)

        st.markdown("##### Legend")
        sev = severity_data.get(selected)
        if sev is not None:
            vmin, vmax = sev.min(), sev.max()
            st.markdown(f"""
            <div style="font-size:0.7rem;color:var(--text-secondary);">
                <span style="color:#22C55E;">●</span> Normal (0–25)<br>
                <span style="color:#EAB308;">●</span> Watch (25–50)<br>
                <span style="color:#F97316;">●</span> Warning (50–75)<br>
                <span style="color:#EF4444;">●</span> Severe (75–90)<br>
                <span style="color:#7F1D1D;">●</span> Extreme (90+)
            </div>
            """, unsafe_allow_html=True)
            st.caption(f"Range: {vmin:.0f} – {vmax:.0f}")

    with col2:
        if selected in severity_data:
            render_hazard_map(gdf, severity_data[selected], selected)


def render_hazard_heatmap(
    gdf: gpd.GeoDataFrame,
    severity_series: pd.Series,
    hazard_name: str,
):
    viz = SpatialVisualizer()
    m = viz.heatmap_layer(gdf, severity_series, hazard_name)
    html = m.get_root().render()
    st.components.v1.html(html, height=500, scrolling=False)


def render_pixel_hazard_overlay(
    lats: List[float],
    lons: List[float],
    severity_grid: List[List[float]],
    title: str = "Pixel-Level Hazard",
):
    fig = go.Figure(data=go.Heatmap(
        z=severity_grid, x=lons, y=lats,
        colorscale=[[0, "#22c55e"], [0.25, "#eab308"], [0.5, "#f97316"],
                    [0.75, "#ef4444"], [1, "#7f1d1d"]],
        zmin=0, zmax=100,
        colorbar=dict(title="Severity", thickness=15, len=0.7),
        hovertemplate="Lat: %{y:.2f}<br>Lon: %{x:.2f}<br>Severity: %{z:.1f}<extra></extra>",
    ))
    fig.update_layout(
        title=dict(text=title, font=dict(size=13)),
        template='plotly_white', height=550,
        xaxis_title="Longitude", yaxis_title="Latitude",
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(t=40, b=20, l=20, r=20),
    )
    st.plotly_chart(fig, use_container_width=True, key=f"pixel_map_{title[:20]}")


def render_hotspot_map(
    hotspots: List[Dict],
    gdf: gpd.GeoDataFrame,
    map_center=(23.5, 78.5),
):
    import folium
    m = folium.Map(location=map_center, zoom_start=7, tiles="CartoDB positron",
                   control_scale=True)

    folium.TileLayer("openstreetmap", name="OpenStreetMap", control=True).add_to(m)
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri", name="Satellite", control=True,
    ).add_to(m)

    for _, row in gdf.iterrows():
        folium.GeoJson(
            row.geometry.__geo_interface__,
            style_function=lambda x: {
                "fillColor": "transparent",
                "color": "#475569",
                "weight": 0.5,
                "fillOpacity": 0,
            },
        ).add_to(m)

    for hotspot in hotspots:
        severity = hotspot["mean_severity"]
        color = "#ef4444" if severity >= 75 else "#f97316" if severity >= 50 else "#eab308"
        size = max(5, min(20, hotspot["pixel_count"] / 10))
        folium.CircleMarker(
            location=[hotspot["centroid_lat"], hotspot["centroid_lon"]],
            radius=size,
            color=color,
            fill=True,
            fillColor=color,
            fillOpacity=0.7,
            popup=folium.Popup(f"Severity: {severity:.1f}<br>Area: {hotspot['pixel_count']} pixels"),
            tooltip=f"⚠️ {severity:.1f}",
        ).add_to(m)

    folium.LayerControl().add_to(m)
    html = m.get_root().render()
    st.components.v1.html(html, height=600, scrolling=False)
