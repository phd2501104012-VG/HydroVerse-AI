from typing import Optional, Dict, List, Tuple, Any
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
import folium
from folium import plugins
from branca.colormap import LinearColormap
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from config import CFG, constants
from utils.logger import log


class SpatialVisualizer:
    def __init__(self):
        self._colormaps = {
            "flood": LinearColormap(["#dbeafe", "#3b82f6", "#1e3a8a"], vmin=0, vmax=100),
            "drought": LinearColormap(["#fef3c7", "#d97706", "#7c2d12"], vmin=0, vmax=100),
            "heatwave": LinearColormap(["#fee2e2", "#dc2626", "#7f1d1d"], vmin=0, vmax=100),
            "agri_stress": LinearColormap(["#dcfce7", "#16a34a", "#14532d"], vmin=0, vmax=100),
            "compound": LinearColormap(["#fae8ff", "#d946ef", "#701a75"], vmin=0, vmax=100),
        }

    def choropleth_map(
        self,
        gdf: gpd.GeoDataFrame,
        severity_series: pd.Series,
        hazard_name: str,
        title: str = "",
        center: Optional[Tuple[float, float]] = None,
        zoom: int = 7,
        save_path: Optional[str] = None,
    ) -> folium.Map:
        center = center or (23.5, 78.5)
        m = folium.Map(location=center, zoom_start=zoom, tiles="CartoDB positron")
        cmap = self._colormaps.get(hazard_name, LinearColormap(["#e2e8f0", "#3b82f6"], vmin=0, vmax=100))
        cmap.caption = f"{title} — severity (0-100)"

        for _, row in gdf.iterrows():
            d = row[CFG.district_col]
            val = float(severity_series.get(d, 0))
            color = cmap(val)

            tooltip_html = f"""
            <div style="font-family: 'Segoe UI', sans-serif; padding: 8px;">
                <b style="font-size: 14px;">{d}</b><br>
                <span style="color: {color}; font-size: 12px;">● {hazard_name.title()}</span><br>
                <span style="font-size: 18px; font-weight: bold;">{val:.1f}</span>
                <span style="font-size: 12px;">/ 100</span>
            </div>
            """

            folium.GeoJson(
                row.geometry.__geo_interface__,
                style_function=lambda x, c=color: {
                    "fillColor": c,
                    "color": "#475569",
                    "weight": 1,
                    "fillOpacity": 0.7,
                },
                highlight_function=lambda x: {
                    "fillOpacity": 0.9,
                    "weight": 2,
                    "color": "#1e293b",
                },
                tooltip=folium.Tooltip(tooltip_html),
                popup=folium.Popup(f"<b>{d}</b><br>{hazard_name}: {val:.1f}"),
            ).add_to(m)

        cmap.add_to(m)
        folium.LayerControl().add_to(m)

        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            m.save(save_path)
            log.info(f"Map saved: {save_path}")
        return m

    def heatmap_layer(
        self,
        gdf: gpd.GeoDataFrame,
        severity_series: pd.Series,
        hazard_name: str,
        center: Optional[Tuple[float, float]] = None,
        zoom: int = 7,
    ) -> folium.Map:
        center = center or (23.5, 78.5)
        m = folium.Map(location=center, zoom_start=zoom, tiles="CartoDB dark_matter")
        cmap = self._colormaps.get(hazard_name, LinearColormap(["#e2e8f0", "#3b82f6"], vmin=0, vmax=100))

        heat_data = []
        for _, row in gdf.iterrows():
            d = row[CFG.district_col]
            val = float(severity_series.get(d, 0))
            centroid = row.geometry.centroid
            intensity = val / 100.0
            heat_data.append([centroid.y, centroid.x, intensity])

        plugins.HeatMap(heat_data, radius=25, blur=10, max_zoom=10).add_to(m)

        for _, row in gdf.iterrows():
            d = row[CFG.district_col]
            folium.GeoJson(
                row.geometry.__geo_interface__,
                style_function=lambda x: {
                    "fillColor": "transparent",
                    "color": "#475569",
                    "weight": 0.5,
                    "fillOpacity": 0,
                },
            ).add_to(m)

        return m

    def severity_timelapse(
        self,
        data: pd.DataFrame,
        district_col: str = "NAME_2",
        date_col: str = "date",
        severity_col: str = "severity",
        title: str = "Severity Timeline",
    ) -> go.Figure:
        fig = px.choropleth(
            data,
            locations=district_col,
            color=severity_col,
            animation_frame=date_col,
            color_continuous_scale="RdYlGn_r",
            range_color=[0, 100],
            title=title,
        )
        fig.update_layout(
            template='plotly_white',
            geo=dict(scope="asia"),
        )
        return fig

    def interactive_hover_map(
        self,
        gdf: gpd.GeoDataFrame,
        severity_data: Dict[str, float],
        hazard_data: Dict[str, Dict[str, float]],
        title: str = "Interactive Hazard Map",
    ) -> go.Figure:
        fig = go.Figure()

        for hazard_name, color in constants.HAZARD_COLORS.items():
            if hazard_name not in severity_data:
                continue
            values = [severity_data[hazard_name].get(d, 0) for d in gdf[CFG.district_col]]
            fig.add_trace(go.Choroplethmapbox(
                geojson=gdf.__geo_interface__,
                locations=gdf.index,
                z=values,
                colorscale=[[0, "#22c55e"], [0.25, "#eab308"], [0.5, "#f97316"], [0.75, "#ef4444"], [1, "#7f1d1d"]],
                zmin=0,
                zmax=100,
                name=hazard_name.title(),
                visible=(hazard_name == list(severity_data.keys())[0]),
                hovertemplate="<b>%{customdata[0]}</b><br>"
                            + f"{hazard_name.title()}: "
                            + "%{z:.1f}<extra></extra>",
                customdata=gdf[[CFG.district_col]].values,
            ))

        buttons = []
        for i, hazard_name in enumerate(severity_data.keys()):
            visibility = [False] * len(severity_data)
            visibility[i] = True
            buttons.append(dict(
                label=hazard_name.title(),
                method="update",
                args=[{"visible": visibility}],
            ))

        fig.update_layout(
            updatemenus=[dict(
                type="buttons",
                direction="down",
                buttons=buttons,
                x=0.05,
                y=0.95,
            )],
            mapbox=dict(
                style="carto-darkmatter",
                center=dict(lat=23.5, lon=78.5),
                zoom=6,
            ),
            title=title,
            height=700,
        )
        return fig

    def hazard_category_map(
        self,
        gdf: gpd.GeoDataFrame,
        classifications: Dict[str, str],
        hazard_name: str,
    ) -> folium.Map:
        m = folium.Map(location=[23.5, 78.5], zoom_start=7, tiles="CartoDB dark_matter")

        category_colors = {
            "Normal": "#22c55e",
            "Moderate": "#eab308",
            "Severe": "#f97316",
            "Extreme": "#ef4444",
        }

        for _, row in gdf.iterrows():
            d = row[CFG.district_col]
            cat = classifications.get(d, "Normal")
            color = category_colors.get(cat, "#22c55e")

            folium.GeoJson(
                row.geometry.__geo_interface__,
                style_function=lambda x, c=color: {
                    "fillColor": c,
                    "color": "#1e293b",
                    "weight": 1,
                    "fillOpacity": 0.8,
                },
                tooltip=folium.Tooltip(f"<b>{d}</b><br>{hazard_name}: <b>{cat}</b>"),
            ).add_to(m)

        legend_html = """
        <div style="position: fixed; bottom: 20px; right: 20px; z-index: 1000;
                    background: rgba(15, 23, 42, 0.9); padding: 12px; border-radius: 8px;
                    font-family: 'Segoe UI', sans-serif;">
        """
        for cat, col in category_colors.items():
            legend_html += f'<p><span style="background:{col}; width:12px; height:12px; display:inline-block; border-radius:2px;"></span> {cat}</p>'
        legend_html += "</div>"
        m.get_root().html.add_child(folium.Element(legend_html))

        return m
