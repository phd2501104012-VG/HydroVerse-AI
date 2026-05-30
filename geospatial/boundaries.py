from typing import Optional, List, Dict, Tuple
from pathlib import Path

import geopandas as gpd
import pandas as pd
import numpy as np

from config import CFG
from utils.logger import log
from utils.gee_utils import shapely_to_ee

try:
    import ee
    _HAS_EE = True
except ImportError:
    ee = None
    _HAS_EE = False


class DistrictBoundaries:
    def __init__(self):
        self._gdf: Optional[gpd.GeoDataFrame] = None
        self._ee_fc: Any = None
        self._centroids: Dict[str, Tuple[float, float]] = {}
        self._ee_geom: Dict[str, Any] = {}

    def load(self, force_reload: bool = False) -> gpd.GeoDataFrame:
        if self._gdf is not None and not force_reload:
            return self._gdf

        self._state_boundary: Optional[gpd.GeoDataFrame] = None

        for p in CFG.shp_candidates:
            if Path(p).exists():
                try:
                    gdf = gpd.read_file(p)
                    if CFG.state_col in gdf.columns and CFG.district_col in gdf.columns:
                        self._gdf = self._filter_state(gdf)
                        log.info(f"Loaded {len(self._gdf)} districts from {p}")
                        return self._gdf
                    elif CFG.state_col in gdf.columns and CFG.district_col not in gdf.columns:
                        # State-level shapefile only — store as boundary backdrop
                        state_gdf = gdf[gdf[CFG.state_col].astype(str).str.contains(CFG.state_name, case=False, na=False)].copy()
                        if state_gdf.crs is not None and state_gdf.crs.to_string() != "EPSG:4326":
                            state_gdf = state_gdf.to_crs("EPSG:4326")
                        self._state_boundary = state_gdf
                        log.info(f"Stored state boundary from {p} ({len(state_gdf)} row(s))")
                except Exception as e:
                    log.warning(f"Failed to load {p}: {e}")

        log.info("No district shapefile found — generating synthetic district hex grid")
        self._gdf = self._generate_hex_gdf()
        return self._gdf

    @staticmethod
    def _generate_hex_gdf():
        """Generate a synthetic GeoDataFrame with hexagon districts for MP."""
        from shapely.geometry import Polygon
        import math

        districts = CFG.all_mp_districts
        n = len(districts)
        # Approximate MP bounding box (lat/lon)
        min_lat, max_lat = 21.5, 26.5
        min_lon, max_lon = 74.0, 82.5
        cols = 8
        rows = math.ceil(n / cols)
        hex_r = 0.35
        w = hex_r * math.sqrt(3)
        h = hex_r * 2
        step_y = hex_r * 1.5

        features = []
        for i, name in enumerate(districts):
            col = i % cols
            row = i // cols
            cx = min_lon + col * w + (0 if row % 2 == 0 else w / 2)
            cy = max_lat - row * step_y
            pts = []
            for a in range(6):
                ang = math.radians(60 * a - 30)
                pts.append((cx + hex_r * math.cos(ang), cy + hex_r * math.sin(ang)))
            poly = Polygon(pts)
            features.append({"NAME_2": name, "NAME_1": "Madhya Pradesh", "geometry": poly})

        gdf = gpd.GeoDataFrame(features, crs="EPSG:4326")
        log.info(f"Generated synthetic hex grid with {len(gdf)} districts")
        return gdf

    def _filter_state(self, gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        mask = gdf[CFG.state_col].astype(str).str.contains(CFG.state_name, case=False, na=False)
        result = gdf[mask].copy()
        result[CFG.district_col] = result[CFG.district_col].astype(str).str.strip().str.title()
        result = result.reset_index(drop=True)
        if result.crs is None:
            result.set_crs("EPSG:4326", inplace=True)
        return result

    def get_filtered(self, district_names: Optional[List[str]] = None) -> gpd.GeoDataFrame:
        gdf = self.load()
        district_col = CFG.district_col
        if district_names is None:
            district_names = CFG.target_districts

        available = gdf[district_col].unique().tolist()
        matched = []
        for d in district_names:
            d_norm = d.strip().lower()
            found = None
            for a in available:
                if a.lower() == d_norm:
                    found = a
                    break
            if found is None:
                for a in available:
                    if d_norm in a.lower() or a.lower() in d_norm:
                        found = a
                        break
            if found:
                matched.append(found)
            else:
                log.warning(f"District '{d}' not found")

        result = gdf[gdf[district_col].isin(matched)].copy().reset_index(drop=True)
        result = result[[district_col, "geometry"]].rename(columns={district_col: "district"})
        return result

    def get_ee_fc(self, district_names: Optional[List[str]] = None) -> Any:
        if not _HAS_EE:
            log.warning("GEE not available — cannot create FeatureCollection")
            return None
        gdf = self.get_filtered(district_names)
        features = []
        for _, row in gdf.iterrows():
            eeg = shapely_to_ee(row.geometry)
            if eeg is not None:
                self._ee_geom[row["district"]] = eeg
                features.append(ee.Feature(eeg, {"district": row["district"]}))
        self._ee_fc = ee.FeatureCollection(features)
        return self._ee_fc

    def get_centroids(self, district_names: Optional[List[str]] = None) -> Dict[str, Tuple[float, float]]:
        gdf = self.get_filtered(district_names)
        return {
            row["district"]: (row.geometry.centroid.y, row.geometry.centroid.x)
            for _, row in gdf.iterrows()
        }

    def get_district_geometry(self, district_name: str):
        gdf = self.get_filtered([district_name])
        if gdf.empty:
            return None
        return gdf.geometry.iloc[0]

    @property
    def district_names(self) -> List[str]:
        return self.all_districts

    @property
    def all_districts(self) -> List[str]:
        gdf = self.load()
        return gdf[CFG.district_col].unique().tolist()

    @property
    def gdf(self) -> gpd.GeoDataFrame:
        return self.load()

    @property
    def state_boundary(self) -> Optional[gpd.GeoDataFrame]:
        """Return the state-level boundary (backdrop) if loaded from shapefile."""
        self.load()
        return self._state_boundary
