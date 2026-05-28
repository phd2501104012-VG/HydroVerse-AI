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

        for p in CFG.shp_candidates:
            if Path(p).exists():
                try:
                    gdf = gpd.read_file(p)
                    if CFG.state_col in gdf.columns and CFG.district_col in gdf.columns:
                        self._gdf = self._filter_state(gdf)
                        log.info(f"Loaded {len(self._gdf)} districts from {p}")
                        return self._gdf
                except Exception as e:
                    log.warning(f"Failed to load {p}: {e}")

        raise FileNotFoundError("No shapefile found")

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
