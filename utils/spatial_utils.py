from typing import Optional, List, Dict, Tuple
import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import shape
from pathlib import Path

from config import CFG
from utils.logger import log
from utils.gee_utils import shapely_to_ee


def load_district_boundaries() -> gpd.GeoDataFrame:
    shp_candidates = CFG.shp_candidates
    state_col = CFG.state_col
    district_col = CFG.district_col
    state_name = CFG.state_name

    gdf_all = None
    for p in shp_candidates:
        if Path(p).exists():
            try:
                gdf_all = gpd.read_file(p)
                if state_col in gdf_all.columns and district_col in gdf_all.columns:
                    log.info(f"Loaded boundaries: {p} ({len(gdf_all)} features)")
                    break
            except Exception as e:
                log.warning(f"Failed to read {p}: {e}")

    if gdf_all is None:
        raise FileNotFoundError("No shapefile found. Check SHP_CANDIDATES paths.")

    mask = gdf_all[state_col].astype(str).str.contains(state_name, case=False, na=False)
    gdf_state = gdf_all[mask].copy()
    gdf_state[district_col] = gdf_state[district_col].astype(str).str.strip().str.title()
    gdf_state = gdf_state.reset_index(drop=True)
    if gdf_state.crs is None:
        gdf_state.set_crs("EPSG:4326", inplace=True)

    log.info(f"{len(gdf_state)} {state_name} districts loaded")
    return gdf_state


def find_district(name: str, available: List[str]) -> Optional[str]:
    name_norm = name.strip().lower()
    for a in available:
        if a.lower() == name_norm:
            return a
    for a in available:
        if name_norm in a.lower() or a.lower() in name_norm:
            return a
    return None


def filter_districts(gdf: gpd.GeoDataFrame, district_names: List[str]) -> gpd.GeoDataFrame:
    district_col = CFG.district_col
    available = gdf[district_col].unique().tolist()
    matched = {}
    for d in district_names:
        m = find_district(d, available)
        if m is not None:
            matched[d] = m
        else:
            log.warning(f"District '{d}' not found in shapefile")

    result = gdf[gdf[district_col].isin(matched.values())].copy().reset_index(drop=True)
    result = result[[district_col, "geometry"]].rename(columns={district_col: "district"})
    return result


def compute_centroids(gdf: gpd.GeoDataFrame) -> Dict[str, Tuple[float, float]]:
    return {
        row["district"]: (row.geometry.centroid.y, row.geometry.centroid.x)
        for _, row in gdf.iterrows()
    }


def build_ee_features(gdf: gpd.GeoDataFrame):
    ee_features = []
    ee_geom = {}
    for _, row in gdf.iterrows():
        d = row["district"]
        eeg = shapely_to_ee(row.geometry)
        if eeg is not None:
            ee_geom[d] = eeg
            ee_features.append(ee_eature(eeg, {"district": d}))
    return ee_geom, ee.FeatureCollection(ee_features) if ee_features else None


def clip_raster_to_district(da, geometry):
    import rioxarray
    if da.rio.crs is None:
        da = da.rio.write_crs("EPSG:4326")
    return da.rio.clip([geometry], from_disk=True)
