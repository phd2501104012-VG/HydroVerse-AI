from typing import Optional, Dict, List, Tuple, Any
import numpy as np
import pandas as pd
import geopandas as gpd
import xarray as xr
from scipy.ndimage import label, binary_dilation, binary_erosion

from config import CFG, HazardType, RiskLevel
from utils.logger import log


class PixelHazardEngine:
    def __init__(self):
        self._severity_maps: Dict[str, np.ndarray] = {}
        self._hotspots: Dict[str, List[Dict]] = {}
        self._clusters: Dict[str, List[Dict]] = {}

    def compute_pixel_severity(
        self,
        da: xr.DataArray,
        hazard_type: HazardType,
        threshold_low: float = 0.25,
        threshold_high: float = 0.75,
    ) -> np.ndarray:
        data = da.values if isinstance(da, xr.DataArray) else da

        if data.ndim == 3:
            data = np.nanmean(data, axis=0)

        normalized = (data - np.nanmin(data)) / (np.nanmax(data) - np.nanmin(data) + 1e-10)
        normalized = np.clip(normalized, 0, 1)

        severity = np.zeros_like(normalized)
        severity[normalized > threshold_low] = normalized[normalized > threshold_low] * 50
        severity[normalized > threshold_high] = 50 + (normalized[normalized > threshold_high] - threshold_high) * 200
        severity = np.clip(severity, 0, 100)

        key = hazard_type.value
        self._severity_maps[key] = severity
        return severity

    def detect_hotspots(
        self,
        severity: np.ndarray,
        lats: np.ndarray,
        lons: np.ndarray,
        threshold: float = 50.0,
        min_area_pixels: int = 3,
    ) -> List[Dict]:
        binary = (severity >= threshold).astype(int)
        structure = np.ones((3, 3), dtype=int)
        labeled, n_features = label(binary, structure)

        hotspots = []
        for i in range(1, n_features + 1):
            mask = labeled == i
            if mask.sum() < min_area_pixels:
                continue
            y_indices, x_indices = np.where(mask)
            severity_values = severity[mask]

            hotspot = {
                "id": i,
                "pixel_count": int(mask.sum()),
                "mean_severity": float(np.mean(severity_values)),
                "max_severity": float(np.max(severity_values)),
                "centroid_lat": float(np.mean(lats[y_indices])),
                "centroid_lon": float(np.mean(lons[x_indices])),
                "bounds": {
                    "lat_min": float(lats[y_indices].min()),
                    "lat_max": float(lats[y_indices].max()),
                    "lon_min": float(lons[x_indices].min()),
                    "lon_max": float(lons[x_indices].max()),
                },
                "severity_class": self._classify_severity(float(np.mean(severity_values))),
            }
            hotspots.append(hotspot)

        hotspots.sort(key=lambda h: h["mean_severity"], reverse=True)
        key = f"hotspot_{threshold}"
        self._hotspots[key] = hotspots
        return hotspots

    def _classify_severity(self, severity: float) -> str:
        if severity >= 90:
            return "Extreme"
        elif severity >= 75:
            return "Severe"
        elif severity >= 50:
            return "Warning"
        elif severity >= 25:
            return "Watch"
        return "Normal"

    def spatial_clustering(
        self,
        severity: np.ndarray,
        lats: np.ndarray,
        lons: np.ndarray,
        threshold: float = 50.0,
        kernel_size: int = 5,
    ) -> Dict[str, Any]:
        binary = (severity >= threshold).astype(float)
        dilated = binary_dilation(binary, structure=np.ones((kernel_size, kernel_size)))
        eroded = binary_erosion(dilated, structure=np.ones((kernel_size, kernel_size)))
        structure = np.ones((3, 3), dtype=int)
        labeled, n_clusters = label(eroded, structure)

        clusters = []
        total_severe_pixels = 0
        for i in range(1, n_clusters + 1):
            mask = labeled == i
            if mask.sum() < 3:
                continue
            total_severe_pixels += mask.sum()
            y_idx, x_idx = np.where(mask)
            clusters.append({
                "cluster_id": i,
                "pixel_count": int(mask.sum()),
                "mean_severity": float(np.mean(severity[mask])),
                "centroid": (float(np.mean(lats[y_idx])), float(np.mean(lons[x_idx]))),
            })

        return {
            "n_clusters": n_clusters,
            "total_severe_pixels": int(total_severe_pixels),
            "severe_fraction": float(total_severe_pixels / severity.size),
            "clusters": clusters,
        }

    def pixel_level_hazard_index(
        self,
        flood_severity: Optional[np.ndarray] = None,
        drought_severity: Optional[np.ndarray] = None,
        heatwave_severity: Optional[np.ndarray] = None,
        weights: Optional[Dict[str, float]] = None,
    ) -> np.ndarray:
        weights = weights or {"flood": 0.25, "drought": 0.25, "heatwave": 0.25, "agri": 0.25}
        composite = np.zeros_like(next(arr for arr in [flood_severity, drought_severity, heatwave_severity] if arr is not None))
        total_weight = 0

        if flood_severity is not None:
            composite += weights.get("flood", 0.25) * flood_severity
            total_weight += weights.get("flood", 0.25)
        if drought_severity is not None:
            composite += weights.get("drought", 0.25) * drought_severity
            total_weight += weights.get("drought", 0.25)
        if heatwave_severity is not None:
            composite += weights.get("heatwave", 0.25) * heatwave_severity
            total_weight += weights.get("heatwave", 0.25)

        if total_weight > 0:
            composite = composite / total_weight
        return np.clip(composite, 0, 100)

    def extract_extreme_polygons(
        self,
        severity: np.ndarray,
        lats: np.ndarray,
        lons: np.ndarray,
        threshold: float = 75.0,
    ) -> List[Dict]:
        from skimage import measure
        binary = (severity >= threshold).astype(np.int8)
        contours = measure.find_contours(binary, level=0.5)

        polygons = []
        for i, contour in enumerate(contours):
            if len(contour) < 4:
                continue
            y_idx = contour[:, 0].astype(int)
            x_idx = contour[:, 1].astype(int)
            y_idx = np.clip(y_idx, 0, len(lats) - 1)
            x_idx = np.clip(x_idx, 0, len(lons) - 1)

            polygon_coords = []
            for j in range(len(contour)):
                yi = min(int(contour[j, 0]), len(lats) - 1)
                xi = min(int(contour[j, 1]), len(lons) - 1)
                polygon_coords.append([float(lons[xi]), float(lats[yi])])

            if len(polygon_coords) >= 3:
                polygons.append({
                    "id": i,
                    "coordinates": polygon_coords,
                    "mean_severity": float(np.mean(severity[y_idx, x_idx])),
                    "area_pixels": len(contour),
                })

        return polygons

    def hazard_density_map(
        self,
        severity: np.ndarray,
        kernel_size: int = 10,
    ) -> np.ndarray:
        from scipy.ndimage import uniform_filter
        return uniform_filter(severity, size=kernel_size)
