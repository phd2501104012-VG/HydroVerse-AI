"""DICRA NDVI loader — reads district-level GeoJSON NDVI rasters."""
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict

import pandas as pd
import numpy as np
import json

from utils.logger import log


class DICRALoader:
    def __init__(self, data_root: str = r"D:\DICRA"):
        self.data_root = Path(data_root)

    def scan_folders(self) -> list:
        folders = sorted(self.data_root.glob("NDVI_*"))
        return [f for f in folders if f.is_dir()]

    def load_all(self, cache_path: Optional[str] = None) -> pd.DataFrame:
        cache_path = cache_path or str(Path.cwd() / "data" / "cache" / "dicra_ndvi.parquet")
        cp = Path(cache_path)
        if cp.exists():
            try:
                df = pd.read_parquet(cp)
                log.info(f"Loaded DICRA NDVI from cache ({len(df)} dates, {len(df.columns)} districts)")
                return df
            except Exception as e:
                log.warning(f"DICRA cache read failed: {e}")

        records = []
        folders = self.scan_folders()
        for folder in folders:
            geo_dir = folder / "VECTOR" / "DISTRICT"
            if not geo_dir.exists():
                continue
            for geojson_path in sorted(geo_dir.glob("*.geojson")):
                date_str = geojson_path.stem  # e.g. "01-01-2013"
                try:
                    dt = datetime.strptime(date_str, "%d-%m-%Y")
                except ValueError:
                    continue
                with open(geojson_path) as f:
                    data = json.load(f)
                for feature in data.get("features", []):
                    props = feature.get("properties", {})
                    district = props.get("district_name")
                    zs = props.get("zonalstat", {})
                    ndvi_val = zs.get("mean")
                    if district and ndvi_val is not None:
                        records.append({"date": dt, "district": district, "ndvi": float(ndvi_val)})

        if not records:
            log.warning("No DICRA NDVI records found")
            return pd.DataFrame()

        raw = pd.DataFrame(records)
        pivot = raw.pivot_table(index="date", columns="district", values="ndvi", aggfunc="first")
        pivot.index = pd.to_datetime(pivot.index)
        pivot = pivot.sort_index()
        pivot.columns.name = None

        cp.parent.mkdir(parents=True, exist_ok=True)
        try:
            pivot.to_parquet(cp)
            log.info(f"Cached DICRA NDVI: {len(pivot)} dates, {len(pivot.columns)} districts")
        except Exception as e:
            log.warning(f"DICRA cache write failed: {e}")

        return pivot

    def merge_into(self, data: pd.DataFrame, district: str) -> pd.DataFrame:
        """Merge DICRA NDVI into the main DataFrame for a given district."""
        ndvi_panel = self.load_all()
        if ndvi_panel.empty or district not in ndvi_panel.columns:
            return data
        ndvi_series = ndvi_panel[district]
        ndvi_series = ndvi_series[~ndvi_series.index.duplicated(keep="first")]
        result = data.copy()
        dicra_vals = result.index.to_series().map(ndvi_series)
        existing = result.get("ndvi")
        if existing is not None:
            result["ndvi"] = dicra_vals.fillna(existing)
        else:
            result["ndvi"] = dicra_vals
        return result
