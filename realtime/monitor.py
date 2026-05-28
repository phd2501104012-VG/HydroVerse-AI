from typing import Optional, Dict, List, Any
from datetime import datetime, timedelta
import time
import pandas as pd
import numpy as np
from pathlib import Path

from config import CFG, constants
from utils.logger import log
from utils.cache import cache_manager


class RealtimeMonitor:
    def __init__(self):
        self._latest_data: Dict[str, pd.DataFrame] = {}
        self._anomaly_scores: Dict[str, pd.Series] = {}
        self._update_count: int = 0
        self._running: bool = False

    def fetch_realtime_precip(self, ee_fc, hours_back: int = 24) -> Optional[pd.DataFrame]:
        try:
            import ee
            coll = (
                ee.ImageCollection(constants.REALTIME_DATASETS["precip"]["collection"])
                .filterBounds(ee_fc)
                .filterDate(
                    (datetime.utcnow() - timedelta(hours=hours_back)).strftime("%Y-%m-%dT%H:%M:%S"),
                    datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S"),
                )
                .select(["precipitationCal"])
            )

            if coll.size().getInfo() == 0:
                log.warning("No recent IMERG data available")
                return None

            def per_image(img):
                d = img.date().format("YYYY-MM-dd HH:mm")
                stats = img.reduceRegions(
                    collection=ee_fc,
                    reducer=ee.Reducer.mean(),
                    scale=5000,
                )
                return stats.map(lambda f: f.set({"datetime": d}).setGeometry(None))

            flat = coll.map(per_image).flatten()
            info = flat.getInfo()
            feats = info.get("features", [])

            rows = []
            for f in feats:
                props = f.get("properties", {})
                row = {
                    "district": props.get("district"),
                    "datetime": props.get("datetime"),
                    "precip_mmhr": props.get("precipitationCal", 0),
                }
                rows.append(row)

            df = pd.DataFrame(rows)
            if not df.empty:
                df["datetime"] = pd.to_datetime(df["datetime"])
                df["precip_mmhr"] = pd.to_numeric(df["precip_mmhr"], errors="coerce")
                df["precip_mmhr"] = df["precip_mmhr"] * 1000
            return df

        except Exception as e:
            log.warning(f"Real-time precip fetch failed: {e}")
            return None

    def fetch_realtime_ndvi(self, ee_fc) -> Optional[pd.DataFrame]:
        try:
            import ee
            today = datetime.utcnow().strftime("%Y-%m-%d")
            past_16 = (datetime.utcnow() - timedelta(days=16)).strftime("%Y-%m-%d")

            img = (
                ee.ImageCollection(constants.REALTIME_DATASETS["ndvi"]["collection"])
                .filterBounds(ee_fc)
                .filterDate(past_16, today)
                .sort("system:time_start", False)
                .first()
            )

            if img is None:
                return None

            stats = img.reduceRegions(
                collection=ee_fc,
                reducer=ee.Reducer.mean(),
                scale=250,
            )

            info = stats.getInfo()
            rows = []
            for f in info.get("features", []):
                props = f.get("properties", {})
                rows.append({
                    "district": props.get("district"),
                    "ndvi": props.get("NDVI", 0) * 0.0001,
                })

            return pd.DataFrame(rows)

        except Exception as e:
            log.warning(f"Real-time NDVI fetch failed: {e}")
            return None

    def fetch_realtime_lst(self, ee_fc) -> Optional[pd.DataFrame]:
        try:
            import ee
            today = datetime.utcnow().strftime("%Y-%m-%d")
            past_day = (datetime.utcnow() - timedelta(days=1)).strftime("%Y-%m-%d")

            img = (
                ee.ImageCollection(constants.REALTIME_DATASETS["lst"]["collection"])
                .filterBounds(ee_fc)
                .filterDate(past_day, today)
                .sort("system:time_start", False)
                .first()
            )

            if img is None:
                return None

            stats = img.reduceRegions(
                collection=ee_fc,
                reducer=ee.Reducer.mean(),
                scale=1000,
            )

            info = stats.getInfo()
            rows = []
            for f in info.get("features", []):
                props = f.get("properties", {})
                rows.append({
                    "district": props.get("district"),
                    "lst_k": props.get("LST_Day_1km", 0) * 0.02,
                })

            return pd.DataFrame(rows)

        except Exception as e:
            log.warning(f"Real-time LST fetch failed: {e}")
            return None

    def fetch_realtime_soil_moisture(self, ee_fc) -> Optional[pd.DataFrame]:
        try:
            import ee
            today = datetime.utcnow().strftime("%Y-%m-%d")
            past_3 = (datetime.utcnow() - timedelta(days=3)).strftime("%Y-%m-%d")

            img = (
                ee.ImageCollection(constants.REALTIME_DATASETS["soil_moisture"]["collection"])
                .filterBounds(ee_fc)
                .filterDate(past_3, today)
                .sort("system:time_start", False)
                .first()
            )

            if img is None:
                return None

            stats = img.reduceRegions(
                collection=ee_fc,
                reducer=ee.Reducer.mean(),
                scale=9000,
            )

            info = stats.getInfo()
            rows = []
            for f in info.get("features", []):
                props = f.get("properties", {})
                rows.append({
                    "district": props.get("district"),
                    "soil_moisture": props.get("sm_surface", 0),
                })

            return pd.DataFrame(rows)

        except Exception as e:
            log.warning(f"Real-time SM fetch failed: {e}")
            return None

    def fetch_all(self, district: str = "") -> Dict[str, bool]:
        """Try GEE if available, fall back to cached snapshot files."""
        try:
            import ee
            from geospatial.boundaries import DistrictBoundaries
            bounds_mgr = DistrictBoundaries()
            gdf = bounds_mgr.gdf
            if district:
                gdf = gdf[gdf[CFG.district_col] == district]
            if not gdf.empty:
                ee_fc = ee.FeatureCollection(gdf.to_json())
                return self.update(ee_fc)
        except Exception as e:
            log.warning(f"GEE fetch_all failed (will check snapshots): {e}")

        snapshot_dir = Path(CFG.realtime_dir)
        if snapshot_dir.exists():
            for var in ["precip", "ndvi", "lst", "soil_moisture"]:
                var_files = sorted(snapshot_dir.glob(f"{var}_*.parquet"))
                if var_files:
                    self._latest_data[var] = pd.read_parquet(var_files[-1])
        return self.get_status()

    def get_status(self) -> Dict[str, Any]:
        return {
            "precip": len(self._latest_data.get("precip", [])) > 0,
            "ndvi": len(self._latest_data.get("ndvi", [])) > 0,
            "lst": len(self._latest_data.get("lst", [])) > 0,
            "soil_moisture": len(self._latest_data.get("soil_moisture", [])) > 0,
            "update_count": self._update_count,
            "timestamp": datetime.utcnow().isoformat(),
            "running": self._running,
        }

    def update(self, ee_fc) -> Dict[str, Any]:
        status = {}
        self._update_count += 1

        precip = self.fetch_realtime_precip(ee_fc)
        if precip is not None:
            self._latest_data["precip"] = precip
            status["precip"] = True
        else:
            status["precip"] = False

        ndvi = self.fetch_realtime_ndvi(ee_fc)
        if ndvi is not None:
            self._latest_data["ndvi"] = ndvi
            status["ndvi"] = True
        else:
            status["ndvi"] = False

        lst = self.fetch_realtime_lst(ee_fc)
        if lst is not None:
            self._latest_data["lst"] = lst
            status["lst"] = True
        else:
            status["lst"] = False

        sm = self.fetch_realtime_soil_moisture(ee_fc)
        if sm is not None:
            self._latest_data["soil_moisture"] = sm
            status["soil_moisture"] = True
        else:
            status["soil_moisture"] = False

        status["update_count"] = self._update_count
        status["timestamp"] = datetime.utcnow().isoformat()
        return status

    def get_latest(self, var: str) -> Optional[pd.DataFrame]:
        return self._latest_data.get(var)

    def save_snapshot(self):
        ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        snapshot_dir = Path(CFG.realtime_dir)
        snapshot_dir.mkdir(parents=True, exist_ok=True)

        for var, df in self._latest_data.items():
            if df is not None and not df.empty:
                path = snapshot_dir / f"{var}_{ts}.parquet"
                df.to_parquet(path)

        log.info(f"Realtime snapshot saved at {ts}")
