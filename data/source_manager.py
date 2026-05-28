from enum import Enum
from typing import Optional, Dict, List, Any, Tuple
from functools import wraps

import pandas as pd
import numpy as np

from config import CFG, DataSource
from utils.logger import log


class DataSourceManager:
    def __init__(self):
        self._active_source: DataSource = CFG.active_data_source
        self._era5_loader = None
        self._imd_loader = None
        self._cmip6_loader = None
        self._data_cache: Dict[str, pd.DataFrame] = {}

    @property
    def active_source(self) -> DataSource:
        return self._active_source

    @active_source.setter
    def active_source(self, source: DataSource):
        log.info(f"Data source switched to: {source.value}")
        self._active_source = source

    def set_era5_loader(self, loader):
        self._era5_loader = loader

    def set_imd_loader(self, loader):
        self._imd_loader = loader

    def set_cmip6_loader(self, loader):
        self._cmip6_loader = loader

    def get_precipitation(
        self,
        district: str,
        start: Optional[str] = None,
        end: Optional[str] = None,
    ) -> Optional[pd.Series]:
        return self._get_variable("precip", district, start, end)

    def get_tmax(
        self,
        district: str,
        start: Optional[str] = None,
        end: Optional[str] = None,
    ) -> Optional[pd.Series]:
        return self._get_variable("tmax", district, start, end)

    def get_tmin(
        self,
        district: str,
        start: Optional[str] = None,
        end: Optional[str] = None,
    ) -> Optional[pd.Series]:
        return self._get_variable("tmin", district, start, end)

    def get_tmean(
        self,
        district: str,
        start: Optional[str] = None,
        end: Optional[str] = None,
    ) -> Optional[pd.Series]:
        return self._get_variable("tmean", district, start, end)

    def _get_variable(
        self,
        var: str,
        district: str,
        start: Optional[str] = None,
        end: Optional[str] = None,
    ) -> Optional[pd.Series]:
        source = self._determine_source(var)
        key = f"{source.value}_{var}_{district}_{start}_{end}"

        if key in self._data_cache:
            return self._data_cache[key]

        result = None
        if source == DataSource.ERA5 and self._era5_loader is not None:
            df = self._era5_loader.fetch_variable(var)
            if not df.empty:
                sub = df[df["district"] == district]
                if not sub.empty:
                    result = sub.set_index("date")["value"]
        elif source == DataSource.IMD and self._imd_loader is not None:
            era5_map = {"precip": "precip", "tmax": "tmax", "tmin": "tmin", "tmean": "tmean"}
            imd_var = era5_map.get(var)
            if imd_var:
                from data.imd_loader import IMDLoader
                try:
                    ds = self._imd_loader.load_variable(imd_var)
                    if ds is not None:
                        from geospatial.boundaries import DistrictBoundaries
                        db = DistrictBoundaries()
                        gdf = db.load()
                        ts = self._imd_loader.district_timeseries(imd_var, gdf, district)
                        if ts is not None:
                            result = ts
                except Exception as e:
                    log.warning(f"IMD fetch failed for {var}/{district}: {e}")
                    if self._era5_loader:
                        df = self._era5_loader.fetch_variable(var)
                        sub = df[df["district"] == district]
                        if not sub.empty:
                            result = sub.set_index("date")["value"]

        if result is not None:
            self._data_cache[key] = result
        return result

    def _determine_source(self, var: str) -> DataSource:
        if self._active_source != DataSource.AUTO:
            if self._active_source == DataSource.ERA5 or self._active_source == DataSource.IMD:
                return self._active_source

        if self._era5_loader is not None:
            try:
                from pathlib import Path
                cp = Path(self._era5_loader._cache_dir) / f"era5_{var}_2000-01-01_2025-10-31.parquet"
                if cp.exists():
                    return DataSource.ERA5
            except Exception:
                pass

        return DataSource.IMD

    def get_district_timeseries(
        self,
        district: str,
        variables: Optional[List[str]] = None,
        start: Optional[str] = None,
        end: Optional[str] = None,
    ) -> Optional[pd.DataFrame]:
        variables = variables or ["tmax", "tmin", "precip"]
        series_list = []
        for var in variables:
            s = self._get_variable(var, district, start, end)
            if s is not None:
                s.name = var
                series_list.append(s)
        if not series_list:
            return None
        result = pd.concat(series_list, axis=1)
        result.index = pd.to_datetime(result.index)
        result.index.name = "date"
        result = result.sort_index().reset_index()
        return result

    def get_available_sources(self) -> List[str]:
        sources = []
        if self._era5_loader is not None:
            sources.append(DataSource.ERA5.value)
        if self._imd_loader is not None:
            sources.append(DataSource.IMD.value)
        return sources

    def clear_cache(self):
        self._data_cache.clear()
        log.info("Data source cache cleared")
