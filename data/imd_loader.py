import glob
import os
from pathlib import Path
from typing import List, Optional, Dict, Tuple, Union
from datetime import datetime

import numpy as np
import pandas as pd
import xarray as xr
import dask.array as da
import geopandas as gpd
from shapely.geometry import mapping

from config import CFG, IMDConfig
from utils.logger import log


class IMDLoader:
    def __init__(self, config: Optional[IMDConfig] = None):
        self.cfg = config or CFG.imd
        self._datasets: Dict[str, xr.Dataset] = {}
        self._merged: Dict[str, xr.Dataset] = {}

    def _find_files(self, directory: str, pattern: str) -> List[str]:
        dir_path = Path(directory)
        if not dir_path.exists():
            log.warning(f"IMD directory not found: {directory}")
            return []
        files = sorted(dir_path.glob(pattern))
        log.info(f"Found {len(files)} IMD files in {directory}")
        return [str(f) for f in files]

    def load_variable(
        self,
        var_name: str,
        directory: Optional[str] = None,
        pattern: Optional[str] = None,
        start_year: Optional[int] = None,
        end_year: Optional[int] = None,
        use_dask: bool = True,
    ) -> Optional[xr.Dataset]:
        cfg_map = {
            "precip": (self.cfg.precip_dir, self.cfg.precip_pattern, self.cfg.precip_var),
            "tmax": (self.cfg.tmax_dir, self.cfg.tmax_pattern, self.cfg.tmax_var),
            "tmin": (self.cfg.tmin_dir, self.cfg.tmin_pattern, self.cfg.tmin_var),
            "tmean": (self.cfg.tmean_dir, self.cfg.tmean_pattern, self.cfg.tmean_var),
        }
        if var_name not in cfg_map:
            raise ValueError(f"Unknown IMD variable: {var_name}. Choose from {list(cfg_map.keys())}")

        dir_path, pat, var = cfg_map[var_name]
        directory = directory or dir_path
        pattern = pattern or pat
        start_year = start_year or self.cfg.start_year
        end_year = end_year or self.cfg.end_year

        files = self._find_files(directory, pattern)
        files = [f for f in files if self._file_year_in_range(f, start_year, end_year)]

        if not files:
            log.warning(f"No IMD {var_name} files found for {start_year}-{end_year}")
            return None

        log.info(f"Loading {len(files)} IMD {var_name} files with dask={use_dask}")
        chunks = {"lat": 100, "lon": 100, "time": -1} if use_dask else None

        try:
            ds = xr.open_mfdataset(
                files,
                combine="by_coords",
                chunks=chunks if use_dask else None,
                engine="netcdf4",
                parallel=use_dask,
            )
            log.info(f"Loaded {var_name}: {ds.sizes}")
            self._datasets[var_name] = ds
            return ds
        except Exception as e:
            log.error(f"Failed to load IMD {var_name}: {e}")
            return None

    def _file_year_in_range(self, filepath: str, start: int, end: int) -> bool:
        import re
        years = re.findall(r"(\d{4})", Path(filepath).stem)
        if years:
            year = int(years[-1])
            return start <= year <= end
        return True

    def load_all_variables(
        self,
        start_year: Optional[int] = None,
        end_year: Optional[int] = None,
        use_dask: bool = True,
    ) -> Dict[str, xr.Dataset]:
        results = {}
        for var in ["precip", "tmax", "tmin", "tmean"]:
            ds = self.load_variable(var, start_year=start_year, end_year=end_year, use_dask=use_dask)
            if ds is not None:
                results[var] = ds
        return results

    def clip_to_geometry(
        self,
        ds: xr.Dataset,
        geometry,
    ) -> xr.Dataset:
        import rioxarray
        ds = ds.rio.set_spatial_dims("lon", "lat", inplace=True)
        ds = ds.rio.write_crs("EPSG:4326", inplace=True)
        try:
            clipped = ds.rio.clip([mapping(geometry)], from_disk=True)
            log.info(f"Clipped to geometry: {clipped.sizes}")
            return clipped
        except Exception as e:
            log.error(f"Clipping failed: {e}")
            return ds

    def clip_to_district(
        self,
        ds: xr.Dataset,
        district_gdf: gpd.GeoDataFrame,
        district_name: str,
    ) -> xr.Dataset:
        district_col = "district" if "district" in district_gdf.columns else CFG.district_col
        row = district_gdf[district_gdf[district_col] == district_name]
        if row.empty:
            raise ValueError(f"District {district_name} not found in column '{district_col}'")
        return self.clip_to_geometry(ds, row.geometry.iloc[0])

    def extract_timeseries(
        self,
        ds: xr.Dataset,
        var_name: str,
        method: str = "mean",
    ) -> pd.Series:
        if method == "mean":
            data = ds[var_name].mean(dim=["lat", "lon"])
        elif method == "median":
            data = ds[var_name].median(dim=["lat", "lon"])
        elif method == "max":
            data = ds[var_name].max(dim=["lat", "lon"])
        elif method == "min":
            data = ds[var_name].min(dim=["lat", "lon"])
        else:
            raise ValueError(f"Unknown method: {method}")

        if hasattr(data, "compute"):
            data = data.compute()
        if isinstance(data, xr.DataArray):
            times = pd.to_datetime(data.time.values)
            values = data.values
        else:
            times = data.index
            values = data.values

        return pd.Series(values, index=times, name=var_name)

    def extract_pixel_timeseries(
        self,
        ds: xr.Dataset,
        var_name: str,
        lat: float,
        lon: float,
    ) -> pd.Series:
        data = ds[var_name].sel(lat=lat, lon=lon, method="nearest")
        if hasattr(data, "compute"):
            data = data.compute()
        return pd.Series(data.values, index=pd.to_datetime(data.time.values), name=var_name)

    def extract_pixel_grid(
        self,
        ds: xr.Dataset,
        var_name: str,
    ) -> pd.DataFrame:
        if hasattr(ds[var_name], "compute"):
            data = ds[var_name].compute()
        else:
            data = ds[var_name]

        lats = data.lat.values
        lons = data.lon.values
        times = pd.to_datetime(data.time.values)
        grid = data.values

        rows = []
        for t_idx in range(len(times)):
            for lat_idx in range(len(lats)):
                for lon_idx in range(len(lons)):
                    rows.append({
                        "time": times[t_idx],
                        "lat": lats[lat_idx],
                        "lon": lons[lon_idx],
                        var_name: float(grid[t_idx, lat_idx, lon_idx]),
                    })
        return pd.DataFrame(rows)

    def _resolve_var_name(self, short_name: str) -> str:
        mapping = {
            "precip": self.cfg.precip_var,
            "tmax": self.cfg.tmax_var,
            "tmin": self.cfg.tmin_var,
            "tmean": self.cfg.tmean_var,
        }
        return mapping.get(short_name, short_name)

    def district_timeseries(
        self,
        var_name: str,
        district_gdf: gpd.GeoDataFrame,
        district_name: str,
        method: str = "mean",
    ) -> pd.Series:
        actual_var = self._resolve_var_name(var_name)
        ds = self._datasets.get(actual_var)
        if ds is None:
            ds = self.load_variable(var_name)
            if ds is None:
                raise RuntimeError(f"IMD {var_name} not loaded")
        clipped = self.clip_to_district(ds, district_gdf, district_name)
        return self.extract_timeseries(clipped, actual_var, method)

    def multi_district_timeseries(
        self,
        var_name: str,
        district_gdf: gpd.GeoDataFrame,
        method: str = "mean",
    ) -> pd.DataFrame:
        ds = self._datasets.get(var_name)
        if ds is None:
            ds = self.load_variable(var_name)
            if ds is None:
                raise RuntimeError(f"IMD {var_name} not loaded")

        results = {}
        district_col = "district" if "district" in district_gdf.columns else CFG.district_col
        for d in district_gdf[district_col].unique():
            try:
                clipped = self.clip_to_district(ds, district_gdf, d)
                ts = self.extract_timeseries(clipped, var_name, method)
                results[d] = ts
            except Exception as e:
                log.warning(f"Failed to extract {d}: {e}")
        return pd.DataFrame(results)

    def compute_imd_climatology(
        self,
        var_name: str,
        start_year: int = 1981,
        end_year: int = 2010,
    ) -> pd.DataFrame:
        ds = self._datasets.get(var_name)
        if ds is None:
            ds = self.load_variable(var_name)
            if ds is None:
                raise RuntimeError(f"IMD {var_name} not loaded")

        da = ds[var_name]
        if hasattr(da, "compute"):
            da = da.compute()

        da_clim = da.sel(time=slice(f"{start_year}-01-01", f"{end_year}-12-31"))
        monthly_clim = da_clim.groupby("time.month").mean("time")
        daily_clim = da_clim.groupby("time.dayofyear").mean("time")

        lats = da.lat.values
        lons = da.lon.values

        df_monthly = pd.DataFrame(
            monthly_clim.values.reshape(12, -1).T,
            columns=[f"month_{m}" for m in range(1, 13)],
        )
        df_doy = pd.DataFrame(
            daily_clim.values.reshape(365, -1).T,
            columns=[f"doy_{d}" for d in range(1, 366)],
        )
        return pd.concat([df_monthly, df_doy], axis=1)
