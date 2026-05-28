from typing import List, Optional, Dict, Any
from pathlib import Path

import pandas as pd
import numpy as np
import time

from config import CFG, constants

try:
    import ee
    _HAS_EE = True
except ImportError:
    ee = None
    _HAS_EE = False
from utils.logger import log
from utils.gee_utils import cmip6_chunked_fetch, cache_key


class CMIP6Loader:
    def __init__(self, ee_fc=None):
        self.ee_fc = ee_fc
        self._cache_dir = Path(CFG.cache_dir)
        self.variables = constants.CMIP6_VARIABLES

    def set_feature_collection(self, ee_fc):
        self.ee_fc = ee_fc

    def fetch_model_year(
        self,
        model: str,
        scenario: str,
        year: int,
        variables: Optional[List[str]] = None,
        chunk_days: Optional[int] = None,
    ) -> pd.DataFrame:
        if self.ee_fc is None:
            raise RuntimeError("EE FeatureCollection not set")

        variables = variables or list(self.variables.keys())
        bands = []
        for v in variables:
            if v in self.variables:
                bands.append(self.variables[v].get("band", v))
            else:
                bands.append(v)

        # Dynamically compute chunk size to stay under GEE's 5000 flatten limit
        if chunk_days is None:
            try:
                n_districts = self.ee_fc.size().getInfo()
            except Exception:
                n_districts = 51
            max_chunk = 4500 // max(n_districts, 1)  # safe buffer
            chunk_days = min(max(max_chunk, 10), 180)

        all_chunks = []
        cur = pd.Timestamp(f"{year}-01-01")
        end_dt = pd.Timestamp(f"{year+1}-01-01")
        while cur < end_dt:
            nxt = min(cur + pd.Timedelta(days=chunk_days), end_dt)
            start = cur.strftime("%Y-%m-%d")
            end = nxt.strftime("%Y-%m-%d")
            try:
                df = cmip6_chunked_fetch(self.ee_fc, model, scenario, bands, start, end)
                if not df.empty:
                    all_chunks.append(df)
                time.sleep(0.3)
            except Exception as e:
                log.warning(f"CMIP6 fetch failed: {model} {start}..{end}: {e}")
            cur = nxt

        if not all_chunks:
            return pd.DataFrame()

        combined = pd.concat(all_chunks, ignore_index=True).drop_duplicates(["district", "date"])
        combined["model"] = model
        combined["scenario"] = scenario

        rename = {k: v["alias"] for k, v in self.variables.items() if k in combined.columns}
        combined.rename(columns=rename, inplace=True)

        for v in variables:
            alias = self.variables[v]["alias"]
            if alias in combined.columns:
                combined[alias] = combined[alias].astype(float)

        combined["tmean_proj"] = combined.get("tmean_proj", np.nan) - 273.15
        combined["tmax_proj"] = combined.get("tmax_proj", np.nan) - 273.15
        combined["tmin_proj"] = combined.get("tmin_proj", np.nan) - 273.15
        if "precip_proj" in combined.columns:
            combined["precip_proj"] = combined["precip_proj"] * 86400

        return combined

    def fetch_all_models(
        self,
        models: Optional[List[str]] = None,
        scenario: Optional[str] = None,
        start_year: Optional[int] = None,
        end_year: Optional[int] = None,
        variables: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        models = models or CFG.forecasting.cmip6_models
        scenario = scenario or CFG.forecasting.ssp_scenario
        start_year = start_year or CFG.forecasting.future_start_year
        end_year = end_year or CFG.forecasting.future_end_year
        # Only fetch tmax/tmin/precip by default — skip hurs, sfcWind, rsds
        variables = variables or ["tasmax", "tasmin", "pr"]

        all_data = []
        for model in models:
            for year in range(start_year, end_year + 1):
                log.info(f"CMIP6: {model} {year}")
                df = self.fetch_model_year(model, scenario, year, variables=variables)
                if not df.empty:
                    all_data.append(df)
                cp = self._cache_dir / f"cmip6_{model}_{scenario}_{year}.parquet"
                if not df.empty:
                    df.to_parquet(cp)
                    log.info(f"Cached: {cp.name} ({len(df)} rows)")

        if not all_data:
            return pd.DataFrame()
        return pd.concat(all_data, ignore_index=True)

    def compute_ensemble(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return pd.DataFrame()

        vars_to_agg = [v["alias"] for v in self.variables.values()]
        vars_to_agg = [v for v in vars_to_agg if v in df.columns and df[v].notna().any()]

        if "scenario" not in df.columns:
            df["scenario"] = CFG.forecasting.ssp_scenario

        agg = {}
        for v in vars_to_agg:
            agg[v] = ["mean", "std", lambda s: s.quantile(0.1), lambda s: s.quantile(0.9)]

        grouped = df.groupby(["district", "date", "scenario"]).agg(agg)
        grouped.columns = [f"{var}_{stat}" for var, stat in grouped.columns]
        grouped = grouped.reset_index()
        grouped.columns = [c.replace("<lambda_0>", "p10").replace("<lambda_1>", "p90") for c in grouped.columns]
        return grouped
