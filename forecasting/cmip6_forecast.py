from typing import Optional, Dict, List
import pandas as pd
import numpy as np

from config import CFG, constants
from utils.logger import log


class CMIP6ForecastGenerator:
    def __init__(self, cmip6_ensemble: Optional[pd.DataFrame] = None):
        self._ensemble = cmip6_ensemble

    def set_ensemble(self, df: pd.DataFrame):
        self._ensemble = df

    def generate(
        self,
        district: str,
        target: str,
        start: str,
        end: str,
    ) -> pd.DataFrame:
        if self._ensemble is None or self._ensemble.empty:
            return pd.DataFrame()

        cmip6_map = {
            "tmax": "tmax_proj",
            "tmin": "tmin_proj",
            "tmean": "tmean_proj",
            "precip": "precip_proj",
        }
        cmip6_var = cmip6_map.get(target)
        if cmip6_var is None:
            return pd.DataFrame()

        mean_col = f"{cmip6_var}_mean"
        p10_col = f"{cmip6_var}_p10"
        p90_col = f"{cmip6_var}_p90"

        if mean_col not in self._ensemble.columns:
            return pd.DataFrame()

        sub = self._ensemble[
            (self._ensemble["district"] == district)
            & (pd.to_datetime(self._ensemble["date"]) >= pd.to_datetime(start))
            & (pd.to_datetime(self._ensemble["date"]) <= pd.to_datetime(end))
        ].copy()

        if sub.empty:
            return pd.DataFrame()

        result = pd.DataFrame({
            "date": pd.to_datetime(sub["date"].values),
            "forecast": sub[mean_col].values,
        })

        if p10_col in sub.columns:
            result["lower"] = sub[p10_col].values
            result["upper"] = sub[p90_col].values
        else:
            std = sub[mean_col].std() if len(sub) > 1 else sub[mean_col].mean() * 0.1
            result["lower"] = sub[mean_col].values - 1.64 * std
            result["upper"] = sub[mean_col].values + 1.64 * std

        return result

    def generate_all_districts(
        self,
        target: str,
        districts: List[str],
        start: str,
        end: str,
    ) -> Dict[str, pd.DataFrame]:
        results = {}
        for d in districts:
            fc = self.generate(d, target, start, end)
            if not fc.empty:
                results[d] = fc
        return results
