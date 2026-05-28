from typing import List, Optional, Dict, Any
from pathlib import Path

import pandas as pd
import numpy as np

from config import CFG, constants

try:
    import ee
    _HAS_EE = True
except ImportError:
    ee = None
    _HAS_EE = False
from utils.logger import log
from utils.gee_utils import chunked_fetch, cache_key


class ERA5Loader:
    def __init__(self, ee_fc=None):
        self.ee_fc = ee_fc
        self.datasets = constants.ERA5_DATASETS
        self._cache_dir = Path(CFG.cache_dir)

    def set_feature_collection(self, ee_fc):
        self.ee_fc = ee_fc

    def fetch_variable(
        self,
        var_name: str,
        start: Optional[str] = None,
        end: Optional[str] = None,
        force_refresh: bool = False,
    ) -> pd.DataFrame:
        if var_name not in self.datasets:
            raise ValueError(f"Unknown ERA5 variable: {var_name}")

        cfg = self.datasets[var_name]
        start = start or CFG.hist_start
        end = end or CFG.hist_end

        cp = self._cache_dir / f"{cache_key('era5', var_name, start, end)}.parquet"
        if cp.exists() and not force_refresh:
            log.info(f"ERA5 cache hit: {var_name}")
            df = pd.read_parquet(cp)
        else:
            if self.ee_fc is None:
                raise RuntimeError("EE FeatureCollection not set and no cache found. Call set_feature_collection() or run the pipeline first.")
            log.info(f"Fetching ERA5 {var_name} ({start} -> {end})")
            df = chunked_fetch(
                self.ee_fc, cfg["collection"], [cfg["band"]],
                start, end,
            )
            if not df.empty:
                df.to_parquet(cp, index=False)

        if df.empty:
            return pd.DataFrame(columns=["district", "date", "value"])

        band = cfg["band"]
        scale = cfg["scale"]
        offset = cfg["offset"]
        vmin, vmax = cfg["valid_range"]

        sub = df[["district", "date", band]].copy()
        sub["value"] = sub[band] * scale + offset
        sub = sub[(sub["value"] >= vmin) & (sub["value"] <= vmax)]
        sub = sub.dropna(subset=["value"]).drop(columns=[band])
        sub = sub.sort_values(["district", "date"]).reset_index(drop=True)
        return sub

    def fetch_all(
        self,
        variables: Optional[List[str]] = None,
        start: Optional[str] = None,
        end: Optional[str] = None,
        force_refresh: bool = False,
    ) -> Dict[str, pd.DataFrame]:
        variables = variables or list(self.datasets.keys())
        results = {}
        for v in variables:
            results[v] = self.fetch_variable(v, start, end, force_refresh)
        return results
