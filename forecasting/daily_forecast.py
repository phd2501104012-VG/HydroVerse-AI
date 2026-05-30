from typing import Optional, Dict, List
import pandas as pd
import numpy as np

from config import CFG
from forecasting.ml_forecast import MLForecastGenerator
from forecasting.cmip6_forecast import CMIP6ForecastGenerator
from forecasting.blending import ForecastBlender
from utils.logger import log


class DailyForecastEngine:
    def __init__(self, cmip6_ensemble: Optional[pd.DataFrame] = None):
        self.ml = MLForecastGenerator()
        self.cmip6 = CMIP6ForecastGenerator(cmip6_ensemble)
        self.blender = ForecastBlender()
        self._forecasts: Dict[tuple, pd.DataFrame] = {}

    def set_ensemble(self, df: pd.DataFrame):
        self.cmip6.set_ensemble(df)

    def set_ensemble(self, df: pd.DataFrame):
        self.cmip6.set_ensemble(df)

    def generate_daily_to_2040(
        self,
        panel: pd.DataFrame,
        target: str,
        district: str,
    ) -> pd.DataFrame:
        ml_horizon = CFG.forecasting.ml_horizon_days
        blend_days = CFG.forecasting.blend_days
        end_year = CFG.forecasting.future_end_year

        last_date = panel.index.max()
        today_str = (last_date + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        end_str = f"{end_year}-12-31"

        log.info(f"  Generating daily forecast: {district}/{target}")
        ml_fc = self.ml.generate_forecast(panel, target, district, horizon_days=ml_horizon)

        # Try CMIP6 for the full range when ML fails
        if ml_fc.empty:
            log.warning(f"    ML forecast failed, trying CMIP6 for {district}/{target}")
            cmip6_fc = self.cmip6.generate(district, target, today_str, end_str)
            if not cmip6_fc.empty:
                log.info(f"    CMIP6 found: {len(cmip6_fc)} days for {district}/{target}")
                cmip6_fc["source"] = "CMIP6"
                cmip6_fc["district"] = district
                cmip6_fc["target"] = target
                self._forecasts[(district, target)] = cmip6_fc
                return cmip6_fc
            log.warning(f"    CMIP6 also empty, using climatology fallback")
            return self.blender.climatology_fallback(panel, target, pd.Timestamp(today_str), pd.Timestamp(end_str))

        long_start = (last_date + pd.Timedelta(days=ml_horizon + 1)).strftime("%Y-%m-%d")
        cmip6_fc = self.cmip6.generate(district, target, long_start, end_str)

        if cmip6_fc.empty:
            log.warning(f"    CMIP6 empty, using climatology fallback")
            long_fc = self.blender.climatology_fallback(panel, target, pd.Timestamp(long_start), pd.Timestamp(end_str))
        else:
            long_fc = cmip6_fc

        combined = self.blender.blend(ml_fc, long_fc)
        combined["district"] = district
        combined["target"] = target

        self._forecasts[(district, target)] = combined
        return combined

    def generate_all(
        self,
        panels: Dict[str, pd.DataFrame],
        targets: Optional[List[str]] = None,
    ) -> Dict[tuple, pd.DataFrame]:
        targets = targets or ["tmax", "precip"]
        results = {}
        for district, panel in panels.items():
            for target in targets:
                if target not in panel.columns:
                    continue
                fc = self.generate_daily_to_2040(panel, target, district)
                if not fc.empty:
                    results[(district, target)] = fc
        return results
