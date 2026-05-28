import numpy as np
import pandas as pd
from typing import Optional, Tuple, List

from utils.logger import log


class ForecastBlender:
    def __init__(self, blend_days: int = 30):
        self.blend_days = blend_days

    def blend(
        self,
        short_term: pd.DataFrame,
        long_term: pd.DataFrame,
        ml_weight: float = 1.0,
    ) -> pd.DataFrame:
        if short_term.empty:
            return long_term.copy()
        if long_term.empty:
            return short_term.copy()

        short_term = short_term.copy()
        long_term = long_term.copy()
        short_term["date"] = pd.to_datetime(short_term["date"])
        long_term["date"] = pd.to_datetime(long_term["date"])

        seam_date = short_term["date"].max()
        overlap_start = seam_date - pd.Timedelta(days=self.blend_days // 2)
        blend_indices = long_term[long_term["date"] >= overlap_start].index[:self.blend_days]

        if len(blend_indices) > 0:
            weights = np.linspace(1.0, 0.0, min(len(blend_indices), self.blend_days))
            for i, idx in enumerate(blend_indices):
                if i >= len(weights):
                    break
                ml_tail = short_term[short_term["date"] >= overlap_start - pd.Timedelta(days=7)]["forecast"].mean() if i < 3 else long_term.loc[idx, "forecast"]
                long_term.loc[idx, "forecast"] = weights[i] * ml_weight * ml_tail + (1 - weights[i]) * long_term.loc[idx, "forecast"]
                long_term.loc[idx, "lower"] = weights[i] * short_term["lower"].tail(1).mean() + (1 - weights[i]) * long_term.loc[idx, "lower"]
                long_term.loc[idx, "upper"] = weights[i] * short_term["upper"].tail(1).mean() + (1 - weights[i]) * long_term.loc[idx, "upper"]

        short_term["source"] = "ML"
        long_term["source"] = long_term.get("source", pd.Series(["CMIP6"] * len(long_term), index=long_term.index))
        long_term.loc[long_term.index[:len(blend_indices)], "source"] = "Blend"

        combined = pd.concat([short_term, long_term], ignore_index=True).sort_values("date").reset_index(drop=True)
        return combined

    def climatology_fallback(
        self,
        panel: pd.DataFrame,
        target: str,
        start: pd.Timestamp,
        end: pd.Timestamp,
        slope: float = 0.0,
    ) -> pd.DataFrame:
        if target not in panel.columns:
            return pd.DataFrame()

        s = panel[target].dropna()
        if len(s) < 5:
            return pd.DataFrame()
        doy_clim = s.groupby(s.index.dayofyear).agg(["mean", "std"])

        if slope == 0:
            annual = s.resample("YE").mean().dropna()
            if len(annual) >= 5:
                years = annual.index.year.values.astype(float)
                slope = np.polyfit(years, annual.values, 1)[0]

        dates = pd.date_range(start, end, freq="D")
        yr_ref = float(np.mean(s.index.year))

        rows = []
        for d in dates:
            doy = d.dayofyear
            if doy in doy_clim.index:
                base = float(doy_clim.loc[doy, "mean"]) + slope * (d.year - yr_ref)
                sd = float(doy_clim.loc[doy, "std"])
            else:
                base = float(s.mean())
                sd = float(s.std())
            rows.append({
                "date": d,
                "forecast": base,
                "lower": base - 1.96 * sd,
                "upper": base + 1.96 * sd,
                "source": "Climatology",
            })

        return pd.DataFrame(rows)
