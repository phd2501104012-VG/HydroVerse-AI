from typing import Optional, Dict, List, Any
import numpy as np
import pandas as pd

from utils.logger import log


class ForecastEnsemble:
    def __init__(self):
        self._members: Dict[str, pd.DataFrame] = {}

    def add_member(self, name: str, forecast: pd.DataFrame):
        self._members[name] = forecast

    def compute_ensemble(self, method: str = "mean") -> pd.DataFrame:
        if not self._members:
            return pd.DataFrame()

        dates = None
        for name, fc in self._members.items():
            fc = fc.copy()
            fc["date"] = pd.to_datetime(fc["date"])
            fc = fc.set_index("date")
            if dates is None:
                dates = fc.index
            dates = dates.union(fc.index)
            self._members[name] = fc

        result = pd.DataFrame(index=sorted(set(dates)))
        result.index.name = "date"

        for name, fc in self._members.items():
            result[f"{name}_forecast"] = fc["forecast"]

        if method == "mean":
            result["ensemble_mean"] = result.mean(axis=1)
        elif method == "median":
            result["ensemble_mean"] = result.median(axis=1)
        elif method == "weighted":
            weights = self._compute_weights()
            result["ensemble_mean"] = sum(
                result[f"{name}_forecast"] * w for name, w in weights.items()
                if f"{name}_forecast" in result.columns
            )

        result["ensemble_std"] = result[[c for c in result.columns if c.endswith("_forecast")]].std(axis=1)
        result["ensemble_p10"] = result[[c for c in result.columns if c.endswith("_forecast")]].quantile(0.1, axis=1)
        result["ensemble_p90"] = result[[c for c in result.columns if c.endswith("_forecast")]].quantile(0.9, axis=1)

        return result.reset_index()

    def _compute_weights(self) -> Dict[str, float]:
        n = len(self._members)
        return {name: 1.0 / n for name in self._members}

    def compute_spread(self) -> pd.DataFrame:
        if not self._members:
            return pd.DataFrame()
        ensemble = self.compute_ensemble()
        if ensemble.empty:
            return ensemble
        spread = pd.DataFrame({
            "date": ensemble["date"],
            "ensemble_mean": ensemble["ensemble_mean"],
            "ensemble_std": ensemble["ensemble_std"],
            "ensemble_range": ensemble["ensemble_p90"] - ensemble["ensemble_p10"],
            "cv": ensemble["ensemble_std"] / (ensemble["ensemble_mean"].abs() + 1e-10),
        })
        return spread
