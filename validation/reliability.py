from typing import Dict, List, Optional, Any
import numpy as np
import pandas as pd

from utils.logger import log


class ReliabilityAnalyzer:
    def __init__(self):
        pass

    def analyze(self, forecasts: np.ndarray, observations: np.ndarray, n_bins: int = 10) -> Dict:
        bins = np.linspace(0, 1, n_bins + 1)
        bin_centers = (bins[:-1] + bins[1:]) / 2
        results = []

        for i in range(n_bins):
            mask = (forecasts >= bins[i]) & (forecasts < bins[i + 1])
            n_samples = mask.sum()
            if n_samples > 0:
                obs_freq = observations[mask].mean()
                fc_avg = forecasts[mask].mean()
                results.append({
                    "bin": i,
                    "bin_start": bins[i],
                    "bin_end": bins[i + 1],
                    "bin_center": bin_centers[i],
                    "n_samples": int(n_samples),
                    "forecast_frequency": float(fc_avg),
                    "observed_frequency": float(obs_freq),
                    "bias": float(fc_avg - obs_freq),
                })
            else:
                results.append({
                    "bin": i,
                    "bin_start": bins[i],
                    "bin_end": bins[i + 1],
                    "bin_center": bin_centers[i],
                    "n_samples": 0,
                    "forecast_frequency": float(bin_centers[i]),
                    "observed_frequency": float("nan"),
                    "bias": float("nan"),
                })

        df = pd.DataFrame(results)
        reliability_error = df["bias"].dropna().abs().mean()
        sharpness = df["forecast_frequency"].std()
        resolution = df["observed_frequency"].dropna().std()

        return {
            "reliability_table": df,
            "reliability_error": float(reliability_error),
            "sharpness": float(sharpness),
            "resolution": float(resolution),
            "reliability_index": float(np.mean(df["bias"].dropna() ** 2)),
        }

    def reliability_diagram_data(self, forecasts: np.ndarray, observations: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
        result = self.analyze(forecasts, observations, n_bins)
        return result["reliability_table"]
