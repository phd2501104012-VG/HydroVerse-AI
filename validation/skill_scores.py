from typing import Dict, List, Optional, Any
import numpy as np
import pandas as pd

from utils.logger import log


class SkillScoreCalculator:
    def __init__(self):
        pass

    def brier_score(self, forecasts: np.ndarray, observations: np.ndarray) -> float:
        return float(np.mean((forecasts - observations) ** 2))

    def brier_skill_score(self, forecasts: np.ndarray, observations: np.ndarray, climatology: Optional[np.ndarray] = None) -> float:
        if climatology is None:
            climatology = np.ones_like(observations) * np.mean(observations)
        bs_ref = self.brier_score(climatology, observations)
        bs_fc = self.brier_score(forecasts, observations)
        return float(1 - bs_fc / bs_ref) if bs_ref > 0 else 0

    def crps(self, ensemble: np.ndarray, observation: float) -> float:
        n = len(ensemble)
        sorted_ens = np.sort(ensemble)
        crps_val = 0.0
        for i in range(n - 1):
            diff = sorted_ens[i + 1] - sorted_ens[i]
            if observation < sorted_ens[i]:
                crps_val += diff * (i + 1) ** 2
            elif observation > sorted_ens[i + 1]:
                crps_val += diff * (n - i - 1) ** 2
            else:
                crps_val += diff * ((i + 1 - n) ** 2)
        crps_val /= n ** 2
        return float(crps_val)

    def continuous_ranked_probability_skill_score(
        self,
        forecasts: np.ndarray,
        observations: np.ndarray,
        climatology_ensemble: Optional[np.ndarray] = None,
    ) -> float:
        if climatology_ensemble is None:
            return 0.0

        crps_fc = np.mean([self.crps(forecasts[i], observations[i]) for i in range(len(observations))])
        crps_clim = np.mean([self.crps(climatology_ensemble[i], observations[i]) for i in range(len(observations))])

        return float(1 - crps_fc / crps_clim) if crps_clim > 0 else 0

    def reliability_curve(self, forecasts: np.ndarray, observations: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
        bins = np.linspace(0, 1, n_bins + 1)
        bin_centers = (bins[:-1] + bins[1:]) / 2
        observed_freqs = []

        for i in range(n_bins):
            mask = (forecasts >= bins[i]) & (forecasts < bins[i + 1])
            if mask.sum() > 0:
                observed_freqs.append(np.mean(observations[mask]))
            else:
                observed_freqs.append(np.nan)

        return pd.DataFrame({
            "bin_center": bin_centers,
            "observed_frequency": observed_freqs,
            "sample_count": [((forecasts >= bins[i]) & (forecasts < bins[i + 1])).sum() for i in range(n_bins)],
        })

    def roc_curve(self, forecasts: np.ndarray, observations: np.ndarray, n_thresholds: int = 20) -> pd.DataFrame:
        thresholds = np.linspace(0, 1, n_thresholds + 1)
        fpr_list = []
        tpr_list = []

        for thr in thresholds:
            pred = (forecasts >= thr).astype(int)
            tp = ((pred == 1) & (observations == 1)).sum()
            fp = ((pred == 1) & (observations == 0)).sum()
            fn = ((pred == 0) & (observations == 1)).sum()
            tn = ((pred == 0) & (observations == 0)).sum()

            tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
            fpr_list.append(fpr)
            tpr_list.append(tpr)

        return pd.DataFrame({"threshold": thresholds, "false_positive_rate": fpr_list, "true_positive_rate": tpr_list})

    def quantile_skill_score(self, forecasts: np.ndarray, observations: np.ndarray, quantiles: List[float] = None) -> Dict:
        quantiles = quantiles or [0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]
        fc_quantiles = np.quantile(forecasts, quantiles)
        obs_quantiles = np.quantile(observations, quantiles)

        return {
            "quantile": quantiles,
            "forecast_quantiles": fc_quantiles.tolist(),
            "observed_quantiles": obs_quantiles.tolist(),
            "qss": (1 - np.mean((fc_quantiles - obs_quantiles) ** 2) / np.var(observations)).tolist(),
        }
