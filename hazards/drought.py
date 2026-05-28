import numpy as np
import pandas as pd

from scipy.stats import gamma, norm

from config import CFG
from hazards.categories import HazardClassifier


class DroughtDetector:
    def __init__(self):
        self.classifier = HazardClassifier()

    def compute_spi(self, precip_daily: pd.Series, scale_months: int = 3) -> pd.Series:
        p_monthly = precip_daily.resample("MS").sum()
        p_n = p_monthly.rolling(scale_months, min_periods=scale_months).sum()
        spi_monthly = pd.Series(np.nan, index=p_n.index)
        for month in range(1, 13):
            mask = (p_n.index.month == month) & p_n.notna()
            if mask.sum() < 5:
                continue
            x = p_n[mask]
            p_zero = (x == 0).mean()
            x_pos = x[x > 0]
            if len(x_pos) < 5:
                continue
            try:
                shape_p, _, scale_p = gamma.fit(x_pos, floc=0)
                cdf_pos = gamma.cdf(x, shape_p, loc=0, scale=scale_p)
                cdf = np.where(x == 0, p_zero / 2, p_zero + (1 - p_zero) * cdf_pos)
                cdf = np.clip(cdf, 1e-6, 1 - 1e-6)
                spi_monthly.loc[mask] = norm.ppf(cdf)
            except Exception:
                pass
        return spi_monthly.reindex(precip_daily.index, method="ffill")

    def compute_imd_deficiency(self, precip_daily: pd.Series) -> pd.Series:
        doy_clim = precip_daily.groupby(precip_daily.index.dayofyear).mean()
        annual_norm = doy_clim.sum()
        annual_actual = precip_daily.resample("YE").sum()
        deficiency = ((annual_actual - annual_norm) / annual_norm * 100).reindex(precip_daily.index, method="ffill")
        return deficiency

    def detect(self, df: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame(index=df.index)
        sev = pd.Series(0.0, index=df.index)
        t = CFG.hazard

        # Compute SPI if not already present
        spi_col = "spi_3m" if "spi_3m" in df.columns else None
        if spi_col is None and "precip" in df.columns:
            self._spi_1m = self.compute_spi(df["precip"].fillna(0), scale_months=1)
            self._spi_3m = self.compute_spi(df["precip"].fillna(0), scale_months=3)
            self._spi_6m = self.compute_spi(df["precip"].fillna(0), scale_months=6)
            df["spi_1m"] = self._spi_1m
            df["spi_3m"] = self._spi_3m
            df["spi_6m"] = self._spi_6m
            spi_col = "spi_3m"

        if spi_col and spi_col in df.columns:
            spi = df[spi_col]
            # SPI-based severity contribution (70% of total)
            sev += ((-spi).clip(lower=0, upper=3) / 3 * 70).fillna(0)

            # SPI Category
            spi_cat = pd.Series("Normal", index=df.index)
            spi_cat = np.where(spi <= -2.0, "Extreme Drought",
                       np.where(spi <= -1.5, "Severe Drought",
                       np.where(spi <= -1.0, "Moderate Drought",
                       np.where(spi < 0, "Mild Drought", "Normal"))))
            out["drought_spi_category"] = spi_cat

            # Sustained drought detection (25+/30 days below threshold)
            in_drought = (spi < t.drought_spi_threshold).astype(int)
            sustained = in_drought.rolling(t.drought_sustained_days, min_periods=20).sum() >= 25
            out["drought_event"] = sustained.astype(int).fillna(0)
        else:
            out["drought_event"] = 0
            out["drought_spi_category"] = "Normal"

        # IMD Rainfall Deficiency
        if "precip" in df.columns:
            deficiency = self.compute_imd_deficiency(df["precip"].fillna(0))
            imd_class = pd.Series("Normal", index=df.index)
            imd_class = np.where(deficiency < -50, "Severe Drought",
                         np.where(deficiency < -26, "Moderate Drought",
                         np.where(deficiency < 0, "Watch", "Normal")))
            out["drought_imd_class"] = imd_class
            out["drought_deficiency"] = deficiency.round(1)
            # Add IMD deficiency to severity (30% of total)
            sev += ((-deficiency).clip(lower=0, upper=50) / 50 * 10).fillna(0)

        # VHI contribution (20%)
        if "vhi" in df.columns:
            sev += ((50 - df["vhi"].clip(0, 50)) / 50 * 20).fillna(0)

        # Soil moisture anomaly (10%)
        if "soil_moisture_anom" in df.columns:
            sev += ((-df["soil_moisture_anom"]).clip(lower=0, upper=3) / 3 * 10).fillna(0)

        out["drought_severity"] = sev.clip(0, 100).round(1)
        out["drought_class"] = self.classifier.classify(out["drought_severity"])
        return out
