from typing import Dict, Optional
import pandas as pd
import numpy as np

from config import CFG, DataSource
from hazards.flood import FloodDetector
from hazards.drought import DroughtDetector
from hazards.heatwave import HeatwaveDetector
from hazards.agri_stress import AgriStressDetector
from hazards.compound import CompoundHazardEngine
from utils.logger import log


class HazardDetector:
    def __init__(self):
        self.flood = FloodDetector()
        self.drought = DroughtDetector()
        self.heatwave = HeatwaveDetector()
        self.agri = AgriStressDetector()
        self.compound = CompoundHazardEngine()

    def compute_climate_indices(self, df: pd.DataFrame) -> pd.DataFrame:
        result = df.copy()
        if not isinstance(result.index, pd.DatetimeIndex):
            if "date" in result.columns:
                result["date"] = pd.to_datetime(result["date"])
                result = result.set_index("date")
            elif "time" in result.columns:
                result = result.set_index("time")
            elif "index" in result.columns:
                result = result.set_index("index")
            else:
                result.index = pd.to_datetime(result.index, errors="ignore")

        if "precip" in result.columns:
            precip = result["precip"].fillna(0)
            if "spi_1m" not in result.columns:
                result["spi_1m"] = self.drought.compute_spi(precip, 1)
            if "spi_3m" not in result.columns:
                result["spi_3m"] = self.drought.compute_spi(precip, 3)
            if "spi_6m" not in result.columns:
                result["spi_6m"] = self.drought.compute_spi(precip, 6)

        clim_end = pd.Timestamp(CFG.climatology_end)

        # TCI (Temperature Condition Index) — computed from tmax alone, no NDVI needed
        if "tmax" in result.columns and "tci" not in result.columns:
            train_tmax = result["tmax"][result.index <= clim_end]
            tmax_doy = result.index.dayofyear
            tmax_clim_95 = train_tmax.groupby(train_tmax.index.dayofyear).agg(lambda x: x.quantile(0.95))
            tmax_clim_05 = train_tmax.groupby(train_tmax.index.dayofyear).agg(lambda x: x.quantile(0.05))
            result["tci"] = 100 * (tmax_doy.map(tmax_clim_95) - result["tmax"]) / (tmax_doy.map(tmax_clim_95) - tmax_doy.map(tmax_clim_05) + 1e-9)
            result["tci"] = result["tci"].clip(0, 100)

        # VCI (Vegetation Condition Index) — requires actual NDVI from MODIS/GEE
        if "ndvi" in result.columns and "vci" not in result.columns:
            train_ndvi = result["ndvi"][result.index <= clim_end]
            doy_clim = pd.DataFrame({
                "doy": train_ndvi.index.dayofyear,
                "ndvi": train_ndvi.values,
            }).groupby("doy")["ndvi"].agg(lambda x: x.quantile(0.05))
            doy_clim_95 = pd.DataFrame({
                "doy": train_ndvi.index.dayofyear,
                "ndvi": train_ndvi.values,
            }).groupby("doy")["ndvi"].agg(lambda x: x.quantile(0.95))
            ndvi_doy = result.index.dayofyear
            result["vci"] = 100 * (result["ndvi"] - ndvi_doy.map(doy_clim)) / (ndvi_doy.map(doy_clim_95) - ndvi_doy.map(doy_clim) + 1e-9)
            result["vci"] = result["vci"].clip(0, 100)

        # VHI (Vegetation Health Index) = VCI + TCI combined — needs VCI (i.e. NDVI)
        if "vci" in result.columns and "tci" in result.columns and "vhi" not in result.columns:
            result["vhi"] = (0.5 * result["vci"] + 0.5 * result["tci"]).clip(0, 100)

        if "cdd" not in result.columns:
            from utils.temporal_utils import compute_cdd_cwd
            if "precip" in result.columns:
                runs = compute_cdd_cwd(result["precip"].fillna(0))
                result["cdd"] = runs["cdd"]
                result["cwd"] = runs["cwd"]

        if "tmax" in result.columns:
            for v in ["tmax", "tmin", "precip", "ndvi", "soil_moisture"]:
                if v in result.columns and f"{v}_anom" not in result.columns:
                    from utils.temporal_utils import compute_anomaly
                    result[f"{v}_anom"] = compute_anomaly(result[v])

        return result

    def detect_all(self, df: pd.DataFrame, district: str | None = None) -> pd.DataFrame:
        df = self.compute_climate_indices(df)

        flood = self.flood.detect(df)
        drought = self.drought.detect(df)
        heatwave = self.heatwave.detect(df, district=district)
        agri = self.agri.detect(df)
        compound = self.compound.compute(df, flood, drought, heatwave, agri)

        # Keep computed climate indices from df
        climate_cols = [c for c in df.columns if c not in ["date"] and c not in ["tmax","tmin","precip"]]
        climate_indices = df[climate_cols]

        result = pd.concat([flood, drought, heatwave, agri, compound, climate_indices], axis=1)
        # Drop duplicate columns (detector outputs take precedence)
        result = result.loc[:, ~result.columns.duplicated()]
        return result
