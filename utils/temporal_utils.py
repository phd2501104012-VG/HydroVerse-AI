import numpy as np
import pandas as pd
from typing import Optional


def add_calendar_features(df) -> pd.DataFrame:
    df = df.copy()
    d = df.index if isinstance(df.index, pd.DatetimeIndex) else pd.to_datetime(df.index)
    df["doy"] = d.dayofyear
    df["month"] = d.month
    df["year"] = d.year
    df["doy_sin"] = np.sin(2 * np.pi * df["doy"] / 365.25)
    df["doy_cos"] = np.cos(2 * np.pi * df["doy"] / 365.25)
    df["is_monsoon"] = df["month"].isin([6, 7, 8, 9]).astype(int)
    df["is_winter"] = df["month"].isin([12, 1, 2]).astype(int)
    df["is_premonsoon"] = df["month"].isin([3, 4, 5]).astype(int)
    return df


def compute_cdd_cwd(precip, wet_threshold_mm=1.0):
    is_wet = (precip >= wet_threshold_mm).astype(int)
    dry_streak = (1 - is_wet).groupby(is_wet.cumsum()).cumsum()
    wet_streak = is_wet.groupby((is_wet == 0).cumsum()).cumsum()
    return pd.DataFrame({"cdd": dry_streak.values, "cwd": wet_streak.values}, index=precip.index)


def compute_anomaly(series, clim_end=None):
    if clim_end is None:
        from config import CFG
        clim_end = pd.Timestamp(CFG.climatology_end)
    train = series[series.index <= clim_end]
    doy_train = train.index.dayofyear
    clim_mean = pd.Series(train.values).groupby(doy_train).mean()
    clim_std = pd.Series(train.values).groupby(doy_train).std()
    anom = (series - series.index.dayofyear.map(clim_mean)) / (series.index.dayofyear.map(clim_std) + 1e-9)
    return anom


def consecutive_run(series, min_run=2):
    groups = (series != series.shift()).cumsum()
    return series.groupby(groups).cumsum() >= min_run
