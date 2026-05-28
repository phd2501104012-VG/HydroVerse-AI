"""Check if hazard severity patterns differ year to year with CMIP6 data."""
import sys; sys.path.insert(0, r"D:\cri")
import pandas as pd
import numpy as np
from pathlib import Path
from config import CFG
from hazards.detection import HazardDetector
from forecasting.daily_forecast import DailyForecastEngine
from functools import reduce

cache_dir = Path(CFG.cache_dir)
district = "Agar Malwa"

# 1. Load ERA5 historical data
era5_map = {"tmax": "temperature_2m_max", "tmin": "temperature_2m_min", "precip": "total_precipitation_sum"}
parts = []
for var in ["tmax","tmin","precip"]:
    files = list(cache_dir.glob(f"era5_{var}_*.parquet"))
    df = pd.read_parquet(files[0])
    col = era5_map[var]
    sub = df[df["district"] == district][["date", col]].copy().rename(columns={col: var})
    sub["date"] = pd.to_datetime(sub["date"]); sub = sub.set_index("date")
    if var in ("tmax","tmin"): sub[var] = sub[var] - 273.15
    if var == "precip": sub[var] = sub[var] * 1000
    parts.append(sub)
data = pd.concat(parts, axis=1).sort_index()

# 2. Load CMIP6 ensemble
ens = pd.read_parquet(cache_dir / "cmip6_ensemble.parquet")
fe = DailyForecastEngine(); fe.set_ensemble(ens)

# 3. Generate forecasts
forecasts = {}
for target in ["tmax","tmin","precip"]:
    fc = fe.generate_daily_to_2040(data, target, district)
    if fc is not None and not fc.empty:
        forecasts[(district, target)] = fc

# 4. Build forecast panel
fc_parts = []
for target in ["tmax","tmin","precip"]:
    fc = forecasts.get((district, target))
    if fc is not None and not fc.empty:
        p = fc[["date","forecast"]].copy().rename(columns={"forecast": target})
        p["date"] = pd.to_datetime(p["date"]); fc_parts.append(p)
fc_panel = reduce(lambda a,b: a.merge(b, on="date", how="outer"), fc_parts)
fc_panel = fc_panel.set_index("date").sort_index()

# 5. Merge + detect
ext = data.copy().combine_first(fc_panel)
det = HazardDetector()
haz = det.detect_all(ext, district=district)
fc_idx = fc_panel.index.intersection(haz.index)
haz_fc = haz.loc[fc_idx]

# 6. Compare same date range across multiple years
print("=== July 1-7 across years ===")
sev_cols = ["flood_severity","drought_severity","heatwave_severity","agri_severity","compound_severity"]
for year in [2026, 2028, 2030, 2035]:
    jul = haz_fc[haz_fc.index.year == year]
    jul = jul[jul.index.month == 7]
    jul = jul[jul.index.day <= 7]
    print(f"\n{year} July 1-7:")
    for c in sev_cols:
        v = pd.to_numeric(jul[c], errors="coerce")
        if len(v) > 0:
            print(f"  {c}: mean={v.mean():.1f}, max={v.max():.1f}")

# 7. Check if ALL years have identical monthly means for each hazard
print("\n=== Year-to-year monthly mean comparison ===")
for c in sev_cols:
    monthly = haz_fc.groupby([haz_fc.index.year, haz_fc.index.month])[c].mean()
    vals_by_year = {}
    for (year, month), val in monthly.items():
        vals_by_year.setdefault(month, []).append(val)
    # For each month, check how many distinct values across years
    diffs = []
    for month, vals in sorted(vals_by_year.items()):
        unique_vals = len(set(round(v,2) for v in vals))
        if unique_vals > 1:
            diffs.append(month)
    print(f"  {c}: {len(diffs)}/12 months have year-to-year variation")
