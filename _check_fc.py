import pandas as pd
from pathlib import Path

fc_dir = Path("D:\\cri\\exports\\forecasts")

tmax_dists = set()
precip_dists = set()
for f in fc_dir.glob("*.csv"):
    parts = f.stem.split("_")
    if "tmax" in parts:
        dist = f.stem.replace("_tmax_era5_forecast", "")
        tmax_dists.add(dist)
    elif "precip" in parts:
        dist = f.stem.replace("_precip_era5_forecast", "")
        precip_dists.add(dist)

print(f"Districts with tmax forecasts: {len(tmax_dists)}")
print(f"Districts with precip forecasts: {len(precip_dists)}")
print(f"Districts missing tmax: {sorted(precip_dists - tmax_dists)}")
print()

# Check date range of forecasts
print("Sample forecast date ranges:")
for d in ["Agar Malwa", "Bhopal", "Indore"]:
    for var in ["tmax", "precip"]:
        fp = fc_dir / f"{d}_{var}_era5_forecast.csv"
        if fp.exists():
            df = pd.read_csv(fp)
            print(f"  {d}/{var}: {df['date'].min()} to {df['date'].max()} ({len(df)} rows)")
